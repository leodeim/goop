package goop

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"time"
)

// DefaultRouteTimeout is the limit on the jev call when Router.Timeout is 0.
const DefaultRouteTimeout = 2 * time.Second

// Route is one provider and model a Router can send a run to.
type Route struct {
	Name     string // the option jev picks, e.g. "fast"
	When     string // which requests belong here; jev reads it as the option's description
	Provider Provider
	Model    string
}

// Routed is the first event of Router.Run: the route the run goes to.
type Routed struct {
	Route         string
	Probabilities map[string]float64 // jev's probability per route, nil when jev was not asked
	Err           error              // why jev's answer could not be used, the run then went to the default route
}

func (Routed) isEvent() {}

// Router asks jev which route fits a request, then runs Agent on that route's provider and model.
// The latest user message goes to the Jev URL as the state, so with the hosted API it leaves the
// machine before any model sees it. Point Jev.URL at a laya-mlx server to keep it local.
type Router struct {
	Jev    *Jev
	Routes []Route
	// Default is the route used when jev fails or is unsure. "" means the first route.
	Default string
	// MinConfidence is the lowest probability jev may give its pick, below it the run goes to Default.
	MinConfidence float64
	Timeout       time.Duration // limit on the jev call, 0 = DefaultRouteTimeout
	// Agent is the template for every run. Leave Provider and Model empty, the route sets them.
	Agent Agent
}

const routeQuestion = "Which model should handle this request?"

// Run picks a route for the conversation and runs Agent on it. It yields Routed first, then
// everything Agent.Run yields. The route holds for the whole run, tool calls included.
// Thinking blocks are dropped from prior because a different provider cannot read them.
func (r *Router) Run(ctx context.Context, prior []Message, input ...Block) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		msgs := prior
		if len(input) > 0 {
			msgs = append(slices.Clip(prior), Message{Role: User, Blocks: input})
		}
		route, routed, err := r.pick(ctx, msgs)
		if err != nil {
			yield(nil, err)
			return
		}
		if !yield(routed, nil) {
			return
		}

		agent := r.Agent
		agent.Provider, agent.Model = route.Provider, route.Model
		for ev, err := range agent.Run(ctx, dropThinking(prior), input...) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

// Pick chooses the route for msgs (the conversation including the new user message) without
// running it. It errors only on an invalid Router or a done ctx, a failing jev falls back to Default.
func (r *Router) Pick(ctx context.Context, msgs []Message) (Routed, error) {
	_, routed, err := r.pick(ctx, msgs)
	return routed, err
}

func (r *Router) pick(ctx context.Context, msgs []Message) (Route, Routed, error) {
	if err := r.validate(); err != nil {
		return Route{}, Routed{}, err
	}

	candidates := r.Routes
	if slices.ContainsFunc(msgs, hasDocument) {
		candidates = slices.DeleteFunc(slices.Clone(candidates), func(rt Route) bool {
			_, chat := rt.Provider.(*OpenAI)
			return chat
		})
		if len(candidates) == 0 {
			return Route{}, Routed{}, errors.New("goop: the conversation has a document and no route can read it")
		}
	}
	fallback := candidates[0]
	if i := slices.IndexFunc(candidates, func(rt Route) bool { return rt.Name == r.Default }); i >= 0 {
		fallback = candidates[i]
	}
	fall := func(probs map[string]float64, err error) (Route, Routed, error) {
		return fallback, Routed{Route: fallback.Name, Probabilities: probs, Err: err}, nil
	}

	text := lastUserText(msgs)
	if len(candidates) == 1 || text == "" {
		return fall(nil, nil)
	}

	criteria := make(map[string]string, len(candidates))
	for _, rt := range candidates {
		criteria[rt.Name] = rt.When
	}
	askCtx, cancel := context.WithTimeout(ctx, cmp.Or(r.Timeout, DefaultRouteTimeout))
	defer cancel()
	answers, err := r.Jev.Ask(askCtx, text, map[string]JevQuestion{
		"route": {Type: "choice", Instructions: routeQuestion, Criteria: criteria},
	})
	if ctx.Err() != nil {
		return Route{}, Routed{}, ctx.Err()
	}
	if err != nil {
		return fall(nil, fmt.Errorf("goop: jev route: %w", err))
	}

	answer := answers["route"]
	i := slices.IndexFunc(candidates, func(rt Route) bool { return rt.Name == answer.Choice })
	if i < 0 {
		return fall(answer.Probabilities, fmt.Errorf("goop: jev picked unknown route %q", answer.Choice))
	}
	p, ok := answer.Probabilities[answer.Choice]
	if !ok && r.MinConfidence > 0 {
		return fall(answer.Probabilities, fmt.Errorf("goop: jev gave no probability for route %q", answer.Choice))
	}
	if p < r.MinConfidence {
		return fall(answer.Probabilities, nil)
	}

	return candidates[i], Routed{Route: answer.Choice, Probabilities: answer.Probabilities}, nil
}

func (r *Router) validate() error {
	switch {
	case r.Jev == nil:
		return errors.New("goop: router needs a Jev")
	case len(r.Routes) == 0:
		return errors.New("goop: router needs at least one route")
	case r.Agent.Provider != nil || r.Agent.Model != "":
		return errors.New("goop: router Agent must leave Provider and Model to the routes")
	case r.MinConfidence < 0 || r.MinConfidence > 1:
		return fmt.Errorf("goop: router MinConfidence %v is outside [0, 1]", r.MinConfidence)
	}

	names := make(map[string]bool, len(r.Routes))
	for _, rt := range r.Routes {
		switch {
		case rt.Name == "" || rt.When == "":
			return fmt.Errorf("goop: route %q needs a Name and a When", rt.Name)
		case rt.Provider == nil || rt.Model == "":
			return fmt.Errorf("goop: route %q needs a Provider and a Model", rt.Name)
		case names[rt.Name]:
			return fmt.Errorf("goop: two routes named %q", rt.Name)
		}
		names[rt.Name] = true
	}
	if r.Default != "" && !names[r.Default] {
		return fmt.Errorf("goop: default route %q is not one of the routes", r.Default)
	}

	return nil
}

// lastUserText is the text of the latest user message that has any, tool results aside.
func lastUserText(msgs []Message) string {
	for _, m := range slices.Backward(msgs) {
		if m.Role != User {
			continue
		}
		var parts []string
		for _, b := range m.Blocks {
			if t, ok := b.(Text); ok && t.Text != "" {
				parts = append(parts, t.Text)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	return ""
}

func hasDocument(m Message) bool {
	return slices.ContainsFunc(m.Blocks, func(b Block) bool {
		_, ok := b.(Document)
		return ok
	})
}

// dropThinking removes Thinking from prior, except from the assistant message of an unfinished
// tool loop, which Anthropic rejects without it.
func dropThinking(prior []Message) []Message {
	keep := -1
	if n := len(prior); n >= 2 && prior[n-1].Role == User && slices.ContainsFunc(prior[n-1].Blocks, isToolResult) {
		keep = n - 2
	}

	out := make([]Message, 0, len(prior))
	for i, m := range prior {
		if i != keep {
			m.Blocks = slices.DeleteFunc(slices.Clone(m.Blocks), isThinking)
		}
		if len(m.Blocks) > 0 { // both APIs reject a message with empty content
			out = append(out, m)
		}
	}
	return out
}

func isThinking(b Block) bool {
	_, ok := b.(Thinking)
	return ok
}

func isToolResult(b Block) bool {
	_, ok := b.(ToolResult)
	return ok
}
