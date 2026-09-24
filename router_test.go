package goop

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// routeServer answers the route question with choice and probs, counts calls and records the
// last request body.
func routeServer(t *testing.T, choice string, probs map[string]float64) (*httptest.Server, *map[string]any, *int) {
	t.Helper()
	var last map[string]any
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &last); err != nil {
			t.Errorf("request body: %v", err)
		}
		json.NewEncoder(w).Encode(map[string]any{"answers": map[string]any{
			"route": map[string]any{"type": "choice", "choice": choice, "probabilities": probs},
		}})
	}))
	t.Cleanup(server.Close)
	return server, &last, &calls
}

func endReply(text string) Reply {
	return Reply{Message: Message{Role: Assistant, Blocks: []Block{Text{Text: text}}}, StopReason: StopEnd}
}

type testRouter struct {
	*Router
	fast, deep *fakeProvider
}

func newTestRouter(url string) testRouter {
	fast := &fakeProvider{replies: []Reply{endReply("fast answer")}}
	deep := &fakeProvider{replies: []Reply{endReply("deep answer")}}
	return testRouter{
		Router: &Router{
			Jev: &Jev{URL: url, HTTP: HTTP{MaxRetries: -1}},
			Routes: []Route{
				{Name: "fast", When: "short factual questions", Provider: fast, Model: "small"},
				{Name: "deep", When: "multi-step reasoning and code", Provider: deep, Model: "big"},
			},
			Default: "deep",
			Agent:   Agent{System: "sys", MaxTurns: 3},
		},
		fast: fast,
		deep: deep,
	}
}

func collect(t *testing.T, events func(func(Event, error) bool)) (Routed, *Done, error) {
	t.Helper()
	var routed Routed
	var done *Done
	first := true
	for ev, err := range events {
		if err != nil {
			return routed, done, err
		}
		switch e := ev.(type) {
		case Routed:
			if !first {
				t.Fatal("Routed was not the first event")
			}
			routed = e
		case Done:
			done = &e
		}
		first = false
	}
	return routed, done, nil
}

func TestRouterRunsJevPick(t *testing.T) {
	server, last, _ := routeServer(t, "fast", map[string]float64{"fast": 0.8, "deep": 0.2})
	r := newTestRouter(server.URL)

	prior := []Message{
		{Role: User, Blocks: []Block{Text{Text: "earlier question"}}},
		{Role: Assistant, Blocks: []Block{Text{Text: "earlier answer"}}},
	}
	routed, done, err := collect(t, r.Run(t.Context(), prior, Text{Text: "What is the capital"}, Text{Text: "of France?"}))
	if err != nil {
		t.Fatal(err)
	}
	if routed.Route != "fast" || routed.Probabilities["fast"] != 0.8 || routed.Err != nil {
		t.Fatalf("routed = %+v", routed)
	}
	if len(r.deep.calls) != 0 || len(r.fast.calls) != 1 {
		t.Fatalf("calls: fast %d, deep %d", len(r.fast.calls), len(r.deep.calls))
	}
	req := r.fast.calls[0]
	if req.Model != "small" || req.System != "sys" || len(req.Messages) != 3 {
		t.Fatalf("request = %+v", req)
	}
	if done == nil || done.Messages[len(done.Messages)-1].Blocks[0].(Text).Text != "fast answer" {
		t.Fatalf("done = %+v", done)
	}

	if (*last)["state"] != "What is the capital\nof France?" {
		t.Fatalf("state sent = %v", (*last)["state"])
	}
	q := (*last)["questions"].(map[string]any)["route"].(map[string]any)
	criteria := q["criteria"].(map[string]any)
	if q["type"] != "choice" || criteria["fast"] != "short factual questions" || criteria["deep"] != "multi-step reasoning and code" {
		t.Fatalf("question sent = %v", q)
	}
}

func TestRouterFallsBack(t *testing.T) {
	tests := []struct {
		name          string
		choice        string
		probs         map[string]float64
		minConfidence float64
		wantErr       string
	}{
		{"low confidence", "fast", map[string]float64{"fast": 0.55, "deep": 0.45}, 0.6, ""},
		{"unknown route", "medium", map[string]float64{"medium": 0.9}, 0, `unknown route "medium"`},
		{"no probability", "fast", nil, 0.6, `no probability for route "fast"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, _, _ := routeServer(t, tt.choice, tt.probs)
			r := newTestRouter(server.URL)
			r.MinConfidence = tt.minConfidence

			routed, _, err := collect(t, r.Run(t.Context(), nil, Text{Text: "hi"}))
			if err != nil {
				t.Fatal(err)
			}
			if routed.Route != "deep" || len(r.deep.calls) != 1 {
				t.Fatalf("routed = %+v, deep calls = %d", routed, len(r.deep.calls))
			}
			if tt.wantErr == "" && routed.Err != nil || tt.wantErr != "" && (routed.Err == nil || !strings.Contains(routed.Err.Error(), tt.wantErr)) {
				t.Fatalf("routed.Err = %v, want %q", routed.Err, tt.wantErr)
			}
		})
	}
}

func TestRouterFailsOpen(t *testing.T) {
	var apiErr *APIError
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	t.Cleanup(down.Close)
	r := newTestRouter(down.URL)
	routed, done, err := collect(t, r.Run(t.Context(), nil, Text{Text: "hi"}))
	if err != nil || done == nil {
		t.Fatalf("err = %v, done = %v", err, done)
	}
	if routed.Route != "deep" || !errors.As(routed.Err, &apiErr) || apiErr.Status != http.StatusServiceUnavailable {
		t.Fatalf("routed = %+v", routed)
	}

	// The router's own timeout, not the caller's context, cuts off a slow jev.
	release := make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(slow.Close)
	t.Cleanup(func() { close(release) })
	r = newTestRouter(slow.URL)
	r.Timeout = 20 * time.Millisecond
	routed, err = r.Pick(t.Context(), []Message{{Role: User, Blocks: []Block{Text{Text: "hi"}}}})
	if err != nil || routed.Route != "deep" || !errors.Is(routed.Err, context.DeadlineExceeded) {
		t.Fatalf("routed = %+v, err = %v", routed, err)
	}
}

func TestRouterCancelledIsError(t *testing.T) {
	server, _, _ := routeServer(t, "fast", map[string]float64{"fast": 1})
	r := newTestRouter(server.URL)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := r.Pick(ctx, []Message{{Role: User, Blocks: []Block{Text{Text: "hi"}}}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled instead of a fallback", err)
	}
}

func TestRouterSkipsJev(t *testing.T) {
	server, _, calls := routeServer(t, "fast", map[string]float64{"fast": 1})

	// A document rules out chat completions, which leaves one route and nothing to ask.
	r := newTestRouter(server.URL)
	r.Routes[1].Provider = &OpenAI{}
	routed, err := r.Pick(t.Context(), []Message{{Role: User, Blocks: []Block{Text{Text: "summarise"}, Document{Data: []byte("%PDF")}}}})
	if err != nil || routed.Route != "fast" || routed.Probabilities != nil {
		t.Fatalf("routed = %+v, err = %v", routed, err)
	}

	r.Routes[0].Provider = &OpenAI{}
	if _, err := r.Pick(t.Context(), []Message{{Role: User, Blocks: []Block{Document{Data: []byte("%PDF")}}}}); err == nil {
		t.Fatal("picked a route that cannot read the document")
	}

	// No user text: nothing for jev to decide on.
	r = newTestRouter(server.URL)
	if routed, err := r.Pick(t.Context(), []Message{{Role: User, Blocks: []Block{Image{URL: "https://x/cat.png"}}}}); err != nil || routed.Route != "deep" {
		t.Fatalf("routed = %+v, err = %v", routed, err)
	}

	if *calls != 0 {
		t.Fatalf("jev called %d times", *calls)
	}
}

func TestRouterStopsAfterRouted(t *testing.T) {
	server, _, _ := routeServer(t, "fast", map[string]float64{"fast": 1})
	r := newTestRouter(server.URL)
	for range r.Run(t.Context(), nil, Text{Text: "hi"}) {
		break
	}
	if len(r.fast.calls) != 0 {
		t.Fatal("provider called after the caller stopped at Routed")
	}
}

func TestRouterDropsThinking(t *testing.T) {
	server, _, _ := routeServer(t, "fast", map[string]float64{"fast": 1})
	r := newTestRouter(server.URL)
	thinking := Thinking{Raw: json.RawMessage(`{"type":"thinking","thinking":"hm","signature":"s"}`)}
	prior := []Message{
		{Role: User, Blocks: []Block{Text{Text: "q1"}}},
		{Role: Assistant, Blocks: []Block{thinking, Text{Text: "a1"}}},
		{Role: User, Blocks: []Block{Text{Text: "q2"}}},
		{Role: Assistant, Blocks: []Block{thinking}}, // cut off while thinking, nothing left once dropped
	}
	if _, _, err := collect(t, r.Run(t.Context(), prior, Text{Text: "q3"})); err != nil {
		t.Fatal(err)
	}
	sent := r.fast.calls[0].Messages
	if len(sent) != 4 {
		t.Fatalf("sent %d messages, want 4: %+v", len(sent), sent)
	}
	for _, m := range sent {
		for _, b := range m.Blocks {
			if _, ok := b.(Thinking); ok {
				t.Fatalf("thinking sent to a new route: %+v", m)
			}
		}
	}
	if _, ok := prior[1].Blocks[0].(Thinking); !ok {
		t.Fatal("prior was modified")
	}
}

func TestDropThinkingKeepsOpenToolLoop(t *testing.T) {
	thinking := Thinking{Raw: json.RawMessage(`{"type":"thinking"}`)}
	prior := []Message{
		{Role: User, Blocks: []Block{Text{Text: "q"}}},
		{Role: Assistant, Blocks: []Block{thinking, ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{}`)}}},
		{Role: User, Blocks: []Block{ToolResult{ToolUseID: "t1", Content: "ok"}}},
	}
	got := dropThinking(prior)
	if len(got) != 3 || len(got[1].Blocks) != 2 {
		t.Fatalf("got = %+v", got)
	}
}

func TestRouterValidate(t *testing.T) {
	p := &fakeProvider{}
	tests := []struct {
		name   string
		mutate func(*Router)
		want   string
	}{
		{"no jev", func(r *Router) { r.Jev = nil }, "needs a Jev"},
		{"no routes", func(r *Router) { r.Routes = nil }, "at least one route"},
		{"agent provider", func(r *Router) { r.Agent.Provider = p }, "leave Provider and Model"},
		{"agent model", func(r *Router) { r.Agent.Model = "m" }, "leave Provider and Model"},
		{"confidence", func(r *Router) { r.MinConfidence = 1.5 }, "outside [0, 1]"},
		{"no when", func(r *Router) { r.Routes[0].When = "" }, `route "fast" needs a Name and a When`},
		{"no model", func(r *Router) { r.Routes[1].Model = "" }, `route "deep" needs a Provider and a Model`},
		{"duplicate", func(r *Router) { r.Routes[1].Name = "fast" }, `two routes named "fast"`},
		{"unknown default", func(r *Router) { r.Default = "medium" }, `default route "medium"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestRouter("http://127.0.0.1:1")
			tt.mutate(r.Router)
			_, _, err := collect(t, r.Run(t.Context(), nil, Text{Text: "hi"}))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}
