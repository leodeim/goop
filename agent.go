package goop

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
)

// DefaultMaxTurns is the turn limit when Agent.MaxTurns is 0.
const DefaultMaxTurns = 10

// ErrMaxTurns is returned when a run hits MaxTurns and the model still wants to call tools.
var ErrMaxTurns = errors.New("goop: run exceeded its turn limit")

// ErrBudget is returned when a run uses up its Budget before the model is done.
var ErrBudget = errors.New("goop: run exceeded its token budget")

// LimitError wraps ErrMaxTurns and ErrBudget. It carries the conversation so far so the caller can keep it.
type LimitError struct {
	Limit    error
	Messages []Message
	Usage    Usage
}

func (e *LimitError) Error() string { return e.Limit.Error() }
func (e *LimitError) Unwrap() error { return e.Limit }

// Agent runs the tool loop. It keeps no state between runs, so one Agent can be used from many goroutines.
type Agent struct {
	Provider  Provider
	Model     string
	System    string
	Tools     []Tool
	MaxTokens int // output tokens per turn, 0 = provider default
	MaxTurns  int // 0 = DefaultMaxTurns
	// Budget limits the total tokens of a run, 0 = unlimited. It is only checked between turns,
	// so the last turn can go over.
	Budget int
	// OnToolCall is called before every tool call. It can change the call, or return an error to
	// refuse it. The error text goes back to the model as the tool result.
	OnToolCall func(ctx context.Context, use ToolUse) (ToolUse, error)
}

// Run appends input to prior as a user message and runs the loop. It yields events and
// always ends with either a single Done or a single error.
func (a *Agent) Run(ctx context.Context, prior []Message, input ...Block) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if a.Provider == nil || a.Model == "" {
			yield(nil, errors.New("goop: agent needs a Provider and a Model"))
			return
		}
		tools := make(map[string]Tool, len(a.Tools))
		defs := make([]ToolDef, len(a.Tools))
		for i, t := range a.Tools {
			if err := t.validate(); err != nil {
				yield(nil, err)
				return
			}
			if _, dup := tools[t.Name]; dup {
				yield(nil, fmt.Errorf("goop: two tools named %q", t.Name))
				return
			}
			tools[t.Name] = t
			defs[i] = ToolDef{Name: t.Name, Description: t.Description, Schema: t.Schema}
		}

		msgs := slices.Clone(prior)
		if len(input) > 0 {
			msgs = append(msgs, Message{Role: User, Blocks: input})
		}

		// cancel when the caller breaks out of the loop, otherwise the provider keeps streaming
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		var usage Usage
		for range cmp.Or(a.MaxTurns, DefaultMaxTurns) {
			if a.Budget > 0 && usage.Input+usage.Output >= a.Budget {
				yield(nil, &LimitError{Limit: ErrBudget, Messages: msgs, Usage: usage})
				return
			}

			live := true
			reply, err := a.Provider.Stream(ctx, Request{
				Model:     a.Model,
				System:    a.System,
				Messages:  msgs,
				Tools:     defs,
				MaxTokens: a.MaxTokens,
			}, func(ev Event) {
				if live && !yield(ev, nil) {
					live = false
					cancel()
				}
			})
			if !live {
				return
			}
			if err != nil {
				yield(nil, err)
				return
			}

			usage.Input += reply.Usage.Input
			usage.Output += reply.Usage.Output

			// Tool calls in a max_tokens reply are usually cut off mid-JSON. Drop them, otherwise
			// the conversation can't be sent back.
			if reply.StopReason == StopMaxTokens {
				reply.Message.Blocks = slices.DeleteFunc(slices.Clone(reply.Message.Blocks), isToolUse)
			}
			// Some OpenAI-compatible servers report "stop" alongside tool calls. Unanswered calls
			// would leave a history the API rejects, so any surviving call means tool use.
			if slices.ContainsFunc(reply.Message.Blocks, isToolUse) {
				reply.StopReason = StopToolUse
			}
			// both APIs reject an assistant message with empty content
			if len(reply.Message.Blocks) > 0 {
				msgs = append(msgs, reply.Message)
			}

			if reply.StopReason != StopToolUse {
				yield(Done{Messages: msgs, StopReason: reply.StopReason, Usage: usage}, nil)
				return
			}

			var results []Block
			for _, b := range reply.Message.Blocks {
				use, ok := b.(ToolUse)
				if !ok {
					continue
				}

				run := use
				var refused error
				if a.OnToolCall != nil {
					if changed, err := a.OnToolCall(ctx, use); err != nil {
						refused = err
					} else {
						run = changed
						run.ID = use.ID // the hook can't change the id, the result has to match it
					}
				}
				if !yield(ToolCall{Use: run}, nil) {
					return
				}

				var result ToolResult
				if refused != nil {
					result = toolError(use.ID, refused.Error())
				} else {
					result = callTool(ctx, tools, run)
				}
				if !yield(ToolReturn{Result: result}, nil) {
					return
				}
				results = append(results, result)
			}

			if len(results) == 0 {
				yield(nil, errors.New("goop: model stopped for tool use without a tool call"))
				return
			}

			msgs = append(msgs, Message{Role: User, Blocks: results})
		}
		yield(nil, &LimitError{Limit: ErrMaxTurns, Messages: msgs, Usage: usage})
	}
}

func isToolUse(b Block) bool {
	_, ok := b.(ToolUse)
	return ok
}

func callTool(ctx context.Context, tools map[string]Tool, use ToolUse) (result ToolResult) {
	tool, ok := tools[use.Name]
	if !ok {
		return toolError(use.ID, "unknown tool "+use.Name)
	}

	defer func() {
		if r := recover(); r != nil {
			result = toolError(use.ID, fmt.Sprintf("tool panicked: %v", r))
		}
	}()

	out, err := tool.Run(ctx, use.Input)
	if err != nil {
		return toolError(use.ID, err.Error())
	}

	return ToolResult{ToolUseID: use.ID, Content: out}
}

func toolError(id, msg string) ToolResult {
	return ToolResult{ToolUseID: id, Content: msg, IsError: true}
}
