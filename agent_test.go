package goop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
)

type fakeProvider struct {
	replies []Reply
	calls   []Request
	lastCtx context.Context
}

func (f *fakeProvider) Stream(ctx context.Context, req Request, emit func(Event)) (Reply, error) {
	f.lastCtx = ctx
	f.calls = append(f.calls, req)
	if len(f.calls) > len(f.replies) {
		return Reply{}, errors.New("fake: out of replies")
	}
	reply := f.replies[len(f.calls)-1]
	if emit != nil {
		for _, b := range reply.Message.Blocks {
			if t, ok := b.(Text); ok && t.Text != "" {
				mid := len(t.Text) / 2
				emit(TextDelta{Text: t.Text[:mid]})
				emit(TextDelta{Text: t.Text[mid:]})
			}
		}
	}
	return reply, nil
}

func echoTool(t *testing.T) Tool {
	t.Helper()
	tool, err := NewTool("echo", "Echo the message back.",
		func(_ context.Context, in struct {
			Message string `json:"message"`
		}) (string, error) {
			if in.Message == "boom" {
				return "", errors.New("refused")
			}
			return "echo: " + in.Message, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

func TestRunToolLoop(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				Text{Text: "checking"},
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
			}},
			StopReason: StopToolUse,
			Usage:      Usage{Input: 10, Output: 5},
		},
		{
			Message:    Message{Role: Assistant, Blocks: []Block{Text{Text: "it said hi"}}},
			StopReason: StopEnd,
			Usage:      Usage{Input: 20, Output: 7},
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", System: "sys", Tools: []Tool{echoTool(t)}}

	var events []string
	var done *Done
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "say hi"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		switch e := ev.(type) {
		case TextDelta:
			events = append(events, "delta:"+e.Text)
		case ToolCall:
			events = append(events, "call:"+e.Use.Name)
		case ToolReturn:
			events = append(events, fmt.Sprintf("return:%s:%v", e.Result.Content, e.Result.IsError))
		case Done:
			done = &e
			events = append(events, "done")
		}
	}

	want := []string{
		"delta:chec", "delta:king",
		"call:echo", "return:echo: hi:false",
		"delta:it sa", "delta:id hi",
		"done",
	}
	if fmt.Sprint(events) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
	if done == nil {
		t.Fatal("no Done event")
	}
	if got := len(done.Messages); got != 4 {
		t.Fatalf("conversation has %d messages, want 4", got)
	}
	if done.Usage != (Usage{Input: 30, Output: 12}) {
		t.Fatalf("usage = %+v", done.Usage)
	}
	if done.StopReason != StopEnd {
		t.Fatalf("stop reason = %s", done.StopReason)
	}
	// second request has to contain the tool result for t1
	second := provider.calls[1]
	last := second.Messages[len(second.Messages)-1]
	result, ok := last.Blocks[0].(ToolResult)
	if !ok || result.ToolUseID != "t1" || result.Content != "echo: hi" {
		t.Fatalf("tool result message = %+v", last)
	}
	if len(second.Tools) != 1 || second.Tools[0].Name != "echo" {
		t.Fatalf("tools sent = %+v", second.Tools)
	}
}

func TestRunToolError(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":"boom"}`)},
			}},
			StopReason: StopToolUse,
		},
		{
			Message:    Message{Role: Assistant, Blocks: []Block{Text{Text: "the tool failed"}}},
			StopReason: StopEnd,
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}}

	var errored *ToolResult
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if r, ok := ev.(ToolReturn); ok {
			errored = &r.Result
		}
	}
	if errored == nil || !errored.IsError || errored.Content != "refused" {
		t.Fatalf("tool return = %+v, want is_error 'refused'", errored)
	}
}

func TestRunUnknownTool(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "nope", Input: json.RawMessage(`{}`)},
			}},
			StopReason: StopToolUse,
		},
		{
			Message:    Message{Role: Assistant, Blocks: []Block{Text{Text: "ok"}}},
			StopReason: StopEnd,
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}}
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if r, ok := ev.(ToolReturn); ok && !r.Result.IsError {
			t.Fatalf("unknown tool did not error: %+v", r.Result)
		}
	}
}

func TestRunMaxTurns(t *testing.T) {
	looping := Reply{
		Message: Message{Role: Assistant, Blocks: []Block{
			ToolUse{ID: "t", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
		}},
		StopReason: StopToolUse,
	}
	provider := &fakeProvider{replies: []Reply{looping, looping, looping}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}, MaxTurns: 3}

	var last error
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		last = err
		if _, ok := ev.(Done); ok {
			t.Fatal("run finished instead of hitting the turn limit")
		}
	}
	if !errors.Is(last, ErrMaxTurns) {
		t.Fatalf("final error = %v, want ErrMaxTurns", last)
	}
	var limit *LimitError
	if !errors.As(last, &limit) || len(limit.Messages) != 7 { // user + 3 x (assistant, tool results)
		t.Fatalf("limit error = %+v", last)
	}
	if len(provider.calls) != 3 {
		t.Fatalf("provider called %d times, want 3", len(provider.calls))
	}
}

func TestRunBudget(t *testing.T) {
	looping := Reply{
		Message: Message{Role: Assistant, Blocks: []Block{
			ToolUse{ID: "t", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
		}},
		StopReason: StopToolUse,
		Usage:      Usage{Input: 100, Output: 50},
	}
	provider := &fakeProvider{replies: []Reply{looping, looping}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}, Budget: 120}

	var last error
	for _, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		last = err
	}
	if !errors.Is(last, ErrBudget) {
		t.Fatalf("final error = %v, want ErrBudget", last)
	}
	var limit *LimitError
	if !errors.As(last, &limit) || limit.Usage != (Usage{Input: 100, Output: 50}) || len(limit.Messages) != 3 {
		t.Fatalf("limit error = %+v", limit)
	}
	// budget is only checked between turns, so the first one still runs
	if len(provider.calls) != 1 {
		t.Fatalf("provider called %d times, want 1", len(provider.calls))
	}
}

func TestRunToolHookModifies(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
			}},
			StopReason: StopToolUse,
		},
		{
			Message:    Message{Role: Assistant, Blocks: []Block{Text{Text: "ok"}}},
			StopReason: StopEnd,
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)},
		OnToolCall: func(_ context.Context, use ToolUse) (ToolUse, error) {
			use.Input = json.RawMessage(`{"message":"hey"}`)
			use.ID = "evil" // the loop has to put the original id back
			return use, nil
		},
	}

	var call *ToolCall
	var ret *ToolReturn
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		switch e := ev.(type) {
		case ToolCall:
			call = &e
		case ToolReturn:
			ret = &e
		}
	}
	if call == nil || string(call.Use.Input) != `{"message":"hey"}` || call.Use.ID != "t1" {
		t.Fatalf("tool call = %+v", call)
	}
	if ret == nil || ret.Result.Content != "echo: hey" || ret.Result.ToolUseID != "t1" {
		t.Fatalf("tool return = %+v", ret)
	}
}

func TestRunToolHookRefuses(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
			}},
			StopReason: StopToolUse,
		},
		{
			Message:    Message{Role: Assistant, Blocks: []Block{Text{Text: "understood"}}},
			StopReason: StopEnd,
		},
	}}
	ran := false
	tool, err := NewTool("echo", "d", func(_ context.Context, in struct {
		Message string `json:"message"`
	}) (string, error) {
		ran = true
		return "should not run", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{tool},
		OnToolCall: func(_ context.Context, use ToolUse) (ToolUse, error) {
			return ToolUse{}, errors.New("not allowed")
		},
	}

	var ret *ToolReturn
	var done bool
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		switch e := ev.(type) {
		case ToolReturn:
			ret = &e
		case Done:
			done = true
		}
	}
	if ran {
		t.Fatal("refused tool still ran")
	}
	if ret == nil || !ret.Result.IsError || ret.Result.Content != "not allowed" || ret.Result.ToolUseID != "t1" {
		t.Fatalf("tool return = %+v", ret)
	}
	if !done {
		t.Fatal("run did not continue past the refusal")
	}
}

func TestRunConfigErrors(t *testing.T) {
	runErr := func(a *Agent) error {
		var last error
		for _, err := range a.Run(t.Context(), nil, Text{Text: "x"}) {
			last = err
		}
		return last
	}
	if err := runErr(&Agent{}); err == nil {
		t.Fatal("zero agent did not error")
	}
	dup := echoTool(t)
	if err := runErr(&Agent{Provider: &fakeProvider{}, Model: "m", Tools: []Tool{dup, dup}}); err == nil {
		t.Fatal("duplicate tool names did not error")
	}

	run := func(context.Context, json.RawMessage) (string, error) { return "", nil }
	for name, tool := range map[string]Tool{
		"empty name":  {Schema: json.RawMessage(`{}`), Run: run},
		"nil run":     {Name: "t", Schema: json.RawMessage(`{}`)},
		"nil schema":  {Name: "t", Run: run},
		"non-object":  {Name: "t", Schema: json.RawMessage(`[]`), Run: run},
		"broken json": {Name: "t", Schema: json.RawMessage(`{"type":`), Run: run},
	} {
		ok := &fakeProvider{replies: []Reply{{Message: Message{Role: Assistant, Blocks: []Block{Text{Text: "hi"}}}, StopReason: StopEnd}}}
		if err := runErr(&Agent{Provider: ok, Model: "m", Tools: []Tool{tool}}); err == nil {
			t.Errorf("%s: tool was accepted", name)
		}
	}
}

func TestRunToolUseWithEndStop(t *testing.T) {
	// some OpenAI-compatible servers send finish_reason "stop" with tool calls
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":"hi"}`)},
			}},
			StopReason: StopEnd,
		},
		{Message: Message{Role: Assistant, Blocks: []Block{Text{Text: "done"}}}, StopReason: StopEnd},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}}

	var done *Done
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if d, ok := ev.(Done); ok {
			done = &d
		}
	}
	if len(provider.calls) != 2 {
		t.Fatalf("provider called %d times, want 2 (the tool call was not answered)", len(provider.calls))
	}
	if done == nil || len(done.Messages) != 4 {
		t.Fatalf("done = %+v", done)
	}
	if res, ok := done.Messages[2].Blocks[0].(ToolResult); !ok || res.Content != "echo: hi" {
		t.Fatalf("tool result = %+v", done.Messages[2])
	}
}

func TestRunToolPanic(t *testing.T) {
	boom := Tool{
		Name:   "boom",
		Schema: json.RawMessage(`{"type":"object"}`),
		Run:    func(context.Context, json.RawMessage) (string, error) { panic("kaboom") },
	}
	provider := &fakeProvider{replies: []Reply{
		{Message: Message{Role: Assistant, Blocks: []Block{ToolUse{ID: "t1", Name: "boom"}}}, StopReason: StopToolUse},
		{Message: Message{Role: Assistant, Blocks: []Block{Text{Text: "ok"}}}, StopReason: StopEnd},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{boom}}

	var ret *ToolReturn
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if r, ok := ev.(ToolReturn); ok {
			ret = &r
		}
	}
	if ret == nil || !ret.Result.IsError || ret.Result.Content != "tool panicked: kaboom" {
		t.Fatalf("tool return = %+v", ret)
	}
}

func TestRunMaxTokensDropsToolUse(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				Text{Text: "partial"},
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"message":`)}, // truncated JSON
			}},
			StopReason: StopMaxTokens,
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}}

	var done *Done
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if d, ok := ev.(Done); ok {
			done = &d
		}
	}
	if done == nil || done.StopReason != StopMaxTokens {
		t.Fatalf("done = %+v", done)
	}
	last := done.Messages[len(done.Messages)-1]
	for _, b := range last.Blocks {
		if _, isUse := b.(ToolUse); isUse {
			t.Fatalf("truncated tool_use survived in %+v", last.Blocks)
		}
	}
	if text, ok := last.Blocks[0].(Text); !ok || text.Text != "partial" {
		t.Fatalf("partial text lost: %+v", last.Blocks)
	}
}

func TestRunMaxTokensDropsEmptiedMessage(t *testing.T) {
	// Only a truncated tool call, nothing else. After dropping it the assistant message is
	// empty and must not be kept, the API rejects empty content.
	provider := &fakeProvider{replies: []Reply{
		{
			Message: Message{Role: Assistant, Blocks: []Block{
				ToolUse{ID: "t1", Name: "echo", Input: json.RawMessage(`{"mess`)},
			}},
			StopReason: StopMaxTokens,
		},
	}}
	agent := &Agent{Provider: provider, Model: "m", Tools: []Tool{echoTool(t)}}

	var done *Done
	for ev, err := range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		if d, ok := ev.(Done); ok {
			done = &d
		}
	}
	if done == nil || len(done.Messages) != 1 || done.Messages[0].Role != User {
		t.Fatalf("done = %+v", done)
	}
}

func TestRunConsumerStops(t *testing.T) {
	provider := &fakeProvider{replies: []Reply{
		{Message: Message{Role: Assistant, Blocks: []Block{Text{Text: "long answer"}}}, StopReason: StopEnd},
	}}
	agent := &Agent{Provider: provider, Model: "m"}
	for range agent.Run(t.Context(), nil, Text{Text: "go"}) {
		break // stop after the first event
	}
	// breaking out of the loop should have canceled the provider context
	if provider.lastCtx.Err() == nil {
		t.Fatal("provider context not canceled after the consumer stopped")
	}
}
