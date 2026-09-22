package goop

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const anthropicFixture = `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"the user wants weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me "}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"check."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}

event: message_stop
data: {"type":"message_stop"}

`

func TestAnthropicStream(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "key" || r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing headers: %v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("request body: %v", err)
		}
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, anthropicFixture)
	}))
	defer server.Close()

	priorThinking := json.RawMessage(`{"type":"thinking","thinking":"prior turn","signature":"s0"}`)
	provider := &Anthropic{APIKey: "key", BaseURL: server.URL}
	var deltas, thoughts []string
	reply, err := provider.Stream(t.Context(), Request{
		Model:  "claude-sonnet-5",
		System: "be terse",
		Messages: []Message{
			{Role: User, Blocks: []Block{
				Text{Text: "weather in the photo's city?"},
				Image{MediaType: "image/png", Data: []byte{1, 2, 3}},
			}},
			{Role: Assistant, Blocks: []Block{
				Thinking{Raw: priorThinking},
				Text{Text: "which photo?"},
			}},
			{Role: User, Blocks: []Block{Text{Text: "the attached one"}}},
		},
		Tools: []ToolDef{{Name: "get_weather", Description: "d", Schema: json.RawMessage(`{"type":"object"}`)}},
	}, func(ev Event) {
		switch e := ev.(type) {
		case TextDelta:
			deltas = append(deltas, e.Text)
		case ThinkingDelta:
			thoughts = append(thoughts, e.Text)
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	// text and thinking come out as different event types
	if strings.Join(deltas, "") != "Let me check." {
		t.Fatalf("deltas = %v", deltas)
	}
	if strings.Join(thoughts, "") != "the user wants weather" {
		t.Fatalf("thoughts = %v", thoughts)
	}
	if reply.StopReason != StopToolUse {
		t.Fatalf("stop = %s", reply.StopReason)
	}
	if reply.Usage != (Usage{Input: 25, Output: 40}) {
		t.Fatalf("usage = %+v", reply.Usage)
	}
	if len(reply.Message.Blocks) != 3 {
		t.Fatalf("blocks = %+v", reply.Message.Blocks)
	}
	thinking, ok := reply.Message.Blocks[0].(Thinking)
	if !ok {
		t.Fatalf("first block = %T, want Thinking", reply.Message.Blocks[0])
	}
	var thought struct{ Type, Thinking, Signature string }
	if err := json.Unmarshal(thinking.Raw, &thought); err != nil {
		t.Fatal(err)
	}
	if thought.Type != "thinking" || thought.Thinking != "the user wants weather" || thought.Signature != "sig123" {
		t.Fatalf("thinking = %+v", thought)
	}
	use, ok := reply.Message.Blocks[2].(ToolUse)
	if !ok || use.ID != "toolu_1" || use.Name != "get_weather" || string(use.Input) != `{"city":"Paris"}` {
		t.Fatalf("tool use = %+v", use)
	}

	// check the request body: system prompt as a cached block, tools, the image as
	// base64 and the earlier thinking block exactly as we got it
	system := gotBody["system"].([]any)[0].(map[string]any)
	if system["text"] != "be terse" || system["cache_control"] == nil || gotBody["stream"] != true {
		t.Fatalf("system = %v", system)
	}
	if gotBody["max_tokens"] != float64(AnthropicMaxTokensFallback) {
		t.Fatalf("max_tokens = %v", gotBody["max_tokens"])
	}
	messages := gotBody["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	image := content[1].(map[string]any)["source"].(map[string]any)
	if image["type"] != "base64" || image["media_type"] != "image/png" || image["data"] != "AQID" {
		t.Fatalf("image source = %v", image)
	}
	replayed := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if replayed["type"] != "thinking" || replayed["thinking"] != "prior turn" || replayed["signature"] != "s0" {
		t.Fatalf("replayed thinking = %v", replayed)
	}
	// second cache breakpoint goes on the last block of the last message
	lastContent := messages[2].(map[string]any)["content"].([]any)
	lastBlock := lastContent[len(lastContent)-1].(map[string]any)
	if lastBlock["cache_control"] == nil {
		t.Fatalf("last block not cached: %v", lastBlock)
	}
	tools := gotBody["tools"].([]any)
	if tools[0].(map[string]any)["name"] != "get_weather" {
		t.Fatalf("tools = %v", tools)
	}
}

func TestAnthropicBodyKnobs(t *testing.T) {
	// thinking config is passed through as-is. With no system prompt the first cache
	// breakpoint ends up on the last tool instead.
	p := &Anthropic{Thinking: json.RawMessage(`{"type":"enabled","budget_tokens":1024}`)}
	body, err := p.body(Request{Model: "m", Tools: []ToolDef{
		{Name: "a", Schema: json.RawMessage(`{}`)},
		{Name: "b", Schema: json.RawMessage(`{}`)},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	thinking := got["thinking"].(map[string]any)
	if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(1024) {
		t.Fatalf("thinking = %v", thinking)
	}
	tools := got["tools"].([]any)
	if tools[0].(map[string]any)["cache_control"] != nil || tools[1].(map[string]any)["cache_control"] == nil {
		t.Fatalf("tool cache breakpoints = %v", tools)
	}
}

func TestAnthropicDocumentBlock(t *testing.T) {
	block, err := anthropicBlock(Document{Data: []byte{1, 2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	source := block.(map[string]any)["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "application/pdf" || source["data"] != "AQID" {
		t.Fatalf("document source = %v", source)
	}
	block, err = anthropicBlock(Document{URL: "https://example.com/a.pdf"})
	if err != nil {
		t.Fatal(err)
	}
	source = block.(map[string]any)["source"].(map[string]any)
	if source["type"] != "url" || source["url"] != "https://example.com/a.pdf" {
		t.Fatalf("document source = %v", source)
	}
}

func TestAnthropicCacheUsageCounted(t *testing.T) {
	stream := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_creation_input_tokens":200,"cache_read_input_tokens":3000,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

`
	reply, err := parseAnthropicStream(strings.NewReader(stream), nil)
	if err != nil {
		t.Fatal(err)
	}
	if reply.Usage != (Usage{Input: 3210, Output: 7}) {
		t.Fatalf("usage = %+v", reply.Usage)
	}
}

func TestRetries(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts <= 2 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "overloaded", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, anthropicFixture)
	}))
	defer server.Close()

	provider := &Anthropic{APIKey: "key", BaseURL: server.URL, HTTP: HTTP{Backoff: time.Millisecond}}
	reply, err := provider.Stream(t.Context(), Request{Model: "m"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if attempts != 3 {
		t.Fatalf("attempts = %d, want 3", attempts)
	}
	if reply.StopReason != StopToolUse {
		t.Fatalf("reply after retries = %+v", reply)
	}
}

func TestRetriesDisabled(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := &Anthropic{APIKey: "key", BaseURL: server.URL, HTTP: HTTP{MaxRetries: -1}}
	_, err := provider.Stream(t.Context(), Request{Model: "m"}, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusServiceUnavailable {
		t.Fatalf("err = %v", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func TestNoRetryOnClientError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	provider := &Anthropic{APIKey: "key", BaseURL: server.URL}
	_, err := provider.Stream(t.Context(), Request{Model: "m"}, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
		t.Fatalf("err = %v", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func TestAnthropicRedactedThinkingVerbatim(t *testing.T) {
	stream := `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque123"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}

`
	reply, err := parseAnthropicStream(strings.NewReader(stream), nil)
	if err != nil {
		t.Fatal(err)
	}
	thinking, ok := reply.Message.Blocks[0].(Thinking)
	if !ok || string(thinking.Raw) != `{"type":"redacted_thinking","data":"opaque123"}` {
		t.Fatalf("block = %+v", reply.Message.Blocks[0])
	}
}

func TestAnthropicTruncatedStream(t *testing.T) {
	// connection drops right after the tool_use block. Without message_stop this has to
	// be an error, not a half-filled Reply.
	cut := strings.Index(anthropicFixture, "event: message_delta")
	_, err := parseAnthropicStream(strings.NewReader(anthropicFixture[:cut]), nil)
	if !errors.Is(err, ErrTruncatedStream) {
		t.Fatalf("err = %v, want ErrTruncatedStream", err)
	}
}

func TestAnthropicNonStreamBody(t *testing.T) {
	// 200 with a plain JSON body, i.e. a server that ignored stream:true. No SSE events at all.
	_, err := parseAnthropicStream(strings.NewReader(`{"id":"msg_1","content":[]}`), nil)
	if !errors.Is(err, ErrTruncatedStream) {
		t.Fatalf("err = %v, want ErrTruncatedStream", err)
	}
}

func TestAnthropicRejectsEmptyThinking(t *testing.T) {
	if _, err := anthropicBlock(Thinking{}); err == nil {
		t.Fatal("empty Thinking block did not error")
	}
}

func TestAnthropicHTTPError(t *testing.T) {
	// 429 every time, we want the real status from the last attempt
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"overloaded"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	provider := &Anthropic{APIKey: "key", BaseURL: server.URL, HTTP: HTTP{Backoff: time.Millisecond}}
	_, err := provider.Stream(t.Context(), Request{Model: "m"}, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusTooManyRequests {
		t.Fatalf("err = %v", err)
	}
}

func TestAnthropicStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try later\"}}\n\n")
	}))
	defer server.Close()

	provider := &Anthropic{APIKey: "key", BaseURL: server.URL}
	_, err := provider.Stream(t.Context(), Request{Model: "m"}, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || !strings.Contains(apiErr.Body, "overloaded_error") {
		t.Fatalf("err = %v", err)
	}
}

func TestAnthropicRejectsUnknownBlock(t *testing.T) {
	_, err := anthropicBlock(fakeBlock{})
	if err == nil {
		t.Fatal("unknown block type did not error")
	}
}

type fakeBlock struct{}

func (fakeBlock) block() {}
