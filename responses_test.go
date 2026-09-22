package goop

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const responsesFixture = `event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","delta":"user wants weather"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Sure, "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"checking."}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","delta":"{\"city\":"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","output":[` +
	`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"user wants weather"}],"encrypted_content":"enc123"},` +
	`{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Sure, checking."}]},` +
	`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}` +
	`],"usage":{"input_tokens":30,"output_tokens":12}}}

`

func TestResponsesStream(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" || r.Header.Get("authorization") != "Bearer key" {
			t.Errorf("request = %s %s auth=%q", r.Method, r.URL.Path, r.Header.Get("authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("request body: %v", err)
		}
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, responsesFixture)
	}))
	defer server.Close()

	priorReasoning := json.RawMessage(`{"type":"reasoning","id":"rs_0","summary":[],"encrypted_content":"e0"}`)
	provider := &OpenAIResponses{APIKey: "key", BaseURL: server.URL, ReasoningEffort: "low"}
	var deltas, thoughts []string
	reply, err := provider.Stream(t.Context(), Request{
		Model:  "gpt-5",
		System: "be terse",
		Messages: []Message{
			{Role: User, Blocks: []Block{
				Text{Text: "weather in the photo's city?"},
				Image{MediaType: "image/png", Data: []byte{1, 2, 3}},
			}},
			{Role: Assistant, Blocks: []Block{
				Thinking{Raw: priorReasoning},
				Text{Text: "thinking"},
				ToolUse{ID: "call_0", Name: "get_weather", Input: json.RawMessage(`{"city":"Oslo"}`)},
			}},
			{Role: User, Blocks: []Block{
				ToolResult{ToolUseID: "call_0", Content: "raining"},
			}},
		},
		Tools:     []ToolDef{{Name: "get_weather", Description: "d", Schema: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens: 500,
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

	if strings.Join(deltas, "") != "Sure, checking." {
		t.Fatalf("deltas = %v", deltas)
	}
	if strings.Join(thoughts, "") != "user wants weather" {
		t.Fatalf("thoughts = %v", thoughts)
	}
	if reply.StopReason != StopToolUse {
		t.Fatalf("stop = %s", reply.StopReason)
	}
	if reply.Usage != (Usage{Input: 30, Output: 12}) {
		t.Fatalf("usage = %+v", reply.Usage)
	}
	if len(reply.Message.Blocks) != 3 {
		t.Fatalf("blocks = %+v", reply.Message.Blocks)
	}
	if thinking, ok := reply.Message.Blocks[0].(Thinking); !ok || !strings.Contains(string(thinking.Raw), `"encrypted_content":"enc123"`) {
		t.Fatalf("first block = %+v, want verbatim reasoning item", reply.Message.Blocks[0])
	}
	if text, ok := reply.Message.Blocks[1].(Text); !ok || text.Text != "Sure, checking." {
		t.Fatalf("text block = %+v", reply.Message.Blocks[1])
	}
	use, ok := reply.Message.Blocks[2].(ToolUse)
	if !ok || use.ID != "call_1" || use.Name != "get_weather" || string(use.Input) != `{"city":"Paris"}` {
		t.Fatalf("tool use = %+v", use)
	}

	// request shape: stateless replay, flat tools, and one input item per block kind
	if gotBody["instructions"] != "be terse" || gotBody["store"] != false || gotBody["stream"] != true {
		t.Fatalf("body = %v", gotBody)
	}
	if include := gotBody["include"].([]any); len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %v", include)
	}
	if gotBody["max_output_tokens"] != float64(500) {
		t.Fatalf("max_output_tokens = %v", gotBody["max_output_tokens"])
	}
	reasoning := gotBody["reasoning"].(map[string]any)
	if reasoning["effort"] != "low" || reasoning["summary"] != "auto" {
		t.Fatalf("reasoning = %v", reasoning)
	}
	tool := gotBody["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" || tool["parameters"] == nil {
		t.Fatalf("tool = %v", tool)
	}

	input := gotBody["input"].([]any)
	if len(input) != 5 {
		t.Fatalf("input has %d items: %v", len(input), input)
	}
	user := input[0].(map[string]any)
	userParts := user["content"].([]any)
	if user["role"] != "user" || userParts[0].(map[string]any)["type"] != "input_text" {
		t.Fatalf("user item = %v", user)
	}
	if img := userParts[1].(map[string]any); img["type"] != "input_image" || !strings.HasPrefix(img["image_url"].(string), "data:image/png;base64,AQID") {
		t.Fatalf("image part = %v", img)
	}
	if replayed := input[1].(map[string]any); replayed["type"] != "reasoning" || replayed["encrypted_content"] != "e0" {
		t.Fatalf("replayed reasoning = %v", replayed)
	}
	assistant := input[2].(map[string]any)
	if assistant["role"] != "assistant" || assistant["content"].([]any)[0].(map[string]any)["text"] != "thinking" {
		t.Fatalf("assistant item = %v", assistant)
	}
	if call := input[3].(map[string]any); call["type"] != "function_call" || call["call_id"] != "call_0" || call["arguments"] != `{"city":"Oslo"}` {
		t.Fatalf("function_call item = %v", call)
	}
	if out := input[4].(map[string]any); out["type"] != "function_call_output" || out["call_id"] != "call_0" || out["output"] != "raining" {
		t.Fatalf("function_call_output item = %v", out)
	}
}

func TestResponsesIncomplete(t *testing.T) {
	stream := `event: response.incomplete
data: {"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},` +
		`"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":9}}}

`
	reply, err := parseResponsesStream(strings.NewReader(stream), nil)
	if err != nil {
		t.Fatal(err)
	}
	if reply.StopReason != StopMaxTokens {
		t.Fatalf("stop = %s", reply.StopReason)
	}
	if text, ok := reply.Message.Blocks[0].(Text); !ok || text.Text != "partial" {
		t.Fatalf("blocks = %+v", reply.Message.Blocks)
	}
}

func TestResponsesFailed(t *testing.T) {
	stream := `event: response.failed
data: {"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"boom"}}}

`
	_, err := parseResponsesStream(strings.NewReader(stream), nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Body != "boom" {
		t.Fatalf("err = %v", err)
	}
}

func TestResponsesTruncated(t *testing.T) {
	stream := `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"half"}

`
	if _, err := parseResponsesStream(strings.NewReader(stream), nil); !errors.Is(err, ErrTruncatedStream) {
		t.Fatalf("err = %v, want ErrTruncatedStream", err)
	}
}

func TestResponsesItems(t *testing.T) {
	items, err := responsesItems(Message{Role: User, Blocks: []Block{
		ToolResult{ToolUseID: "c", Content: "no such city", IsError: true},
		Document{Data: []byte{1, 2, 3}},
		Document{URL: "https://example.com/a.pdf"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if out := items[0].(map[string]any); out["output"] != "error: no such city" {
		t.Fatalf("tool output = %v", out)
	}
	parts := items[1].(map[string]any)["content"].([]map[string]any)
	if parts[0]["type"] != "input_file" || parts[0]["filename"] != "document.pdf" || parts[0]["file_data"] != "data:application/pdf;base64,AQID" {
		t.Fatalf("inline document = %v", parts[0])
	}
	if parts[1]["type"] != "input_file" || parts[1]["file_url"] != "https://example.com/a.pdf" {
		t.Fatalf("url document = %v", parts[1])
	}

	if _, err := responsesItems(Message{Role: Assistant, Blocks: []Block{Thinking{}}}); err == nil {
		t.Fatal("empty Thinking block did not error")
	}
}
