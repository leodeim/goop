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

const openaiFixture = `data: {"choices":[{"delta":{"reasoning":"user wants weather"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"Sure, "},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"checking."},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"ci"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":12}}

data: [DONE]

`

func TestOpenAIStream(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("authorization") != "Bearer key" {
			t.Errorf("auth header = %q", r.Header.Get("authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("request body: %v", err)
		}
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, openaiFixture)
	}))
	defer server.Close()

	provider := &OpenAI{APIKey: "key", BaseURL: server.URL, ReasoningEffort: "low"}
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
				Text{Text: "thinking"},
				ToolUse{ID: "call_0", Name: "get_weather", Input: json.RawMessage(`{"city":"Oslo"}`)},
			}},
			{Role: User, Blocks: []Block{
				ToolResult{ToolUseID: "call_0", Content: "raining", IsError: false},
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
	use, ok := reply.Message.Blocks[1].(ToolUse)
	if !ok || use.ID != "call_1" || use.Name != "get_weather" || string(use.Input) != `{"city":"Paris"}` {
		t.Fatalf("tool use = %+v", use)
	}

	// check the request body: system message first, image as a data URL, the assistant
	// tool call, and the tool result as a separate role:"tool" message
	messages := gotBody["messages"].([]any)
	if messages[0].(map[string]any)["role"] != "system" {
		t.Fatalf("first message = %v", messages[0])
	}
	userParts := messages[1].(map[string]any)["content"].([]any)
	imageURL := userParts[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(imageURL, "data:image/png;base64,AQID") {
		t.Fatalf("image url = %q", imageURL)
	}
	assistant := messages[2].(map[string]any)
	calls := assistant["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["arguments"] != `{"city":"Oslo"}` {
		t.Fatalf("assistant tool call = %v", fn)
	}
	toolMsg := messages[3].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "call_0" || toolMsg["content"] != "raining" {
		t.Fatalf("tool message = %v", toolMsg)
	}
	if gotBody["max_completion_tokens"] != float64(500) || gotBody["max_tokens"] != nil {
		t.Fatalf("max tokens fields = %v / %v", gotBody["max_completion_tokens"], gotBody["max_tokens"])
	}
	if gotBody["stream_options"].(map[string]any)["include_usage"] != true {
		t.Fatalf("stream_options = %v", gotBody["stream_options"])
	}
	if gotBody["reasoning_effort"] != "low" {
		t.Fatalf("reasoning_effort = %v", gotBody["reasoning_effort"])
	}
}

func TestOpenAILegacyMaxTokens(t *testing.T) {
	body, err := (&OpenAI{LegacyMaxTokens: true}).body(Request{Model: "m", MaxTokens: 5})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got["max_tokens"] != float64(5) || got["max_completion_tokens"] != nil {
		t.Fatalf("max tokens fields = %v / %v", got["max_tokens"], got["max_completion_tokens"])
	}
}

func TestOpenAIToolCallsWithoutIndex(t *testing.T) {
	// Mistral and older Ollama send each call in one chunk with no index. A new id has to
	// start a new call instead of being merged into the first one.
	stream := `data: {"choices":[{"delta":{"tool_calls":[` +
		`{"id":"a","function":{"name":"f1","arguments":"{\"x\":1}"}},` +
		`{"id":"b","function":{"name":"f2","arguments":"{\"y\":2}"}}` +
		`]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	reply, err := parseOpenAIStream(strings.NewReader(stream), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(reply.Message.Blocks) != 2 {
		t.Fatalf("blocks = %+v", reply.Message.Blocks)
	}
	first := reply.Message.Blocks[0].(ToolUse)
	second := reply.Message.Blocks[1].(ToolUse)
	if first.ID != "a" || string(first.Input) != `{"x":1}` || second.ID != "b" || string(second.Input) != `{"y":2}` {
		t.Fatalf("calls = %+v, %+v", first, second)
	}
}

func TestOpenAITruncatedStream(t *testing.T) {
	stream := `data: {"choices":[{"delta":{"content":"half"},"finish_reason":null}]}

`
	if _, err := parseOpenAIStream(strings.NewReader(stream), nil); !errors.Is(err, ErrTruncatedStream) {
		t.Fatalf("err = %v, want ErrTruncatedStream", err)
	}
	// 200 with a plain JSON body, i.e. a server that ignored stream:true
	if _, err := parseOpenAIStream(strings.NewReader(`{"choices":[{"message":{"content":"hi"}}]}`), nil); !errors.Is(err, ErrTruncatedStream) {
		t.Fatalf("err = %v, want ErrTruncatedStream", err)
	}
}

func TestOpenAIErrorResultPrefix(t *testing.T) {
	msgs, err := openaiMessages(Message{Role: User, Blocks: []Block{
		ToolResult{ToolUseID: "c", Content: "no such city", IsError: true},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if msgs[0]["content"] != "error: no such city" {
		t.Fatalf("content = %v", msgs[0]["content"])
	}
}

func TestOpenAIPlainStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`)
	}))
	defer server.Close()

	provider := &OpenAI{APIKey: "key", BaseURL: server.URL}
	reply, err := provider.Stream(t.Context(), Request{Model: "m", Messages: []Message{
		{Role: User, Blocks: []Block{Text{Text: "hello"}}},
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if reply.StopReason != StopEnd {
		t.Fatalf("stop = %s", reply.StopReason)
	}
	if text, ok := reply.Message.Blocks[0].(Text); !ok || text.Text != "hi" {
		t.Fatalf("blocks = %+v", reply.Message.Blocks)
	}
}
