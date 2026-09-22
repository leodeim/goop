package goop

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAI is a Provider for the Chat Completions API. Set BaseURL to use an OpenAI-compatible server instead.
type OpenAI struct {
	APIKey  string
	BaseURL string // default https://api.openai.com/v1
	HTTP
	// LegacyMaxTokens sends max_tokens instead of max_completion_tokens. Some compatible servers only know the old name.
	LegacyMaxTokens bool
	// ReasoningEffort sets reasoning_effort ("low", "medium", "high") if not empty.
	ReasoningEffort string
}

func (p *OpenAI) Stream(ctx context.Context, req Request, emit func(Event)) (Reply, error) {
	body, err := p.body(req)
	if err != nil {
		return Reply{}, err
	}

	base := strings.TrimSuffix(cmp.Or(p.BaseURL, "https://api.openai.com/v1"), "/")

	resp, err := p.send(ctx, func() (*http.Request, error) {
		hr, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		hr.Header.Set("content-type", "application/json")
		if p.APIKey != "" {
			hr.Header.Set("authorization", "Bearer "+p.APIKey)
		}
		return hr, nil
	})
	if err != nil {
		return Reply{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Reply{}, readAPIError(resp)
	}

	return parseOpenAIStream(resp.Body, emit)
}

func (p *OpenAI) body(req Request) ([]byte, error) {
	var messages []map[string]any
	if req.System != "" {
		messages = append(messages, map[string]any{"role": "system", "content": req.System})
	}
	for _, m := range req.Messages {
		converted, err := openaiMessages(m)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	body := map[string]any{
		"model":    req.Model,
		"messages": messages,
		"stream":   true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
	}
	if req.MaxTokens > 0 {
		field := "max_completion_tokens"
		if p.LegacyMaxTokens {
			field = "max_tokens"
		}
		body[field] = req.MaxTokens
	}
	if p.ReasoningEffort != "" {
		body["reasoning_effort"] = p.ReasoningEffort
	}
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Schema,
				},
			}
		}
		body["tools"] = tools
	}

	return json.Marshal(body)
}

// openaiMessages converts a Message to chat completion messages. A tool result becomes its own
// role:"tool" message. Thinking blocks are dropped, chat completions has nowhere to put them.
func openaiMessages(m Message) ([]map[string]any, error) {
	if m.Role == Assistant {
		msg := map[string]any{"role": "assistant"}
		var text strings.Builder
		var calls []map[string]any
		for _, b := range m.Blocks {
			switch b := b.(type) {
			case Text:
				text.WriteString(b.Text)
			case ToolUse:
				calls = append(calls, map[string]any{
					"id":   b.ID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": cmp.Or(string(b.Input), "{}"),
					},
				})
			case Thinking:
			default:
				return nil, fmt.Errorf("goop: openai cannot send a %T block from the assistant", b)
			}
		}

		if text.Len() > 0 {
			msg["content"] = text.String()
		}
		if len(calls) > 0 {
			msg["tool_calls"] = calls
		}

		return []map[string]any{msg}, nil
	}

	var out []map[string]any
	var parts []map[string]any
	for _, b := range m.Blocks {
		switch b := b.(type) {
		case ToolResult:
			content := b.Content
			if b.IsError {
				content = "error: " + content
			}
			out = append(out, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": content})

		case Text:
			parts = append(parts, map[string]any{"type": "text", "text": b.Text})

		case Image:
			url := b.URL
			if url == "" {
				url = "data:" + b.MediaType + ";base64," + base64.StdEncoding.EncodeToString(b.Data)
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})

		case Thinking:
		default:
			return nil, fmt.Errorf("goop: openai cannot send a %T block from the user", b)
		}
	}

	if len(parts) > 0 {
		out = append(out, map[string]any{"role": "user", "content": parts})
	}

	return out, nil
}

// openaiCall collects the argument deltas of one tool call.
type openaiCall struct {
	id, name string
	args     strings.Builder
}

func parseOpenAIStream(r io.Reader, emit func(Event)) (Reply, error) {
	var reply Reply
	var text strings.Builder
	var calls []*openaiCall
	err := scanSSE(r, func(data string) error {
		if data == "[DONE]" {
			return ErrStreamDone
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					Reasoning        string `json:"reasoning"`         // OpenRouter
					ReasoningContent string `json:"reasoning_content"` // DeepSeek and some others
					ToolCalls        []struct {
						Index    *int   `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("goop: openai stream: %w", err)
		}

		if chunk.Error != nil {
			return &APIError{Body: chunk.Error.Message}
		}
		if chunk.Usage != nil {
			reply.Usage.Input = chunk.Usage.PromptTokens
			reply.Usage.Output = chunk.Usage.CompletionTokens
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				text.WriteString(choice.Delta.Content)
				if emit != nil {
					emit(TextDelta{Text: choice.Delta.Content})
				}
			}
			if reasoning := cmp.Or(choice.Delta.Reasoning, choice.Delta.ReasoningContent); reasoning != "" && emit != nil {
				emit(ThinkingDelta{Text: reasoning})
			}

			for _, tc := range choice.Delta.ToolCalls {
				// Mistral and older Ollama versions send tool calls without an index. Treat a new id
				// as a new call, anything else continues the last one.
				var call *openaiCall
				switch {
				case tc.Index != nil:
					for len(calls) <= *tc.Index {
						calls = append(calls, &openaiCall{})
					}
					call = calls[*tc.Index]
				case tc.ID != "" || len(calls) == 0:
					call = &openaiCall{}
					calls = append(calls, call)
				default:
					call = calls[len(calls)-1]
				}
				call.id = cmp.Or(call.id, tc.ID)
				call.name = cmp.Or(call.name, tc.Function.Name)
				call.args.WriteString(tc.Function.Arguments)
			}

			if choice.FinishReason != "" {
				reply.StopReason = openaiStop(choice.FinishReason)
			}
		}
		return nil
	})
	if err != nil {
		return Reply{}, err
	}

	reply.Message = Message{Role: Assistant}
	if text.Len() > 0 {
		reply.Message.Blocks = append(reply.Message.Blocks, Text{Text: text.String()})
	}
	for _, call := range calls {
		reply.Message.Blocks = append(reply.Message.Blocks, ToolUse{
			ID:    call.id,
			Name:  call.name,
			Input: json.RawMessage(cmp.Or(call.args.String(), "{}")),
		})
	}

	return reply, nil
}

func openaiStop(s string) StopReason {
	switch s {
	case "stop":
		return StopEnd
	case "tool_calls", "function_call":
		return StopToolUse
	case "length":
		return StopMaxTokens
	}
	return StopReason(s)
}
