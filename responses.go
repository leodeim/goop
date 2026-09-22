package goop

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// OpenAIResponses is a Provider for the OpenAI Responses API. Reasoning items are replayed
// statelessly (store:false plus encrypted_content), so the conversation lives in Done.Messages.
type OpenAIResponses struct {
	APIKey  string
	BaseURL string // default https://api.openai.com/v1
	HTTP
	// ReasoningEffort sets reasoning.effort ("low", "medium", "high") if not empty and turns on
	// reasoning summaries, which stream as ThinkingDelta.
	ReasoningEffort string
}

func (p *OpenAIResponses) Stream(ctx context.Context, req Request, emit func(Event)) (Reply, error) {
	body, err := p.body(req)
	if err != nil {
		return Reply{}, err
	}

	base := strings.TrimSuffix(cmp.Or(p.BaseURL, "https://api.openai.com/v1"), "/")
	resp, err := p.post(ctx, base+"/responses", bearer(p.APIKey), body)
	if err != nil {
		return Reply{}, err
	}
	defer resp.Body.Close()

	return parseResponsesStream(resp.Body, emit)
}

func (p *OpenAIResponses) body(req Request) ([]byte, error) {
	input := []any{}
	for _, m := range req.Messages {
		items, err := responsesItems(m)
		if err != nil {
			return nil, err
		}
		input = append(input, items...)
	}

	body := map[string]any{
		"model":   req.Model,
		"input":   input,
		"stream":  true,
		"store":   false,
		"include": []string{"reasoning.encrypted_content"},
	}
	if req.System != "" {
		body["instructions"] = req.System
	}
	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if p.ReasoningEffort != "" {
		body["reasoning"] = map[string]any{"effort": p.ReasoningEffort, "summary": "auto"}
	}
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = map[string]any{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Schema,
			}
		}
		body["tools"] = tools
	}

	return json.Marshal(body)
}

// responsesItems converts a Message to Responses input items. Tool calls, tool results and
// reasoning are items of their own; a user message's text, images and files share one item.
func responsesItems(m Message) ([]any, error) {
	var items []any
	if m.Role == Assistant {
		for _, b := range m.Blocks {
			switch b := b.(type) {
			case Text:
				items = append(items, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []map[string]any{{"type": "output_text", "text": b.Text}},
				})

			case ToolUse:
				items = append(items, map[string]any{
					"type":      "function_call",
					"call_id":   b.ID,
					"name":      b.Name,
					"arguments": cmp.Or(string(b.Input), "{}"),
				})

			case Thinking:
				if len(b.Raw) == 0 {
					return nil, fmt.Errorf("goop: responses cannot send an empty Thinking block")
				}
				items = append(items, b.Raw)

			default:
				return nil, fmt.Errorf("goop: responses cannot send a %T block from the assistant", b)
			}
		}
		return items, nil
	}

	var parts []map[string]any
	for _, b := range m.Blocks {
		switch b := b.(type) {
		case ToolResult:
			output := b.Content
			if b.IsError {
				output = "error: " + output
			}
			items = append(items, map[string]any{"type": "function_call_output", "call_id": b.ToolUseID, "output": output})

		case Text:
			parts = append(parts, map[string]any{"type": "input_text", "text": b.Text})

		case Image:
			url := b.URL
			if url == "" {
				url = dataURL(b.MediaType, b.Data)
			}
			parts = append(parts, map[string]any{"type": "input_image", "image_url": url})

		case Document:
			if b.URL != "" {
				parts = append(parts, map[string]any{"type": "input_file", "file_url": b.URL})
				continue
			}
			mediaType := cmp.Or(b.MediaType, "application/pdf")
			name := "document"
			if mediaType == "application/pdf" {
				name += ".pdf"
			}
			parts = append(parts, map[string]any{"type": "input_file", "filename": name, "file_data": dataURL(mediaType, b.Data)})

		case Thinking:
		default:
			return nil, fmt.Errorf("goop: responses cannot send a %T block from the user", b)
		}
	}

	if len(parts) > 0 {
		items = append(items, map[string]any{"role": "user", "content": parts})
	}

	return items, nil
}

// parseResponsesStream emits text and reasoning deltas as they arrive and builds the Reply from
// the complete output items in the final response.completed / response.incomplete event.
func parseResponsesStream(r io.Reader, emit func(Event)) (Reply, error) {
	var reply Reply
	err := scanSSE(r, func(data string) error {
		var ev struct {
			Type     string          `json:"type"`
			Delta    json.RawMessage `json:"delta"`
			Message  string          `json:"message"` // type "error"
			Response struct {
				Status            string            `json:"status"`
				Output            []json.RawMessage `json:"output"`
				IncompleteDetails struct {
					Reason string `json:"reason"`
				} `json:"incomplete_details"`
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
				Usage struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return fmt.Errorf("goop: responses stream: %w", err)
		}

		switch ev.Type {
		case "response.output_text.delta", "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			var text string
			if err := json.Unmarshal(ev.Delta, &text); err != nil {
				return fmt.Errorf("goop: responses stream: %s: %w", ev.Type, err)
			}
			if emit == nil || text == "" {
				return nil
			}
			if ev.Type == "response.output_text.delta" {
				emit(TextDelta{Text: text})
			} else {
				emit(ThinkingDelta{Text: text})
			}

		case "response.completed", "response.incomplete":
			res := ev.Response
			msg, sawTool, err := responsesOutput(res.Output)
			if err != nil {
				return err
			}
			reply = Reply{
				Message:    msg,
				StopReason: responsesStop(res.Status, res.IncompleteDetails.Reason, sawTool),
				Usage:      Usage{Input: res.Usage.InputTokens, Output: res.Usage.OutputTokens},
			}
			return ErrStreamDone

		case "response.failed":
			return &APIError{Body: ev.Response.Error.Message}
		case "error":
			return &APIError{Body: ev.Message}
		}
		return nil
	})
	if err != nil {
		return Reply{}, err
	}

	return reply, nil
}

// responsesOutput turns output items into a Message. Item types goop does not model (reasoning,
// built-in tool calls) are kept verbatim as Thinking so they replay unchanged.
func responsesOutput(items []json.RawMessage) (Message, bool, error) {
	msg := Message{Role: Assistant}
	sawTool := false
	for _, raw := range items {
		var item struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Content   []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return Message{}, false, fmt.Errorf("goop: responses stream: output item: %w", err)
		}

		switch item.Type {
		case "message":
			var text strings.Builder
			for _, c := range item.Content {
				if c.Type == "output_text" {
					text.WriteString(c.Text)
				}
			}
			if text.Len() > 0 {
				msg.Blocks = append(msg.Blocks, Text{Text: text.String()})
			}

		case "function_call":
			sawTool = true
			msg.Blocks = append(msg.Blocks, ToolUse{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(cmp.Or(item.Arguments, "{}")),
			})

		default:
			msg.Blocks = append(msg.Blocks, Thinking{Raw: raw})
		}
	}

	return msg, sawTool, nil
}

func responsesStop(status, reason string, sawTool bool) StopReason {
	switch {
	case status == "incomplete" && reason == "max_output_tokens":
		return StopMaxTokens
	case status == "incomplete":
		return StopReason(cmp.Or(reason, "incomplete"))
	case sawTool:
		return StopToolUse
	}
	return StopEnd
}
