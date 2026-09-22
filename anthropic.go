package goop

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// AnthropicMaxTokensFallback is the max_tokens sent when Request.MaxTokens is 0. The API requires the field.
const AnthropicMaxTokensFallback = 64000

// Anthropic is a Provider for the Messages API.
type Anthropic struct {
	APIKey  string
	BaseURL string // default https://api.anthropic.com
	HTTP
	// Thinking is passed through as the "thinking" request parameter, e.g. {"type":"enabled","budget_tokens":2048}.
	Thinking json.RawMessage
}

func (p *Anthropic) Stream(ctx context.Context, req Request, emit func(Event)) (Reply, error) {
	body, err := p.body(req)
	if err != nil {
		return Reply{}, err
	}

	base := strings.TrimSuffix(cmp.Or(p.BaseURL, "https://api.anthropic.com"), "/")
	resp, err := p.post(ctx, base+"/v1/messages", map[string]string{
		"anthropic-version": "2023-06-01",
		"x-api-key":         p.APIKey,
	}, body)
	if err != nil {
		return Reply{}, err
	}
	defer resp.Body.Close()

	return parseAnthropicStream(resp.Body, emit)
}

func (p *Anthropic) body(req Request) ([]byte, error) {
	ephemeral := map[string]any{"type": "ephemeral"}
	messages := make([]map[string]any, len(req.Messages))
	var lastContent []any
	for i, m := range req.Messages {
		content := make([]any, len(m.Blocks))
		for j, b := range m.Blocks {
			block, err := anthropicBlock(b)
			if err != nil {
				return nil, err
			}
			content[j] = block
		}
		messages[i] = map[string]any{"role": string(m.Role), "content": content}
		lastContent = content
	}

	// cache_control on the system prompt (or last tool) and on the last message, so the next
	// call gets a cache hit on everything up to here.
	if n := len(lastContent); n > 0 {
		if block, ok := lastContent[n-1].(map[string]any); ok {
			block["cache_control"] = ephemeral
		}
	}

	body := map[string]any{
		"model":      req.Model,
		"max_tokens": cmp.Or(req.MaxTokens, AnthropicMaxTokensFallback),
		"messages":   messages,
		"stream":     true,
	}
	if req.System != "" {
		body["system"] = []map[string]any{{"type": "text", "text": req.System, "cache_control": ephemeral}}
	}
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.Schema,
			}
		}
		if req.System == "" {
			tools[len(tools)-1]["cache_control"] = ephemeral
		}
		body["tools"] = tools
	}
	if len(p.Thinking) > 0 {
		body["thinking"] = p.Thinking
	}

	return json.Marshal(body)
}

func anthropicBlock(b Block) (any, error) {
	switch b := b.(type) {
	case Text:
		return map[string]any{"type": "text", "text": b.Text}, nil

	case Image:
		if b.URL != "" {
			return map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": b.URL}}, nil
		}
		return map[string]any{"type": "image", "source": map[string]any{
			"type":       "base64",
			"media_type": b.MediaType,
			"data":       base64.StdEncoding.EncodeToString(b.Data),
		}}, nil

	case Document:
		if b.URL != "" {
			return map[string]any{"type": "document", "source": map[string]any{"type": "url", "url": b.URL}}, nil
		}
		return map[string]any{"type": "document", "source": map[string]any{
			"type":       "base64",
			"media_type": cmp.Or(b.MediaType, "application/pdf"),
			"data":       base64.StdEncoding.EncodeToString(b.Data),
		}}, nil

	case ToolUse:
		input := b.Input
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return map[string]any{"type": "tool_use", "id": b.ID, "name": b.Name, "input": input}, nil

	case ToolResult:
		block := map[string]any{"type": "tool_result", "tool_use_id": b.ToolUseID, "content": b.Content}
		if b.IsError {
			block["is_error"] = true
		}
		return block, nil

	case Thinking:
		if len(b.Raw) == 0 {
			return nil, fmt.Errorf("goop: anthropic cannot send an empty Thinking block")
		}
		return b.Raw, nil

	default:
		return nil, fmt.Errorf("goop: anthropic cannot send a %T block", b)
	}
}

// anthropicBuild collects the deltas of one content block.
type anthropicBuild struct {
	kind      string
	id, name  string
	raw       json.RawMessage // the content_block_start event, for block types we don't parse
	text      strings.Builder
	inputJSON strings.Builder
	thinking  strings.Builder
	signature strings.Builder
}

func parseAnthropicStream(r io.Reader, emit func(Event)) (Reply, error) {
	var reply Reply
	var blocks []*anthropicBuild
	err := scanSSE(r, func(data string) error {
		var ev struct {
			Type    string `json:"type"`
			Index   int    `json:"index"`
			Message struct {
				Usage struct {
					InputTokens              int `json:"input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					OutputTokens             int `json:"output_tokens"`
				} `json:"usage"`
			} `json:"message"`
			ContentBlock json.RawMessage `json:"content_block"`
			Delta        struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
				Thinking    string `json:"thinking"`
				Signature   string `json:"signature"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return fmt.Errorf("goop: anthropic stream: %w", err)
		}

		switch ev.Type {
		case "message_start":
			// input_tokens excludes cached tokens, add them so Usage.Input is the real prompt size
			u := ev.Message.Usage
			reply.Usage.Input = u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
			reply.Usage.Output = u.OutputTokens

		case "content_block_start":
			for len(blocks) <= ev.Index {
				blocks = append(blocks, &anthropicBuild{})
			}
			var cb struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal(ev.ContentBlock, &cb); err != nil {
				return fmt.Errorf("goop: anthropic stream: content block: %w", err)
			}
			b := blocks[ev.Index]
			b.kind = cb.Type
			b.id = cb.ID
			b.name = cb.Name
			b.raw = ev.ContentBlock
			b.text.WriteString(cb.Text)

		case "content_block_delta":
			if ev.Index >= len(blocks) {
				return fmt.Errorf("goop: anthropic stream: delta for unknown block %d", ev.Index)
			}
			b := blocks[ev.Index]
			switch ev.Delta.Type {
			case "text_delta":
				b.text.WriteString(ev.Delta.Text)
				if emit != nil && ev.Delta.Text != "" {
					emit(TextDelta{Text: ev.Delta.Text})
				}
			case "input_json_delta":
				b.inputJSON.WriteString(ev.Delta.PartialJSON)
			case "thinking_delta":
				b.thinking.WriteString(ev.Delta.Thinking)
				if emit != nil && ev.Delta.Thinking != "" {
					emit(ThinkingDelta{Text: ev.Delta.Thinking})
				}
			case "signature_delta":
				b.signature.WriteString(ev.Delta.Signature)
			}

		case "message_delta":
			if ev.Delta.StopReason != "" {
				reply.StopReason = anthropicStop(ev.Delta.StopReason)
			}
			if ev.Usage.OutputTokens > 0 {
				reply.Usage.Output = ev.Usage.OutputTokens
			}

		case "message_stop":
			return ErrStreamDone
		case "error":
			return &APIError{Body: ev.Error.Type + ": " + ev.Error.Message}
		}
		return nil
	})
	if err != nil {
		return Reply{}, err
	}

	reply.Message = Message{Role: Assistant}
	for _, b := range blocks {
		switch b.kind {
		case "text":
			reply.Message.Blocks = append(reply.Message.Blocks, Text{Text: b.text.String()})
		case "tool_use":
			reply.Message.Blocks = append(reply.Message.Blocks, ToolUse{
				ID:    b.id,
				Name:  b.name,
				Input: json.RawMessage(cmp.Or(b.inputJSON.String(), "{}")),
			})

		case "thinking":
			raw, err := json.Marshal(map[string]string{
				"type":      "thinking",
				"thinking":  b.thinking.String(),
				"signature": b.signature.String(),
			})
			if err != nil {
				return Reply{}, err
			}
			reply.Message.Blocks = append(reply.Message.Blocks, Thinking{Raw: raw})

		default:
			// anything else (redacted_thinking etc.) is kept verbatim and sent back unchanged
			reply.Message.Blocks = append(reply.Message.Blocks, Thinking{Raw: b.raw})
		}
	}

	return reply, nil
}

func anthropicStop(s string) StopReason {
	switch s {
	case "end_turn", "stop_sequence":
		return StopEnd
	case "tool_use":
		return StopToolUse
	case "max_tokens":
		return StopMaxTokens
	}
	return StopReason(s)
}
