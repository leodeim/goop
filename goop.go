// Package goop implements a small agent loop on top of the Anthropic and OpenAI
// streaming APIs. It has no dependencies outside the standard library.
package goop

import (
	"context"
	"encoding/json"
	"fmt"
)

// Role is the author of a message, "user" or "assistant".
type Role string

const (
	User      Role = "user"
	Assistant Role = "assistant"
)

// Block is a piece of message content (Text, Image, Document, ToolUse, ToolResult or Thinking).
type Block interface{ block() }

// Text is a plain text block.
type Text struct {
	Text string
}

// Image is an image block, either inline Data with a MediaType or a URL.
type Image struct {
	MediaType string // required with Data, e.g. "image/png"
	Data      []byte
	URL       string
}

// Document is a file (PDF etc.) for the model to read. Anthropic and OpenAIResponses support it.
type Document struct {
	MediaType string // defaults to application/pdf
	Data      []byte
	URL       string
}

// ToolUse is a tool call requested by the model.
type ToolUse struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult is the output of a tool call, matched to its ToolUse by ID.
type ToolResult struct {
	ToolUseID string
	Content   string
	IsError   bool
}

// Thinking is a reasoning block kept as the provider's raw JSON. It has to be sent back
// unchanged, Anthropic rejects a replayed tool call without it.
type Thinking struct {
	Raw json.RawMessage
}

func (Text) block()       {}
func (Image) block()      {}
func (Document) block()   {}
func (ToolUse) block()    {}
func (ToolResult) block() {}
func (Thinking) block()   {}

// Message is a single user or assistant message.
type Message struct {
	Role   Role
	Blocks []Block
}

// StopReason says why the model stopped.
type StopReason string

const (
	StopEnd       StopReason = "end"        // normal end of the response
	StopToolUse   StopReason = "tool_use"   // waiting for tool results
	StopMaxTokens StopReason = "max_tokens" // hit MaxTokens
)

// Usage is a token count. Agent.Run sums it over all turns of a run.
type Usage struct {
	Input  int
	Output int
}

// ToolDef is the tool definition sent to the model.
type ToolDef struct {
	Name        string
	Description string
	Schema      json.RawMessage // JSON Schema for the input object
}

// Request is a single call to the model.
type Request struct {
	Model     string
	System    string
	Messages  []Message
	Tools     []ToolDef
	MaxTokens int // 0 means provider default
}

// Provider is a model API. Stream sends one Request and calls emit (may be nil) for
// every TextDelta and ThinkingDelta as it comes in.
type Provider interface {
	Stream(ctx context.Context, req Request, emit func(Event)) (Reply, error)
}

// Reply is the model's complete response to a Request.
type Reply struct {
	Message    Message
	StopReason StopReason
	Usage      Usage
}

// Event is anything Agent.Run yields: TextDelta, ThinkingDelta, ToolCall, ToolReturn or Done.
type Event interface{ isEvent() }

// TextDelta is a chunk of streamed answer text.
type TextDelta struct {
	Text string
}

// ThinkingDelta is a chunk of streamed reasoning. Only good for display, it is not part of the conversation.
type ThinkingDelta struct {
	Text string
}

// ToolCall is yielded right before a tool runs.
type ToolCall struct {
	Use ToolUse
}

// ToolReturn is yielded after a tool ran, whether it succeeded or not.
type ToolReturn struct {
	Result ToolResult
}

// Done is the last event of a successful run. Messages holds the full conversation including the new turns.
type Done struct {
	Messages   []Message
	StopReason StopReason
	Usage      Usage
}

func (TextDelta) isEvent()     {}
func (ThinkingDelta) isEvent() {}
func (ToolCall) isEvent()      {}
func (ToolReturn) isEvent()    {}
func (Done) isEvent()          {}

// APIError is an error from the provider, either a non-2xx status or an error event inside the stream.
type APIError struct {
	Status int // 0 for errors reported inside the stream
	Body   string
}

func (e *APIError) Error() string {
	if e.Status == 0 {
		return "goop: api error: " + e.Body
	}
	return fmt.Sprintf("goop: api error: status %d: %s", e.Status, e.Body)
}
