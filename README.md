<p align="center">
  <img src="logo.png" width="300" alt="goop logo">
</p>

# goop

A minimal tool-using agent loop for Go, standard library only. Give it a
provider, a model and some Go functions. It streams the answer, runs the
tools the model asks for, feeds the results back and loops until the model
is done. Providers: Anthropic (Messages API) and OpenAI (Chat Completions,
which also covers OpenRouter, LM Studio, Ollama and other compatible servers).

## Features

- `Agent.Run` is an iterator yielding `TextDelta`, `ThinkingDelta`,
  `ToolCall` and `ToolReturn`, ending in one `Done` or one error.
- `Done` carries the whole conversation and token usage; pass it back in to
  continue the chat.
- `NewTool` infers a JSON schema from a Go struct: json tag names, `desc`
  tags as descriptions, pointer and `omitempty` fields optional.
- Tool errors go back to the model as error results; `OnToolCall` can
  inspect, rewrite or refuse any call.
- `MaxTurns` and `Budget` (total tokens) bound a run; both return a
  `LimitError` that keeps the conversation so far.
- Text, images and documents (Anthropic only) as input; thinking blocks from
  reasoning models are carried through so tool loops keep working.
- Streaming over SSE for both providers, retries with backoff on 408/429/5xx
  (`HTTP.MaxRetries`, default 2), and a truncated stream is an error rather
  than a half answer.
- Anthropic requests set prompt-cache breakpoints; OpenAI requests use
  `max_completion_tokens` and stream reasoning deltas where servers send them.
- Breaking out of the `range` loop cancels the request.
- `Jev` calls TypeSafe's System One model (or a local laya-mlx server) for
  calibrated yes/no, choice and score decisions. `Jev.Gate` plugs into
  `OnToolCall` as a fast approval check; `Jev.Tool` lets the model ask it.

## Example

```go
weather, _ := goop.NewTool("get_weather", "Look up the weather in a city.",
	func(ctx context.Context, in struct {
		City string `json:"city" desc:"the city to look up"`
	}) (string, error) {
		return `{"temperature_c":18,"conditions":"light rain"}`, nil
	})

agent := &goop.Agent{
	Provider: &goop.Anthropic{APIKey: os.Getenv("ANTHROPIC_API_KEY")},
	Model:    "claude-sonnet-5",
	Tools:    []goop.Tool{weather},
}

var conv []goop.Message
for ev, err := range agent.Run(ctx, conv, goop.Text{Text: "Will it rain in Paris?"}) {
	if err != nil {
		log.Fatal(err)
	}
	switch e := ev.(type) {
	case goop.TextDelta:
		fmt.Print(e.Text)
	case goop.Done:
		conv = e.Messages // continue from here next time
	}
}
```

Swap the provider for `&goop.OpenAI{APIKey: ..., BaseURL: ...}` and set a
model; nothing else changes.

## Try it

```bash
cp cmd/try/.env.example cmd/try/.env   # set GOOP_PROVIDER and GOOP_API_KEY
go run ./cmd/try "Will it rain in Paris?"
go run ./cmd/try                        # REPL; ctrl-c stops a turn, ctrl-d quits
```
