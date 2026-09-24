<p align="center">
  <img src="logo.png" width="300" alt="goop logo">
</p>

# goop

A minimal tool-using agent loop for Go, standard library only. Give it a
provider, a model and some Go functions; it streams the answer, runs the tools
the model asks for and loops until the model is done.

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
		conv = e.Messages // pass back in to continue the chat
	}
}
```

## Features

| Feature | How |
| --- | --- |
| Streaming events | `Agent.Run` yields `TextDelta`, `ThinkingDelta`, `ToolCall`, `ToolReturn`, then one `Done` or one error |
| Typed tools | `NewTool` builds the JSON schema from a Go struct: `json` names, `desc` descriptions, pointer / `omitempty` / `omitzero` fields optional |
| Input validation | Unknown or missing required fields go back to the model as an error; the tool never runs |
| Tool errors | Returned errors and panics become error results the model can react to |
| Approval hook | `OnToolCall` can inspect, rewrite or refuse any call |
| Limits | `MaxTurns` (default 10) and `Budget` (total tokens) return a `LimitError` that keeps the conversation |
| Retries | 408, 429 and 5xx with backoff and `Retry-After`; `HTTP.MaxRetries`, default 2 |
| Safe streams | A dropped connection is `ErrTruncatedStream`, never a half answer |
| Cancellation | Breaking out of the `range` loop cancels the request |
| Jev | `Jev.Gate` approves tool calls with TypeSafe's System One model; `Jev.Tool` lets the model ask it |
| Routing | `Router` asks jev which of your models fits each request and runs the agent there, see [Routing](#routing) |

## Providers

| | `Anthropic` | `OpenAIResponses` | `OpenAI` |
| --- | :---: | :---: | :---: |
| API | Messages | Responses | Chat Completions, plus compatible servers (OpenRouter, LM Studio, Ollama, …) |
| Images | ✓ | ✓ | ✓ |
| Documents (PDF) | ✓ | ✓ | – |
| Streamed reasoning | ✓ | ✓ | ✓ |
| Reasoning kept across tool calls | ✓ | ✓ | – |
| Prompt caching | ✓ | server side | server side |
| Options | `Thinking` | `ReasoningEffort` | `ReasoningEffort`, `LegacyMaxTokens` |

Every provider takes `APIKey`, `BaseURL` and `HTTP`. Swapping one for another
changes nothing else.

## Routing

`Router` sends each run to one of several provider/model pairs. Jev picks the
route from the `When` descriptions and returns a probability for each one.

```go
router := &goop.Router{
	Jev: &goop.Jev{APIKey: os.Getenv("TYPESAFE_API_KEY"), Model: "jev-1.13.0"},
	Routes: []goop.Route{
		{Name: "fast", When: "short factual questions and small talk",
			Provider: anthropic, Model: "claude-haiku-4-5"},
		{Name: "deep", When: "multi-step reasoning, code and analysis",
			Provider: anthropic, Model: "claude-opus-5-5"},
	},
	Default:       "deep", // when jev fails, times out or is unsure
	MinConfidence: 0.6,
	Agent:         goop.Agent{System: system, Tools: tools},
}

for ev, err := range router.Run(ctx, conv, goop.Text{Text: question}) {
	// the first event is goop.Routed{Route, Probabilities, Err}, then the usual events
}
```

- **Data**: the latest user message goes to the jev API before any model sees
  it. Set `Jev.URL` to a [laya-mlx](https://github.com/mizorewww/laya-mlx)
  server to keep it on your machine.
- **Cost**: every run adds one jev call (2s limit, `Router.Timeout`). It pays
  off when the routes differ a lot in price or speed.
- **Fails open**: if jev errors, times out or is below `MinConfidence`, the run
  goes to `Default`. `Routed.Err` says why jev's answer was not used, so log it.
- **Pin the jev model**: `jev-latest` changes over time, and your routing
  changes with it.
- **One route per run**: tool calls stay on the chosen route. Thinking blocks are
  dropped from earlier turns, because a different provider cannot read them.
- **Hard rules first**: conversations with a `Document` never go to `OpenAI`
  (chat completions cannot read them).
- **Check it on your traffic**: `Router.Pick` routes a conversation without
  running it, so you can review picks before turning routing on.

## Try it

```bash
cp cmd/try/.env.example cmd/try/.env
go run ./cmd/try
```
