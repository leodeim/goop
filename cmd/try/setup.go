package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/leodeim/goop"
)

var defaultEnvFile = func() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ".env"
	}
	return filepath.Join(filepath.Dir(file), ".env")
}()

func loadEnv(path string) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) && path == defaultEnvFile {
		return nil
	}
	if err != nil {
		return err
	}
	for n, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(strings.TrimPrefix(line, "export "), "=")
		if !ok {
			return fmt.Errorf("%s:%d: expected KEY=VALUE", path, n+1)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if len(value) >= 2 && (value[0] == '"' || value[0] == '\'') && value[len(value)-1] == value[0] {
			value = value[1 : len(value)-1]
		}
		if _, set := os.LookupEnv(key); !set {
			os.Setenv(key, value)
		}
	}
	return nil
}

func newAgent(name, model string) (*goop.Agent, error) {
	name = cmp.Or(name, os.Getenv("GOOP_PROVIDER"), "anthropic")
	key, baseURL := os.Getenv("GOOP_API_KEY"), os.Getenv("GOOP_BASE_URL")
	model = cmp.Or(model, os.Getenv("GOOP_MODEL"))
	agent := &goop.Agent{
		System: cmp.Or(os.Getenv("GOOP_SYSTEM"), "You are a terse assistant. Use the tools when they help."),
		Tools:  tools(),
	}
	if jevKey, jevURL := os.Getenv("GOOP_JEV_KEY"), os.Getenv("GOOP_JEV_URL"); jevKey != "" || jevURL != "" {
		jev := &goop.Jev{APIKey: jevKey, URL: jevURL, Model: os.Getenv("GOOP_JEV_MODEL")}
		agent.Tools = append(agent.Tools, jev.Tool())
		agent.OnToolCall = jev.Gate("This tool call is reasonable for an assistant to make on the user's behalf.", 0.5)
	}
	switch name {
	case "anthropic":
		if key == "" {
			return nil, fmt.Errorf("GOOP_API_KEY is not set; copy .env.example to %s", defaultEnvFile)
		}
		agent.Provider = &goop.Anthropic{APIKey: key, BaseURL: baseURL}
		agent.Model = cmp.Or(model, "claude-sonnet-5")
	case "openai", "responses":
		if key == "" && baseURL == "" { // a local server needs no key
			return nil, fmt.Errorf("GOOP_API_KEY or GOOP_BASE_URL is not set; copy .env.example to %s", defaultEnvFile)
		}
		if name == "openai" {
			agent.Provider = &goop.OpenAI{APIKey: key, BaseURL: baseURL}
		} else {
			agent.Provider = &goop.OpenAIResponses{APIKey: key, BaseURL: baseURL}
		}
		agent.Model = cmp.Or(model, "gpt-5")
	default:
		return nil, fmt.Errorf("unknown provider %q (want anthropic, openai or responses)", name)
	}
	return agent, nil
}

func tools() []goop.Tool {
	weather := must(goop.NewTool("get_weather", "Current weather in a city (fake data).",
		func(_ context.Context, in struct {
			City string `json:"city" desc:"city name"`
		}) (string, error) {
			return fmt.Sprintf(`{"city":%q,"temperature_c":%d,"conditions":"light rain"}`, in.City, 10+len(in.City)%15), nil
		}))
	now := must(goop.NewTool("now", "The current date and time in UTC.",
		func(context.Context, struct{}) (string, error) {
			return time.Now().UTC().Format(time.RFC3339), nil
		}))
	return []goop.Tool{weather, now}
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
