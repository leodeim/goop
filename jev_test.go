package goop

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// jevServer answers every question with the given noul and records the last request body.
func jevServer(t *testing.T, noul float64) (*httptest.Server, *map[string]any) {
	t.Helper()
	var last map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("authorization") != "Bearer key" {
			t.Errorf("auth header = %q", r.Header.Get("authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &last); err != nil {
			t.Errorf("request body: %v", err)
		}
		answers := map[string]any{}
		for name := range last["questions"].(map[string]any) {
			answers[name] = map[string]any{"type": "noul", "noul": noul}
		}
		json.NewEncoder(w).Encode(map[string]any{"model": "jev-1.13.0", "answers": answers, "usage": map[string]int{"input_tokens": 5}})
	}))
	t.Cleanup(server.Close)
	return server, &last
}

func TestJevAsk(t *testing.T) {
	server, last := jevServer(t, 0.95)
	jev := &Jev{APIKey: "key", URL: server.URL}
	answers, err := jev.Ask(t.Context(), "Help! My payouts have been failing for 3 days.", map[string]JevQuestion{
		"is_urgent": {Type: "noul", Instructions: "Does this convey urgency?"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answers["is_urgent"].Noul != 0.95 {
		t.Fatalf("answers = %+v", answers)
	}
	// A custom URL is a local server: no model unless one is set.
	if _, has := (*last)["model"]; has {
		t.Fatalf("model sent to local server: %v", (*last)["model"])
	}
	q := (*last)["questions"].(map[string]any)["is_urgent"].(map[string]any)
	if q["type"] != "noul" || q["instructions"] != "Does this convey urgency?" || q["criteria"] != nil {
		t.Fatalf("question sent = %v", q)
	}
}

func TestJevDefaultModel(t *testing.T) {
	if (&Jev{}).model() != "jev-latest" {
		t.Fatal("hosted default model missing")
	}
	if (&Jev{URL: "http://local"}).model() != "" {
		t.Fatal("local server should get no model by default")
	}
	if (&Jev{URL: "http://local", Model: "laya"}).model() != "laya" {
		t.Fatal("explicit model dropped")
	}
}

func TestJevGate(t *testing.T) {
	use := ToolUse{ID: "t1", Name: "rm", Input: json.RawMessage(`{"path":"/"}`)}

	server, last := jevServer(t, 0.2)
	gate := (&Jev{APIKey: "key", URL: server.URL}).Gate("The call is safe.", 0.5)
	if _, err := gate(t.Context(), use); err == nil || !strings.Contains(err.Error(), "0.20") {
		t.Fatalf("err = %v, want refusal with probability", err)
	}
	state := (*last)["state"].(map[string]any)
	if state["tool"] != "rm" || state["input"].(map[string]any)["path"] != "/" {
		t.Fatalf("gate state = %v", state)
	}

	server, _ = jevServer(t, 0.9)
	gate = (&Jev{APIKey: "key", URL: server.URL}).Gate("The call is safe.", 0.5)
	got, err := gate(t.Context(), use)
	if err != nil || got.ID != "t1" || string(got.Input) != `{"path":"/"}` {
		t.Fatalf("got = %+v, err = %v", got, err)
	}

	// Fail closed: an unreachable jev refuses the call.
	gate = (&Jev{APIKey: "key", URL: "http://127.0.0.1:1", HTTP: HTTP{MaxRetries: -1}}).Gate("The call is safe.", 0.5)
	if _, err := gate(t.Context(), use); err == nil {
		t.Fatal("gate passed a call while jev was unreachable")
	}
}

func TestJevTool(t *testing.T) {
	server, last := jevServer(t, 0.7)
	tool := (&Jev{APIKey: "key", URL: server.URL}).Tool()
	if tool.Name != "jev" {
		t.Fatalf("name = %q", tool.Name)
	}
	var schema struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			AdditionalProperties struct {
				Required []string `json:"required"`
			} `json:"additionalProperties"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(tool.Schema, &schema); err != nil {
		t.Fatal(err)
	}
	if len(schema.Required) != 2 || strings.Join(schema.Properties["questions"].AdditionalProperties.Required, ",") != "type,instructions" {
		t.Fatalf("schema = %s", tool.Schema)
	}

	input := `{"state":{"review":"a delight"},"questions":{"positive":{"type":"noul","instructions":"The review is positive."}}}`
	out, err := tool.Run(t.Context(), json.RawMessage(input))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, `"noul":0.7`) {
		t.Fatalf("out = %s", out)
	}
	if (*last)["state"].(map[string]any)["review"] != "a delight" {
		t.Fatalf("state sent = %v", (*last)["state"])
	}
}
