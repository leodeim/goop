package goop

import (
	"context"
	"encoding/json"
	"net/netip"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestSchemaInference(t *testing.T) {
	type nested struct {
		Depth int `json:"depth"`
	}
	tests := []struct {
		name string
		make func() (Tool, error)
		want string
	}{
		{
			name: "flat fields with tags",
			make: func() (Tool, error) {
				return NewTool("t", "d", func(_ context.Context, in struct {
					City  string  `json:"city" desc:"the city"`
					Count int     `json:"count"`
					Ratio float64 `json:"ratio"`
					Sure  bool    `json:"sure"`
				}) (string, error) {
					return "", nil
				})
			},
			want: `{"additionalProperties":false,"properties":{` +
				`"city":{"description":"the city","type":"string"},"count":{"type":"integer"},` +
				`"ratio":{"type":"number"},"sure":{"type":"boolean"}},` +
				`"required":["city","count","ratio","sure"],"type":"object"}`,
		},
		{
			name: "optional pointer, skipped and unexported fields",
			make: func() (Tool, error) {
				return NewTool("t", "d", func(_ context.Context, in struct {
					Name   string `json:"name"`
					Limit  *int   `json:"limit"`
					Hidden string `json:"-"`
				}) (string, error) {
					return "", nil
				})
			},
			want: `{"additionalProperties":false,"properties":{"limit":{"type":"integer"},"name":{"type":"string"}},"required":["name"],"type":"object"}`,
		},
		{
			name: "arrays, maps and nesting",
			make: func() (Tool, error) {
				return NewTool("t", "d", func(_ context.Context, in struct {
					Tags   []string       `json:"tags"`
					Counts map[string]int `json:"counts"`
					Inner  nested         `json:"inner"`
				}) (string, error) {
					return "", nil
				})
			},
			want: `{"additionalProperties":false,"properties":{` +
				`"counts":{"additionalProperties":{"type":"integer"},"type":"object"},` +
				`"inner":{"additionalProperties":false,"properties":{"depth":{"type":"integer"}},"required":["depth"],"type":"object"},` +
				`"tags":{"items":{"type":"string"},"type":"array"}},` +
				`"required":["tags","counts","inner"],"type":"object"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, err := tt.make()
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(tool.Schema, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tt.want), &want); err != nil {
				t.Fatal(err)
			}
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(want)
			if string(gotJSON) != string(wantJSON) {
				t.Fatalf("schema =\n%s\nwant\n%s", gotJSON, wantJSON)
			}
		})
	}
}

func TestSchemaAnyAndOmitempty(t *testing.T) {
	tool, err := NewTool("t", "d", func(_ context.Context, in struct {
		Value any    `json:"value"`
		Note  string `json:"note,omitempty"`
	}) (string, error) {
		return "", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Properties map[string]map[string]any `json:"properties"`
		Required   []string                  `json:"required"`
	}
	if err := json.Unmarshal(tool.Schema, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Properties["value"]) != 0 {
		t.Fatalf("any field schema = %v, want empty (any value)", got.Properties["value"])
	}
	if len(got.Required) != 1 || got.Required[0] != "value" {
		t.Fatalf("required = %v, want only value", got.Required)
	}
}

func TestSchemaRejectsUnsupported(t *testing.T) {
	_, err := NewTool("t", "d", func(_ context.Context, in struct {
		Ch chan int `json:"ch"`
	}) (string, error) {
		return "", nil
	})
	if err == nil {
		t.Fatal("channel field did not error")
	}
}

func TestSchemaRejectsRecursion(t *testing.T) {
	type node struct {
		Value    string `json:"value"`
		Children []node `json:"children"`
	}
	_, err := NewTool("t", "d", func(_ context.Context, in node) (string, error) {
		return "", nil
	})
	if err == nil || !strings.Contains(err.Error(), "recursive") {
		t.Fatalf("err = %v, want recursion error", err)
	}
}

func TestSchemaAllowsRepeatedSiblings(t *testing.T) {
	type point struct {
		X int `json:"x"`
		Y int `json:"y"`
	}
	_, err := NewTool("t", "d", func(_ context.Context, in struct {
		From point `json:"from"`
		To   point `json:"to"`
	}) (string, error) {
		return "", nil
	})
	if err != nil {
		t.Fatalf("same struct twice on separate paths errored: %v", err)
	}
}

func TestSchemaRejectsEmbedded(t *testing.T) {
	type base struct {
		ID string `json:"id"`
	}
	_, err := NewTool("t", "d", func(_ context.Context, in struct {
		base
		Name string `json:"name"`
	}) (string, error) {
		return "", nil
	})
	if err == nil || !strings.Contains(err.Error(), "embedded") {
		t.Fatalf("err = %v, want embedded struct error", err)
	}
}

func TestNewToolRejectsNonStruct(t *testing.T) {
	_, err := NewTool("t", "d", func(_ context.Context, in string) (string, error) {
		return "", nil
	})
	if err == nil {
		t.Fatal("non-struct input type did not error")
	}
}

func TestToolRunUnmarshals(t *testing.T) {
	tool, err := NewTool("sum", "adds", func(_ context.Context, in struct {
		A int `json:"a"`
		B int `json:"b"`
	}) (string, error) {
		return strconv.Itoa(in.A + in.B), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.Run(t.Context(), json.RawMessage(`{"a":2,"b":3}`))
	if err != nil {
		t.Fatal(err)
	}
	if out != "5" {
		t.Fatalf("out = %q, want 5", out)
	}
	if _, err := tool.Run(t.Context(), json.RawMessage(`{"a":"x"}`)); err == nil {
		t.Fatal("bad input did not error")
	}
}

func TestSchemaCustomEncodings(t *testing.T) {
	tool, err := NewTool("t", "d", func(_ context.Context, in struct {
		Blob []byte          `json:"blob"`
		Raw  json.RawMessage `json:"raw"`
		When time.Time       `json:"when"`
		Addr netip.Addr      `json:"addr"`
		Opt  int             `json:"opt,omitzero"`
	}) (string, error) {
		return "", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"additionalProperties":false,"properties":{` +
		`"addr":{"type":"string"},"blob":{"contentEncoding":"base64","type":"string"},"opt":{"type":"integer"},` +
		`"raw":{},"when":{"format":"date-time","type":"string"}},` +
		`"required":["blob","raw","when","addr"],"type":"object"}`
	if string(tool.Schema) != want {
		t.Fatalf("schema\n got %s\nwant %s", tool.Schema, want)
	}
	if _, err := tool.Run(t.Context(), json.RawMessage(`{"blob":"aGk=","raw":[1],"when":"2026-01-02T03:04:05Z","addr":"10.0.0.1"}`)); err != nil {
		t.Fatalf("input matching the schema failed: %v", err)
	}
}

func TestToolRunValidatesInput(t *testing.T) {
	type item struct {
		ID   int     `json:"id"`
		Note *string `json:"note"`
	}
	var got []item
	tool, err := NewTool("t", "d", func(_ context.Context, in struct {
		Name  string          `json:"name"`
		Inner *item           `json:"inner"`
		Items []item          `json:"items,omitempty"`
		ByKey map[string]item `json:"by_key,omitempty"`
	}) (string, error) {
		got = in.Items
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		input, wantErr string
	}{
		{`{"name":"a"}`, ""},
		{`{"name":"a","inner":null,"items":[{"id":1,"note":null}]}`, ""},
		{``, "missing required field name"},
		{`{"name":null}`, "missing required field name"},
		{`{"name":"a","typo":1}`, "unknown field"},
		{`{"name":"a","inner":{"note":"x"}}`, "missing required field inner.id"},
		{`{"name":"a","items":[{"id":1},{}]}`, "missing required field items.1.id"},
		{`{"name":"a","by_key":{"k":{}}}`, "missing required field by_key.k.id"},
		{`{"name":"a","items":[{"id":1,"extra":true}]}`, "unknown field"},
	}
	for _, tt := range tests {
		_, err := tool.Run(t.Context(), json.RawMessage(tt.input))
		switch {
		case tt.wantErr == "" && err != nil:
			t.Errorf("%s: unexpected error %v", tt.input, err)
		case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
			t.Errorf("%s: error = %v, want %q", tt.input, err, tt.wantErr)
		}
	}
	if len(got) != 1 || got[0].ID != 1 {
		t.Fatalf("items = %+v", got)
	}
}
