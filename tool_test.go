package goop

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
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
