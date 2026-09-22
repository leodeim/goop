package goop

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// Tool is a function exposed to the model.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Run         func(ctx context.Context, input json.RawMessage) (string, error)
}

// NewTool builds a Tool whose input schema is derived from the struct type In. Field names
// come from the json tag, descriptions from the desc tag, and pointer or omitempty fields are optional.
func NewTool[In any](name, description string, fn func(ctx context.Context, in In) (string, error)) (Tool, error) {
	rt := reflect.TypeFor[In]()
	if rt.Kind() != reflect.Struct {
		return Tool{}, fmt.Errorf("goop: tool %s: input type %s is not a struct", name, rt)
	}

	schema, err := schemaOf(rt, map[reflect.Type]bool{})
	if err != nil {
		return Tool{}, fmt.Errorf("goop: tool %s: %w", name, err)
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return Tool{}, fmt.Errorf("goop: tool %s: %w", name, err)
	}

	return Tool{
		Name:        name,
		Description: description,
		Schema:      raw,
		Run: func(ctx context.Context, input json.RawMessage) (string, error) {
			var in In
			if len(input) > 0 {
				if err := json.Unmarshal(input, &in); err != nil {
					return "", fmt.Errorf("bad input: %w", err)
				}
			}
			return fn(ctx, in)
		},
	}, nil
}

func schemaOf(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	switch t.Kind() {
	case reflect.Pointer:
		return schemaOf(t.Elem(), seen)
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Interface:
		return map[string]any{}, nil // any JSON value

	case reflect.Slice, reflect.Array:
		items, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil

	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("map key must be string, not %s", t.Key())
		}
		values, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": values}, nil

	case reflect.Struct:
		if seen[t] {
			return nil, fmt.Errorf("recursive type %s", t)
		}
		seen[t] = true
		defer delete(seen, t)
		return structSchema(t, seen)

	default:
		return nil, fmt.Errorf("cannot express %s as a JSON schema", t)
	}
}

func structSchema(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	properties := map[string]any{}
	required := []string{}
	for i := range t.NumField() {
		f := t.Field(i)
		if f.Anonymous {
			return nil, fmt.Errorf("field %s: embedded structs are not supported", f.Name)
		}
		if !f.IsExported() {
			continue
		}

		name, opts, _ := strings.Cut(f.Tag.Get("json"), ",")
		if name == "-" {
			continue
		}
		if name == "" {
			name = f.Name
		}

		prop, err := schemaOf(f.Type, seen)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", f.Name, err)
		}
		if desc := f.Tag.Get("desc"); desc != "" {
			prop["description"] = desc
		}

		properties[name] = prop
		if f.Type.Kind() != reflect.Pointer && !strings.Contains(opts, "omitempty") {
			required = append(required, name)
		}
	}

	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}, nil
}
