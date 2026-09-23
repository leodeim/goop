package goop

import (
	"bytes"
	"context"
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"time"
)

// Tool is a function exposed to the model.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Run         func(ctx context.Context, input json.RawMessage) (string, error)
}

func (t Tool) validate() error {
	switch {
	case t.Name == "":
		return errors.New("goop: tool with an empty name")
	case t.Run == nil:
		return fmt.Errorf("goop: tool %s has no Run func", t.Name)
	case !json.Valid(t.Schema) || !bytes.HasPrefix(bytes.TrimSpace(t.Schema), []byte("{")):
		return fmt.Errorf("goop: tool %s: Schema must be a JSON object", t.Name)
	}
	return nil
}

// NewTool builds a Tool whose input schema is derived from the struct type In. Field names
// come from the json tag, descriptions from the desc tag, and pointer, omitempty or omitzero
// fields are optional. Input with unknown or missing required fields is rejected before fn runs.
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
			if len(bytes.TrimSpace(input)) == 0 {
				input = json.RawMessage("{}")
			}
			if err := checkRequired(schema, input, ""); err != nil {
				return "", fmt.Errorf("bad input: %w", err)
			}
			var in In
			dec := json.NewDecoder(bytes.NewReader(input))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&in); err != nil {
				return "", fmt.Errorf("bad input: %w", err)
			}
			return fn(ctx, in)
		},
	}, nil
}

// checkRequired reports the first required property missing (or null) anywhere in input.
// Type mismatches and unknown fields are left to the decoder.
func checkRequired(schema map[string]any, input json.RawMessage, path string) error {
	if string(bytes.TrimSpace(input)) == "null" {
		return nil // an absent optional value, required ones are caught by the parent
	}
	switch schema["type"] {
	case "object":
		var fields map[string]json.RawMessage
		if json.Unmarshal(input, &fields) != nil {
			return nil
		}
		props, _ := schema["properties"].(map[string]any)
		if props == nil { // a map: every value has the additionalProperties schema
			values, _ := schema["additionalProperties"].(map[string]any)
			for name, v := range fields {
				if err := checkRequired(values, v, path+name+"."); err != nil {
					return err
				}
			}
			return nil
		}
		required, _ := schema["required"].([]string)
		for _, name := range required {
			if v, ok := fields[name]; !ok || string(v) == "null" {
				return fmt.Errorf("missing required field %s%s", path, name)
			}
		}
		for name, v := range fields {
			if prop, ok := props[name].(map[string]any); ok {
				if err := checkRequired(prop, v, path+name+"."); err != nil {
					return err
				}
			}
		}

	case "array":
		items, _ := schema["items"].(map[string]any)
		var elems []json.RawMessage
		if items == nil || json.Unmarshal(input, &elems) != nil {
			return nil
		}
		for i, v := range elems {
			if err := checkRequired(items, v, fmt.Sprintf("%s%d.", path, i)); err != nil {
				return err
			}
		}
	}
	return nil
}

var (
	rawMessageType    = reflect.TypeFor[json.RawMessage]()
	timeType          = reflect.TypeFor[time.Time]()
	jsonMarshalerType = reflect.TypeFor[json.Marshaler]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
)

func implements(t, iface reflect.Type) bool {
	return t.Implements(iface) || reflect.PointerTo(t).Implements(iface)
}

func schemaOf(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	// Custom encodings take precedence over the kind, as they do in encoding/json.
	switch {
	case t == rawMessageType:
		return map[string]any{}, nil // any JSON value
	case t == timeType:
		return map[string]any{"type": "string", "format": "date-time"}, nil
	case t.Kind() != reflect.Pointer && implements(t, jsonMarshalerType):
		return map[string]any{}, nil // encoding unknown, accept any JSON value
	case t.Kind() != reflect.Pointer && implements(t, textMarshalerType):
		return map[string]any{"type": "string"}, nil
	case t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8:
		return map[string]any{"type": "string", "contentEncoding": "base64"}, nil
	}

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

		name, rest, _ := strings.Cut(f.Tag.Get("json"), ",")
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
		opts := strings.Split(rest, ",")
		if f.Type.Kind() != reflect.Pointer && !slices.Contains(opts, "omitempty") && !slices.Contains(opts, "omitzero") {
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
