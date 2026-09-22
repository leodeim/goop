package goop

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// Jev is a client for TypeSafe's System One model: typed, calibrated decisions from one call.
// URL defaults to the hosted API. Point it at a laya-mlx server's /v1/predict to run locally.
type Jev struct {
	APIKey string
	URL    string // default https://api.typesafe.ai/v1/systemone
	Model  string // default jev-latest on the hosted API, omitted otherwise
	HTTP
}

// JevQuestion is one typed decision. Criteria is an option->description map for "choice", an
// ordered list of rubric levels for "score", and unused for "noul".
type JevQuestion struct {
	Type         string `json:"type" desc:"noul (yes/no), choice or score"`
	Instructions string `json:"instructions" desc:"the question, phrased for the type"`
	Criteria     any    `json:"criteria,omitempty" desc:"choice: map of option to description; score: ordered list of levels"`
}

// JevAnswer is the decision for one JevQuestion. Which fields are set depends on Type.
type JevAnswer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// Ask answers every question about state. State may be a string, an object or a list.
func (j *Jev) Ask(ctx context.Context, state any, questions map[string]JevQuestion) (map[string]JevAnswer, error) {
	req := map[string]any{"state": state, "questions": questions}
	if model := j.model(); model != "" {
		req["model"] = model
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := j.post(ctx, cmp.Or(j.URL, "https://api.typesafe.ai/v1/systemone"), bearer(j.APIKey), body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out struct {
		Answers map[string]JevAnswer `json:"answers"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, fmt.Errorf("goop: jev response: %w", err)
	}

	return out.Answers, nil
}

func (j *Jev) model() string {
	if j.Model == "" && j.URL == "" {
		return "jev-latest"
	}
	return j.Model
}

// Gate returns an Agent.OnToolCall that asks jev whether statement holds for each pending call
// (state: tool name and input) and refuses the call when P(yes) is below threshold. A failed
// request also refuses, so the gate fails closed.
func (j *Jev) Gate(statement string, threshold float64) func(context.Context, ToolUse) (ToolUse, error) {
	return func(ctx context.Context, use ToolUse) (ToolUse, error) {
		answers, err := j.Ask(ctx, map[string]any{"tool": use.Name, "input": use.Input},
			map[string]JevQuestion{"ok": {Type: "noul", Instructions: statement}})
		if err != nil {
			return ToolUse{}, fmt.Errorf("jev gate: %w", err)
		}
		if p := answers["ok"].Noul; p < threshold {
			return ToolUse{}, fmt.Errorf("refused: jev puts P(%q) at %.2f", statement, p)
		}
		return use, nil
	}
}

// Tool exposes jev to the model: it supplies a state and named questions and gets the answers
// back as JSON.
func (j *Jev) Tool() Tool {
	tool, err := NewTool("jev", "Get calibrated decisions about a state from a specialised model: "+
		"noul gives P(yes) for a statement, choice picks an option with probabilities, score rates on a rubric. "+
		"Use it when a judgment should be a probability rather than a guess.",
		func(ctx context.Context, in struct {
			State     any                    `json:"state" desc:"the text or facts to decide about"`
			Questions map[string]JevQuestion `json:"questions" desc:"named questions to answer about the state"`
		}) (string, error) {
			answers, err := j.Ask(ctx, in.State, in.Questions)
			if err != nil {
				return "", err
			}
			out, err := json.Marshal(answers)
			return string(out), err
		})
	if err != nil {
		panic(err) // the input type is fixed, so this cannot fail at run time
	}
	return tool
}
