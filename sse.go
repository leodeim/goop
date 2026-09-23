package goop

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrTruncatedStream means the connection closed before the stream's final event arrived.
var ErrTruncatedStream = errors.New("goop: stream ended before its terminal event")

// ErrStreamDone is returned by an SSE callback to stop scanning without an error.
var ErrStreamDone = errors.New("goop: stream done")

// maxSSELine bounds one SSE line. Responses sends the whole output, encrypted reasoning
// included, in a single response.completed line, so this has to fit the largest answer.
const maxSSELine = 64 << 20

func scanSSE(r io.Reader, fn func(data string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), maxSSELine)
	var data []string

	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			if len(data) > 0 {
				err := fn(strings.Join(data, "\n"))
				data = data[:0]
				if errors.Is(err, ErrStreamDone) {
					return nil
				}
				if err != nil {
					return err
				}
			}
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("goop: sse: %w", err)
	}
	return ErrTruncatedStream
}
