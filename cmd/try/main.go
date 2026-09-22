// Command try is a small CLI for trying the library against a real API.
// Keys are read from a .env file in this directory.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"goop"
)

func main() {
	envFile := flag.String("env", defaultEnvFile, "KEY=VALUE file loaded into the environment")
	provider := flag.String("provider", "", "anthropic or openai (default: GOOP_PROVIDER, then anthropic)")
	model := flag.String("model", "", "model id (default: claude-sonnet-5 or gpt-5)")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: try [flags] [prompt...]")
		fmt.Fprintln(os.Stderr, "With no prompt, reads one prompt per line from stdin and keeps the conversation.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if err := run(*envFile, *provider, *model, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintln(os.Stderr, "try:", err)
		os.Exit(1)
	}
}

func run(envFile, providerName, model, prompt string) error {
	if err := loadEnv(envFile); err != nil {
		return err
	}
	agent, err := newAgent(providerName, model)
	if err != nil {
		return err
	}
	ctx := context.Background()
	var conv []goop.Message
	if prompt != "" {
		_, err := turn(ctx, agent, conv, prompt)
		return err
	}
	fmt.Fprintln(os.Stderr, dim(fmt.Sprintf("[%T] [%s] chat:", agent.Provider, agent.Model)))
	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !in.Scan() {
			fmt.Println()
			return in.Err()
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		if conv, err = turn(ctx, agent, conv, line); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
		}
	}
}

func turn(ctx context.Context, agent *goop.Agent, conv []goop.Message, prompt string) ([]goop.Message, error) {
	// Only catch ctrl-c while a turn is running so it cancels the turn. At the
	// prompt the default handler kills the process, which is what we want.
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()

	start := time.Now()
	var out printer
	out.wait()
	for ev, err := range agent.Run(ctx, conv, goop.Text{Text: prompt}) {
		if err != nil {
			out.endLine()
			if ctx.Err() != nil {
				out.note("(interrupted)")
				return conv, nil
			}
			var limit *goop.LimitError
			if errors.As(err, &limit) {
				return limit.Messages, err
			}
			return conv, err
		}
		switch e := ev.(type) {
		case goop.TextDelta:
			out.text(e.Text)
		case goop.ThinkingDelta:
			out.thought(e.Text)
		case goop.ToolCall:
			out.note(fmt.Sprintf("[%s %s]", e.Use.Name, e.Use.Input))
		case goop.ToolReturn:
			mark := "->"
			if e.Result.IsError {
				mark = "!!"
			}
			out.note(fmt.Sprintf("[%s %s]", mark, e.Result.Content))
			out.wait()
		case goop.Done:
			out.note(fmt.Sprintf("(%s: %d in, %d out, %s)",
				e.StopReason, e.Usage.Input, e.Usage.Output, time.Since(start).Round(time.Millisecond)))
			return e.Messages, nil
		}
	}
	return conv, errors.New("run ended without Done or an error")
}

// printer writes the answer to stdout and everything else dimmed to stderr. Both
// end up on the same terminal, so it inserts a newline when switching mid-line.
type printer struct {
	open io.Writer // writer that last wrote without a trailing newline, nil if at column 0
	spin *spinner  // running spinner, nil if none
}

// wait starts the spinner. Skipped when stderr is not a terminal, nobody wants
// spinner frames in a log file.
func (p *printer) wait() {
	if p.spin != nil {
		return
	}
	if info, err := os.Stderr.Stat(); err == nil && info.Mode()&os.ModeCharDevice != 0 {
		p.spin = startSpinner(os.Stderr)
	}
}

func (p *printer) settle() {
	if p.spin != nil {
		p.spin.stop()
		p.spin = nil
	}
}

func (p *printer) text(s string)    { p.write(os.Stdout, s) }
func (p *printer) thought(s string) { p.write(os.Stderr, dim(s)) }

// note prints s dimmed on its own line.
func (p *printer) note(s string) {
	p.endLine()
	fmt.Fprintln(os.Stderr, dim(s))
}

func (p *printer) write(w io.Writer, s string) {
	if s == "" {
		return
	}
	p.settle()
	if p.open != nil && p.open != w {
		p.endLine()
	}
	fmt.Fprint(w, s)
	p.open = w
	if strings.HasSuffix(s, "\n") {
		p.open = nil
	}
}

func (p *printer) endLine() {
	p.settle()
	if p.open != nil {
		fmt.Fprintln(p.open)
		p.open = nil
	}
}

// dim wraps s in dim escape codes. A trailing newline stays outside so write can still see it.
func dim(s string) string {
	body := strings.TrimSuffix(s, "\n")
	return "\033[2m" + body + "\033[0m" + s[len(body):]
}

// spinner animates a single dimmed character on w until stop is called, then
// clears it. It assumes the cursor is at column 0.
type spinner struct {
	stop func()
}

func startSpinner(w io.Writer) *spinner {
	frames := []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
	stopped, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		tick := time.NewTicker(80 * time.Millisecond)
		defer tick.Stop()
		for i := 0; ; i++ {
			select {
			case <-stopped:
				fmt.Fprint(w, "\r\033[K")
				return
			case <-tick.C:
				fmt.Fprint(w, "\r"+dim(string(frames[i%len(frames)])))
			}
		}
	}()
	return &spinner{stop: func() { close(stopped); <-done }}
}
