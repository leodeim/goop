package goop

import (
	"bytes"
	"cmp"
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// HTTP is how a provider talks to its API. The zero value uses http.DefaultClient, 2 retries
// and a 500ms backoff (doubled per retry). MaxRetries < 0 disables retries.
type HTTP struct {
	Client     *http.Client
	MaxRetries int
	Backoff    time.Duration
}

// post sends a JSON body and returns the response if it is a 200. Any other status is an *APIError.
func (h HTTP) post(ctx context.Context, url string, headers map[string]string, body []byte) (*http.Response, error) {
	resp, err := h.send(ctx, func() (*http.Request, error) {
		hr, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		hr.Header.Set("content-type", "application/json")
		for k, v := range headers {
			hr.Header.Set(k, v)
		}
		return hr, nil
	})
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, readAPIError(resp)
	}

	return resp, nil
}

func readAPIError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return &APIError{Status: resp.StatusCode, Body: strings.TrimSpace(string(body))}
}

func (h HTTP) send(ctx context.Context, build func() (*http.Request, error)) (*http.Response, error) {
	client := cmp.Or(h.Client, http.DefaultClient)
	backoff := cmp.Or(h.Backoff, 500*time.Millisecond)
	retries := h.MaxRetries
	switch {
	case retries == 0:
		retries = 2
	case retries < 0:
		retries = 0
	}

	var lastErr error
	for attempt := 0; ; attempt++ {
		req, err := build()
		if err != nil {
			return nil, err
		}

		resp, err := client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, err
			}
			lastErr = err
		} else if !retryStatus(resp.StatusCode) || attempt == retries {
			return resp, nil
		}
		if attempt == retries {
			return nil, lastErr
		}

		wait := backoff << attempt
		if resp != nil {
			if after, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil {
				wait = time.Duration(after) * time.Second
			}
			resp.Body.Close()
		}
		if err := sleep(ctx, wait); err != nil {
			return nil, err
		}
	}
}

func retryStatus(code int) bool {
	return code == http.StatusRequestTimeout || code == http.StatusTooManyRequests || code >= 500
}

func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}

	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
