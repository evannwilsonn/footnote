// Package ollama is a minimal client for a local Ollama server: embeddings and chat.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Client struct {
	BaseURL    string // e.g. http://localhost:11434
	EmbedModel string // e.g. nomic-embed-text
	ChatModel  string // e.g. qwen2.5:1.5b
	HTTP       *http.Client
}

func New(baseURL, embedModel, chatModel string) *Client {
	return &Client{BaseURL: baseURL, EmbedModel: embedModel, ChatModel: chatModel, HTTP: &http.Client{Timeout: 10 * time.Minute}}
}

// post retries a server error twice: Ollama restarts a crashed model runner on the next request.
func (c *Client) post(ctx context.Context, path string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		retry, err := c.try(ctx, path, body, out)
		if err == nil || !retry || attempt == 2 {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt+1) * 5 * time.Second):
		}
	}
}

func (c *Client) try(ctx context.Context, path string, body []byte, out any) (retry bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return false, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return false, fmt.Errorf("ollama %s: %w (is `ollama serve` running?)", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return resp.StatusCode >= 500, fmt.Errorf("ollama %s: %s: %s", path, resp.Status, bytes.TrimSpace(msg))
	}
	return false, json.NewDecoder(resp.Body).Decode(out)
}

// Embed returns one vector per input text.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := c.post(ctx, "/api/embed", map[string]any{"model": c.EmbedModel, "input": texts}, &out); err != nil {
		return nil, err
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama returned %d embeddings for %d texts", len(out.Embeddings), len(texts))
	}
	return out.Embeddings, nil
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResult struct {
	Text          string
	PromptTokens  int
	AnswerTokens  int
	TotalDuration time.Duration
}

// Chat runs one deterministic (temperature 0) completion.
func (c *Client) Chat(ctx context.Context, messages []Message, maxTokens int) (ChatResult, error) {
	return c.ChatFormat(ctx, messages, maxTokens, nil)
}

// ChatFormat is Chat with Ollama's structured output: format is a JSON schema the reply must match.
func (c *Client) ChatFormat(ctx context.Context, messages []Message, maxTokens int, format any) (ChatResult, error) {
	var out struct {
		Message         Message `json:"message"`
		PromptEvalCount int     `json:"prompt_eval_count"`
		EvalCount       int     `json:"eval_count"`
		TotalDuration   int64   `json:"total_duration"`
	}
	in := map[string]any{
		"model":    c.ChatModel,
		"messages": messages,
		"stream":   false,
		"options":  map[string]any{"temperature": 0, "seed": 1, "num_predict": maxTokens, "num_ctx": 4096},
	}
	if format != nil {
		in["format"] = format
	}
	if err := c.post(ctx, "/api/chat", in, &out); err != nil {
		return ChatResult{}, err
	}
	return ChatResult{Text: out.Message.Content, PromptTokens: out.PromptEvalCount, AnswerTokens: out.EvalCount, TotalDuration: time.Duration(out.TotalDuration)}, nil
}
