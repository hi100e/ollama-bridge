package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// Version is set at release time by updating this line.
var Version = "v0.3.0"

// Config holds the bridge configuration.
type Config struct {
	ListenAddr string            `json:"listen_addr"`
	BaseURL    string            `json:"base_url"` // OpenAI-compatible endpoint, e.g. http://127.0.0.1:1234/v1
	ModelMap   map[string]string `json:"model_map"` // ollama model name -> openai model name
}

// ─── Ollama request types ──────────────────────────────────────────────

type ChatRequest struct {
	Model    string      `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Tools    []ToolDef   `json:"tools,omitempty"`
	Format   any         `json:"format,omitempty"`
	Options  json.RawMessage `json:"options,omitempty"`
	Stream   bool        `json:"stream,omitempty"` // default true in Ollama
	Think    any         `json:"think,omitempty"`
	Raw      bool        `json:"raw,omitempty"`
	Logprobs *bool       `json:"logprobs,omitempty"`
	TopLogprobs *int    `json:"top_logprobs,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

type ChatMessage struct {
	Role      string     `json:"role"`
	Content   any        `json:"content"` // string or []interface{} (multimodal)
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolDef struct {
	Type     string          `json:"type"`
	Function FunctionDef     `json:"function"`
}

type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolCall struct {
	Function ToolFunctionCall `json:"function"`
}

type ToolFunctionCall struct {
	Name      string     `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ─── Ollama response types ─────────────────────────────────────────────

type ChatResponse struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         MessageOut    `json:"message"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason,omitempty"`
	TotalDuration   int64         `json:"total_duration,omitempty"` // nanoseconds
	LoadDuration    int64         `json:"load_duration,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDur   int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount       int           `json:"eval_count,omitempty"`
	EvalDur         int64         `json:"eval_duration,omitempty"`
	Logprobs        []Logprob     `json:"logprobs,omitempty"`
}

type MessageOut struct {
	Role      string     `json:"role"`
	Content   any        `json:"content"` // string or []interface{} (multimodal)
	Thinking  string     `json:"thinking,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Logprob struct {
	Token       string    `json:"token"`
	LogProb     float64   `json:"logprob"`
	Bytes       []byte    `json:"bytes,omitempty"`
	BestOfLogprobs []BestOfLogprob `json:"best_of_logprobs,omitempty"`
}

type BestOfLogprob struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes,omitempty"`
}

// ─── Ollama generate types ─────────────────────────────────────────────

type GenerateRequest struct {
	Model     string      `json:"model"`
	Prompt    string      `json:"prompt"`
	Suffix    string      `json:"suffix,omitempty"` // fill-in-the-middle (not supported by OpenAI)
	Images    []string    `json:"images,omitempty"`
	Format    any         `json:"format,omitempty"`
	System    string      `json:"system,omitempty"`
	Stream    bool        `json:"stream,omitempty"` // default true in Ollama
	Think     any         `json:"think,omitempty"`
	Raw       bool        `json:"raw,omitempty"`
	Logprobs  *bool       `json:"logprobs,omitempty"`
	TopLogprobs *int      `json:"top_logprobs,omitempty"`
	Options   json.RawMessage `json:"options,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

type GenerateResponse struct {
	Model           string     `json:"model"`
	CreatedAt       string     `json:"created_at"`
	Response        string     `json:"response"`
	Thinking        string     `json:"thinking,omitempty"`
	Done            bool       `json:"done"`
	DoneReason      string     `json:"done_reason,omitempty"`
	TotalDuration   int64      `json:"total_duration,omitempty"` // nanoseconds
	LoadDuration    int64      `json:"load_duration,omitempty"`
	PromptEvalCount int        `json:"prompt_eval_count,omitempty"`
	PromptEvalDur   int64      `json:"prompt_eval_duration,omitempty"`
	EvalCount       int        `json:"eval_count,omitempty"`
	EvalDur         int64      `json:"eval_duration,omitempty"`
	Logprobs        []Logprob  `json:"logprobs,omitempty"`
}

// ─── OpenAI types ──────────────────────────────────────────────────────

type OpenAIChatRequest struct {
	Model            string           `json:"model"`
	Messages         []OpenAIMessage  `json:"messages"`
	Tools            []OpenAITool     `json:"tools,omitempty"`
	ResponseFormat   *ResponseFormat  `json:"response_format,omitempty"`
	Temperature      *float64         `json:"temperature,omitempty"`
	MaxTokens        *int             `json:"max_tokens,omitempty"`
	TopP             *float64         `json:"top_p,omitempty"`
	FrequencyPenalty *float64         `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64         `json:"presence_penalty,omitempty"`
	Seed             *int             `json:"seed,omitempty"`
	Stream           bool             `json:"stream,omitempty"`
}

type OpenAIMessage struct {
	Role      string     `json:"role"`
	Content   any        `json:"content"`
	Name      string     `json:"name,omitempty"`
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JsonSchema *JsonSchema `json:"json_schema,omitempty"`
}

type JsonSchema struct {
	Schema json.RawMessage `json:"schema"`
}

type OpenAITool struct {
	Type     string        `json:"type"`
	Function FunctionDef   `json:"function"`
}

type OpenAIChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []OpenAIChoice     `json:"choices"`
	Usage   *UsageInfo         `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int               `json:"index"`
	Delta        OpenAIDelta       `json:"delta"`
	Message      OpenAIMessage     `json:"message,omitempty"`
	FinishReason *string           `json:"finish_reason"`
	LogProbs     *json.RawMessage  `json:"logprobs,omitempty"`
}

type OpenAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIToolCall struct {
	Index    int             `json:"index"`
	ID       string          `json:"id,omitempty"`
	Type     string          `json:"type"`
	Function OpenAIFuncCall  `json:"function,omitempty"`
}

type OpenAIFuncCall struct {
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type UsageInfo struct {
	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	PromptTokensDetails      *struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails  *struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"completion_tokens_details,omitempty"`
}

// ─── Ollama /api/tags response ────────────────────────────────────────

type TagsResponse struct {
	Models []ModelSummary `json:"models"`
}

type ModelSummary struct {
	Name       string   `json:"name"`
	Model      string   `json:"model"`
	RemoteModel string  `json:"remote_model,omitempty"`
	RemoteHost  string  `json:"remote_host,omitempty"`
	Size       int64    `json:"size"`
	Digest     string   `json:"digest"`
	ModifiedAt string   `json:"modified_at"`
	Details    ModelDetails `json:"details,omitempty"`
}

type ModelDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// ─── Ollama /api/version response ──────────────────────────────────────

type VersionResponse struct {
	Version string `json:"version"`
}

// ─── Config & helpers ──────────────────────────────────────────────────

var config Config

func resolveModel(name string) string {
	if mapped, ok := config.ModelMap[name]; ok {
		return mapped
	}
	return name
}

func ollamaTime() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

// toNanoseconds converts a duration to nanoseconds (Ollama spec).
func toNanoseconds(d time.Duration) int64 {
	return d.Nanoseconds()
}

// parseOptions extracts numeric options from Ollama's raw JSON into OpenAI params.
func parseOptions(raw json.RawMessage, req *OpenAIChatRequest) {
	if raw == nil {
		return
	}
	var opts map[string]json.RawMessage
	if err := json.Unmarshal(raw, &opts); err != nil {
		return
	}

	if v, ok := opts["temperature"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			req.Temperature = &f
		}
	}
	if v, ok := opts["top_p"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			req.TopP = &f
		}
	}
	if v, ok := opts["num_predict"]; ok {
		var n int
		if err := json.Unmarshal(v, &n); err == nil {
			req.MaxTokens = &n
		}
	}
	if v, ok := opts["seed"]; ok {
		var s int
		if err := json.Unmarshal(v, &s); err == nil {
			req.Seed = &s
		}
	}
	if v, ok := opts["frequency_penalty"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			req.FrequencyPenalty = &f
		}
	}
	if v, ok := opts["presence_penalty"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			req.PresencePenalty = &f
		}
	}
}

// convertTools converts Ollama tool definitions to OpenAI format.
func convertTools(ollamaTools []ToolDef) []OpenAITool {
	if len(ollamaTools) == 0 {
		return nil
	}
	result := make([]OpenAITool, len(ollamaTools))
	for i, t := range ollamaTools {
		result[i] = OpenAITool{
			Type:     "function",
			Function: FunctionDef(t.Function),
		}
	}
	return result
}

// convertFormat converts Ollama format to OpenAI response_format.
func convertFormat(ollamaFmt any) *ResponseFormat {
	if ollamaFmt == nil {
		return nil
	}
	switch v := ollamaFmt.(type) {
	case string:
		if strings.EqualFold(v, "json") {
			// Use json_schema with a generic object schema for broader compatibility.
			// Some backends (like LM Studio) don't support "json_object" type.
			return &ResponseFormat{
				Type: "json_schema",
				JsonSchema: &JsonSchema{
					Schema: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":true}`),
				},
			}
		}
	case map[string]any:
		b, _ := json.Marshal(v)
		return &ResponseFormat{
			Type:       "json_schema",
			JsonSchema: &JsonSchema{Schema: b},
		}
	}
	return nil
}

// buildOpenAIMessages converts Ollama messages to OpenAI format.
func buildOpenAIMessages(msgs []ChatMessage) []OpenAIMessage {
	result := make([]OpenAIMessage, len(msgs))
	for i, m := range msgs {
		msg := OpenAIMessage{Role: m.Role}

		switch c := m.Content.(type) {
		case string:
			msg.Content = c
		default:
			// Multimodal or other — pass through as-is.
			msg.Content = m.Content
		}

		if len(m.Images) > 0 {
			// For multimodal models, images are typically base64 data URIs.
			// We'll attach them to the content if it's a string (append image references).
			// This is a best-effort approach since OpenAI expects specific formats.
			if _, ok := msg.Content.(string); ok {
				msg.Content = m.Content // pass through; backend may handle images differently
			}
		}

		result[i] = msg
	}
	return result
}

// ─── Streaming helpers ─────────────────────────────────────────────────

func writeNDJSON(w http.ResponseWriter, flusher http.Flusher, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintln(w, string(b))
	flusher.Flush()
}

// ─── /api/chat handler ────────────────────────────────────────────────

func chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	openaiModel := resolveModel(req.Model)
	openaiMsgs := buildOpenAIMessages(req.Messages)

	openaiReq := OpenAIChatRequest{
		Model:        openaiModel,
		Messages:     openaiMsgs,
		Tools:        convertTools(req.Tools),
		ResponseFormat: convertFormat(req.Format),
		Stream:       req.Stream,
	}

	parseOptions(req.Options, &openaiReq)

	bodyBytes, _ := json.Marshal(openaiReq)
	openaiURL := config.BaseURL + "/chat/completions"

	resp, err := http.Post(openaiURL, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("Error calling OpenAI API: %v", err)
		http.Error(w, fmt.Sprintf("backend error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	start := time.Now()

	if !req.Stream {
		// Non-streaming response.
		var openaiResp OpenAIChatResponse
		if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
			http.Error(w, fmt.Sprintf("parse error: %v", err), http.StatusBadGateway)
			return
		}

		content := ""
		reason := "stop"
		var toolCalls []ToolCall
		var thinking string

		if len(openaiResp.Choices) > 0 {
			ch := openaiResp.Choices[0]
			switch v := ch.Message.Content.(type) {
			case string:
				content = v
			default:
				// Multimodal — pass through.
				content = ""
			}

			if ch.FinishReason != nil {
				reason = *ch.FinishReason
			}

		// Extract tool calls if present.
		for _, tc := range ch.Message.ToolCalls {
			toolCall := ToolCall{
				Function: ToolFunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
			toolCalls = append(toolCalls, toolCall)
		}

			// Extract thinking if present (some backends return it separately).
			if thinkContent, ok := ch.Message.Content.(map[string]any); ok {
				if t, ok2 := thinkContent["thinking"].(string); ok2 {
					thinking = t
				}
			}
		}

		totalDur := toNanoseconds(time.Since(start))
		promptTokens := 0
		completionTokens := 0
		if openaiResp.Usage != nil {
			promptTokens = openaiResp.Usage.PromptTokens
			completionTokens = openaiResp.Usage.CompletionTokens
		}

		ollamaResp := ChatResponse{
			Model:           req.Model,
			CreatedAt:       ollamaTime(),
			Message: MessageOut{
				Role:      "assistant",
				Content:   content,
				Thinking:  thinking,
				ToolCalls: toolCalls,
			},
			Done:            true,
			DoneReason:      reason,
			TotalDuration:   totalDur,
			PromptEvalCount: promptTokens,
			EvalCount:       completionTokens,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ollamaResp)
		return
	}

	// Streaming response — NDJSON.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	var fullContent strings.Builder
	var toolCallBuffers []strings.Builder // one per tool call index
	toolCallIDs := make(map[int]string)  // index -> ID (from first chunk)
	finishReason := ""
	promptTokens := 0
	completionTokens := 0

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			// Final done chunk.
			respOut := ChatResponse{
				Model:           req.Model,
				CreatedAt:       ollamaTime(),
				Done:            true,
				DoneReason:      finishReason,
				TotalDuration:   toNanoseconds(time.Since(start)),
				PromptEvalCount: promptTokens,
				EvalCount:       completionTokens,
			}
			writeNDJSON(w, flusher, respOut)
			return
		}

		var openaiResp OpenAIChatResponse
		if err := json.Unmarshal([]byte(data), &openaiResp); err != nil {
			continue
		}

		for _, choice := range openaiResp.Choices {
			// Track usage.
			if openaiResp.Usage != nil {
				promptTokens = openaiResp.Usage.PromptTokens
				completionTokens = openaiResp.Usage.CompletionTokens
			}

			delta := choice.Delta

			// Handle tool calls.
			if len(delta.ToolCalls) > 0 {
				for _, tc := range delta.ToolCalls {
					idx := tc.Index
					if idx < 0 {
						idx = 0
					}
					if tc.ID != "" && toolCallIDs[idx] == "" {
						toolCallIDs[idx] = tc.ID
					}

					for len(toolCallBuffers) <= idx {
						toolCallBuffers = append(toolCallBuffers, strings.Builder{})
					}
					if tc.Function.Arguments != nil {
						toolCallBuffers[idx].Write(tc.Function.Arguments)
					}

					// Send tool call chunk.
					chunkOut := ChatResponse{
						Model:     req.Model,
						CreatedAt: ollamaTime(),
						Message: MessageOut{
							Role:      "assistant",
							Content:   "",
							ToolCalls: []ToolCall{{Function: ToolFunctionCall{Name: tc.Function.Name}}},
						},
						Done: false,
					}
					writeNDJSON(w, flusher, chunkOut)
				}
			}

			// Handle content.
			if delta.Content != "" {
				fullContent.WriteString(delta.Content)
				chunkOut := ChatResponse{
					Model:     req.Model,
					CreatedAt: ollamaTime(),
					Message: MessageOut{
						Role:    "assistant",
						Content: delta.Content,
					},
					Done: false,
				}
				writeNDJSON(w, flusher, chunkOut)
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = *choice.FinishReason
			}
		}
	}

	// Build final tool calls from accumulated buffers.
	var finalToolCalls []ToolCall
	for _, buf := range toolCallBuffers {
		if buf.Len() > 0 {
			finalToolCalls = append(finalToolCalls, ToolCall{
				Function: ToolFunctionCall{
					Name:      "", // We lost the name after sending chunks; this is a limitation.
					Arguments: json.RawMessage(buf.String()),
				},
			})
		}
	}

	// Send final done chunk with accumulated content and tool calls.
	respOut := ChatResponse{
		Model:           req.Model,
		CreatedAt:       ollamaTime(),
		Message: MessageOut{
			Role:      "assistant",
			Content:   fullContent.String(),
			ToolCalls: finalToolCalls,
		},
		Done:            true,
		DoneReason:      finishReason,
		TotalDuration:   toNanoseconds(time.Since(start)),
		PromptEvalCount: promptTokens,
		EvalCount:       completionTokens,
	}
	writeNDJSON(w, flusher, respOut)

	if err := scanner.Err(); err != nil {
		log.Printf("Stream read error: %v", err)
	}
}

// ─── /api/generate handler ────────────────────────────────────────────

func generateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	openaiModel := resolveModel(req.Model)

	// Build messages from prompt + system.
	var msgs []OpenAIMessage
	if req.System != "" && !req.Raw {
		msgs = append(msgs, OpenAIMessage{Role: "system", Content: req.System})
	}
	msgs = append(msgs, OpenAIMessage{Role: "user", Content: req.Prompt})

	openaiReq := OpenAIChatRequest{
		Model:        openaiModel,
		Messages:     msgs,
		ResponseFormat: convertFormat(req.Format),
		Stream:       req.Stream, // respect the stream flag (default true in Ollama)
	}

	parseOptions(req.Options, &openaiReq)

	bodyBytes, _ := json.Marshal(openaiReq)
	openaiURL := config.BaseURL + "/chat/completions"

	resp, err := http.Post(openaiURL, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("Error calling OpenAI API: %v", err)
		http.Error(w, fmt.Sprintf("backend error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	start := time.Now()

	if !req.Stream {
		// Non-streaming response.
		var openaiResp OpenAIChatResponse
		if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
			http.Error(w, fmt.Sprintf("parse error: %v", err), http.StatusBadGateway)
			return
		}

		content := ""
		reason := "stop"
		promptTokens := 0
		completionTokens := 0

		if len(openaiResp.Choices) > 0 {
			ch := openaiResp.Choices[0]
			switch v := ch.Message.Content.(type) {
			case string:
				content = v
			default:
				content = ""
			}
			if ch.FinishReason != nil {
				reason = *ch.FinishReason
			}
		}

		if openaiResp.Usage != nil {
			promptTokens = openaiResp.Usage.PromptTokens
			completionTokens = openaiResp.Usage.CompletionTokens
		}

		totalDur := toNanoseconds(time.Since(start))

		ollamaResp := GenerateResponse{
			Model:           req.Model,
			CreatedAt:       ollamaTime(),
			Response:        content,
			Done:            true,
			DoneReason:      reason,
			TotalDuration:   totalDur,
			PromptEvalCount: promptTokens,
			EvalCount:       completionTokens,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ollamaResp)
		return
	}

	// Streaming response — NDJSON.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	var fullContent strings.Builder
	promptTokens := 0
	completionTokens := 0
	finishReason := ""

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			respOut := GenerateResponse{
				Model:           req.Model,
				CreatedAt:       ollamaTime(),
				Response:        fullContent.String(),
				Done:            true,
				DoneReason:      finishReason,
				TotalDuration:   toNanoseconds(time.Since(start)),
				PromptEvalCount: promptTokens,
				EvalCount:       completionTokens,
			}
			writeNDJSON(w, flusher, respOut)
			return
		}

		var openaiResp OpenAIChatResponse
		if err := json.Unmarshal([]byte(data), &openaiResp); err != nil {
			continue
		}

		for _, choice := range openaiResp.Choices {
			if openaiResp.Usage != nil {
				promptTokens = openaiResp.Usage.PromptTokens
				completionTokens = openaiResp.Usage.CompletionTokens
			}

			content := choice.Delta.Content
			if content != "" {
				fullContent.WriteString(content)
				chunkOut := GenerateResponse{
					Model:     req.Model,
					CreatedAt: ollamaTime(),
					Response:  content,
					Done:      false,
				}
				writeNDJSON(w, flusher, chunkOut)
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = *choice.FinishReason
			}
		}
	}

	// Send final done chunk.
	respOut := GenerateResponse{
		Model:           req.Model,
		CreatedAt:       ollamaTime(),
		Response:        fullContent.String(),
		Done:            true,
		DoneReason:      finishReason,
		TotalDuration:   toNanoseconds(time.Since(start)),
		PromptEvalCount: promptTokens,
		EvalCount:       completionTokens,
	}
	writeNDJSON(w, flusher, respOut)

	if err := scanner.Err(); err != nil {
		log.Printf("Stream read error: %v", err)
	}
}

// ─── /api/tags handler ────────────────────────────────────────────────

func tagsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var models []ModelSummary
	for name := range config.ModelMap {
		models = append(models, ModelSummary{
			Name:       name,
			Model:      name,
			Size:       0, // unknown for remote backends
			Digest:     "sha256:" + strings.ReplaceAll(name, "/", ""),
			ModifiedAt: time.Now().UTC().Format(time.RFC3339),
			Details: ModelDetails{
				Format:            "gguf",
				Family:            "",
				Families:          []string{},
				ParameterSize:     "unknown",
				QuantizationLevel: "F16",
			},
		})
	}

	resp := TagsResponse{Models: models}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ─── /api/version handler ─────────────────────────────────────────────

func versionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := VersionResponse{Version: "0.21.0"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ─── /api/ps handler (placeholder — no running models in bridge) ──────

func psHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := struct{ Models []any `json:"models"` }{Models: []any{}}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ─── /api/show handler (placeholder — returns minimal info) ───────────

func showHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct{ Model string `json:"model"` }
	json.NewDecoder(r.Body).Decode(&req)

	resp := map[string]any{
		"details": map[string]any{
			"format":            "gguf",
			"families":          []string{},
			"parameter_size":    "unknown",
			"quantization_level": "F16",
		},
		"template":      "",
		"modified_at":   time.Now().UTC().Format(time.RFC3339),
		"model_info":    map[string]any{"general.architecture": ""},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ─── /api/embed handler (placeholder — returns zero embeddings) ──────

func embedHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Model      string `json:"model"`
		Input      any    `json:"input"` // string or []string
		Truncate   *bool  `json:"truncate,omitempty"`
		Dimensions *int   `json:"dimensions,omitempty"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	// Try to get embeddings from the OpenAI backend if it supports /embeddings.
	var embeddings [][]float64
	openaiModel := resolveModel(req.Model)
	embedURL := config.BaseURL + "/embeddings"

	inputStrs := []string{}
	switch v := req.Input.(type) {
	case string:
		inputStrs = append(inputStrs, v)
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputStrs = append(inputStrs, s)
			}
		}
	}

	if len(inputStrs) > 0 {
		type embedReq struct {
			Model      string   `json:"model"`
			Input      []string `json:"input"`
			Dimensions *int     `json:"dimensions,omitempty"`
		}
		type embedResp struct {
			Data []struct {
				Embedding []float64 `json:"embedding"`
			} `json:"data"`
		}

		body, _ := json.Marshal(embedReq{Model: openaiModel, Input: inputStrs, Dimensions: req.Dimensions})
		resp2, err := http.Post(embedURL, "application/json", bytes.NewReader(body))
		if err == nil {
			defer resp2.Body.Close()
			var er embedResp
			if json.NewDecoder(resp2.Body).Decode(&er) == nil {
				for _, d := range er.Data {
					embeddings = append(embeddings, d.Embedding)
				}
			}
		}
	}

	if len(embeddings) == 0 {
		// Fallback: return zero embeddings.
		for range inputStrs {
			embeddings = append(embeddings, []float64{})
		}
	}

	out := map[string]any{
		"model":        req.Model,
		"embeddings":   embeddings,
		"total_duration": int64(0),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// ─── /api/copy handler (placeholder — no-op) ──────────────────────────

func copyHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// ─── /api/delete handler (placeholder — no-op) ────────────────────────

func deleteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// ─── /api/create handler (placeholder — no-op) ────────────────────────

func createHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// ─── /api/pull handler (placeholder — no-op) ──────────────────────────

func pullHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// ─── /api/push handler (placeholder — no-op) ──────────────────────────

func pushHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// ─── main ──────────────────────────────────────────────────────────────

func main() {
	// Handle -v/--version before anything else.
	for _, arg := range os.Args[1:] {
		if arg == "-v" || arg == "--version" {
			fmt.Println("ollama-bridge", Version)
			os.Exit(0)
		}
	}

	configFile := os.Getenv("OLLAMA_BRIDGE_CONFIG")
	if configFile == "" {
		configFile = "/home/hi/.opencode/ollama-bridge.json"
	}

	if data, err := os.ReadFile(configFile); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			log.Fatalf("Failed to parse config: %v", err)
		}
	} else {
		log.Printf("No config file at %s, using defaults", configFile)
	}

	if config.ListenAddr == "" {
		config.ListenAddr = ":11434"
	}
	if config.BaseURL == "" {
		config.BaseURL = "http://127.0.0.1:1234/v1"
	}
	if config.ModelMap == nil {
		config.ModelMap = map[string]string{}
	}

	// Register all Ollama API endpoints.
	http.HandleFunc("/api/chat", chatHandler)
	http.HandleFunc("/api/generate", generateHandler)
	http.HandleFunc("/api/tags", tagsHandler)
	http.HandleFunc("/api/version", versionHandler)
	http.HandleFunc("/api/ps", psHandler)
	http.HandleFunc("/api/show", showHandler)
	http.HandleFunc("/api/embed", embedHandler)
	http.HandleFunc("/api/copy", copyHandler)
	http.HandleFunc("/api/delete", deleteHandler)
	http.HandleFunc("/api/create", createHandler)
	http.HandleFunc("/api/pull", pullHandler)
	http.HandleFunc("/api/push", pushHandler)

	log.Printf("ollama-bridge listening on %s", config.ListenAddr)
	log.Printf("  → OpenAI backend: %s", config.BaseURL)
	log.Printf("  → Model mappings: %d entries", len(config.ModelMap))
	for k, v := range config.ModelMap {
		log.Printf("    %s → %s", k, v)
	}

	if err := http.ListenAndServe(config.ListenAddr, nil); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
