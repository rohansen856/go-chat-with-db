package rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type LlamaLLM struct {
	Opts     LLMOpts
	Query    string
	Response string
}

func NewLlamaLLM(opts LLMOpts) LLM {
	return &LlamaLLM{
		Opts: opts,
	}
}

func (llm *LlamaLLM) GenerateQuery(que string) (string, error) {
	prompt := fmt.Sprintf(`You are an expert SQL generator. Based on the schema:
%v

Generate only SELECT queries or queries to read data. Never include sensitive fields like password, hashed_password etc. Only output valid SQL, no explanations. If the question is not answerable using the schema, return an error message.

Question: %s`, llm.Opts.Context, que)

	res, err := callOllama("llama3", prompt)
	if err != nil {
		return "", err
	}
	llm.Query = res
	return res, nil
}

func (llm *LlamaLLM) GenerateResponse(data interface{}, que string) (string, error) {
	prompt := fmt.Sprintf(`Summarize the following data in a conversational manner.

Context: %s

Data: %v

Write in a concise and human-like way.`, que, data)

	res, err := callOllama("llama3", prompt)
	if err != nil {
		return "", err
	}
	llm.Response = res
	return res, nil
}

func callOllama(model, prompt string) (string, error) {
	body := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	}
	jsonData, _ := json.Marshal(body)

	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to contact Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		content, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama error (%d): %s", resp.StatusCode, string(content))
	}

	var result struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse ollama response: %w", err)
	}

	return result.Response, nil
}
