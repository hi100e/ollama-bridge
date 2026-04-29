# ollama-bridge

A lightweight Go server that exposes any OpenAI-compatible API as an Ollama API.

Use it to make coding agents (OpenCode, Claude Code, etc.) talk to local LM Studio models through the familiar Ollama interface — without installing or managing actual Ollama models.

## Why?

- Your coding agent expects `http://localhost:11434` but your model runs on LM Studio at `http://127.0.0.1:1234/v1`
- No need to pull GGUF files — the bridge maps Ollama model names directly to OpenAI model IDs
- Supports streaming, tool calling, structured outputs, and all numeric options

## Quick Start

### Build from source

Requires Go 1.21+.

```bash
git clone https://github.com/hi100e/ollama-bridge.git
cd ollama-bridge
go build -o ollama-bridge .
```

### Configure (optional — defaults work out of the box)
cat > config.json <<EOF
{
  "listen_addr": ":11435",
  "base_url": "http://127.0.0.1:1234/v1",
  "model_map": {
    "qwen3.6:35b": "unsloth/qwen3.6-35b-a3b",
    "qwen3.6-coder:30b": "qwen/qwen3-coder-30b"
  }
}
EOF

# Run
OLLAMA_BRIDGE_CONFIG=config.json ./ollama-bridge &

# Test
curl http://localhost:11435/api/tags | jq .
```

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `listen_addr` | `:11434` | Port to listen on (use a different port if real Ollama is running) |
| `base_url` | `http://127.0.0.1:1234/v1` | OpenAI-compatible API endpoint |
| `model_map` | `{}` | Map of Ollama-style names → actual model IDs on the backend |

Set via JSON config file or the `OLLAMA_BRIDGE_CONFIG` environment variable.

## Supported Endpoints

### Fully supported (feature-complete)

- **POST `/api/chat`** — Chat completions with streaming, tool calling, structured outputs
- **POST `/api/generate`** — Text generation with streaming and options
- **GET `/api/tags`** — Lists configured models
- **GET `/api/version`** — Returns Ollama version `0.5.7`

### Implemented (placeholder)

- **GET `/api/ps`** — Empty model list (no running models in bridge context)
- **POST `/api/show`** — Minimal model info
- **POST `/api/embed`** — Attempts OpenAI embeddings, falls back to zeros
- **POST `/api/copy`**, **DELETE `/api/delete`**, **POST `/api/create`**, **POST `/api/pull`**, **POST `/api/push`** — No-op (model management is Ollama-specific)

## Usage with Coding Agents

### OpenCode

```bash
export OLLAMA_BASE_URL=http://localhost:11435
opencode run 'fix the auth bug' --model qwen3.6-coder:30b
```

Or configure permanently in `~/.opencode/opencode.json`:

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama Bridge",
      "options": {
        "baseURL": "http://localhost:11435/v1"
      },
      "models": {
        "qwen3.6-coder:30b": { "name": "qwen3.6-coder:30b" }
      }
    }
  }
}
```

### Claude Code / Codex

Set the `OPENAI_BASE_URL` and `OPENAI_API_KEY` (any non-empty value) to point at the bridge:

```bash
export OPENAI_BASE_URL=http://localhost:11435/v1
export OPENAI_API_KEY=ollama-bridge
claude 'refactor this module' --model qwen3.6-coder:30b
```

## License

MIT
