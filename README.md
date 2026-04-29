# ollama-bridge

A lightweight Go server that exposes any OpenAI-compatible API as an Ollama API.

Use it to make coding agents (OpenCode, Claude Code, etc.) talk to local LM Studio models through the familiar Ollama interface — without installing or managing actual Ollama models.

## Why?

- Your coding agent expects `http://localhost:11434` but your model runs on LM Studio at `http://127.0.0.1:1234/v1`
- No need to pull GGUF files — the bridge maps Ollama model names directly to OpenAI model IDs
- Supports streaming, tool calling, structured outputs, and all numeric options

## Installation

### Build from source

Requires Go 1.21+.

```bash
git clone https://github.com/hi100e/ollama-bridge.git
cd ollama-bridge
go build -o ollama-bridge .
```

The version is hardcoded in `main.go` and updated at each release. Check `--version` to confirm:

```bash
./ollama-bridge --version   # → ollama-bridge v0.3.0
```

### Install via `go install` (latest tagged release)

```bash
go install github.com/hi100e/ollama-bridge@v0.3.0
# Binary is placed in $GOPATH/bin or ~/go/bin
# Version reflects the hardcoded value from that tag's source code.
```

## Quick Start

### Configure

Copy and edit the example config:

```bash
cp config.example.json config.json
# Edit config.json with your LM Studio URL and model mappings
```

See [`config.example.json`](config.example.json) for a full reference.

### Run

```bash
OLLAMA_BRIDGE_CONFIG=config.json ollama-bridge &
curl http://localhost:11435/api/tags | jq .
```

Print the binary version:

```bash
ollama-bridge --version   # or -v
```

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
|| `listen_addr` | `:21434` | Port to listen on (use a different port if real Ollama is running) |
| `base_url` | `http://127.0.0.1:1234/v1` | OpenAI-compatible API endpoint |
| `model_map` | `{}` | Map of Ollama-style names → actual model IDs on the backend |

Set via JSON config file or the `OLLAMA_BRIDGE_CONFIG` environment variable.

## Supported Endpoints

### Fully supported (feature-complete)

- **POST `/api/chat`** — Chat completions with streaming, tool calling, structured outputs
- **POST `/api/generate`** — Text generation with streaming and options
- **GET `/api/tags`** — Lists models from the OpenAI backend's `/v1/models`, merged with `model_map` aliases
- **GET `/api/version`** — Returns Ollama version `0.21.0`

### Implemented (placeholder)

- **GET `/api/ps`** — Empty model list (no running models in bridge context)
- **POST `/api/show`** — Minimal model info
- **POST `/api/embed`** — Attempts OpenAI embeddings, falls back to zeros
- **POST `/api/copy`**, **DELETE `/api/delete`**, **POST `/api/create`**, **POST `/api/pull`**, **POST `/api/push`** — No-op (model management is Ollama-specific)

## Usage with Coding Agents

### OpenCode

```bash
export OLLAMA_BASE_URL=http://localhost:21434
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
        "baseURL": "http://localhost:21434/v1"
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
export OPENAI_BASE_URL=http://localhost:21434/v1
export OPENAI_API_KEY=ollama-bridge
claude 'refactor this module' --model qwen3.6-coder:30b
```

## License

MIT
