# Model Proxy

A proxy server for LLM APIs that supports both OpenAI and Anthropic backends. It hides real API keys from clients, maps model names to remote endpoints, and records all requests and responses.

## Features

- **Dual API support** — exposes both `/v1/chat/completions` (OpenAI format) and `/v1/messages` (Anthropic format). Automatically converts between formats when a client's request format doesn't match the backend type.
- **Model mapping** — clients use local model names; the proxy replaces them with remote model/URL/token.
- **Thinking support** — configurable per model via YAML defaults:
  - Anthropic backends: injects native `thinking` parameter when client doesn't specify one
  - OpenAI backends: `<think>` tag parsing → `reasoning_content` conversion
- **Transparent proxying** — client request bodies are forwarded as-is to their native API format. Each API uses its own standard fields (e.g. OpenAI `reasoning_effort`, Anthropic `thinking`).
- **Request recording** — saves request/response JSON to disk with automatic cleanup of old records.
- **Streaming** — full SSE streaming support for both backends, including cross-format streaming conversion.
- **YAML config** — all settings in a single `config.yaml` file.

## Build

Requires Rust toolchain. Install via [rustup](https://rustup.rs/).

```bash
# Build release binary
make build
# Output: dist/model-proxy

# Cross-compile for Linux x86_64
make linux
# Output: dist/linux-x86/model-proxy
```

## Configuration

Copy the example config and fill in your API tokens:

```bash
mkdir -p ~/.model-proxy
cp config.example.yaml ~/.model-proxy/config.yaml
```

See [`config.example.yaml`](config.example.yaml) for all available options. A minimal config:

```yaml
backends:
  openai:
    - name: gpt-4o
      url: https://api.openai.com/v1/chat/completions
      api_token: sk-xxx
      model: gpt-4o

  anthropic:
    - name: claude-sonnet
      url: https://api.anthropic.com/v1/messages
      api_token: sk-ant-xxx
      model: claude-sonnet-4-20250514
```

## Usage

```bash
# Run with default config (~/.model-proxy/config.yaml)
./dist/model-proxy

# Specify config file, port, and debug mode
./dist/model-proxy --config /path/to/config.yaml --port 9000 --debug
```

### CLI flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config` | `-c` | `~/.model-proxy/config.yaml` | Path to YAML config file |
| `--port` | `-p` | `8000` | Port to listen on (overrides config) |
| `--debug` | `-d` | `false` | Enable debug logging (overrides config) |

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/messages` | POST | Anthropic-compatible messages API |
| `/v1/models` | GET | List all configured models |

If `url_prefix` is set in config (e.g. `/api`), all routes are prefixed: `/api/v1/chat/completions`.

### Examples

```bash
# OpenAI format (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Anthropic format
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Anthropic format with thinking
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "What is 15 * 37?"}],
    "thinking": {"type": "enabled", "budget_tokens": 5000}
  }'

# List models
curl http://localhost:8000/v1/models
```

### Thinking

Thinking defaults are set per model in `config.yaml`. Clients can always override by including the native thinking field in their request — the proxy passes it through as-is:

- **Anthropic** (`/v1/messages`): use `"thinking": {"type": "enabled", "budget_tokens": 8000}` per Anthropic's API
- **OpenAI** (`/v1/chat/completions`): use `"reasoning_effort": "high"` per OpenAI's API (for o-series models)

If the client omits thinking fields, the proxy applies the YAML config defaults for Anthropic backends. For OpenAI backends, the config controls `<think>` tag parsing in streaming responses.

## Linux systemd service

```bash
# Create service user
sudo useradd -r -s /sbin/nologin model-server

# Deploy
sudo mkdir -p /opt/model-server/records
sudo cp dist/model-proxy /opt/model-server/
sudo cp config.yaml /opt/model-server/
sudo chown -R model-server:model-server /opt/model-server

# Install and start service
sudo cp model-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now model-server

# Manage
sudo systemctl status model-server
sudo journalctl -u model-server -f
sudo systemctl restart model-server
```
