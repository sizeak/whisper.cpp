# OpenAI API Compatible Server

An OpenAI API compatible server for whisper.cpp that implements the `/v1/audio/transcriptions` endpoint.

## Building

```bash
cmake -B build
cmake --build build --target whisper-openai-server
```

## Usage

```bash
# Basic usage with default settings
./build/bin/whisper-openai-server --models-dir ./models

# With custom port and authentication
./build/bin/whisper-openai-server --models-dir ./models --port 8000 --api-key your-secret-key

# Disable GPU acceleration
./build/bin/whisper-openai-server --models-dir ./models --no-gpu
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --help` | Show help message | |
| `-t N, --threads N` | Number of threads | 4 |
| `--models-dir PATH` | Models directory | ./models |
| `-l, --language LANG` | Default language | en |
| `--host HOST` | Server hostname | 127.0.0.1 |
| `--port PORT` | Server port | 8080 |
| `--api-key KEY` | Require Bearer token authentication | (none) |
| `-ng, --no-gpu` | Disable GPU acceleration | false |
| `-dev, --device N` | GPU device ID | 0 |
| `-fa, --flash-attn` | Enable flash attention | true |
| `-nfa, --no-flash-attn` | Disable flash attention | false |

## Endpoints

### POST /v1/audio/transcriptions

Transcribes audio into text. Compatible with OpenAI's Audio API.

**Request** (multipart/form-data):

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Audio file (WAV format) |
| `model` | string | Yes | Model name (e.g., "base.en", "large-v3-turbo") |
| `language` | string | No | ISO language code (default: server default) |
| `prompt` | string | No | Optional guiding prompt |
| `response_format` | string | No | `json`, `verbose_json`, or `text` (default: json) |
| `temperature` | float | No | Sampling temperature (default: 0.0) |

**Response formats:**

- `json`: `{"text": "transcribed text"}`
- `text`: Plain text
- `verbose_json`: Full OpenAI-style response with segments, timestamps, and tokens

### GET /v1/models

Lists available models in the models directory.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "base.en", "object": "model", "owned_by": "whisper.cpp"},
    {"id": "large-v3-turbo", "object": "model", "owned_by": "whisper.cpp"}
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

## Examples

### Basic transcription

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@samples/jfk.wav" \
  -F model="base.en"
```

### With verbose output

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@samples/jfk.wav" \
  -F model="base.en" \
  -F response_format="verbose_json"
```

### With authentication

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret-key" \
  -F file="@samples/jfk.wav" \
  -F model="base.en"
```

### List available models

```bash
curl http://localhost:8080/v1/models
```

## Model Files

Models should be placed in the models directory with the naming convention `ggml-{model_name}.bin`:

- `ggml-base.en.bin` → model name: `base.en`
- `ggml-large-v3-turbo.bin` → model name: `large-v3-turbo`

Download models using the provided script:

```bash
./models/download-ggml-model.sh base.en
./models/download-ggml-model.sh large-v3-turbo
```

## Integration with OpenAI clients

This server is compatible with OpenAI client libraries. Example with Python:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-secret-key"  # or any string if auth is disabled
)

audio_file = open("audio.wav", "rb")
transcription = client.audio.transcriptions.create(
    model="base.en",
    file=audio_file
)
print(transcription.text)
```
