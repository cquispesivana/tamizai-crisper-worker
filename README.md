# TamizAI Crisper Worker

RunPod Serverless worker for TamizAI speech-to-text with CrisperWhisper 2.0.

This repo is intentionally separate from `tamizai-backend`. The backend should keep the scoring pipeline and call this worker over HTTP only when it needs Crisper transcription.

Entrypoint:

- `rp_handler.py` starts the queue-based RunPod worker.
- `handler.py` contains the transcription logic imported by the entrypoint.

## Runtime Contract

Preferred request shape:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "language": "es",
    "mode": "verbatim",
    "word_timestamps": true
  }
}
```

Small local/debug payloads can use base64:

```json
{
  "input": {
    "audio_base64": "BASE64_AUDIO_HERE",
    "audio_format": "wav",
    "language": "es",
    "mode": "verbatim",
    "word_timestamps": true
  }
}
```

Response shape:

```json
{
  "provider": "crisper_whisper",
  "model": "nyralabs/CrisperWhisper2.0_medium",
  "backend": "ct2",
  "device": "cuda",
  "compute_type": "float16",
  "language": "es",
  "mode": "verbatim",
  "text": "transcribed text",
  "duration_seconds": null,
  "processing_time_seconds": 3.42,
  "words": [
    {
      "word": "transcribed",
      "start": 0.12,
      "end": 0.48,
      "confidence": null
    }
  ]
}
```

Errors are returned as:

```json
{
  "error": {
    "code": "MISSING_AUDIO_INPUT",
    "message": "Send either input.audio_url or input.audio_base64.",
    "status_code": 400
  }
}
```

## Environment Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `CRISPER_MODEL_SIZE` | `medium` | Alias for the bundled Crisper model. |
| `CRISPER_MODEL_ID` | empty | Overrides `CRISPER_MODEL_SIZE` with a full Hugging Face model id. |
| `CRISPER_BACKEND` | `ct2` | GPU-friendly backend for RunPod. |
| `CRISPER_DEVICE` | `cuda` | Use `cpu` only for local debugging. |
| `CRISPER_COMPUTE_TYPE` | `float16` | Recommended for GPU inference. |
| `CRISPER_DOWNLOAD_ROOT` | empty | Optional explicit model download/cache directory. |
| `CRISPER_LANGUAGE` | `es` | Default language. |
| `CRISPER_MODE` | `verbatim` | Keeps hesitations/repetitions for dyslexia-risk speech analysis. |
| `CRISPER_WORD_TIMESTAMPS` | `true` | Include word timing when available. |
| `MAX_AUDIO_BYTES` | `26214400` | Maximum accepted audio size. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Download timeout for `audio_url`. |

## Build

```bash
docker build -t tamizai-crisper-worker:latest .
```

Tag for GHCR:

```bash
docker tag tamizai-crisper-worker:latest ghcr.io/TU_USUARIO/tamizai-crisper-worker:latest
docker push ghcr.io/TU_USUARIO/tamizai-crisper-worker:latest
```

Or Docker Hub:

```bash
docker tag tamizai-crisper-worker:latest TU_USUARIO/tamizai-crisper-worker:latest
docker push TU_USUARIO/tamizai-crisper-worker:latest
```

## RunPod Serverless

Create a queue-based Serverless endpoint using the pushed image.

Suggested starting point:

- GPU: L4, RTX 3090, RTX 4090, A4000, A5000, or better.
- Minimum workers: `0` while validating cost.
- Maximum workers: `1` until the backend queue and timeout behavior are tuned.
- Container disk: enough for the model cache.
- Environment: keep the defaults unless testing another model.

After deployment, RunPod will expose an endpoint URL. The backend integration should store:

```text
CRISPER_REMOTE_ENDPOINT_URL=
CRISPER_REMOTE_API_KEY=
CRISPER_REMOTE_TIMEOUT_SECONDS=120
```

## Local Notes

This image is meant for NVIDIA GPU runtime. Local CPU testing is possible by overriding:

```bash
docker run --rm \
  -e CRISPER_DEVICE=cpu \
  -e CRISPER_COMPUTE_TYPE=int8 \
  tamizai-crisper-worker:latest
```

For real validation, deploy it on RunPod and test with `audio_url`, because the backend will normally send a signed URL for the uploaded exercise audio.

## Backend Integration Plan

The next backend branch should add a remote Crisper adapter that calls this RunPod endpoint and maps this response into the existing TamizAI speech-to-text port. No database migration should be needed because the scoring pipeline already consumes transcript text from an adapter boundary.
