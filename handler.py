import base64
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import runpod


MODEL_ALIASES = {
    "small": "nyralabs/CrisperWhisper2.0_small",
    "medium": "nyralabs/CrisperWhisper2.0_medium",
    "large": "nyralabs/CrisperWhisper2.0_large",
    "turbo": "nyralabs/CrisperWhisper2.0_turbo",
}

DEFAULT_MODEL = os.getenv("CRISPER_MODEL_SIZE", "medium")
MODEL_ID = os.getenv("CRISPER_MODEL_ID", MODEL_ALIASES.get(DEFAULT_MODEL, DEFAULT_MODEL))
BACKEND = os.getenv("CRISPER_BACKEND", "ct2")
DEVICE = os.getenv("CRISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("CRISPER_COMPUTE_TYPE", "float16")
DEFAULT_LANGUAGE = os.getenv("CRISPER_LANGUAGE", "es")
DEFAULT_MODE = os.getenv("CRISPER_MODE", "verbatim")
DEFAULT_WORD_TIMESTAMPS = os.getenv("CRISPER_WORD_TIMESTAMPS", "true").lower() == "true"
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
DOWNLOAD_ROOT = os.getenv("CRISPER_DOWNLOAD_ROOT")

_MODEL: Any | None = None
_MODEL_LOCK = Lock()


class WorkerError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def get_model() -> Any:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        from crisperwhisper import CrisperWhisperModel

        kwargs: dict[str, Any] = {
            "backend": BACKEND,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        }
        if DOWNLOAD_ROOT:
            kwargs["download_root"] = DOWNLOAD_ROOT

        _MODEL = CrisperWhisperModel(
            MODEL_ID,
            **kwargs,
        )
        return _MODEL


def normalize_input(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        raise WorkerError("INVALID_INPUT", "RunPod input must be a JSON object.")
    return payload


def download_audio(audio_url: str, target: Path) -> None:
    try:
        with requests.get(audio_url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            total = 0
            with target.open("wb") as audio_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise WorkerError(
                            "AUDIO_TOO_LARGE",
                            f"Audio is larger than MAX_AUDIO_BYTES={MAX_AUDIO_BYTES}.",
                            413,
                        )
                    audio_file.write(chunk)
    except WorkerError:
        raise
    except requests.RequestException as exc:
        raise WorkerError("AUDIO_DOWNLOAD_FAILED", str(exc), 422) from exc


def decode_audio(audio_base64: str, target: Path) -> None:
    try:
        raw_audio = base64.b64decode(audio_base64, validate=True)
    except ValueError as exc:
        raise WorkerError("INVALID_AUDIO_BASE64", "audio_base64 is not valid base64.") from exc

    if len(raw_audio) > MAX_AUDIO_BYTES:
        raise WorkerError(
            "AUDIO_TOO_LARGE",
            f"Audio is larger than MAX_AUDIO_BYTES={MAX_AUDIO_BYTES}.",
            413,
        )

    target.write_bytes(raw_audio)


def materialize_audio(payload: dict[str, Any], temp_dir: Path) -> Path:
    suffix = str(payload.get("audio_format") or "wav").strip().lstrip(".")
    if not suffix:
        suffix = "wav"

    audio_path = temp_dir / f"audio.{suffix}"
    audio_url = payload.get("audio_url")
    audio_base64 = payload.get("audio_base64")

    if audio_url:
        download_audio(str(audio_url), audio_path)
        return audio_path

    if audio_base64:
        decode_audio(str(audio_base64), audio_path)
        return audio_path

    raise WorkerError(
        "MISSING_AUDIO_INPUT",
        "Send either input.audio_url or input.audio_base64.",
    )


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def serialize_word(word: Any) -> dict[str, Any]:
    if isinstance(word, dict):
        return {
            "word": word.get("word") or word.get("text"),
            "start": to_float(word.get("start")),
            "end": to_float(word.get("end")),
            "confidence": to_float(word.get("confidence") or word.get("score")),
        }

    return {
        "word": getattr(word, "word", None) or getattr(word, "text", None),
        "start": to_float(getattr(word, "start", None)),
        "end": to_float(getattr(word, "end", None)),
        "confidence": to_float(getattr(word, "confidence", None) or getattr(word, "score", None)),
    }


def serialize_result(result: Any, elapsed_seconds: float, language: str, mode: str) -> dict[str, Any]:
    words = getattr(result, "words", None)
    if words is None and isinstance(result, dict):
        words = result.get("words")

    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    if text is None:
        text = str(result)

    return {
        "provider": "crisper_whisper",
        "model": MODEL_ID,
        "model_version": getattr(result, "model_version", None),
        "backend": BACKEND,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": getattr(result, "language", None) or (result.get("language") if isinstance(result, dict) else None) or language,
        "mode": mode,
        "text": text,
        "duration_seconds": to_float(getattr(result, "duration", None) or (result.get("duration") if isinstance(result, dict) else None)),
        "processing_time_seconds": elapsed_seconds,
        "words": [serialize_word(word) for word in (words or [])],
    }


def transcribe(audio_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    model = get_model()
    language = str(payload.get("language") or DEFAULT_LANGUAGE)
    mode = str(payload.get("mode") or DEFAULT_MODE)
    word_timestamps = bool(payload.get("word_timestamps", DEFAULT_WORD_TIMESTAMPS))

    started_at = time.perf_counter()
    result = model.transcribe(
        str(audio_path),
        language=language,
        mode=mode,
        word_timestamps=word_timestamps,
    )
    elapsed = time.perf_counter() - started_at

    return serialize_result(result, elapsed, language, mode)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = normalize_input(job)
        with tempfile.TemporaryDirectory(prefix="tamizai-crisper-") as temp_dir_name:
            audio_path = materialize_audio(payload, Path(temp_dir_name))
            return transcribe(audio_path, payload)
    except WorkerError as exc:
        return {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "status_code": exc.status_code,
            }
        }
    except Exception as exc:
        return {
            "error": {
                "code": "TRANSCRIPTION_FAILED",
                "message": str(exc),
                "status_code": 500,
            }
        }


runpod.serverless.start({"handler": handler})
