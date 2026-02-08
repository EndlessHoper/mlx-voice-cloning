import asyncio
import hashlib
import json
import logging
import os
import shutil
import struct
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_TITLE = "Meowl Voice API"

# Backend: "mlx" (default, Apple Silicon GPU, fp32) or "pytorch" (CPU/MPS, needs different transformers)
BACKEND = os.getenv("MEOWLVOICE_BACKEND", "mlx").lower()

PYTORCH_MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
MLX_MODELS = {
    "1.7B": "cr2k2/Qwen3-TTS-12Hz-1.7B-Base-fp32",
    "0.6B": "cr2k2/Qwen3-TTS-12Hz-0.6B-Base-fp32",
}
AVAILABLE_MODELS = MLX_MODELS if BACKEND == "mlx" else PYTORCH_MODELS
DEFAULT_MODEL_KEY = "1.7B"

ROOT_DIR = Path(__file__).resolve().parent.parent
REFS_DIR = ROOT_DIR / "refs"
OUT_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"

for directory in (REFS_DIR, OUT_DIR, LOGS_DIR):
    directory.mkdir(exist_ok=True)

LANGUAGE_OPTIONS = [
    "Auto",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

SCRIPT_LIBRARY = [
    (
        "Voice cloning has become incredibly easy. Scarily easy, in fact. This is running entirely"
        " on a regular Mac, locally, completely for free. A few years ago this would have been pretty"
        " much impossible for a normal consumer, simply unthinkable. Now it takes about thirty seconds"
        " of audio. The model runs offline, the data never leaves my machine, and"
        " the results are good enough to fool most people on a phone call. All it needs is a short"
        " recording. Thirty seconds of you talking naturally, and it picks up your pitch, your rhythm,"
        " the way you emphasize certain words. It even captures the little imperfections that make a"
        " voice sound human. And the quality is only going to get better. Smaller models, faster"
        " hardware, lower latency. What do you do about that? Honestly, I am not sure anyone has a"
        " good answer yet."
    ),
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("meowlvoice.fastapi")
    if logger.handlers:
        return logger

    raw_level = os.getenv("MEOWLVOICE_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, raw_level, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "fastapi.log",
        maxBytes=2_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


LOGGER = setup_logger()
LOGGER.info("Backend engine: %s", BACKEND)


@dataclass
class VoiceProfile:
    profile_id: str
    ref_audio_path: str
    ref_text: str
    x_vector_only: bool
    device: str
    created_at: float
    model_key: str = DEFAULT_MODEL_KEY
    backend: str = BACKEND
    # PyTorch only — cached prompt items from create_voice_clone_prompt()
    prompt_items: object = field(default=None, repr=False)


PROMPT_CACHE: dict[str, object] = {}
VOICE_PROFILES: dict[str, VoiceProfile] = {}
PROFILE_LOCK = Lock()

_MODEL = None
_MODEL_ID: str | None = None
_MODEL_DEVICE: str | None = None
MODEL_LOCK = Lock()


class SynthesizeRequest(BaseModel):
    profile_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="English")
    max_new_tokens: Optional[int] = Field(default=None, ge=200, le=4096)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    temperature: Optional[float] = Field(default=None, ge=0.01, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.01, le=1.0)


WARMUP_TEXT = "Hello, this is a warmup pass."
WARMUP_REF_TEXT = "Warmup reference."


def _run_warmup():
    """Run a short dummy generation to warm up the model (lazy graph init, JIT, etc.)."""
    model_key = DEFAULT_MODEL_KEY
    model_id = AVAILABLE_MODELS.get(model_key)
    if not model_id:
        LOGGER.warning("Warmup skipped: no model ID for key '%s'", model_key)
        return

    LOGGER.info("Warmup: loading model %s (%s backend)...", model_id, BACKEND)
    t0 = time.time()

    try:
        if BACKEND == "mlx":
            model = get_mlx_model(model_id)
            import mlx.core as mx

            # Use in-memory audio so warmup matches runtime MLX call path.
            warmup_audio = mx.zeros((TARGET_SR,), dtype=mx.float32)  # 1 second silence
            # Run a short generation to trigger lazy graph compilation
            for _ in model.generate(
                text=WARMUP_TEXT,
                ref_audio=warmup_audio,
                ref_text=WARMUP_REF_TEXT,
                max_tokens=50,
            ):
                break  # just need the first chunk to trigger compilation
        else:
            # PyTorch: just load the model — actual generation requires a trained prompt
            get_pytorch_model(model_id, "auto")

        elapsed = time.time() - t0
        LOGGER.info("Warmup complete in %.1fs", elapsed)
    except Exception:
        LOGGER.exception("Warmup failed (non-fatal, will retry on first request)")


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup: run warmup in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_warmup)
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

origins_raw = os.getenv(
    "MEOWLVOICE_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
allow_origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TARGET_SR = 24000


def script_stats(script_text: str) -> tuple[int, int]:
    words = len((script_text or "").split())
    est_seconds = max(1, int(words / 2.8))
    return words, est_seconds


def save_upload_file(audio: UploadFile) -> Path:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    raw_path = REFS_DIR / f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    LOGGER.info("Upload saved | filename=%s path=%s", audio.filename, raw_path)
    return raw_path


def prepare_recording(raw_path: Path) -> tuple[Path, float, tuple[np.ndarray, int]]:
    """Load, resample to 24kHz mono, save WAV, return (path, duration, (wav_array, sr)) tuple."""
    wav, sr = torchaudio.load(str(raw_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    out_path = REFS_DIR / f"script_ref_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    torchaudio.save(str(out_path), wav, TARGET_SR)
    duration = wav.shape[1] / TARGET_SR
    audio_tuple = (wav.squeeze(0).numpy(), TARGET_SR)
    LOGGER.info("Prepared recording | input=%s output=%s duration=%.2fs", raw_path, out_path, duration)
    return out_path, duration, audio_tuple


def prompt_cache_key(ref_audio_path: str, ref_text: str, x_vector_only: bool, device: str) -> str:
    try:
        stat = Path(ref_audio_path).stat()
        file_sig = f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        file_sig = "unknown"
    text_sig = hashlib.sha1(ref_text.encode("utf-8")).hexdigest() if ref_text else "empty"
    return f"{ref_audio_path}|{file_sig}|{x_vector_only}|{device}|{text_sig}"


# ── MLX Backend ──────────────────────────────────────────────────────────────


def get_mlx_model(model_id: str):
    global _MODEL, _MODEL_ID, _MODEL_DEVICE

    with MODEL_LOCK:
        if _MODEL is not None and _MODEL_ID == model_id:
            LOGGER.info("Reusing MLX model | model=%s", model_id)
            return _MODEL

        from mlx_audio.tts.utils import load_model

        _MODEL = load_model(model_id)
        _MODEL_ID = model_id
        _MODEL_DEVICE = "mlx"
        LOGGER.info("Loaded MLX model | model=%s", model_id)
        return _MODEL


def mlx_synthesize(
    model,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Generate speech using MLX backend. Returns (wav_array, sample_rate)."""
    results = list(model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        **kwargs,
    ))
    if not results:
        raise RuntimeError("MLX generation returned no audio segments")

    # Non-streaming MLX may still yield multiple segments (e.g., newline-split input).
    # Concatenate them into a single waveform for parity with PyTorch output handling.
    parts = [np.array(item.audio, copy=False) for item in results]
    wav = parts[0] if len(parts) == 1 else np.concatenate(parts)
    # Normalize if needed
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav, TARGET_SR


def mlx_synthesize_stream(
    model,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    **kwargs,
):
    """Generate speech using MLX backend with streaming. Yields (chunk_np, is_final) tuples."""
    got_any = False
    for result in model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        stream=True,
        streaming_interval=2.0,
        **kwargs,
    ):
        got_any = True
        chunk = np.array(result.audio, copy=False)
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        is_final = getattr(result, "is_final_chunk", False)
        yield chunk, is_final
    if not got_any:
        raise RuntimeError("MLX generation returned no audio segments")


# ── PyTorch Backend ──────────────────────────────────────────────────────────


def get_pytorch_model(model_id: str, device: str):
    global _MODEL, _MODEL_ID, _MODEL_DEVICE

    with MODEL_LOCK:
        if _MODEL is not None and _MODEL_ID == model_id and _MODEL_DEVICE == device:
            LOGGER.info("Reusing PyTorch model | model=%s device=%s", model_id, device)
            return _MODEL

        import torch
        from qwen_tts import Qwen3TTSModel

        if device == "auto":
            if torch.cuda.is_available():
                device_map = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = "cuda:0" if device == "cuda" else device

        is_cuda = device_map.startswith("cuda")
        dtype = torch.bfloat16 if is_cuda else torch.float32

        attn_impl = None
        if is_cuda:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                LOGGER.info("Using flash_attention_2")
            except ImportError:
                LOGGER.info("flash_attn not installed, using default attention")

        _MODEL = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        _MODEL_ID = model_id
        _MODEL_DEVICE = device
        LOGGER.info(
            "Loaded PyTorch model | model=%s device=%s device_map=%s dtype=%s attn=%s",
            model_id, device, device_map, dtype, attn_impl or "default",
        )
        return _MODEL


# ── Shared Helpers ───────────────────────────────────────────────────────────


def safe_audio_response(base_dir: Path, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    candidate = (base_dir / safe_name).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(candidate)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"ok": True, "app": APP_TITLE, "backend": BACKEND}


@app.get("/api/config")
def config():
    scripts = []
    for index, script in enumerate(SCRIPT_LIBRARY):
        words, est_seconds = script_stats(script)
        scripts.append(
            {
                "id": index,
                "text": script,
                "word_count": words,
                "estimated_seconds": est_seconds,
            }
        )
    return {
        "backend": BACKEND,
        "models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL_KEY,
        "languages": LANGUAGE_OPTIONS,
        "scripts": scripts,
        "recommended_seconds": {"min": 10, "max": 30},
    }


@app.post("/api/train")
def train_voice_profile(
    audio: UploadFile = File(...),
    script_text: str = Form(""),
    device: Literal["auto", "cpu", "cuda"] = Form("auto"),
    x_vector_only: bool = Form(False),
    cache_voice: bool = Form(True),
    model_key: str = Form(DEFAULT_MODEL_KEY),
):
    if not x_vector_only and not script_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Script text is required unless x_vector_only_mode is enabled.",
        )

    model_id = AVAILABLE_MODELS.get(model_key)
    if model_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Available: {', '.join(AVAILABLE_MODELS)}",
        )

    raw_path = save_upload_file(audio)
    try:
        ref_audio_path, duration, audio_tuple = prepare_recording(raw_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to prepare uploaded recording")
        raise HTTPException(status_code=500, detail=f"Failed to prepare recording: {exc}") from exc
    finally:
        if raw_path.exists():
            raw_path.unlink(missing_ok=True)

    warning = None
    if duration < 3.0:
        warning = "Recording is very short. Aim for at least 10-30 seconds for best results."

    try:
        ref_text = " ".join(script_text.split())
        prompt_items = None
        cache_hit = False

        if BACKEND == "pytorch":
            # PyTorch: extract speaker embeddings once and cache them
            model = get_pytorch_model(model_id, device)
            cache_key = prompt_cache_key(str(ref_audio_path), ref_text, x_vector_only, device)

            if cache_voice and cache_key in PROMPT_CACHE:
                prompt_items = PROMPT_CACHE[cache_key]
                cache_hit = True

            if prompt_items is None:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=audio_tuple,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                )
                if cache_voice:
                    PROMPT_CACHE[cache_key] = prompt_items
        else:
            # MLX: no separate prompt extraction — ref audio is passed at generate time.
            # Just validate the model loads OK.
            get_mlx_model(model_id)

        profile_id = f"voice_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        profile = VoiceProfile(
            profile_id=profile_id,
            ref_audio_path=str(ref_audio_path),
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            device=device,
            created_at=time.time(),
            model_key=model_key,
            backend=BACKEND,
            prompt_items=prompt_items,
        )
        with PROFILE_LOCK:
            VOICE_PROFILES[profile_id] = profile

        LOGGER.info(
            "Training complete | backend=%s profile=%s duration=%.2fs cache_hit=%s",
            BACKEND,
            profile_id,
            duration,
            cache_hit,
        )
        return {
            "profile_id": profile_id,
            "duration_seconds": duration,
            "status": "Voice profile trained successfully.",
            "warning": warning,
            "prompt_cached": bool(cache_voice),
            "cache_hit": cache_hit,
            "reference_audio_url": f"/api/audio/refs/{Path(ref_audio_path).name}",
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/api/synthesize")
def synthesize_from_profile(request: SynthesizeRequest):
    with PROFILE_LOCK:
        profile = VOICE_PROFILES.get(request.profile_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found. Train a profile first.")

    if request.language not in LANGUAGE_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported language option.")

    text = " ".join(request.text.split())

    try:
        model_id = AVAILABLE_MODELS.get(profile.model_key)
        if model_id is None:
            raise HTTPException(status_code=500, detail="Profile references unknown model.")

        # Only pass generation params that the user explicitly set (non-None).
        gen_kwargs: dict = {}
        if request.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = request.max_new_tokens
        if request.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = request.repetition_penalty
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            gen_kwargs["top_p"] = request.top_p

        LOGGER.info(
            "Synthesis started | backend=%s profile=%s model=%s language=%s chars=%s overrides=%s",
            profile.backend,
            request.profile_id,
            profile.model_key,
            request.language,
            len(request.text),
            gen_kwargs or "SDK defaults",
        )

        if profile.backend == "mlx":
            model = get_mlx_model(model_id)
            wav, sr = mlx_synthesize(
                model,
                text=text,
                ref_audio_path=profile.ref_audio_path,
                ref_text=profile.ref_text,
                **gen_kwargs,
            )
        else:
            model = get_pytorch_model(model_id, profile.device)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=request.language,
                voice_clone_prompt=profile.prompt_items,
                non_streaming_mode=True,
                **gen_kwargs,
            )
            wav = wavs[0]

        out_path = OUT_DIR / f"script_clone_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        sf.write(out_path, wav, sr)
        LOGGER.info("Synthesis complete | backend=%s profile=%s output=%s", profile.backend, request.profile_id, out_path)

        return {
            "status": "Generated speech.",
            "output_audio_url": f"/api/audio/outputs/{out_path.name}",
            "sample_rate": sr,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Synthesis failed | profile=%s", request.profile_id)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


@app.post("/api/synthesize-stream")
def synthesize_stream(request: SynthesizeRequest):
    """Stream audio chunks as they're generated (MLX only). Falls back to non-streaming for PyTorch."""
    with PROFILE_LOCK:
        profile = VOICE_PROFILES.get(request.profile_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found. Train a profile first.")
    if request.language not in LANGUAGE_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported language option.")
    if profile.backend != "mlx":
        raise HTTPException(status_code=400, detail="Streaming only supported with MLX backend.")

    text = " ".join(request.text.split())
    model_id = AVAILABLE_MODELS.get(profile.model_key)
    if model_id is None:
        raise HTTPException(status_code=500, detail="Profile references unknown model.")

    gen_kwargs: dict = {}
    if request.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = request.max_new_tokens
    if request.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = request.repetition_penalty
    if request.temperature is not None:
        gen_kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        gen_kwargs["top_p"] = request.top_p

    def generate_chunks():
        """Yield binary frames: [4-byte uint32 length][payload]. Audio = raw PCM float32. Final = JSON."""
        LOGGER.info(
            "Stream synthesis started | profile=%s model=%s",
            request.profile_id, profile.model_key,
        )
        all_chunks = []
        model = get_mlx_model(model_id)
        try:
            for chunk, is_final in mlx_synthesize_stream(
                model,
                text=text,
                ref_audio_path=profile.ref_audio_path,
                ref_text=profile.ref_text,
                **gen_kwargs,
            ):
                all_chunks.append(chunk)
                pcm_bytes = chunk.tobytes()
                yield struct.pack("<I", len(pcm_bytes)) + pcm_bytes
        except Exception as exc:
            LOGGER.exception("Stream synthesis failed | profile=%s", request.profile_id)
            error_json = json.dumps({"error": str(exc)}).encode("utf-8")
            yield struct.pack("<I", len(error_json)) + error_json
            return

        # Save full audio to disk for generation history
        if all_chunks:
            wav = np.concatenate(all_chunks) if len(all_chunks) > 1 else all_chunks[0]
            out_path = OUT_DIR / f"script_clone_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
            sf.write(out_path, wav, TARGET_SR)
            audio_dur = len(wav) / TARGET_SR
            LOGGER.info("Stream synthesis complete | profile=%s output=%s duration=%.1fs", request.profile_id, out_path, audio_dur)
            meta = json.dumps({
                "done": True,
                "output_audio_url": f"/api/audio/outputs/{out_path.name}",
                "sample_rate": TARGET_SR,
                "audio_duration": audio_dur,
            }).encode("utf-8")
            yield struct.pack("<I", len(meta)) + meta

    return StreamingResponse(generate_chunks(), media_type="application/octet-stream")


@app.get("/api/audio/refs/{filename}")
def get_reference_audio(filename: str):
    return safe_audio_response(REFS_DIR, filename)


@app.get("/api/audio/outputs/{filename}")
def get_output_audio(filename: str):
    return safe_audio_response(OUT_DIR, filename)


if FRONTEND_DIST_DIR.exists():
    LOGGER.info("Serving frontend build from %s", FRONTEND_DIST_DIR)
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:
    LOGGER.info(
        "Frontend build directory not found (%s). API-only mode active.",
        FRONTEND_DIST_DIR,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MEOWLVOICE_API_HOST", "127.0.0.1")
    port = int(os.getenv("MEOWLVOICE_API_PORT", "8000"))
    LOGGER.info("Launching API | host=%s port=%s backend=%s", host, port, BACKEND)
    uvicorn.run(app, host=host, port=port, reload=False)
