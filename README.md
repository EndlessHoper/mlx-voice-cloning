# Meowl Voice

Voice cloning running locally on Apple Silicon. Record ~30 seconds of speech, then generate anything in that voice.

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and [MLX](https://github.com/Blaizzy/mlx-audio). FastAPI backend, React frontend.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.10+
- Node.js 18+

## Setup

```bash
# Install dependencies
make install

# Run (builds frontend, serves everything on :8000)
make run
```

Then open [http://localhost:8000](http://localhost:8000).

For development (hot-reload frontend on :5173, API on :8000):

```bash
make dev
```

## How it works

1. **Record** — Read the script out loud (~10-30 seconds). The app captures your voice via the browser mic.
2. **Train** — The recording is processed into a voice profile using Qwen3-TTS speaker embeddings.
3. **Generate** — Type anything and generate speech in the cloned voice.

The model runs entirely on-device via MLX on the Apple GPU. No data leaves your machine.

## Models

Uses [cr2k2/Qwen3-TTS-12Hz-1.7B-Base-fp32](https://huggingface.co/cr2k2/Qwen3-TTS-12Hz-1.7B-Base-fp32) by default (~4.5GB, downloaded automatically on first run). The 0.6B model is available but not recommended for voice cloning quality.

## Performance

On Apple Silicon (M5 MacBook Pro), expect ~29 seconds wall time for ~10 seconds of audio (RTF ~2.9x). The first generation is slower due to MLX graph compilation — a warmup pass runs automatically at startup.

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEOWLVOICE_BACKEND` | `mlx` | Backend engine (`mlx` or `pytorch`) |
| `MEOWLVOICE_API_HOST` | `127.0.0.1` | API bind address |
| `MEOWLVOICE_API_PORT` | `8000` | API port |
| `MEOWLVOICE_CORS_ORIGINS` | `localhost:5173,...` | Allowed CORS origins |
| `MEOWLVOICE_LOG_LEVEL` | `INFO` | Log level |

## Tech stack

- **Backend**: FastAPI + MLX Audio + torchaudio
- **Frontend**: React + Framer Motion + Vite
- **Model**: Qwen3-TTS 1.7B (fp32 MLX conversion)

## License

MIT
