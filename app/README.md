# Local tiny-voca interface

A small local website for [tiny-voca](https://github.com/n4ze3m/tiny-voca), a
4.5M-parameter English text-to-speech model with two voices. The released ONNX
model runs on your CPU with ONNX Runtime; the page talks to a FastAPI server on
`127.0.0.1`. It does not need the PyTorch training code.

## Start

```bash
./run.sh
```

The first run creates `app/.venv`, installs the Python packages and downloads
the weights (about 40 MB) from Hugging Face. Then it opens
<http://127.0.0.1:8000>.

Requirements: Python 3.10+ (3.12 recommended), about 150 MB of disk for the
runtime, and `uv` if you have it (the script falls back to `python3 -m venv`).

To use a different port, or to open it from a phone on the same network:

```bash
TINY_VOCA_PORT=8080 ./run.sh
TINY_VOCA_HOST=0.0.0.0 ./run.sh
```

The second one exposes the server to your local network, so only use it on a
network you trust.

## What it does

- Type or paste text, choose **F2** or **M2**, and adjust speed.
- Open **Voice shaping** for pitch and pitch range.
- Audio starts playing sentence by sentence while the rest is still being
  generated, and the player keeps the whole clip.
- Download the result as a WAV, and inspect the phonemes and timings.
- Everything stays on this machine.

## API

The page uses the streaming endpoint, but the server also exposes a plain WAV
endpoint:

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from the command line.","voice":"F2","speed":1.0}' \
  -o hello.wav
```

`POST /api/tts/stream` returns newline-delimited JSON: a `start` event, one
`sentence` event per generated utterance with base64 float32 audio, and a
`done` event with the timings.

## Notes

- English only, two fixed voices, no voice cloning.
- The model is small and fast: roughly 70× real time on four CPU threads.
- The upstream browser demo is in `web/`; this interface is a local addition.
