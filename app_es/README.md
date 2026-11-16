# tiny-voca español — local web interface

The same page as `../app/`, pointed at the Spanish checkpoint this repository's
run produced. Same HTML layout, same `app.js`, same player, same API; only the
phonemizer and the weights differ.

```bash
./run-es.sh
```

Then open <http://127.0.0.1:8000>. Nothing is downloaded — the weights are the
artifacts from the run, in `~/tiny-voca-artifacts/es-1-anneal`.

## What differs from the English interface

| | `app/` (English) | `app_es/` (Spanish) |
|---|---|---|
| phonemizer | `dizione` | eSpeak NG 1.52, 42 phones, `es-ES` |
| weights | downloaded from Hugging Face | local artifacts |
| voices | F2, M2 (American English) | F2, M2 (Castilian) |
| frontend files | its own | reuses `app/static/*.js` and `style.css` |

The phonemizer is imported from `src/tiny_voca`, not reimplemented. A hand-written
eSpeak wrapper would be a second tokenizer, and the failure mode is producing
tokens that differ from training in one punctuation mark. At startup the server
compares `frontend_fingerprint()` against the value the export recorded and
refuses to run if they disagree, so a wrong eSpeak version is an error on the
page rather than plausible-sounding nonsense.

## Choosing a model

`TINY_VOCA_ES_DIR` selects the run. To compare against the pre-anneal baseline:

```bash
TINY_VOCA_ES_DIR=~/tiny-voca-artifacts/es-1 \
TINY_VOCA_ES_MODEL=~/tiny-voca-artifacts/es-1/eval/es-1/inference-best.pt.onnx \
  ./run-es.sh
```

Or set `TINY_VOCA_ES_MODEL` and `TINY_VOCA_ES_CONFIG` directly. The baseline's
ONNX is `eval/es-1/tiny-voca-es-best.onnx` with `tiny-voca-es-best.json`.

## Requirements

- Python with `fastapi`, `uvicorn`, `onnxruntime`, `soundfile` (the repository's
  `.venv` already has them)
- `espeak-ng` 1.52 on `PATH`. A different version changes the phonemes and the
  fingerprint check will stop the server.

## API

Same as the English interface:

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"¿Cree que el comité aprobará la propuesta antes del verano?","voice":"F2","speed":1.0}' \
  -o hola.wav
```

`POST /api/tts/stream` returns newline-delimited JSON, one event per sentence,
so playback starts before the paragraph is finished.

## Measured

On this machine, four CPU threads: the model generates 3.6 s of audio in
76–290 ms, and eSpeak adds 120–530 ms per sentence. End to end that is 4–14×
realtime depending on how many sentences are in flight.
