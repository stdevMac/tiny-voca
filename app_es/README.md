# tiny-voca español — local web interface

The same page as `../app/`, pointed at the Spanish model. Same HTML layout,
same `app.js`, same player, same API; only the phonemizer and the weights
differ.

```bash
./run-es.sh
```

Then open <http://127.0.0.1:8000>. The first run creates `app_es/.venv` with
the packages in `requirements.txt` (no PyTorch: the page runs the ONNX export).
If `weights/es/tiny-voca-es.onnx` and `tiny-voca-es.json` are not there, it
downloads them, about 21 MB, from
[marcosmaceo/tiny-voca-es](https://huggingface.co/marcosmaceo/tiny-voca-es).
Later runs start in a second or two and download nothing.

## What differs from the English interface

| | `app/` (English) | `app_es/` (Spanish) |
|---|---|---|
| phonemizer | `dizione` | eSpeak NG 1.52.0, 42 phones, `es-ES` |
| weights | [n4ze3m/tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) | [marcosmaceo/tiny-voca-es](https://huggingface.co/marcosmaceo/tiny-voca-es) |
| voices | F2, M2 (American English) | F2, M2 (Castilian) |
| frontend files | its own | reuses `app/static/*.js` and `style.css` |

The phonemizer is imported from `src/tiny_voca`, not reimplemented. A hand-written
eSpeak wrapper would be a second tokenizer, and the failure mode is producing
tokens that differ from training in one punctuation mark. At startup the server
compares `frontend_fingerprint()` against the value the export recorded and
refuses to run if they disagree, so a wrong eSpeak version is an error on the
page rather than plausible-sounding nonsense.

The released export speaks with the mixed excitation described in
[`docs/spanish.md`](../docs/spanish.md#excitation-at-inference).

## Options

| Variable | Default | Meaning |
|---|---|---|
| `TINY_VOCA_ES_MODEL` | `weights/es/tiny-voca-es.onnx` | another ONNX export |
| `TINY_VOCA_ES_CONFIG` | `weights/es/tiny-voca-es.json` | its JSON sidecar |
| `TINY_VOCA_PORT` | 8000 | port |
| `TINY_VOCA_HOST` | 127.0.0.1 | address to listen on |
| `TINY_VOCA_NO_OPEN` | 0 | 1 stops the script from opening a browser |

When `TINY_VOCA_ES_MODEL` or `TINY_VOCA_ES_CONFIG` is set, nothing is
downloaded; a missing file is an error. To try your own export:

```bash
.venv/bin/python -m tiny_voca.export --checkpoint weights/es/tiny-voca-es.pt --output out/tiny-voca-es.onnx
TINY_VOCA_ES_MODEL=out/tiny-voca-es.onnx TINY_VOCA_ES_CONFIG=out/tiny-voca-es.json ./run-es.sh
```

## Requirements

- Python 3.10 or newer. `run-es.sh` uses `uv` when it is installed, else
  `python3 -m venv` and `pip`.
- `espeak-ng` 1.52.0 on `PATH`. A different version changes the phonemes and the
  fingerprint check will stop the server. On Linux, `scripts/host_setup.sh`
  builds it from source.

## API

Same as the English interface:

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"¿Cree que el comité aprobará la propuesta antes del verano?","voice":"F2","speed":1.0}' \
  -o pregunta.wav
```

`POST /api/tts/stream` returns newline-delimited JSON, one event per sentence,
so playback starts before the paragraph is finished.

## Measured

On a laptop, four CPU threads, with the earlier pulses-only export: the model
generates 3.6 s of audio in 76–290 ms, and eSpeak adds 120–530 ms per sentence.
End to end that is 4–14× realtime depending on how many sentences are in
flight. The released export's excitation takes about a tenth longer.
