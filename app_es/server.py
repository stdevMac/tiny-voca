"""Local web interface for the Spanish tiny-voca model.

This is the same page as ``app/`` — same HTML, same ``app.js``, same player —
pointed at the Spanish checkpoint instead of the English one. The differences
are all in this file:

* the phonemizer is the Spanish front end (eSpeak NG, 42 phones) rather than
  ``dizione``, so the tokens match the ones the model was trained on;
* the weights are ``weights/es/tiny-voca-es.onnx`` and ``.json``, which
  ``run-es.sh`` downloads from https://huggingface.co/marcosmaceo/tiny-voca-es.

The front end is imported from ``src/tiny_voca`` rather than reimplemented. A
hand-written eSpeak wrapper would be a second implementation of the tokenizer,
and the whole failure mode here is producing tokens that differ from training in
one punctuation mark. The fingerprint check below makes that a startup error.

    ./run-es.sh                      # or, directly:
    .venv/bin/python -m uvicorn app_es.server:app --host 127.0.0.1 --port 8000

Environment:
    TINY_VOCA_ES_MODEL   .onnx path (default weights/es/tiny-voca-es.onnx)
    TINY_VOCA_ES_CONFIG  .json path (default weights/es/tiny-voca-es.json)
    TINY_VOCA_PORT       port to listen on (used by run-es.sh)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Iterator, Literal

# The front end is chosen once, at import, from this variable. It has to be set
# before tiny_voca.text is imported or the English profile is loaded instead.
os.environ.setdefault("TINY_VOCA_LANGUAGE", "es")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from tiny_voca.text import NothingToPronounce, finished, frontend_fingerprint, phonemize, sentences, shorter

APP_STATIC = ROOT / "app" / "static"
ES_STATIC = Path(__file__).resolve().parent / "static"

WEIGHTS = ROOT / "weights" / "es"
ONNX = Path(os.environ.get("TINY_VOCA_ES_MODEL", WEIGHTS / "tiny-voca-es.onnx")).expanduser()
CONFIG = Path(os.environ.get("TINY_VOCA_ES_CONFIG", WEIGHTS / "tiny-voca-es.json")).expanduser()

MAX_TEXT = 5000
SPANISH_ONLY = "tiny-voca solo lee español por ahora."
# One request at a time: the model is tiny, but two generations would compete for
# the same cores. eSpeak runs as a subprocess per piece, so it is serialised too.
INFER_LOCK = Lock()
PHONEME_LOCK = Lock()


class UtteranceTooLong(ValueError):
    """The utterance needs more frames than the model's limit: split it."""


class ModelState:
    def __init__(self) -> None:
        self.session = None
        self.config: dict = {}
        self.error: str | None = None
        self.load_ms: float | None = None
        self.sample_rate = 24000
        self.hop = 256
        self.max_tokens = 512
        self.max_frames = 3000
        self.voices = ["F2", "M2"]
        self.parameters = 0

    def run(self, phones, voice: str, speed: float, pitch_shift: float, pitch_range: float) -> np.ndarray:
        feed = {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([self.voices.index(voice)], dtype=np.int64),
            "speed": np.array([speed], dtype=np.float32),
            "pitch_shift": np.array([pitch_shift], dtype=np.float32),
            "pitch_range": np.array([pitch_range], dtype=np.float32),
        }
        audio, durations = self.session.run(["audio", "durations"], feed)
        frames = int(np.asarray(durations).sum())
        if frames > self.max_frames:
            raise UtteranceTooLong("utterance exceeds the model's frame limit; split it")
        wave = np.asarray(audio, dtype=np.float32).reshape(-1)
        expected = frames * self.hop
        if expected and len(wave) > expected:
            wave = wave[:expected]
        if not np.isfinite(wave).all():
            raise RuntimeError("the model produced non-finite audio")
        return wave

    def load(self) -> None:
        started = time.perf_counter()
        try:
            for path in (ONNX, CONFIG):
                if not path.exists():
                    raise FileNotFoundError(f"missing {path}; run ./run-es.sh to download the weights")
            self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
            self.sample_rate = int(self.config["sample_rate"])
            self.hop = int(self.config["hop"])
            self.max_tokens = int(self.config["max_tokens"])
            self.max_frames = int(self.config["max_frames"])
            self.voices = list(self.config["voices"])

            # The exported config records the fingerprint of the front end that
            # made the training tokens. If this machine's eSpeak or accent rules
            # differ, the model would be fed symbols it has never seen, so refuse
            # to start rather than produce plausible-sounding nonsense.
            if self.config.get("frontend_fingerprint") != frontend_fingerprint():
                raise RuntimeError(
                    "this machine's Spanish front end does not match the checkpoint "
                    f"({frontend_fingerprint()} != {self.config.get('frontend_fingerprint')}); "
                    "install eSpeak NG 1.52 and check TINY_VOCA_ACCENT"
                )

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(ONNX), sess_options=options,
                                                providers=["CPUExecutionProvider"])
            # The export does not record the parameter count and ONNX Runtime
            # does not expose the initializers, so count them from the graph.
            try:
                import onnx
                graph = onnx.load(str(ONNX), load_external_data=False).graph
                self.parameters = sum(int(np.prod(t.dims)) if t.dims else 1 for t in graph.initializer)
            except Exception:
                self.parameters = int(self.config.get("parameters") or 0)
            # Warm up at a realistic length. ONNX Runtime allocates its arena for
            # the shapes it has seen, so warming up on a three-word sentence left
            # the first real request paying two seconds of allocation.
            with PHONEME_LOCK:
                warm = phonemize("El tren de las ocho y media sale del andén número cuatro, "
                                 "y el viaje hasta la costa dura casi una hora.")
            self.run(warm, self.voices[0], 1.0, 0.0, 1.0)
            self.load_ms = (time.perf_counter() - started) * 1000
        except Exception as error:  # surfaced by /api/health and on the page
            self.error = str(error)
            self.session = None

    def require(self):
        if self.session is None:
            raise HTTPException(status_code=503, detail=self.error or "the model is still loading")
        return self


state = ModelState()


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    voice: Literal["F2", "M2"] = "F2"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch_semitones: float = Field(0.0, ge=-12.0, le=12.0)
    pitch_range: float = Field(1.0, ge=0.0, le=2.0)

    @field_validator("text")
    @classmethod
    def text_must_have_something(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @property
    def pitch_shift(self) -> float:
        return self.pitch_semitones * math.log(2) / 12


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    await asyncio.to_thread(state.load)
    yield


app = FastAPI(title="tiny-voca español", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/api/health")
@app.get("/health")
def health():
    if state.error:
        return JSONResponse({"status": "error", "message": state.error}, status_code=503)
    if state.session is None:
        return {"status": "loading"}
    return {
        "status": "ready",
        "model": "tiny-voca-es",
        "language": "es-ES",
        "voices": state.voices,
        "sampleRate": state.sample_rate,
        "parameters": state.parameters,
        "checkpointStep": state.config.get("step"),
        "loadMs": round(state.load_ms or 0, 1),
        "device": "cpu",
        "runtime": f"onnxruntime {ort.__version__}",
        "modelBytes": ONNX.stat().st_size,
        "maxText": MAX_TEXT,
    }


def limit_peak(wave: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """``tiny_voca.infer.limit_peak`` takes a torch tensor; this path is numpy."""
    loudest = float(np.abs(wave).max()) if wave.size else 0.0
    return wave * (peak / loudest) if loudest > peak else wave


def trim_silence(wave: np.ndarray, sample_rate: int, lead_keep: float = 0.03,
                 trail_keep: float = 0.15, floor_db: float = -40.0) -> np.ndarray:
    """Cut the dead air the model puts around every utterance.

    The recordings it learned from begin with about 0.6 s of lead-in silence, and
    the duration predictor reproduces that faithfully: a four-sentence paragraph
    came out 29% silence, with 0.72 s before the first word. That silence is not
    digital zero either — it carries a faint tone inherited from the teacher, so
    it is both dead air and a low-level artefact.

    A little of it is kept on purpose: 30 ms at the start so the first consonant
    is not clipped, 150 ms at the end so sentences in a paragraph still have a
    natural gap between them.
    """
    frame = max(1, int(0.010 * sample_rate))
    count = len(wave) // frame
    if count < 4:
        return wave
    env = np.sqrt(np.mean(wave[:count * frame].reshape(count, frame) ** 2, axis=1))
    peak = float(env.max())
    if peak <= 0:
        return wave
    audible = env > peak * 10 ** (floor_db / 20)
    if not audible.any():
        return wave
    first = int(np.argmax(audible))
    last = count - int(np.argmax(audible[::-1]))
    start = max(0, first * frame - int(lead_keep * sample_rate))
    end = min(len(wave), last * frame + int(trail_keep * sample_rate))
    return wave[start:end] if end > start else wave


def _phonemize(text: str):
    try:
        return phonemize(text)
    except NothingToPronounce:
        return None


def _prepare(text: str, timing: dict[str, float]) -> list[tuple[str, object]]:
    prepared: list[tuple[str, object]] = []
    for sentence in sentences(finished(text)):
        started = time.perf_counter()
        with PHONEME_LOCK:
            phones = _phonemize(sentence)
        timing["phoneme_ms"] += (time.perf_counter() - started) * 1000
        if phones is not None:
            prepared.append((sentence, phones))
    return prepared


def _utterances(sentence, phones, voice, speed, pitch_shift, pitch_range, timing):
    """One sentence -> one or more utterances the model can fit, yielded as soon
    as each is made so the page can start playing before the sentence ends."""
    limit = state.max_tokens - 12
    if len(phones) <= limit:
        started = time.perf_counter()
        try:
            with INFER_LOCK:
                wave = state.run(phones, voice, speed, pitch_shift, pitch_range)
        except UtteranceTooLong:
            pass
        else:
            timing["model_ms"] += (time.perf_counter() - started) * 1000
            yield trim_silence(wave, state.sample_rate), phones
            return
        timing["model_ms"] += (time.perf_counter() - started) * 1000

    pieces = shorter(sentence)
    if not pieces:
        raise ValueError(f"one word is too long to speak: {sentence[:40]!r}")
    for piece in pieces:
        started = time.perf_counter()
        with PHONEME_LOCK:
            piece_phones = _phonemize(piece)
        timing["phoneme_ms"] += (time.perf_counter() - started) * 1000
        if piece_phones is not None:
            yield from _utterances(piece, piece_phones, voice, speed, pitch_shift, pitch_range, timing)


@app.post("/api/tts")
def tts(request: SpeechRequest):
    model = state.require()
    started = time.perf_counter()
    timing = {"phoneme_ms": 0.0, "model_ms": 0.0}
    try:
        prepared = _prepare(request.text, timing)
        waves, spoken = [], []
        for sentence, phones in prepared:
            for wave, utterance in _utterances(sentence, phones, request.voice, request.speed,
                                               request.pitch_shift, request.pitch_range, timing):
                waves.append(limit_peak(wave))
                spoken.append(utterance)
        if not waves:
            raise HTTPException(status_code=422, detail=SPANISH_ONLY)
        audio = np.concatenate(waves)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    buffer = io.BytesIO()
    sf.write(buffer, audio, model.sample_rate, format="WAV", subtype="PCM_16")
    seconds = len(audio) / model.sample_rate
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Audio-Seconds": f"{seconds:.3f}",
            "X-Generation-Ms": f"{(time.perf_counter() - started) * 1000:.1f}",
            "X-Utterances": str(len(spoken)),
            "Content-Disposition": 'inline; filename="tiny-voca-es.wav"',
        },
    )


def _event(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@app.post("/api/tts/stream")
def tts_stream(request: SpeechRequest):
    model = state.require()

    def generate() -> Iterator[bytes]:
        started = time.perf_counter()
        timing = {"phoneme_ms": 0.0, "model_ms": 0.0}
        try:
            prepared = _prepare(request.text, timing)
            if not prepared:
                yield _event({"type": "error", "message": SPANISH_ONLY})
                return
            total, rate = len(prepared), model.sample_rate
            yield _event({"type": "start", "total": total, "sampleRate": rate, "voices": model.voices})

            samples_total, utterances, phonemes = 0, 0, []
            for index, (sentence, phones) in enumerate(prepared):
                for wave, spoken in _utterances(sentence, phones, request.voice, request.speed,
                                                request.pitch_shift, request.pitch_range, timing):
                    wave = limit_peak(wave)
                    samples = np.ascontiguousarray(wave, dtype="<f4")
                    samples_total += len(samples)
                    utterances += 1
                    text = " ".join(spoken.symbols)
                    phonemes.append(text)
                    yield _event({"type": "sentence", "index": index, "total": total, "phonemes": text,
                                  "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
                                  "sampleRate": rate})

            if not utterances:
                yield _event({"type": "error", "message": SPANISH_ONLY})
                return
            yield _event({"type": "done", "audioSeconds": samples_total / rate,
                          "generationMs": (time.perf_counter() - started) * 1000,
                          "phonemeMs": timing["phoneme_ms"], "modelMs": timing["model_ms"],
                          "utterances": utterances, "sentences": total, "phonemes": phonemes})
        except Exception as error:
            yield _event({"type": "error", "message": str(error)})

    return StreamingResponse(generate(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/")
@app.get("/index.html")
def index():
    return FileResponse(ES_STATIC / "index.html")


# Everything else (app.js, player.js, style.css) is the English interface's own
# file, unchanged: only the copy and the model differ.
app.mount("/", StaticFiles(directory=APP_STATIC, html=True), name="static")
