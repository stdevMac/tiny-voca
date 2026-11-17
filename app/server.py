"""Local web interface for tiny-voca.

The released ONNX model runs on this machine with ONNX Runtime on the CPU. The
browser only sends text to this server and plays the audio it gets back;
nothing leaves the computer. This is the same model file the official browser
demo uses, so it does not depend on the PyTorch training code.

Start it with ``./run.sh`` from the repository root, or::

    app/.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator, Literal

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
ONNX = WEIGHTS / "onnx" / "model.onnx"
CONFIG = WEIGHTS / "config.json"
PHONEMIZER = WEIGHTS / "phonemizer"
STATIC = Path(__file__).resolve().parent / "static"
os.environ["DIZIONE_DATA"] = str(PHONEMIZER)

import dizione
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

MAX_TEXT = 5000
ENGLISH_ONLY = "tiny-voca only reads English for now."
# One request at a time: the model is tiny, but two generations would compete
# for the same CPU cores and make both slower. The phonemizer keeps a small
# mutable cache, so its calls are serialised too.
INFER_LOCK = Lock()
PHONEME_LOCK = Lock()


class UtteranceTooLong(ValueError):
    """The utterance needs more frames than the model's limit: split it."""


# Sentence splitting, kept the same as tiny_voca.infer and the browser demo.
_SPACE = re.compile("[\\s" + chr(0xFEFF) + "]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_NOT_AN_END = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$",
    re.IGNORECASE | re.ASCII,
)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile("[\\s" + chr(0xFEFF) + "]+$")
_CLOSING = "\"'”’)]»"


def sentences(text: str) -> list[str]:
    parts: list[str] = []
    current = ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def finished(text: str) -> str:
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def shorter(text: str) -> list[str]:
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def limit_peak(wave: np.ndarray, peak: float = 0.99) -> np.ndarray:
    loudest = float(np.abs(wave).max()) if wave.size else 0.0
    return wave * (peak / loudest) if loudest > peak else wave


class ModelState:
    """The ONNX model, its config and the phonemizer, loaded once at startup."""

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

    def run(self, phones, voice: str, speed: float, pitch_shift: float, pitch_range: float) -> np.ndarray:
        """Phonemes -> float32 waveform, one call of the ONNX model."""
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
            raise RuntimeError("model produced non-finite audio")
        return wave

    def load(self) -> None:
        started = time.perf_counter()
        try:
            if not ONNX.exists():
                raise FileNotFoundError(
                    f"missing {ONNX}; run ./run.sh to download the weights"
                )
            self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
            model = self.config["model"]
            self.sample_rate = int(model["sample_rate"])
            self.hop = int(model["hop"])
            self.max_tokens = int(model["max_tokens"])
            self.max_frames = int(model["max_frames"])
            self.voices = list(model["speakers"])

            # The phonemizer must produce the exact inventory the model was
            # trained on. The data files in weights/phonemizer are the ones
            # shipped with the released checkpoint.
            if int(self.config["phonemizer"]["version"]) != dizione.VERSION:
                raise RuntimeError("the installed dizione version does not match the model")
            expected = list(self.config["vocab"].keys())
            if list(dizione.symbols.SYMBOLS) != expected:
                raise RuntimeError("the phoneme inventory does not match the model")

            with PHONEME_LOCK:
                dizione.phonemize("Warm up.")

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(ONNX), sess_options=options, providers=["CPUExecutionProvider"]
            )
            # Run once so the first real request does not pay for lazy setup.
            with PHONEME_LOCK:
                warm = dizione.phonemize("Warm up.")
            self.run(warm, self.voices[0], 1.0, 0.0, 1.0)
            self.load_ms = (time.perf_counter() - started) * 1000
        except Exception as error:  # shown by /api/health and the page
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
    # The model takes log-F0 units; people think in semitones.
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
    await asyncio.to_thread(state.load)
    yield


app = FastAPI(title="tiny-voca local", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health")
@app.get("/api/health")
def health():
    if state.error:
        return JSONResponse({"status": "error", "message": state.error}, status_code=503)
    if state.session is None:
        return {"status": "loading"}
    return {
        "status": "ready",
        "model": "tiny-voca",
        "voices": state.voices,
        "sampleRate": state.sample_rate,
        "parameters": 4537291,
        "checkpointStep": state.config.get("step"),
        "loadMs": round(state.load_ms or 0, 1),
        "device": "cpu",
        "runtime": f"onnxruntime {ort.__version__}",
        "modelBytes": ONNX.stat().st_size,
        "maxText": MAX_TEXT,
    }


def _phonemize(text: str):
    """``dizione.phonemize`` but ``None`` when the text has nothing to say."""
    try:
        return dizione.phonemize(text)
    except ValueError:
        if any(word for mark, words in dizione.pieces(text) if not mark for word in words):
            raise
        return None


def _prepare(text: str, timing: dict[str, float]) -> list[tuple[str, object]]:
    """Sentences that can be read, with their phonemes. Unreadable pieces
    (another script, emoji only) are skipped, as in the reference demo."""
    prepared: list[tuple[str, object]] = []
    for sentence in sentences(finished(text)):
        started = time.perf_counter()
        with PHONEME_LOCK:
            phones = _phonemize(sentence)
        timing["phoneme_ms"] += (time.perf_counter() - started) * 1000
        if phones is not None:
            prepared.append((sentence, phones))
    return prepared


def _utterances(
    sentence: str,
    phones,
    voice: str,
    speed: float,
    pitch_shift: float,
    pitch_range: float,
    timing: dict[str, float],
) -> Iterator[tuple[np.ndarray, object]]:
    """One sentence -> one or more utterances the model can fit. This mirrors
    ``tiny_voca.infer.speak_text`` but yields each utterance as soon as it is
    made, so the page can start playing before the sentence is finished."""
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
            yield wave, phones
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
            yield from _utterances(
                piece, piece_phones, voice, speed, pitch_shift, pitch_range, timing
            )


@app.post("/api/tts")
def tts(request: SpeechRequest):
    """The whole clip as one WAV file. The page uses the streaming endpoint."""
    model = state.require()
    started = time.perf_counter()
    timing = {"phoneme_ms": 0.0, "model_ms": 0.0}
    try:
        prepared = _prepare(request.text, timing)
        waves: list[np.ndarray] = []
        spoken = []
        for sentence, phones in prepared:
            for wave, utterance_phones in _utterances(
                sentence,
                phones,
                request.voice,
                request.speed,
                request.pitch_shift,
                request.pitch_range,
                timing,
            ):
                waves.append(limit_peak(wave))
                spoken.append(utterance_phones)
        if not waves:
            raise HTTPException(status_code=422, detail=ENGLISH_ONLY)
        audio = np.concatenate(waves)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    buffer = io.BytesIO()
    sf.write(buffer, audio, model.sample_rate, format="WAV", subtype="PCM_16")
    seconds = len(audio) / model.sample_rate
    elapsed = (time.perf_counter() - started) * 1000
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Audio-Seconds": f"{seconds:.3f}",
            "X-Generation-Ms": f"{elapsed:.1f}",
            "X-Utterances": str(len(spoken)),
            "Content-Disposition": 'inline; filename="tiny-voca.wav"',
        },
    )


def _event(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@app.post("/api/tts/stream")
def tts_stream(request: SpeechRequest):
    """Newline-delimited JSON: one event per sentence, with base64 float32
    audio, so the browser can play while the rest is still being made."""
    model = state.require()

    def generate() -> Iterator[bytes]:
        started = time.perf_counter()
        timing = {"phoneme_ms": 0.0, "model_ms": 0.0}
        try:
            prepared = _prepare(request.text, timing)
            if not prepared:
                yield _event({"type": "error", "message": ENGLISH_ONLY})
                return
            total = len(prepared)
            rate = model.sample_rate
            yield _event(
                {
                    "type": "start",
                    "total": total,
                    "sampleRate": rate,
                    "voices": model.voices,
                }
            )

            samples_total = 0
            utterances = 0
            phonemes: list[str] = []
            for index, (sentence, phones) in enumerate(prepared):
                for wave, spoken in _utterances(
                    sentence,
                    phones,
                    request.voice,
                    request.speed,
                    request.pitch_shift,
                    request.pitch_range,
                    timing,
                ):
                    wave = limit_peak(wave)
                    samples = np.ascontiguousarray(wave, dtype="<f4")
                    samples_total += len(samples)
                    utterances += 1
                    text = " ".join(spoken.symbols)
                    phonemes.append(text)
                    yield _event(
                        {
                            "type": "sentence",
                            "index": index,
                            "total": total,
                            "phonemes": text,
                            "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
                            "sampleRate": rate,
                        }
                    )

            if not utterances:
                yield _event({"type": "error", "message": ENGLISH_ONLY})
                return

            yield _event(
                {
                    "type": "done",
                    "audioSeconds": samples_total / rate,
                    "generationMs": (time.perf_counter() - started) * 1000,
                    "phonemeMs": timing["phoneme_ms"],
                    "modelMs": timing["model_ms"],
                    "utterances": utterances,
                    "sentences": total,
                    "phonemes": phonemes,
                }
            )
        except Exception as error:
            yield _event({"type": "error", "message": str(error)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# API routes are registered above; everything else is the page.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
