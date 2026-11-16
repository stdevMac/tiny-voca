"""Measure question intonation: does the voice rise at the end of a question?

Spanish marks a yes/no question with intonation, not word order: *Vienes mañana.*
and *¿Vienes mañana?* differ only in the final contour. Word error rate cannot
see this, and the mel distance averages it away, so a model that reads every
question as a statement still scores well. This probe isolates the one number
that matters: the slope of F0 over the last part of the utterance.

It runs on any directory of recordings with their texts, so the same code scores
the teacher (the target) and the trained model (the result):

    # the target: what the corpus actually sounds like
    python scripts/probe_intonation.py --manifest data/audio-es/manifest.jsonl \\
        --audio-dir data/audio-es/audio --output data/eval/intonation-teacher.json

    # the model: what it produces from text alone
    python scripts/probe_intonation.py --report data/eval/es-1/report.json \\
        --audio-dir data/eval/es-1/audio --output data/eval/intonation-es-1.json

A rising final contour has a positive slope. The useful comparison is the gap
between questions and statements: the teacher's gap is the target, and a model
that learned the pattern reproduces most of it.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLE_RATE = 16000
# The final contour of a Spanish question lives in the last stressed syllable
# and everything after it, which is roughly the last third of the utterance.
TAIL = 0.35
MIN_VOICED_SECONDS = 0.25


def final_slope(path, sample_rate=SAMPLE_RATE):
    """Slope of log-F0 in semitones per second over the tail of a recording.

    Returns None when too little of the tail is voiced to fit a line: a whisper
    or a clipped ending says nothing about intonation and must not be counted
    as a flat contour.
    """
    import librosa
    wave, rate = librosa.load(str(path), sr=sample_rate, mono=True)
    if wave.size < int(0.3 * sample_rate) or not np.isfinite(wave).all():
        return None
    f0, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=sample_rate,
                                 frame_length=1024, hop_length=160)
    if f0 is None:
        return None
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    if voiced.sum() < 5:
        return None
    times = np.arange(f0.size) * (160 / sample_rate)
    last = times[voiced][-1]
    first = times[voiced][0]
    span = last - first
    if span < MIN_VOICED_SECONDS:
        return None
    tail = voiced & (times >= last - TAIL * span)
    if tail.sum() < 5:
        return None
    x = times[tail]
    y = 12 * np.log2(np.maximum(f0[tail], 1e-3) / 200.0)
    slope = float(np.polyfit(x - x.mean(), y, 1)[0])
    return {"slope": slope, "tail_seconds": round(float(x[-1] - x[0]), 3), "voiced_frames": int(tail.sum())}


# Spanish asks two different tunes. A yes/no question ends on a rise; a
# question that opens with an interrogative word ends on a fall, because the
# rise has already happened on that first stressed syllable. Scoring all
# questions together hides the difference and lets a model that raises pitch
# after every "¿" look correct, so the two are counted apart.
WH = {"qué", "quién", "quiénes", "dónde", "cuándo", "cómo", "cuánto", "cuánta", "cuántos", "cuántas",
      "cuál", "cuáles", "adónde"}
WH_PAIR = {"por qué", "para qué", "a dónde", "de dónde", "desde cuándo", "hasta cuándo"}


def classify(text):
    text = text.strip()
    if text.endswith("?"):
        body = text.lstrip("¿¡").strip()
        words = body.split()
        first = words[0].lower().strip(",.") if words else ""
        pair = " ".join(words[:2]).lower().strip(",.") if len(words) > 1 else ""
        return "question_wh" if first in WH or pair in WH_PAIR else "question_yesno"
    if text.endswith("!"):
        return "exclamation"
    return "statement"


def load_jobs(args):
    """(id, text, generated path, reference path or None) for every recording."""
    jobs = []
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        for sample in report["samples"]:
            generated = Path(args.audio_dir) / sample["generated"]
            reference = Path(args.audio_dir) / f"{sample['id']}-reference.wav"
            jobs.append((sample["id"], sample["text"], generated, reference if reference.is_file() else None))
    else:
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            jobs.append((row["id"], row["text"], Path(args.audio_dir) / Path(row["audio"]).name, None))
    return jobs


def summarize(values):
    if not values:
        return {"samples": 0, "mean_slope": None, "median_slope": None, "rising_fraction": None}
    slopes = np.array([v["slope"] for v in values])
    return {"samples": int(slopes.size), "mean_slope": round(float(slopes.mean()), 3),
            "median_slope": round(float(np.median(slopes)), 3),
            "rising_fraction": round(float((slopes > 0).mean()), 3),
            "mean_tail_seconds": round(float(np.mean([v["tail_seconds"] for v in values])), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Corpus manifest (teacher audio)")
    source.add_argument("--report", type=Path, help="evaluate report.json (model audio)")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Cap the recordings per kind, for a quick look")
    args = parser.parse_args()

    jobs = load_jobs(args)
    kinds = ("question_yesno", "question_wh", "statement", "exclamation")
    buckets = {kind: [] for kind in kinds}
    reference_buckets = {kind: [] for kind in kinds}
    for identifier, text, generated, reference in jobs:
        kind = classify(text)
        if len(buckets[kind]) >= (args.limit or 10 ** 9):
            continue
        if not generated.is_file():
            continue
        measured = final_slope(generated)
        if measured is None:
            continue
        buckets[kind].append(measured)
        if reference is not None:
            measured_reference = final_slope(reference)
            if measured_reference is not None:
                reference_buckets[kind].append(measured_reference)

    def gap(values, against):
        if not values or not against:
            return None
        return round(summarize(values)["mean_slope"] - summarize(against)["mean_slope"], 3)

    result = {"source": str(args.manifest or args.report), "audio_dir": str(args.audio_dir),
              "generated": {kind: summarize(values) for kind, values in buckets.items()}}
    if any(reference_buckets.values()):
        result["reference"] = {kind: summarize(values) for kind, values in reference_buckets.items()}
    for label, values in (("yesno", buckets["question_yesno"]), ("wh", buckets["question_wh"])):
        value = gap(values, buckets["statement"])
        if value is not None:
            result[f"{label}_minus_statement_semitones_per_second"] = value
    for label, values in (("yesno", reference_buckets["question_yesno"]), ("wh", reference_buckets["question_wh"])):
        reference_gap = gap(values, reference_buckets["statement"])
        if reference_gap is None:
            continue
        result[f"reference_{label}_minus_statement_semitones_per_second"] = reference_gap
        measured_gap = result.get(f"{label}_minus_statement_semitones_per_second")
        if measured_gap is not None and reference_gap != 0:
            result[f"{label}_gap_recovered"] = round(measured_gap / reference_gap, 3)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    for kind in kinds:
        summary = summarize(buckets[kind])
        if summary["samples"]:
            print(f"{kind:16} n={summary['samples']:3}  {summary['mean_slope']:+.2f} st/s  "
                  f"rising {summary['rising_fraction']:.0%}")
    for label in ("yesno", "wh"):
        measured_gap = result.get(f"{label}_minus_statement_semitones_per_second")
        reference_gap = result.get(f"reference_{label}_minus_statement_semitones_per_second")
        if measured_gap is not None:
            detail = f" (teacher {reference_gap:+.2f}, {result.get(f'{label}_gap_recovered', 0):.0%} recovered)" if reference_gap is not None else ""
            print(f"{label:16} gap vs statement {measured_gap:+.2f} st/s{detail}")


if __name__ == "__main__":
    main()
