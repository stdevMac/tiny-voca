#!/usr/bin/env python3
"""Measure how periodic the model is against the recording it was taught from.

HNR (harmonic-to-noise ratio) is the share of a frame's energy that sits on the
partials of its own F0. A model that leaves energy between the partials sounds
rough at the edges of words even when its spectrum is right on average, and
that is exactly what the multi-resolution STFT loss cannot see.

The comparison is always model-versus-its-own-teacher on the same sentence,
because HNR is confounded by F0: a low voice packs more harmonics under 8 kHz
and scores higher for the same timbre. F2 and M2 must never be compared with
each other, only each with its own recording.

    ./scripts/probe_periodicity.py --checkpoint data/runs/es-1-anneal/inference.pt \
        --annotations data/annotations-es --output data/eval/periodicity.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from tiny_voca.data import AnnotatedDataset                            # noqa: E402
from tiny_voca.infer import load_model                                 # noqa: E402

SAMPLE_RATE, HOP = 24000, 160


def hnr(wave, sample_rate=SAMPLE_RATE, hop=HOP, ceiling=8000.0, tolerance=1.5):
    """Mean HNR in dB over the voiced frames of one utterance."""
    import librosa
    f0, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=sample_rate, frame_length=1024, hop_length=hop)
    voiced = np.asarray(voiced, bool) & np.isfinite(f0)
    if not voiced.any():
        return float("nan")
    spectrum = np.abs(librosa.stft(wave, n_fft=1024, hop_length=hop)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
    width = tolerance * sample_rate / 1024
    values = []
    for i in range(min(spectrum.shape[1], len(f0))):
        if not voiced[i] or not np.isfinite(f0[i]):
            continue
        partials = np.zeros_like(frequencies, dtype=bool)
        harmonic = 1
        while harmonic * f0[i] < ceiling:
            partials |= np.abs(frequencies - harmonic * f0[i]) <= width
            harmonic += 1
        band = frequencies <= ceiling
        noise = band & ~partials
        if spectrum[noise, i].sum() > 0:
            values.append(10 * np.log10(spectrum[partials, i].sum() / spectrum[noise, i].sum()))
    return float(np.mean(values)) if values else float("nan")


def pick(dataset, per_voice):
    """First `per_voice` utterances of each voice, in manifest order."""
    chosen, counts = [], {}
    for index, row in enumerate(dataset.rows):
        voice = row["speaker"]
        if counts.get(voice, 0) < per_voice:
            chosen.append(index)
            counts[voice] = counts.get(voice, 0) + 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="directory holding validation.jsonl")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-voice", type=int, default=12)
    parser.add_argument("--pitch-range", type=float, default=1.5, help="the web interface default")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    config = model.config
    dataset = AnnotatedDataset(args.annotations / f"{args.split}.jsonl", config)

    rows = []
    for index in pick(dataset, args.per_voice):
        item = dataset[index]
        tokens, stress = item["tokens"][None], item["stress"][None]
        length = torch.tensor([tokens.shape[1]])
        speaker = torch.tensor([item["speaker"]])
        with torch.inference_mode():
            wave, lengths, _ = model.synthesize(tokens, stress, length, speaker, speed=args.speed,
                                                seed=args.seed, pitch_range=args.pitch_range)
        generated = wave[0, : int(lengths[0])].cpu().numpy()
        recorded = item["wave"].numpy()
        rows.append({"id": dataset.rows[index]["id"], "speaker": config.speakers[item["speaker"]],
                     "teacher": hnr(recorded), "model": hnr(generated)})

    summary = {}
    for voice in config.speakers:
        values = [row for row in rows if row["speaker"] == voice and np.isfinite(row["model"])]
        if not values:
            continue
        teacher = float(np.mean([row["teacher"] for row in values]))
        model_hnr = float(np.mean([row["model"] for row in values]))
        summary[voice] = {"teacher_hnr": teacher, "model_hnr": model_hnr, "deficit": model_hnr - teacher,
                          "samples": len(values)}
    report = {"checkpoint": str(args.checkpoint), "pitch_range": args.pitch_range, "summary": summary, "rows": rows}
    print(f"{'voice':6}{'teacher':>9}{'model':>8}{'deficit':>9}{'n':>4}")
    for voice, value in summary.items():
        print(f"{voice:6}{value['teacher_hnr']:9.2f}{value['model_hnr']:8.2f}{value['deficit']:+9.2f}{value['samples']:4}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
