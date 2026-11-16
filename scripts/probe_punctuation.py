"""Does the model take the question tune from the punctuation, or from the words?

The corpus's questions are conversational ("¿Dónde dejaste las llaves?") and its
statements are mostly encyclopedic, so a model can learn "chatty words mean rise"
and still score well on held-out questions. That model would read a formal
yes/no question flat, which is exactly the case a Spanish voice has to get right.

This removes the vocabulary from the test. It takes validation statements and
speaks each one twice: as written, and with the same words turned into a yes/no
question by changing only the punctuation ("El tren sale a las ocho." becomes
"¿El tren sale a las ocho?"). Same words, same voice, same seed. Any difference
in the final contour comes from the `¿` and `?` alone.

    python scripts/probe_punctuation.py --checkpoint data/runs/es-1/inference.pt \\
        --annotations data/annotations-es --output data/eval/punctuation-es-1

A model that learned the cue raises the question version well above the
statement version. The teacher's gap on real questions is about +8 st/s; the
useful number here is the share of pairs where the question rises more than the
statement, which a vocabulary shortcut cannot fake.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tiny_voca.infer import load_model, speak
from probe_intonation import final_slope


def as_question(text):
    body = text.strip()
    if not body.endswith("."):
        return None
    body = body[:-1].rstrip()
    if not body or body.endswith((",", ";", ":")):
        return None
    return "¿" + body + "?"


def measure(wave, rate, path):
    import soundfile as sf
    samples = np.asarray(wave, dtype=np.float32)
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0:
        samples = samples / peak * 0.95      # the model can peak above 1.0; clipping would skew the pitch track
    sf.write(path, samples, rate, subtype="PCM_16")
    return final_slope(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-voice", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    rate = model.config.sample_rate
    rows = [json.loads(line) for line in (args.annotations / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    statements = [row for row in rows if row["text"].strip().endswith(".") and as_question(row["text"])]
    if not statements:
        raise SystemExit("no statements to convert")
    # Spread over the length distribution instead of taking the shortest.
    statements.sort(key=lambda row: len(row["text"]))
    chosen = [statements[int(q * (len(statements) - 1))] for q in np.linspace(0.1, 0.9, args.per_voice)]

    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for number, row in enumerate(chosen):
        question = as_question(row["text"])
        for voice in model.config.speakers:
            with torch.no_grad():
                flat, _, _ = speak(model, row["text"], voice, seed=args.seed)
                risen, _, _ = speak(model, question, voice, seed=args.seed)
            stem = f"s{number:02d}-{voice}"
            a = measure(flat.numpy(), rate, args.output / f"{stem}-statement.wav")
            b = measure(risen.numpy(), rate, args.output / f"{stem}-question.wav")
            if a is None or b is None:
                continue
            pairs.append({"statement": row["text"], "question": question, "voice": voice,
                          "statement_slope": round(a["slope"], 3), "question_slope": round(b["slope"], 3),
                          "rise": round(b["slope"] - a["slope"], 3)})

    if not pairs:
        raise SystemExit("no pair had a voiced tail in both versions")
    rises = np.array([p["rise"] for p in pairs])
    summary = {"pairs": len(pairs), "mean_statement_slope": round(float(np.mean([p["statement_slope"] for p in pairs])), 3),
               "mean_question_slope": round(float(np.mean([p["question_slope"] for p in pairs])), 3),
               "mean_rise": round(float(rises.mean()), 3), "median_rise": round(float(np.median(rises)), 3),
               "share_question_rises_more": round(float((rises > 0).mean()), 3),
               "share_question_above_zero": round(float(np.mean([p["question_slope"] > 0 for p in pairs])), 3)}
    result = {"checkpoint": str(args.checkpoint), "split": args.split, "seed": args.seed, "summary": summary, "pairs": pairs}
    (args.output / "punctuation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n{summary['share_question_rises_more']:.0%} of pairs rise more as a question "
          f"({summary['mean_rise']:+.2f} st/s); {summary['share_question_above_zero']:.0%} of questions end above flat")


if __name__ == "__main__":
    main()
