"""Compare numbered checkpoints by evaluation, not by step number.

After the adversarial phase starts, the last checkpoint is not necessarily the
best one: the GAN can sharpen the voice or make it unstable, and the mel
distance does not fall monotonically. `train.py` already keeps a numbered
checkpoint every `--keep-every` steps; this exports each one to inference form,
runs `tiny_voca.evaluate` on it, and prints the scores side by side.

    python scripts/compare_checkpoints.py --run data/runs/es-1 --annotations data/annotations-es \\
        --steps 60000 80000 100000 --output data/eval/compare

The recommendation is the checkpoint with the lowest mean word error rate,
ties broken by the DTW mel distance. Listen to the winner before believing it.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(command)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, help="Checkpoint steps to compare; default all found")
    parser.add_argument("--per-voice", type=int, default=3, help="Clips per voice per split (keep it small)")
    parser.add_argument("--novel", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    found = sorted(int(path.stem.split("-")[1]) for path in args.run.glob("step-*.pt") if path.stem.split("-")[1].isdigit())
    steps = args.steps or found
    if not steps:
        raise SystemExit(f"no numbered checkpoints in {args.run}; train with --keep-every")
    args.output.mkdir(parents=True, exist_ok=True)

    def resolve(step):
        """The numbered checkpoint, or last.pt when it holds that same step.

        The final step is not a multiple of --keep-every, so it is only ever
        written as last.pt. That is usually the checkpoint worth comparing.
        """
        numbered = args.run / f"step-{step:08d}.pt"
        if numbered.is_file():
            return numbered
        last = args.run / "last.pt"
        if last.is_file():
            state = torch.load(last, map_location="cpu", weights_only=True)
            if int(state["step"]) == step:
                return last
        return None

    rows = []
    for step in steps:
        checkpoint = resolve(step)
        if checkpoint is None:
            print(f"skip {step}: no checkpoint at that step", flush=True)
            continue
        inference = args.output / f"inference-{step:08d}.pt"
        if not inference.is_file():
            run([sys.executable, str(Path(__file__).resolve().parent / "export_checkpoint.py"),
                 "--checkpoint", str(checkpoint), "--output", str(inference)])
        report = args.output / f"report-{step:08d}.json"
        if not report.is_file():
            run([sys.executable, "-m", "tiny_voca.evaluate", "--checkpoint", str(inference),
                 "--annotations", str(args.annotations), "--output", str(args.output / f"audio-{step:08d}"),
                 "--tag", f"step-{step}", "--device", args.device, "--splits", "validation", "test",
                 "--per-voice", str(args.per_voice), "--novel", str(args.novel)])
            source = args.output / f"audio-{step:08d}" / "report.json"
            report.write_text(source.read_text(encoding="utf-8"))
        data = json.loads(report.read_text(encoding="utf-8"))
        summary = data["summary"]
        values = [summary[group]["all"] for group in ("validation", "test", "novel") if group in summary]
        wer = [v["generated_wer"] for v in values if v.get("generated_wer") is not None]
        mel = [v["mel_dtw"] for v in values if v.get("mel_dtw") is not None]
        rows.append({"step": step, "wer": sum(wer) / len(wer) if wer else None,
                     "mel_dtw": sum(mel) / len(mel) if mel else None,
                     "samples": sum(v["samples"] for v in values)})
        print(json.dumps(rows[-1]), flush=True)
    if not rows:
        raise SystemExit("nothing to compare")
    ranked = sorted(rows, key=lambda r: (r["wer"] if r["wer"] is not None else 9, r["mel_dtw"] if r["mel_dtw"] is not None else 9))
    print("\nstep        wer     mel_dtw  samples")
    for row in rows:
        mark = "  <- best" if row is ranked[0] else ""
        print(f"{row['step']:>8}  {row['wer'] if row['wer'] is None else round(row['wer'], 4):>7}  "
              f"{row['mel_dtw'] if row['mel_dtw'] is None else round(row['mel_dtw'], 4):>7}  {row['samples']:>7}{mark}")
    (args.output / "comparison.json").write_text(json.dumps({"rows": rows, "best": ranked[0]}, indent=2) + "\n", encoding="utf-8")
    winner = args.output / f"inference-{ranked[0]['step']:08d}.pt"
    print(f"\nbest: step {ranked[0]['step']}  ({winner})")


if __name__ == "__main__":
    main()
