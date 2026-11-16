#!/usr/bin/env bash
# Fine-tune the shipped model with the inter-harmonic loss, and keep it only if it wins.
#
# Round 1 measured the model against its own teacher: it loses 1.0 dB of
# periodicity on F2 and 2.2 dB on M2, and the loss has no term that can see
# that. The multi-resolution STFT and mel distances are satisfied by a spectrum
# that is right on average, including one whose partials are smeared into the
# gaps between them. `HarmonicLoss` compares the inter-harmonic share of the
# prediction with the recording's own share, per voiced frame.
#
# This branches from the shipped anneal rather than resuming it, because the
# recipe changed (a new loss weight). `--init-from` takes over the weights, the
# critics and the optimizer moments; `--resume` would refuse a changed recipe.
#
# The rate is chosen to re-enter the band the anneal finished in, not to jump:
#
#     base 4.30e-3, half-life 20,000
#     step 142,000 -> 4.30e-3 * 0.5**7.1 = 3.1e-5   (the anneal's own start)
#     step 162,000 -> 4.30e-3 * 0.5**8.1 = 1.6e-5   (where the anneal ended)
#
#   ./scripts/harmonic_es.sh            # train
#   ./scripts/harmonic_es.sh --compare  # evaluate both runs and say which won
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
PY=${PY:-.venv/bin/python}
BASE_RUN=${BASE_RUN:-es-1-anneal}
RUN=${RUN:-es-1-harmonic}
STEPS=${STEPS:-162000}
HARMONIC=${HARMONIC:-4.0}
BATCH_FRAMES=${BATCH_FRAMES:-9000}
MAX_BATCH=${MAX_BATCH:-32}
TRAIN_SECONDS=${TRAIN_SECONDS:-14400}
OUT="$DATA/runs/$RUN"

if [[ "${1:-}" == "--compare" ]]; then
  echo "== evaluating $BASE_RUN and $RUN with the same protocol"
  for name in "$BASE_RUN" "$RUN"; do
    checkpoint="$DATA/runs/$name/inference.pt"
    [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
    "$PY" -m tiny_voca.evaluate --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --output "$DATA/eval/harmonic-$name" --tag "$name" --device cuda --splits validation test \
      --per-voice 20 --asr-model large-v3-turbo
    "$PY" scripts/probe_periodicity.py --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --per-voice 12 --output "$DATA/eval/harmonic-$name/periodicity.json"
  done
  "$PY" - "$DATA/eval/harmonic-$BASE_RUN" "$DATA/eval/harmonic-$RUN" <<'PY'
import json, sys
names, rows = ("base", "harmonic"), []
for name, root in zip(names, sys.argv[1:]):
    summary = json.load(open(f"{root}/report.json"))["summary"]
    values = [summary[g]["all"] for g in ("validation", "test") if g in summary]
    wer = [v["generated_wer"] for v in values if v.get("generated_wer") is not None]
    mel = [v["mel_dtw"] for v in values if v.get("mel_dtw") is not None]
    periodicity = json.load(open(f"{root}/periodicity.json"))["summary"]
    rows.append({"run": name, "wer": sum(wer) / len(wer) if wer else None,
                 "mel_dtw": sum(mel) / len(mel) if mel else None,
                 "hnr": {voice: value["model_hnr"] for voice, value in periodicity.items()},
                 "deficit": {voice: value["deficit"] for voice, value in periodicity.items()},
                 "samples": sum(v["samples"] for v in values)})
for row in rows:
    hnr = "  ".join(f"{voice} {value:5.2f} dB ({row['deficit'][voice]:+.2f})" for voice, value in row["hnr"].items())
    print(f"{row['run']:9} wer {row['wer']:.4f}  mel_dtw {row['mel_dtw']:.4f}  HNR {hnr}  n={row['samples']}")
base, tuned = rows
better = []
for voice in base["hnr"]:
    if tuned["hnr"][voice] > base["hnr"][voice] + 0.3:
        better.append(f"{voice} +{tuned['hnr'][voice] - base['hnr'][voice]:.2f} dB")
print()
print("periodicity improved on: " + (", ".join(better) if better else "neither voice"))
print(f"wer {base['wer']:.4f} -> {tuned['wer']:.4f}   mel {base['mel_dtw']:.4f} -> {tuned['mel_dtw']:.4f}")
PY
  exit 0
fi

if [[ -e "$OUT" ]]; then
  echo "$OUT already exists; remove it or set RUN" >&2
  exit 1
fi

"$PY" -m tiny_voca.train --train "$DATA/annotations-es/train.jsonl" \
  --validation "$DATA/annotations-es/validation.jsonl" --output "$OUT" --device cuda \
  --steps "$STEPS" --init-from "$DATA/runs/$BASE_RUN/last.pt" \
  --batch-frames "$BATCH_FRAMES" --max-batch "$MAX_BATCH" \
  --learning-rate 4.30e-3 --lr-half-life 20000 --warmup 1 \
  --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --keep-every 5000 \
  --weights "{\"harmonic\": $HARMONIC}" \
  --max-seconds "$TRAIN_SECONDS" --deadline-file "$DATA/budget.json"

echo
echo "done: $OUT/inference.pt"
