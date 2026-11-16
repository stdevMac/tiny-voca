#!/usr/bin/env bash
# Polish the finished run with a low learning rate, and keep it only if it wins.
#
# The run never annealed. `--lr-half-life` is 100,000 and training stopped at
# 122,000, so the learning rate was still 1.46e-4 at the last step, about half of
# peak. A flat validation curve at that rate is a noise floor, not convergence:
# the usual fix is to decay the rate and let the weights settle.
#
# `--resume` refuses a changed recipe, so this branches with `--init-from`, which
# takes over the weights, the critics and the optimizer moments. The schedule is
# absolute-step, `base * 0.5 ** (step / half_life)`, so the base is chosen to
# land on the intended rate at the branch point rather than being a peak:
#
#     base 2.05e-3, half-life 20,000
#     step 122,000 -> 2.05e-3 * 0.5**6.1  = 3.0e-5   (10% of the original peak)
#     step 142,000 -> 2.05e-3 * 0.5**7.1  = 1.5e-5   (halved again)
#
# Twenty thousand steps is about 80 minutes on a 4090, which is the cheapest
# experiment available: it needs no new audio, no new alignment and no teacher.
#
#   ./scripts/anneal_es.sh            # train
#   ./scripts/anneal_es.sh --compare  # evaluate both and say which won
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
PY=${PY:-.venv/bin/python}
BASE_RUN=${BASE_RUN:-es-1}
RUN=${RUN:-es-1-anneal}
STEPS=${STEPS:-142000}
TRAIN_SECONDS=${TRAIN_SECONDS:-10800}
OUT="$DATA/runs/$RUN"

if [[ "${1:-}" == "--compare" ]]; then
  echo "== evaluating $BASE_RUN and $RUN with the same protocol"
  for name in "$BASE_RUN" "$RUN"; do
    checkpoint="$DATA/runs/$name/inference.pt"
    [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
    "$PY" -m tiny_voca.evaluate --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --output "$DATA/eval/anneal-$name" --tag "$name" --device cuda --splits validation test \
      --per-voice 20 --asr-model large-v3-turbo
  done
  "$PY" - "$DATA/eval/anneal-$BASE_RUN/report.json" "$DATA/eval/anneal-$RUN/report.json" <<'PY'
import json, sys
names, rows = ("base", "anneal"), []
for name, path in zip(names, sys.argv[1:]):
    summary = json.load(open(path))["summary"]
    values = [summary[g]["all"] for g in ("validation", "test") if g in summary]
    wer = [v["generated_wer"] for v in values if v.get("generated_wer") is not None]
    mel = [v["mel_dtw"] for v in values if v.get("mel_dtw") is not None]
    rows.append({"run": name, "wer": sum(wer) / len(wer) if wer else None,
                 "mel_dtw": sum(mel) / len(mel) if mel else None,
                 "samples": sum(v["samples"] for v in values)})
for row in rows:
    print(f"{row['run']:7} wer {row['wer']:.4f}  mel_dtw {row['mel_dtw']:.4f}  n={row['samples']}")
best = sorted(rows, key=lambda r: (r["wer"] if r["wer"] is not None else 9,
                                   r["mel_dtw"] if r["mel_dtw"] is not None else 9))[0]
print(f"\nbetter: {best['run']}")
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
  --learning-rate 2.05e-3 --lr-half-life 20000 --warmup 1 \
  --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
  --max-seconds "$TRAIN_SECONDS" --deadline-file "$DATA/budget.json"

echo
echo "done: $OUT/inference.pt"
