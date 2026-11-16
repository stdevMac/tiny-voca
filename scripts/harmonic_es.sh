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
  # Everything round 1 measured, on both checkpoints, so the fine-tune is judged
  # on the same battery that shipped the anneal rather than on HNR alone. A
  # periodicity win that costs word error rate or the question rise is not a win.
  measure() {
    local name="$1" checkpoint="$2" out="$DATA/eval/harmonic-$1"
    mkdir -p "$out"
    echo "== $name: $checkpoint"
    "$PY" -m tiny_voca.evaluate --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --output "$out" --tag "$name" --device cuda --splits validation test \
      --per-voice 20 --asr-model large-v3-turbo
    "$PY" scripts/probe_periodicity.py --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --per-voice 12 --output "$out/periodicity.json"
    "$PY" scripts/probe_punctuation.py --checkpoint "$checkpoint" --annotations "$DATA/annotations-es" \
      --output "$out/punctuation" --per-voice 12 --device cuda
    "$PY" scripts/probe_intonation.py --report "$out/report.json" --audio-dir "$out/audio" \
      --output "$out/intonation.json" --limit 40
    "$PY" scripts/probe_voices.py --report "$out/report.json" --audio-dir "$out/audio" \
      --references "$DATA/voices-es/references" --output "$out/voices.json"
    "$PY" scripts/demo_es.py --checkpoint "$checkpoint" --output "$out/demo" --device cuda
  }
  measure base "$DATA/runs/$BASE_RUN/inference.pt"
  measure harmonic "$DATA/runs/$RUN/inference.pt"
  "$PY" - "$DATA/eval/harmonic-base" "$DATA/eval/harmonic-harmonic" <<'PY'
import json, sys
names, rows = ("base", "harmonic"), []
for name, root in zip(names, sys.argv[1:]):
    summary = json.load(open(f"{root}/report.json"))["summary"]
    values = [summary[g]["all"] for g in ("validation", "test") if g in summary]
    wer = [v["generated_wer"] for v in values if v.get("generated_wer") is not None]
    mel = [v["mel_dtw"] for v in values if v.get("mel_dtw") is not None]
    periodicity = json.load(open(f"{root}/periodicity.json"))["summary"]
    punctuation = json.load(open(f"{root}/punctuation.json"))
    intonation = json.load(open(f"{root}/intonation.json"))
    voices = json.load(open(f"{root}/voices.json"))
    rows.append({"run": name, "wer": sum(wer) / len(wer) if wer else None,
                 "mel_dtw": sum(mel) / len(mel) if mel else None,
                 "hnr": {v: x["model_hnr"] for v, x in periodicity.items()},
                 "deficit": {v: x["deficit"] for v, x in periodicity.items()},
                 "punctuation": punctuation, "intonation": intonation, "voices": voices,
                 "samples": sum(v["samples"] for v in values)})
for row in rows:
    hnr = "  ".join(f"{v} {x:5.2f} dB ({row['deficit'][v]:+.2f})" for v, x in row["hnr"].items())
    print(f"{row['run']:9} wer {row['wer']:.4f}  mel_dtw {row['mel_dtw']:.4f}  HNR {hnr}  n={row['samples']}")
base, tuned = rows
print()
for voice in base["hnr"]:
    print(f"{voice} periodicity {base['hnr'][voice]:5.2f} -> {tuned['hnr'][voice]:5.2f} dB "
          f"({tuned['hnr'][voice] - base['hnr'][voice]:+.2f}), deficit {base['deficit'][voice]:+.2f} -> {tuned['deficit'][voice]:+.2f}")
print(f"wer {base['wer']:.4f} -> {tuned['wer']:.4f}   mel {base['mel_dtw']:.4f} -> {tuned['mel_dtw']:.4f}")
print(f"punctuation {json.dumps(base['punctuation'])[:160]}")
print(f"            {json.dumps(tuned['punctuation'])[:160]}")
print(f"intonation  {json.dumps(base['intonation'])[:160]}")
print(f"            {json.dumps(tuned['intonation'])[:160]}")
print(f"voices      {json.dumps(base['voices'])[:160]}")
print(f"            {json.dumps(tuned['voices'])[:160]}")
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
