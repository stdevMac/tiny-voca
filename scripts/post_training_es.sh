#!/usr/bin/env bash
# Everything that happens after training, in the order it has to happen.
#
# The last checkpoint is not necessarily the best one: once the adversarial
# phase starts the voice can sharpen or destabilise, and the mel distance does
# not fall monotonically. So the numbered checkpoints are ranked first, the
# winner is re-evaluated with more samples than the ranking pass, and only then
# is it measured for the two things word error rate cannot see: whether a
# question rises at the end, and whether a listener hears a Castilian speaker.
#
#   MISTRAL_API_KEY=... ./scripts/post_training_es.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-es-1}
DATA=${DATA:-data}
PY=${PY:-.venv/bin/python}
STEPS=${STEPS:-"60000 80000 100000 110000 120000 122000"}
COMPARE="$DATA/eval/compare-$RUN"
OUT="$DATA/eval/$RUN"

mkdir -p "$OUT"
step() { echo; echo "== $*"; }

step "ranking checkpoints $STEPS"
# shellcheck disable=SC2086
"$PY" scripts/compare_checkpoints.py --run "$DATA/runs/$RUN" --annotations "$DATA/annotations-es" \
  --output "$COMPARE" --steps $STEPS --per-voice 6 --novel 4 --device cuda

BEST=$("$PY" -c "import json;print(json.load(open('$COMPARE/comparison.json'))['best']['step'])")
BEST_CKPT="$COMPARE/inference-$(printf '%08d' "$BEST").pt"
echo "best checkpoint: step $BEST"

step "evaluating step $BEST with 20 samples per voice per split"
"$PY" -m tiny_voca.evaluate --checkpoint "$BEST_CKPT" --annotations "$DATA/annotations-es" \
  --output "$OUT" --tag "$RUN-step$BEST" --device cuda --splits validation test --per-voice 20 --novel 10

step "exporting the winner"
# The pipeline exported the final checkpoint. If the comparison picked an
# earlier one, that export is the wrong model, so the winner is exported here
# and fetch_run.sh prefers these files.
cp "$BEST_CKPT" "$OUT/inference-best.pt"
"$PY" -m tiny_voca.export --checkpoint "$BEST_CKPT" --output "$OUT/tiny-voca-es-best.onnx"
"$PY" -m tiny_voca.export --checkpoint "$BEST_CKPT" --output "$OUT/tiny-voca-es-best-exact.onnx" --external-noise
"$PY" scripts/check_onnx.py --checkpoint "$BEST_CKPT" --onnx "$OUT/tiny-voca-es-best-exact.onnx" \
  --release "$OUT/tiny-voca-es-best.onnx" --output "$OUT/onnx-check" --threads 4

step "question intonation"
"$PY" scripts/probe_intonation.py --report "$OUT/report.json" --audio-dir "$OUT/audio" \
  --output "$OUT/intonation.json" --limit 40

step "punctuation-only intonation (same words, only the marks change)"
"$PY" scripts/probe_punctuation.py --checkpoint "$BEST_CKPT" --annotations "$DATA/annotations-es" \
  --output "$OUT/punctuation" --per-voice 12 --device cuda

step "teacher intonation, for the target"
if [ -f "$DATA/audio-es/manifest.jsonl" ]; then
  "$PY" scripts/probe_intonation.py --manifest "$DATA/audio-es/manifest.jsonl" \
    --audio-dir "$DATA/audio-es/audio" --output "$OUT/intonation-teacher.json" --limit 40
fi

if [[ -n "${MISTRAL_API_KEY:-}" ]]; then
  step "listening test (Voxtral)"
  "$PY" scripts/validate_voice.py --root "$OUT/audio" --report "$OUT/report.json" \
    --references "$DATA/voices-es/references" --limit 10 --output "$OUT/voice.json"
else
  echo
  echo "-- listening test skipped: set MISTRAL_API_KEY"
fi

echo
echo "done: $BEST_CKPT"
"$PY" - "$BEST_CKPT" <<'PY'
import json, sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(f"step {state['step']}, {sum(v.numel() for v in state['model'].values())} parameters")
PY
