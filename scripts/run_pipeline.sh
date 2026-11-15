#!/usr/bin/env bash
# End-to-end Spanish run on a GPU host. Stops at the first failure.
#
#   ./scripts/run_pipeline.sh pilot          ~200 clips, gate on the alignment probe
#   ./scripts/run_pipeline.sh full           9,000 sentences, the real thing
#   ./scripts/run_pipeline.sh full --skip-audio   resume from the audio stage
#
# Every stage logs to logs/<stage>.log and prints one line. The budget deadline
# is written to data/budget.json and honoured by generate_audio.py and train.py,
# so a run cannot outlive its deadline even if this script is killed.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-pilot}
shift || true
DATA=${DATA:-data}
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-$(nproc)}
ASR_DEVICE=${ASR_DEVICE:-cuda}
DEADLINE_HOURS=${DEADLINE_HOURS:-30}
RUN=${RUN:-es-1}
SKIP=" $* "

mkdir -p logs "$DATA"

echo "mode=$MODE data=$DATA workers=$WORKERS asr=$ASR_DEVICE deadline=${DEADLINE_HOURS}h"
"$PY" - "$DATA" "$DEADLINE_HOURS" <<'PY'
import json, sys, time
from pathlib import Path
data, hours = Path(sys.argv[1]), float(sys.argv[2])
deadline = time.time() + hours * 3600
(data / "budget.json").write_text(json.dumps({"deadline": deadline}))
print(f"budget: stops at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))} local")
PY

stage() {
  local name=$1; shift
  if [[ "$SKIP" == *" --skip-$name "* ]]; then echo "-- $name skipped"; return 0; fi
  local start; start=$(date +%s)
  echo "== $name"
  if "$@" > >(tee "logs/$name.log") 2>&1; then
    echo "-- $name ok ($(( $(date +%s) - start ))s)"
  else
    echo "-- $name FAILED; see logs/$name.log" >&2
    exit 1
  fi
}

if [[ "$MODE" == "pilot" ]]; then
  COUNT=400; LIMIT=200; AUDIO="$DATA/pilot/audio"; FEATURES="$DATA/pilot/features"; ANN="$DATA/pilot/annotations"
  STEPS=0
else
  COUNT=9000; LIMIT=; AUDIO="$DATA/audio-es"; FEATURES="$DATA/features-es"; ANN="$DATA/annotations-es"
  STEPS=${STEPS:-104000}
fi

stage preflight   "$PY" scripts/preflight.py --data "$DATA"
stage scripts     "$PY" scripts/build_text_es.py \
                    --collector "$DATA/common-voice/sentence-collector.txt" \
                    --wiki "$DATA/common-voice/wiki.es.txt" \
                    --output "$DATA/scripts-$MODE.jsonl" --count "$COUNT" \
                    --validation $(( COUNT / 25 )) --test $(( COUNT / 30 ))
stage voices      "$PY" scripts/design_voices.py --root "$DATA/voices-es" --config voices/es-ES.json
stage audio       "$PY" scripts/generate_audio.py --scripts "$DATA/scripts-$MODE.jsonl" --root "$AUDIO" \
                    --references "$DATA/voices-es/references" --voices "$DATA/voices-es/voices.json" \
                    --asr-device "$ASR_DEVICE" --deadline-file "$DATA/budget.json" ${LIMIT:+--limit "$LIMIT"}
stage split       "$PY" scripts/split_manifest.py --root "$AUDIO" --force
run_prepare() {
  for split in train validation test; do
    "$PY" -m tiny_voca.prepare --inputs "$AUDIO/$split.jsonl" --split "$split" \
      --output "$FEATURES" --workers "$WORKERS"
  done
}
stage prepare     run_prepare
stage annotate    "$PY" -m tiny_voca.annotate --features "$FEATURES" --output "$ANN" \
                    --dataset-revision "$RUN" --device cuda
stage gate        "$PY" scripts/check_annotations.py --annotations "$ANN" --strict

if [[ "$MODE" == "pilot" ]]; then
  echo
  echo "pilot passed. Listen to $DATA/pilot/audio/ before the full run."
  exit 0
fi

stage train       "$PY" -m tiny_voca.train --train "$ANN/train.jsonl" --validation "$ANN/validation.jsonl" \
                    --output "$DATA/runs/$RUN" --device cuda --steps "$STEPS" \
                    --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --deadline-file "$DATA/budget.json"
stage evaluate    "$PY" -m tiny_voca.evaluate --checkpoint "$DATA/runs/$RUN/inference.pt" \
                    --annotations "$ANN" --output "$DATA/eval/$RUN" --tag "$RUN" --device cuda
stage export      "$PY" -m tiny_voca.export --checkpoint "$DATA/runs/$RUN/inference.pt" \
                    --output "$DATA/runs/$RUN/tiny-voca-es.onnx"

echo
echo "done: $DATA/runs/$RUN/inference.pt"
"$PY" -c "import torch; s=torch.load('$DATA/runs/$RUN/inference.pt', map_location='cpu', weights_only=True); print('steps', s['step'], 'params', sum(v.numel() for v in s['model'].values()))"
