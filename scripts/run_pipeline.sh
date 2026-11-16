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
BATCH=${BATCH:-32}
BATCH_CHARS=${BATCH_CHARS:-2500}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DEADLINE_HOURS=${DEADLINE_HOURS:-30}
RUN=${RUN:-es-1}
TRAIN_SECONDS=${TRAIN_SECONDS:-7200}
SKIP=" $* "

mkdir -p logs "$DATA"
# cuDNN for CTranslate2, when host_setup.sh managed to install it.
# Sourced with `set +u`: the file appends to LD_LIBRARY_PATH, which may be unset.
if [ -f .venv/nvidia.env ]; then set +u; . .venv/nvidia.env; set -u; fi

echo "mode=$MODE data=$DATA workers=$WORKERS asr=$ASR_DEVICE batch=$BATCH/$BATCH_CHARS deadline=${DEADLINE_HOURS}h"
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
                    --asr-device "$ASR_DEVICE" --batch "$BATCH" --batch-chars "$BATCH_CHARS" --deadline-file "$DATA/budget.json" ${LIMIT:+--limit "$LIMIT"}
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

# train.py saves and exits at --max-seconds (default one hour), which is right for
# a cloud box that might vanish but wrong here: without this loop the pipeline
# would evaluate a one-hour checkpoint and call it done. Resume until --steps.
run_train() {
  local out="$DATA/runs/$RUN" resume="" step
  for attempt in 1 2 3 4 5 6 7 8; do
    "$PY" -m tiny_voca.train --train "$ANN/train.jsonl" --validation "$ANN/validation.jsonl" \
      --output "$out" --device cuda --steps "$STEPS" --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
      --max-seconds "$TRAIN_SECONDS" --deadline-file "$DATA/budget.json" $resume || return 1
    step=$("$PY" -c "import torch;print(torch.load('$out/last.pt',map_location='cpu',weights_only=True)['step'])" 2>/dev/null || echo 0)
    echo "  training at step $step of $STEPS"
    [ "$step" -ge "$STEPS" ] && return 0
    resume="--resume $out/last.pt"
  done
  return 1
}
stage train       run_train
stage evaluate    "$PY" -m tiny_voca.evaluate --checkpoint "$DATA/runs/$RUN/inference.pt" \
                    --annotations "$ANN" --output "$DATA/eval/$RUN" --tag "$RUN" --device cuda
stage export      "$PY" -m tiny_voca.export --checkpoint "$DATA/runs/$RUN/inference.pt" \
                    --output "$DATA/runs/$RUN/tiny-voca-es.onnx"
if [[ -n "${MISTRAL_API_KEY:-}" ]]; then
  stage listen    "$PY" scripts/validate_voice.py --root "$DATA/eval/$RUN/audio" \
                    --report "$DATA/eval/$RUN/report.json" --references "$DATA/voices-es/references" \
                    --limit 8 --output "$DATA/eval/$RUN/voice.json"
else
  echo "-- listen skipped (no MISTRAL_API_KEY)"
fi

echo
echo "done: $DATA/runs/$RUN/inference.pt"
"$PY" -c "import torch; s=torch.load('$DATA/runs/$RUN/inference.pt', map_location='cpu', weights_only=True); print('steps', s['step'], 'params', sum(v.numel() for v in s['model'].values()))"
