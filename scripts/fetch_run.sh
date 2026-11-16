#!/usr/bin/env bash
# Pull everything worth keeping off the GPU host before it is deleted.
#
# The expensive things are not the weights: they are the corpus, the teacher's
# audio and the annotations. Losing those means paying for the teacher again.
# This copies the run, the evaluation, the voice references and the corpus
# metadata, then writes a manifest of checksums so a later round can prove the
# local copy is the one that produced the weights.
#
#   ./scripts/fetch_run.sh --host root@203.57.40.129 --port 10151 \
#       --key ~/.ssh/tiny-voca-es-run --run es-1 --output ~/tiny-voca-es-1
#
# Add --audio to also copy the 1.6 GB of teacher FLAC, or --features/--annotations
# to copy the derived data (large; only needed to retrain without the teacher).
set -euo pipefail

HOST=${HOST:-}
PORT=${PORT:-22}
KEY=${KEY:-}
RUN=${RUN:-es-1}
OUTPUT=${OUTPUT:-}
REMOTE=${REMOTE:-/workspace/tiny-voca}
WANT_AUDIO=0
WANT_FEATURES=0
WANT_ANNOTATIONS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --key) KEY=$2; shift 2 ;;
    --run) RUN=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --remote) REMOTE=$2; shift 2 ;;
    --audio) WANT_AUDIO=1; shift ;;
    --features) WANT_FEATURES=1; shift ;;
    --annotations) WANT_ANNOTATIONS=1; shift ;;
    --all) WANT_AUDIO=1; WANT_FEATURES=1; WANT_ANNOTATIONS=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$HOST" || -z "$OUTPUT" ]]; then
  echo "usage: $0 --host user@host --port PORT --key KEY --output DIR [--all]" >&2
  exit 2
fi

SSH=(ssh -o StrictHostKeyChecking=no -p "$PORT")
[[ -n "$KEY" ]] && SSH+=(-i "$KEY")
SCP=(scp -o StrictHostKeyChecking=no -P "$PORT")
[[ -n "$KEY" ]] && SCP+=(-i "$KEY")

mkdir -p "$OUTPUT"
copy() {
  local remote=$1 local=$2
  echo "  $remote"
  mkdir -p "$(dirname "$OUTPUT/$local")"
  "${SCP[@]}" -q -r "$HOST:$REMOTE/$remote" "$OUTPUT/$local"
}

echo "== model and run"
copy "data/runs/$RUN/inference.pt" "runs/$RUN/inference.pt"
# Numbered checkpoints are large and only useful for comparison; the best one is
# already exported to inference.pt.
"${SSH[@]}" "$HOST" "cd $REMOTE && ls data/runs/$RUN/tiny-voca-es.onnx 2>/dev/null" >/dev/null && \
  copy "data/runs/$RUN/tiny-voca-es.onnx" "runs/$RUN/tiny-voca-es.onnx"
for name in config.json recipe.json metrics.jsonl; do
  "${SSH[@]}" "$HOST" "cd $REMOTE && test -f data/runs/$RUN/$name" 2>/dev/null && copy "data/runs/$RUN/$name" "runs/$RUN/$name"
done

echo "== evaluation"
copy "data/eval/$RUN" "eval/$RUN"
if "${SSH[@]}" "$HOST" "cd $REMOTE && test -d data/eval/compare-$RUN" 2>/dev/null; then
  copy "data/eval/compare-$RUN" "eval/compare-$RUN"
fi

echo "== voices and corpus text"
copy "data/voices-es" "voices-es"
copy "data/scripts-full.jsonl" "scripts-full.jsonl"

echo "== teacher audio metadata"
copy "data/audio-es/manifest.jsonl" "audio-es/manifest.jsonl"
copy "data/audio-es/review.jsonl" "audio-es/review.jsonl"
copy "data/audio-es/progress.json" "audio-es/progress.json"

if [[ "$WANT_AUDIO" == 1 ]]; then
  echo "== teacher audio (large)"
  copy "data/audio-es/audio" "audio-es/audio"
fi
if [[ "$WANT_FEATURES" == 1 ]]; then
  echo "== features (large)"
  copy "data/features-es" "features-es"
fi
if [[ "$WANT_ANNOTATIONS" == 1 ]]; then
  echo "== annotations (large)"
  copy "data/annotations-es" "annotations-es"
fi

echo "== checksums"
( cd "$OUTPUT" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 shasum -a 256 | sort -k2 > SHA256SUMS )
echo "  $(wc -l < "$OUTPUT/SHA256SUMS" | tr -d ' ') files, $(du -sh "$OUTPUT" | cut -f1)"

# The weights are only useful with the front end that made them.
echo "== verifying the checkpoint loads with this code"
.venv/bin/python - "$OUTPUT/runs/$RUN/inference.pt" <<'PY'
import sys, torch
sys.path.insert(0, "src")
from tiny_voca.text import frontend_fingerprint
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if state.get("frontend") != frontend_fingerprint():
    raise SystemExit("front end does not match: the checkpoint was made with different symbols")
print(f"  step {state['step']}, {sum(v.numel() for v in state['model'].values())} parameters, front end matches")
PY
