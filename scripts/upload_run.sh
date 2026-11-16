#!/usr/bin/env bash
# Push everything a run needs onto a GPU host, the mirror of fetch_run.sh.
#
# The features are not sent: `prepare` rebuilds them from the teacher FLAC in
# about ten minutes on a many-core host, and they are six times the size of the
# audio. The annotations are sent because re-deriving them means paying the
# aligner again, and they are valid as long as the audio and the front end are
# the same.
#
#   ./scripts/upload_run.sh --host root@1.2.3.4 --port 47453 --key ~/.ssh/k \
#       --artifacts ~/tiny-voca-artifacts --run es-1 --base es-1-anneal
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${HOST:-}
PORT=${PORT:-22}
KEY=${KEY:-}
ARTIFACTS=${ARTIFACTS:-$HOME/tiny-voca-artifacts}
RUN=${RUN:-es-1}
BASE=${BASE:-es-1-anneal}
REMOTE=${REMOTE:-/workspace/tiny-voca}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --key) KEY=$2; shift 2 ;;
    --artifacts) ARTIFACTS=$2; shift 2 ;;
    --run) RUN=$2; shift 2 ;;
    --base) BASE=$2; shift 2 ;;
    --remote) REMOTE=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$HOST" ]] || { echo "--host is required" >&2; exit 2; }
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no)
RSYNC=(rsync -a -e "ssh -p $PORT -o StrictHostKeyChecking=no")
[[ -n "$KEY" ]] && { SSH+=(-i "$KEY"); RSYNC=(-e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no"); }

echo "== code"
"${SSH[@]}" "$HOST" "mkdir -p $REMOTE/data"
"${RSYNC[@]}" --exclude '.venv' --exclude '.git' --exclude 'data' --exclude 'weights' --exclude 'app' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.linode' ./ "$HOST:$REMOTE/"

echo "== corpus and annotations"
"${RSYNC[@]}" "$ARTIFACTS/$RUN/audio-es/" "$HOST:$REMOTE/data/audio-es/"
"${RSYNC[@]}" "$ARTIFACTS/$RUN/annotations-es/" "$HOST:$REMOTE/data/annotations-es/"
"${RSYNC[@]}" "$ARTIFACTS/$RUN/scripts-full.jsonl" "$HOST:$REMOTE/data/"

echo "== the checkpoint to branch from"
"${SSH[@]}" "$HOST" "mkdir -p $REMOTE/data/runs/$BASE"
"${RSYNC[@]}" "$ARTIFACTS/$BASE/runs/$BASE/last.pt" "$ARTIFACTS/$BASE/runs/$BASE/inference.pt" "$HOST:$REMOTE/data/runs/$BASE/"

echo "== budget file, so train.py does not refuse to start"
"${SSH[@]}" "$HOST" "printf '{\"deadline\": %s}\n' \$(( \$(date +%s) + 86400 )) > $REMOTE/data/budget.json"

echo "uploaded to $HOST:$REMOTE"
