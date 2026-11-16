#!/usr/bin/env bash
# Start the local tiny-voca web interface with the Spanish model.
#
#   ./run-es.sh
#
# Same page as ./run.sh, same frontend files, pointed at the Spanish checkpoint
# in ~/tiny-voca-artifacts. Nothing is downloaded: the weights are the ones this
# run produced, and the server refuses to start if this machine's eSpeak does not
# match the front end the model was trained on.
#
#   TINY_VOCA_ES_DIR=~/tiny-voca-artifacts/es-1 ./run-es.sh   # the baseline instead
#   TINY_VOCA_PORT=8080 ./run-es.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PYTHON="$VENV/bin/python"
DEFAULT_DIR="$HOME/tiny-voca-artifacts/es-1-anneal"
RUN_DIR="${TINY_VOCA_ES_DIR:-$DEFAULT_DIR}"
EVAL_DIR="$RUN_DIR/eval/anneal-es-1-anneal"
MODEL="${TINY_VOCA_ES_MODEL:-$EVAL_DIR/tiny-voca-es-anneal.onnx}"
CONFIG="${TINY_VOCA_ES_CONFIG:-$EVAL_DIR/tiny-voca-es-anneal.json}"

if [ ! -x "$PYTHON" ]; then
  echo "No $VENV. Run 'make venv' first, or use the training environment." >&2
  exit 1
fi

if ! "$PYTHON" -c "import fastapi, uvicorn, onnxruntime, soundfile" >/dev/null 2>&1; then
  echo "Missing web packages. Install them with:" >&2
  echo "  $PYTHON -m pip install fastapi 'uvicorn[standard]' onnxruntime soundfile" >&2
  exit 1
fi

if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "espeak-ng is not on PATH; the Spanish front end needs it (version 1.52)." >&2
  exit 1
fi

for path in "$MODEL" "$CONFIG"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    echo "Set TINY_VOCA_ES_DIR to the artifacts directory, or TINY_VOCA_ES_MODEL and TINY_VOCA_ES_CONFIG." >&2
    exit 1
  fi
done

export TINY_VOCA_LANGUAGE=es
export TINY_VOCA_ES_MODEL="$MODEL"
export TINY_VOCA_ES_CONFIG="$CONFIG"

HOST="${TINY_VOCA_HOST:-127.0.0.1}"
PORT="${TINY_VOCA_PORT:-8000}"
BROWSER_HOST="$HOST"
if [ "$HOST" = "0.0.0.0" ]; then
  BROWSER_HOST="127.0.0.1"
fi
URL="http://$BROWSER_HOST:$PORT"

open_browser() {
  if command -v open >/dev/null 2>&1; then
    open "$1" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 || true
  fi
}

if [ "${TINY_VOCA_NO_OPEN:-0}" != "1" ]; then
  (
    for _ in $(seq 1 120); do
      if curl -fsS "http://$BROWSER_HOST:$PORT/health" >/dev/null 2>&1; then
        open_browser "$URL"
        break
      fi
      sleep 0.5
    done
  ) &
fi

echo
echo "  tiny-voca (español) is starting at $URL"
echo "  Model: $MODEL"
echo "  Press Ctrl+C to stop."
echo

exec "$PYTHON" -m uvicorn app_es.server:app --host "$HOST" --port "$PORT"
