#!/usr/bin/env bash
# Start the local tiny-voca web interface with the Spanish model.
#
#   ./run-es.sh
#
# The first run creates app_es/.venv, installs the Python packages (no torch:
# the app runs the ONNX export) and downloads the model, about 21 MB, from
# https://huggingface.co/marcosmaceo/tiny-voca-es into weights/es/. Later runs
# start in a second or two.
#
# The Spanish front end needs eSpeak NG 1.52.0 on PATH, the version the model
# was trained with:
#   macOS:  brew install espeak-ng
#   Linux:  scripts/host_setup.sh builds it from source (distributions often ship 1.51)
#
#   TINY_VOCA_PORT=8080 ./run-es.sh
#   TINY_VOCA_ES_MODEL=path.onnx TINY_VOCA_ES_CONFIG=path.json ./run-es.sh   # another export
set -euo pipefail
cd "$(dirname "$0")"

VENV="app_es/.venv"
PYTHON="$VENV/bin/python"
REPO="marcosmaceo/tiny-voca-es"
MODEL="${TINY_VOCA_ES_MODEL:-weights/es/tiny-voca-es.onnx}"
CONFIG="${TINY_VOCA_ES_CONFIG:-weights/es/tiny-voca-es.json}"

if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "espeak-ng is not on PATH. The Spanish front end needs eSpeak NG 1.52.0:" >&2
  echo "  macOS: brew install espeak-ng    Linux: scripts/host_setup.sh builds it" >&2
  exit 1
fi
ESPEAK_VERSION="$(espeak-ng --version | sed -n 's/.*text-to-speech: \([0-9.]*\).*/\1/p')"
if [ "$ESPEAK_VERSION" != "1.52.0" ]; then
  echo "Warning: eSpeak NG ${ESPEAK_VERSION:-of unknown version} found; the model was trained with 1.52.0." >&2
  echo "         Other versions spell some words differently, and the model will mispronounce them." >&2
fi

install_packages() {
  echo "Installing the Python packages…"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$PYTHON" -r app_es/requirements.txt
  else
    "$PYTHON" -m pip install --quiet -r app_es/requirements.txt
  fi
}

if [ ! -x "$PYTHON" ]; then
  echo "Creating a Python environment…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --quiet --python 3.12 "$VENV"
  else
    python3 -m venv "$VENV"
    "$PYTHON" -m pip install --quiet --upgrade pip
  fi
  install_packages
fi

if ! "$PYTHON" -c "import fastapi, uvicorn, onnxruntime, soundfile, huggingface_hub" >/dev/null 2>&1; then
  install_packages
fi

if [ ! -f "$MODEL" ] || [ ! -f "$CONFIG" ]; then
  if [ -n "${TINY_VOCA_ES_MODEL:-}${TINY_VOCA_ES_CONFIG:-}" ]; then
    echo "Missing $MODEL or $CONFIG" >&2
    exit 1
  fi
  echo "Downloading the model from https://huggingface.co/$REPO (about 21 MB)…"
  "$PYTHON" - "$REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], local_dir="weights/es", allow_patterns=["tiny-voca-es.onnx", "tiny-voca-es.json"])
PY
fi

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
