#!/usr/bin/env bash
# Start the local tiny-voca web interface.
#
#   ./run.sh
#
# The first run creates app/.venv, installs the Python packages and downloads
# the model weights from Hugging Face. Later runs start in a second or two.
set -euo pipefail
cd "$(dirname "$0")"

VENV="app/.venv"
PYTHON="$VENV/bin/python"

install_packages() {
  echo "Installing the Python packages…"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" -r app/requirements.txt
  else
    "$PYTHON" -m pip install -r app/requirements.txt
  fi
}

if [ ! -x "$PYTHON" ]; then
  echo "Creating a Python environment…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    python3 -m venv "$VENV"
    "$PYTHON" -m pip install --upgrade pip
  fi
  install_packages
fi

if ! "$PYTHON" -c "import fastapi, uvicorn, dizione, soundfile, onnxruntime" >/dev/null 2>&1; then
  install_packages
fi

if [ ! -f weights/tiny-voca.pt ] || [ ! -f weights/phonemizer/lexicon.tsv ]; then
  echo "Downloading the model weights (about 40 MB)…"
  "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("n4ze3m/tiny-voca", local_dir="weights")
PY
fi

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
echo "  tiny-voca is starting at $URL"
if [ "$HOST" = "0.0.0.0" ]; then
  echo "  Reachable from other devices at http://<this-machine-ip>:$PORT"
fi
echo "  Press Ctrl+C to stop."
echo

exec "$PYTHON" -m uvicorn app.server:app --host "$HOST" --port "$PORT"
