#!/usr/bin/env bash
# Provision an Ubuntu 24.04 host for the Spanish run. Idempotent.
#
#   ./scripts/host_setup.sh
#
# Two things here are not obvious and both were measured:
#
# * eSpeak 1.52.0 is built from source. Ubuntu ships 1.51, whose Spanish
#   phonemes differ from the 1.52 build used on macOS: 1.51 writes "informe" as
#   i m f o ɾ m e and 1.52 as i n f o ɾ m e. Training on one and speaking with
#   the other would mispronounce a whole class of words. The probe hash is
#   identical once both are 1.52 (scripts/espeak_probe.py).
# * cuDNN is optional. faster-whisper can screen on the GPU, but CTranslate2
#   needs cuDNN 9 and the pip wheel only works with the right LD_LIBRARY_PATH.
#   Screening overlaps generation on a background thread, so the CPU path is
#   not on the critical path and is the default.
set -euo pipefail
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12-venv python3-pip git build-essential autoconf automake libtool \
  pkg-config ffmpeg libsndfile1 jq curl ca-certificates

if ! /usr/local/bin/espeak-ng --version 2>/dev/null | grep -q "1.52.0"; then
  echo "building espeak-ng 1.52.0"
  rm -rf /tmp/espeak-ng
  git clone -q --depth 1 --branch 1.52.0 https://github.com/espeak-ng/espeak-ng.git /tmp/espeak-ng
  ( cd /tmp/espeak-ng && ./autogen.sh >/dev/null 2>&1 && ./configure --prefix=/usr/local >/dev/null 2>&1 \
    && make -j"$(nproc)" >/dev/null 2>&1 && make install >/dev/null 2>&1 )
  ldconfig
fi
/usr/local/bin/espeak-ng --version | head -1

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[prepare,annotate,evaluate,export]"
uv pip install --python .venv/bin/python -U qwen-tts faster-whisper

# cuDNN 9 for CTranslate2. Harmless if it does not work: preflight warns and
# screening stays on the CPU.
uv pip install --python .venv/bin/python nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 >/dev/null 2>&1 || true
LIBS=$(ls -d "$PWD"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':' || true)
if [ -n "$LIBS" ]; then
  printf 'export LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:-}\n' "$LIBS" > .venv/nvidia.env
fi

mkdir -p data logs
echo "host ready: $(/usr/local/bin/espeak-ng --version | head -1)"
