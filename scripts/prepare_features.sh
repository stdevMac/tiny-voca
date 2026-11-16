#!/usr/bin/env bash
# Rebuild the acoustic features from the teacher FLAC on a fresh host.
#
# `fetch_run.sh` leaves the features out on purpose: they are six times the size
# of the audio and `prepare` rebuilds them in minutes on a many-core host. The
# annotations are NOT rebuilt, because that would mean paying the aligner again;
# they stay valid as long as the audio and the front end are unchanged, and the
# frame count of every feature file is checked against them below.
#
#   WORKERS=48 ./scripts/prepare_features.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-48}
FEATURES="$DATA/features-es"
ANN="$DATA/annotations-es"

for split in train validation test; do
  echo "== prepare $split"
  "$PY" -m tiny_voca.prepare --inputs "$DATA/audio-es/$split.jsonl" --split "$split" \
    --output "$FEATURES" --workers "$WORKERS" --resume
done

echo "== checking every annotation against its feature file"
"$PY" - "$FEATURES" "$ANN" <<'PY'
import json, sys
from pathlib import Path
import torch
features, annotations = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for split in ("train", "validation", "test"):
    rows = [json.loads(line) for line in (annotations / f"{split}.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    frames = 0
    for row in rows:
        path = features / row["path"]
        if not path.exists():
            print(f"missing {path}"); bad += 1; continue
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if stored["mel"].shape[0] != row["frames"] or stored["wave"].shape[0] != row["frames"] * 256:
            print(f"frame mismatch {path}: {stored['mel'].shape[0]} vs {row['frames']}"); bad += 1
        frames += row["frames"]
    print(f"{split:11} {len(rows):6} utterances  {frames:9} frames  {frames / 93.75 / 3600:6.2f} h")
if bad:
    raise SystemExit(f"{bad} annotation rows do not match their features")
print("features match the annotations")
PY
