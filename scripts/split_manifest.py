"""Split a manifest into one JSONL per split, which is what ``tiny_voca.prepare`` reads.

``prepare`` takes one ``--split`` per run and refuses rows that belong to
another one, so a single manifest with a ``split`` column has to be cut up
first. Rows without a split are an error: guessing would silently move a
sentence between train and test.

    python scripts/split_manifest.py --root data/audio-es
"""

import argparse
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory holding manifest.jsonl")
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-split files")
    args = parser.parse_args()
    source = args.root / args.manifest
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [row.get("id") for row in rows if row.get("split") not in SPLITS]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no valid split (first: {missing[:3]})")
    counts = {}
    for split in SPLITS:
        chosen = [row for row in rows if row["split"] == split]
        path = args.root / f"{split}.jsonl"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; pass --force to overwrite")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen), encoding="utf-8")
        counts[split] = len(chosen)
    print(json.dumps({"rows": len(rows), "splits": counts}))


if __name__ == "__main__":
    main()
