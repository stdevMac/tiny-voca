"""Local stand-in teacher: macOS ``say`` instead of Qwen3-TTS. Laptop only.

This is a smoke-test generator, not the real dataset recipe. It exists so the
whole chain (prepare -> annotate -> train -> infer) can be exercised in Spanish
on a laptop, without a GPU, before paying for one. The voices are the system's
Spanish voices, so the audio is intelligible but not studio quality, and it is
NOT what the released model should be trained on.

    python scripts/generate_audio_local.py --scripts data/scripts-es.jsonl \\
        --root data/pilot --voices "Mónica|Grandpa (Spanish (Spain))" --limit 200

Output: <root>/audio/<id>_<voice>.wav (24 kHz mono 16-bit) and
<root>/manifest.jsonl in the JSONL form ``tiny_voca.prepare`` reads.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPEAKERS = ("F2", "M2")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--voices", required=True, help="Two macOS voice names, separated by |")
    parser.add_argument("--rate", type=int, default=180, help="Words per minute")
    parser.add_argument("--limit", type=int, help="Only the first N scripts (smoke test)")
    args = parser.parse_args()
    voices = [voice.strip() for voice in args.voices.split("|")]
    if len(voices) != len(SPEAKERS):
        parser.error(f"--voices needs exactly {len(SPEAKERS)} names separated by |")
    audio = args.root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.scripts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: (r["character_count"], r["id"]))[:args.limit]
    manifest = []
    for row in rows:
        for speaker, voice in zip(SPEAKERS, voices):
            identifier = f"{row['id']}_{speaker}"
            path = audio / f"{identifier}.wav"
            if not path.exists():
                subprocess.run(["say", "-v", voice, "-r", str(args.rate), "-o", str(path),
                                "--data-format=LEI16@24000", row["text"]], check=True)
            manifest.append({"id": identifier, "text_id": row["id"], "voice_id": speaker, "text": row["text"],
                             "split": row["split"], "category": row["category"], "origin": row["origin"],
                             "source_id": row["source_id"], "source_license": row["source_license"],
                             "source_url": row["source_url"], "teacher_model": f"macos-say:{voice}",
                             "reference_audio": None, "audio": f"audio/{identifier}.wav"})
    (args.root / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    print(json.dumps({"clips": len(manifest), "scripts": len(rows), "voices": voices}))


if __name__ == "__main__":
    main()
