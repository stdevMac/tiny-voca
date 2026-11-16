"""Check that the two voices are still two voices, by pitch rather than by label.

A listener model asked "is this male or female" answers from timbre, and it
answered "female" for a 90 Hz voice. Pitch is not a matter of opinion: the two
reference recordings sit about an octave and a half apart, and if the trained
model collapsed them onto one speaker the median F0 would show it immediately.

    python scripts/probe_voices.py --report data/eval/es-1/report.json \\
        --audio-dir data/eval/es-1/audio --references data/voices-es/references \\
        --output data/eval/es-1/voices.json

The useful number is the gap between the voices: it should match the gap
between the references, and each voice should stay near its own reference.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLE_RATE = 16000


def median_f0(path):
    import librosa
    wave, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if wave.size < int(0.3 * SAMPLE_RATE):
        return None
    f0, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=rate, frame_length=1024, hop_length=160)
    if f0 is None:
        return None
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    if voiced.sum() < 5:
        return None
    values = f0[voiced]
    return {"median_hz": round(float(np.median(values)), 1),
            "p10_hz": round(float(np.percentile(values, 10)), 1),
            "p90_hz": round(float(np.percentile(values, 90)), 1),
            "voiced_fraction": round(float(voiced.mean()), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        jobs = [(sample["voice"], Path(args.audio_dir) / sample["generated"]) for sample in report["samples"]]
    else:
        jobs = [(row.get("voice_id"), Path(args.audio_dir) / Path(row["audio"]).name)
                for row in (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip())]

    measured = {}
    for voice, path in jobs:
        if voice is None or not path.is_file():
            continue
        value = median_f0(path)
        if value is not None:
            measured.setdefault(voice, []).append(value["median_hz"])

    result = {"voices": {}}
    for voice, values in sorted(measured.items()):
        result["voices"][voice] = {"samples": len(values), "median_hz": round(float(np.median(values)), 1),
                                   "min_hz": round(float(np.min(values)), 1), "max_hz": round(float(np.max(values)), 1)}
    if args.references and args.references.is_dir():
        result["references"] = {}
        for path in sorted(args.references.glob("*.wav")):
            value = median_f0(path)
            if value is not None:
                result["references"][path.stem] = value["median_hz"]
    if len(result["voices"]) >= 2:
        low, high = sorted(result["voices"].items(), key=lambda item: item[1]["median_hz"])
        result["separation_hz"] = round(high[1]["median_hz"] - low[1]["median_hz"], 1)
        result["separation_semitones"] = round(12 * float(np.log2(high[1]["median_hz"] / low[1]["median_hz"])), 2)
        if len(result.get("references", {})) >= 2:
            ref_low, ref_high = sorted(result["references"].items(), key=lambda item: item[1])
            result["reference_separation_hz"] = round(ref_high[1] - ref_low[1], 1)
            result["reference_separation_semitones"] = round(12 * float(np.log2(ref_high[1] / ref_low[1])), 2)
            result["separation_ratio"] = round(result["separation_hz"] / max(result["reference_separation_hz"], 1e-9), 3)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    for voice, value in result["voices"].items():
        reference = result.get("references", {}).get(voice)
        detail = f", reference {reference:.1f} Hz" if reference else ""
        print(f"{voice}: median {value['median_hz']:.1f} Hz over {value['samples']} clips{detail}")
    if "separation_hz" in result:
        print(f"separation {result['separation_hz']:.1f} Hz ({result['separation_semitones']:.2f} semitones)"
              + (f", reference {result['reference_separation_hz']:.1f} Hz "
                 f"({result['separation_ratio']:.0%} of it)" if "reference_separation_hz" in result else ""))


if __name__ == "__main__":
    main()
