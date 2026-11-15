"""Decide whether an annotation run is good enough to train on.

Reads ``summary.json`` and applies the thresholds from ``docs/training-es.md``.
The pilot uses this as a gate: if the teacher and the front end disagree, the
timing probe says so here, before 19 hours of audio are generated.

    python scripts/check_annotations.py --annotations data/annotations-es --strict
"""

import argparse
import json
from pathlib import Path
import sys

# Against the English corpus: 99.7% accepted, 81-86% vowel->voiceless and
# 86-89% fricative->vowel within two frames, 0.2% one-frame phones.
LIMITS = {
    "accepted_fraction": 0.95,
    "vowel_to_voiceless_within_2": 0.75,
    "fricative_to_vowel_within_2": 0.80,
    "one_frame_phone_fraction": 0.02,
    "mean_phone_frames": (5.0, 10.0),
}


def measure(summary):
    splits = summary["splits"]
    accepted = sum(s["accepted"] for s in splits.values())
    rejected = sum(s["rejected"] for s in splits.values())
    probe = summary.get("timing_probe", {}).get(summary.get("stage", "ctc"), {})
    values = {}
    for kind in ("vowel_to_voiceless", "fricative_to_vowel"):
        scores = [probe[voice][kind]["within_2_frames"] for voice in probe if kind in probe[voice]]
        values[kind] = sum(scores) / len(scores) if scores else None
    frames = [entry["mean"] for voice in summary.get("phone_duration_frames", {}).values() for entry in voice.values()]
    one_frame = [entry["one_frame"] for voice in summary.get("phone_duration_frames", {}).values() for entry in voice.values()]
    return {
        "accepted": accepted, "rejected": rejected,
        "accepted_fraction": accepted / max(accepted + rejected, 1),
        "vowel_to_voiceless_within_2": values["vowel_to_voiceless"],
        "fricative_to_vowel_within_2": values["fricative_to_vowel"],
        "mean_phone_frames": sum(frames) / len(frames) if frames else None,
        "one_frame_phone_fraction": sum(one_frame) / len(one_frame) if one_frame else None,
    }


def verdict(measured, strict=True):
    problems = []
    if measured["accepted_fraction"] < LIMITS["accepted_fraction"]:
        problems.append(f"only {measured['accepted_fraction']:.1%} of clips were accepted "
                        f"(need {LIMITS['accepted_fraction']:.0%}); the teacher and the front end disagree")
    for kind, limit in (("vowel_to_voiceless_within_2", LIMITS["vowel_to_voiceless_within_2"]),
                        ("fricative_to_vowel_within_2", LIMITS["fricative_to_vowel_within_2"])):
        value = measured[kind]
        if value is not None and value < limit:
            problems.append(f"{kind} is {value:.1%} (need {limit:.0%}); boundaries are misplaced")
    if measured["one_frame_phone_fraction"] is not None and measured["one_frame_phone_fraction"] > LIMITS["one_frame_phone_fraction"]:
        problems.append(f"{measured['one_frame_phone_fraction']:.1%} of phones last one frame "
                        f"(limit {LIMITS['one_frame_phone_fraction']:.0%}); the alignment collapsed")
    low, high = LIMITS["mean_phone_frames"]
    if measured["mean_phone_frames"] is not None and not low <= measured["mean_phone_frames"] <= high:
        problems.append(f"mean phone length is {measured['mean_phone_frames']:.1f} frames (expected {low:.0f}-{high:.0f})")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when a threshold is missed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.annotations / "summary.json").read_text())
    measured = measure(summary)
    problems = verdict(measured)
    if args.json:
        print(json.dumps({"measured": measured, "problems": problems}, indent=2))
    else:
        for key, value in measured.items():
            print(f"{key:34} {value if not isinstance(value, float) else round(value, 4)}")
        for problem in problems:
            print(f"PROBLEM  {problem}")
        if not problems:
            print("annotation looks trainable")
    if problems and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
