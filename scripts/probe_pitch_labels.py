#!/usr/bin/env python3
"""Where the pitch tracker lost the voice in the teacher recordings.

The oscillator of the waveform head is driven, in training, by the pitch that
`tiny_voca.prepare` measured on each recording (`librosa.pyin`, 43 ms frames,
50 Hz floor). Where the tracker found no pitch the curve is interpolated from
the neighbours (`tiny_voca.data.continuous_pitch`), and a vowel is still marked
voiced. The head is then asked to match a voiced recording with harmonics at a
pitch nobody measured.

A 43 ms frame holds ten periods of F2 and fewer than four of M2, so the tracker
has least to go on where a low voice starts and stops. This probe asks, on the
teacher recordings and with the annotation's own word boundaries:

* how many vowel frames have no tracked pitch, at the edges of words and inside
* whether those frames are in fact periodic (the tracker failed) or not (creak)
* how far the interpolated pitch the model was trained with is from the pitch
  that is there

Periodicity is measured independently of the tracker: the normalised
cross-correlation between one pitch period and the next (as in RAPT), which
needs two periods of signal rather than a 43 ms frame.

Compare each voice with itself (edge against interior). The voices differ in
pitch, and so in how much signal every one of these measures gets.

    TINY_VOCA_LANGUAGE=es ./scripts/probe_pitch_labels.py --annotations ~/tiny-voca-artifacts/es-1/annotations-es \
        --audio ~/tiny-voca-artifacts/es-1/audio-es/audio --per-voice 120 --output ~/tiny-voca-artifacts/pitch-labels/labels.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EDGE_FRAMES = 4          # 43 ms at 93.75 frames/s
PERIODIC = 0.8           # cross-correlation of adjacent periods above which a frame is called periodic
PLACES = ("initial", "final", "interior")
SEMITONE_LIMITS = (0.5, 1.0, 2.0)


def word_layout(tokens, durations, phones):
    """Per frame: owner token index, frames since its word began, frames until
    the word ends, and whether silence touches the word's start / end. A word
    is a run of phoneme tokens; frames outside words get -1."""
    total = int(sum(durations))
    owner = np.repeat(np.arange(len(tokens)), durations)
    since, until = np.full(total, -1), np.full(total, -1)
    pause_before, pause_after = np.zeros(total, bool), np.zeros(total, bool)
    starts = np.concatenate(([0], np.cumsum(durations)))
    index = 0
    while index < len(tokens):
        if tokens[index] not in phones:
            index += 1
            continue
        last = index
        while last + 1 < len(tokens) and tokens[last + 1] in phones:
            last += 1
        first_frame, end_frame = int(starts[index]), int(starts[last + 1])
        # Silence next to the word: frames owned by the boundary tokens around it.
        before = index
        while before > 0 and tokens[before - 1] not in phones:
            before -= 1
        after = last + 1
        while after < len(tokens) and tokens[after] not in phones:
            after += 1
        frames = np.arange(first_frame, end_frame)
        since[frames], until[frames] = frames - first_frame, end_frame - 1 - frames
        pause_before[frames] = starts[index] - starts[before] > 0 or before == 0
        pause_after[frames] = starts[after] - starts[last + 1] > 0 or after == len(tokens)
        index = last + 1
    return owner, since, until, pause_before, pause_after


def adjacent_periods(wave, rate, hop, frames, low, high, window):
    """Per frame: the best normalised cross-correlation between a window and the
    same window one lag later, over lags from rate/high to rate/low, and the
    pitch of that lag. The pair of windows is centred on the frame."""
    lags = np.arange(int(rate / high), int(np.ceil(rate / low)) + 1)
    reach = (window + int(lags[-1])) // 2 + 1
    padded = np.pad(wave.astype(np.float64), (reach, reach + window + int(lags[-1])))
    energy = np.concatenate(([0.0], np.cumsum(padded ** 2)))
    centres = np.arange(frames) * hop + reach
    index = np.arange(window)
    scores = np.empty((len(lags), frames))
    for i, lag in enumerate(lags):
        start = centres - (window + int(lag)) // 2
        first, second = padded[start[:, None] + index], padded[start[:, None] + int(lag) + index]
        power = (energy[start + window] - energy[start]) * (energy[start + lag + window] - energy[start + lag])
        scores[i] = (first * second).sum(1) / np.sqrt(power + 1e-12)
    best = scores.argmax(0)
    peak = scores[best, np.arange(frames)]
    # Parabola through the peak and its neighbours: the lag to a fraction of a sample.
    inner = np.clip(best, 1, len(lags) - 2)
    left, middle, right = (scores[inner + offset, np.arange(frames)] for offset in (-1, 0, 1))
    curvature = left - 2 * middle + right
    shift = np.where((np.abs(curvature) > 1e-9) & (best == inner), 0.5 * (left - right) / np.where(np.abs(curvature) > 1e-9, curvature, 1.0), 0.0)
    return peak, rate / (lags[best] + np.clip(shift, -1, 1))


def examine(task):
    row, audio, edge = task
    import librosa
    import soundfile as sf
    import torch
    from tiny_voca.config import Config
    from tiny_voca.data import VOWEL_IDS, continuous_pitch
    from tiny_voca.prepare import extract
    from tiny_voca.text import PHONE_IDS

    torch.set_num_threads(1)
    config = Config()
    wave, rate = sf.read(audio / f"{row['id']}.flac", dtype="float32")
    sample = extract(wave, config)
    frames = row["frames"]
    if rate != config.sample_rate or sample["prosody"].shape[0] != frames:
        raise ValueError(f"{row['id']}: audio does not match the annotation")
    tracked = sample["prosody"][:, 1].numpy() > 0.5
    curve, trusted = continuous_pitch(sample["prosody"][:, 0], sample["prosody"][:, 1])
    trained = np.clip(config.pitch_reference_hz * np.exp(curve.numpy()), config.f0_min, config.f0_max)
    trusted = trusted.numpy() > 0.5
    if trusted.sum() < 10:
        return None
    typical = float(np.exp(np.median(np.log(trained[trusted]))))

    owner, since, until, pause_before, pause_after = word_layout(row["tokens"], row["durations"], PHONE_IDS)
    vowel = np.isin(np.asarray(row["tokens"])[owner], list(VOWEL_IDS))
    peak, measured = adjacent_periods(sample["wave"].numpy(), rate, config.hop, frames, typical / 1.8, typical * 1.8,
                                      int(round(rate / typical)))
    # The same tracker with twice the frame and a lower floor, for comparison.
    longer, flags, _ = librosa.pyin(sample["wave"].numpy(), fmin=40.0, fmax=config.f0_max, sr=rate, frame_length=2048,
                                    hop_length=config.hop, center=True, pad_mode="constant")
    longer_tracked = np.asarray(flags[:frames], bool)

    place = np.where(until < edge, 1, np.where(since < edge, 0, 2))   # final wins a tie in a short word
    error = np.abs(12 * np.log2(measured / trained))
    periodic = peak >= PERIODIC
    result = {"id": row["id"], "voice": row["speaker"], "typical_hz": typical, "frames": int(frames), "vowel_frames": int(vowel.sum())}
    groups = {name: vowel & (place == number) for number, name in enumerate(PLACES)}
    groups.update({"initial_after_pause": groups["initial"] & pause_before, "initial_inside_phrase": groups["initial"] & ~pause_before,
                   "final_before_pause": groups["final"] & pause_after, "final_inside_phrase": groups["final"] & ~pause_after,
                   "all": vowel})
    for name, member in groups.items():
        lost = member & ~tracked
        found = member & tracked & trusted
        entry = {"frames": int(member.sum()), "untracked": int(lost.sum()), "untrusted": int((member & ~trusted).sum()),
                 "untracked_periodic": int((lost & periodic).sum()), "tracked_periodic": int((found & periodic).sum()),
                 "tracked": int(found.sum()), "untracked_found_by_longer_frame": int((lost & longer_tracked).sum()),
                 "untracked_error": error[lost & periodic].round(3).tolist(), "tracked_error": error[found & periodic].round(3).tolist(),
                 "untracked_peak": peak[lost].round(3).tolist(), "tracked_peak_mean": float(peak[found].mean()) if found.any() else None}
        result[name] = entry
    return result


def share(part, whole):
    return part / whole if whole else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True, help="directory of <id>.flac teacher recordings")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--per-voice", type=int, default=120)
    parser.add_argument("--edge-frames", type=int, default=EDGE_FRAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from scipy import stats
    from tiny_voca.text import frontend_fingerprint
    rows = []
    for split in args.splits:
        rows += [json.loads(line) for line in (args.annotations / f"{split}.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    if any(row["frontend"] != frontend_fingerprint() for row in rows):
        parser.error("the annotations were made with another front end; set TINY_VOCA_LANGUAGE to match")
    chosen = []
    for voice in sorted({row["speaker"] for row in rows}):
        mine = sorted((row for row in rows if row["speaker"] == voice), key=lambda row: row["id"])
        random.Random(args.seed).shuffle(mine)
        chosen += mine[:args.per_voice]
    with ProcessPoolExecutor(args.workers) as pool:
        results = [r for r in pool.map(examine, [(row, args.audio, args.edge_frames) for row in chosen], chunksize=2) if r]

    summary = {}
    for voice in sorted({r["voice"] for r in results}):
        mine = [r for r in results if r["voice"] == voice]
        print(f"\n{voice}: {len(mine)} recordings, typical pitch {np.exp(np.mean([np.log(r['typical_hz']) for r in mine])):.0f} Hz, "
              f"{sum(r['vowel_frames'] for r in mine)} vowel frames")
        print(f"  {'vowel frames':24}{'frames':>8}{'no pitch':>10}{'untrusted':>11}{'periodic*':>11}{'longer':>8}"
              f"{'median err':>12}{'>0.5 st':>9}{'>1 st':>7}{'>2 st':>7}   control: tracked frames {'periodic':>9}{'median err':>12}{'>0.5 st':>9}")
        summary[voice] = {"recordings": len(mine)}
        for name in ("all", *PLACES, "initial_after_pause", "initial_inside_phrase", "final_before_pause", "final_inside_phrase"):
            total = {key: sum(r[name][key] for r in mine) for key in ("frames", "untracked", "untrusted", "untracked_periodic",
                                                                       "tracked", "tracked_periodic", "untracked_found_by_longer_frame")}
            lost_error = np.concatenate([r[name]["untracked_error"] for r in mine] or [[]])
            kept_error = np.concatenate([r[name]["tracked_error"] for r in mine] or [[]])
            entry = {**total, "untracked_share": share(total["untracked"], total["frames"]),
                     "untrusted_share": share(total["untrusted"], total["frames"]),
                     "untracked_periodic_share": share(total["untracked_periodic"], total["untracked"]),
                     "untracked_found_by_longer_frame_share": share(total["untracked_found_by_longer_frame"], total["untracked"]),
                     "tracked_periodic_share": share(total["tracked_periodic"], total["tracked"]),
                     "untracked_error_median": float(np.median(lost_error)) if lost_error.size else None,
                     "tracked_error_median": float(np.median(kept_error)) if kept_error.size else None,
                     **{f"untracked_error_over_{limit}": float(np.mean(lost_error > limit)) if lost_error.size else None for limit in SEMITONE_LIMITS},
                     **{f"tracked_error_over_{limit}": float(np.mean(kept_error > limit)) if kept_error.size else None for limit in SEMITONE_LIMITS}}
            summary[voice][name] = entry
            if lost_error.size and kept_error.size:
                print(f"  {name:24}{total['frames']:8d}{entry['untracked_share']:10.1%}{entry['untrusted_share']:11.1%}"
                      f"{entry['untracked_periodic_share']:11.1%}{entry['untracked_found_by_longer_frame_share']:8.1%}"
                      f"{entry['untracked_error_median']:12.2f}{entry['untracked_error_over_0.5']:9.1%}{entry['untracked_error_over_1.0']:7.1%}"
                      f"{entry['untracked_error_over_2.0']:7.1%}{'':27}{entry['tracked_periodic_share']:9.1%}"
                      f"{entry['tracked_error_median']:12.2f}{entry['tracked_error_over_0.5']:9.1%}")
            else:
                print(f"  {name:24}{total['frames']:8d}{entry['untracked_share']:10.1%}{entry['untrusted_share']:11.1%}")
        # Paired within recordings: is a word's edge lost more often than its inside?
        summary[voice]["paired"] = {}
        for name in ("initial", "final", "initial_after_pause", "final_before_pause", "initial_inside_phrase", "final_inside_phrase"):
            pairs = [(share(r[name]["untracked"], r[name]["frames"]), share(r["interior"]["untracked"], r["interior"]["frames"]))
                     for r in mine if r[name]["frames"] >= 5 and r["interior"]["frames"] >= 5]
            if len(pairs) < 3:
                continue
            edge, inside = np.array(pairs).T
            difference = edge - inside
            half = stats.t.ppf(0.975, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
            test = {"n": len(difference), "edge": float(edge.mean()), "interior": float(inside.mean()), "difference": float(difference.mean()),
                    "ci95": [float(difference.mean() - half), float(difference.mean() + half)],
                    "wilcoxon_p": float(stats.wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0,
                    "share_edge_worse": float(np.mean(difference > 0))}
            summary[voice]["paired"][name] = test
            print(f"  paired, {name:22} no pitch {test['edge']:6.1%} vs interior {test['interior']:6.1%}: {test['difference']:+7.1%} "
                  f"[{test['ci95'][0]:+6.1%}, {test['ci95'][1]:+6.1%}]  wilcoxon p={test['wilcoxon_p']:.2g}  "
                  f"edge worse in {test['share_edge_worse']:.0%} of {test['n']} recordings")
    print("\n* of the frames without a tracked pitch, the share where one period matches the next (correlation >= "
          f"{PERIODIC}), i.e. the voice was periodic and the tracker missed it.")
    print("  'longer' is the share of them the same tracker finds with an 85 ms frame and a 40 Hz floor.")
    print("  'err' is the distance in semitones between the pitch the model was trained with and the pitch of the matching period.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"edge_frames": args.edge_frames, "periodic": PERIODIC, "summary": summary, "rows": results}) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
