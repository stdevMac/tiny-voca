#!/usr/bin/env python3
"""Measure how peaky the model's voiced waveform is against its own teacher.

A pulse-excited source-filter head puts each period's energy next to the pulse.
A recorded voice spreads it over the period. Magnitude and periodicity measures
cannot see the difference, because it is a matter of phase; the shape of the
waveform can. This compares paired `<id>_<voice>-reference.wav` and
`-generated.wav` files, per utterance, and tests the paired difference.

Three measures are fixed in advance, all on voiced samples only:

* ``crest``      peak over RMS of all voiced samples of the utterance
* ``kurtosis``   fourth moment over squared second moment of the same samples
* ``local_crest`` peak over RMS inside a window of two pitch periods, in dB,
                 averaged over the voiced frames

The first two pool the utterance, so they move with the spread of loudness as
well as with the shape of a period: a clip with one loud word scores high on
both. ``local_crest`` is normalised inside every window, so only the shape is
left. The rest is descriptive: the same local measure at the edges and in the
interior of voiced runs, and the peakiness of the envelope in bands above the
resolved harmonics, which is where the ear hears phase.

Only model-versus-its-own-teacher is valid. Every measure here moves with F0,
so the two voices must never be compared with each other. The generated audio
is free-running and its pitch is not the teacher's, so the difference is also
reported after removing what the pitch difference explains.

    ./scripts/probe_peakiness.py --pairs ~/tiny-voca-artifacts/es-1-anneal/eval/anneal-es-1-anneal/audio \
        --output ~/tiny-voca-artifacts/phase-dispersion/peakiness-shipped.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HOP_SECONDS = 0.010
EDGE_SECONDS = 0.040
PERIODS = 2.0
BANDS = ((300, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
PRIMARY = ("crest", "kurtosis", "local_crest_db")


def voiced_runs(flags):
    """[(first, last+1)] frame ranges of consecutive voiced frames."""
    padded = np.concatenate(([False], flags, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[0::2], change[1::2]))


def pitch_track(wave, rate):
    """(F0 per 10 ms frame, voiced flags), by the same tracker for teacher and
    model, so neither is given an advantage."""
    import librosa
    f0, flags, _ = librosa.pyin(wave, fmin=50, fmax=500, sr=rate, frame_length=1024, hop_length=int(round(HOP_SECONDS * rate)))
    return f0, np.asarray(flags, bool) & np.isfinite(f0)


def measure(path):
    """Every measure for one file, with voicing and F0 from the file itself."""
    import soundfile as sf
    wave, rate = sf.read(path, dtype="float64")
    return shape(wave, rate, *pitch_track(wave - wave.mean(), rate))


def shape(wave, rate, f0, flags):
    """Every measure for one waveform, given its pitch track. Renders of one
    utterance that differ only in the phase of the excitation share a track, so
    they are compared over exactly the same frames."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    wave = wave - wave.mean()
    hop = int(round(HOP_SECONDS * rate))
    if not flags.any():
        return None

    # Pooled over the voiced samples of the utterance: each voiced frame owns
    # the hop of samples centred on it.
    owned = np.zeros(len(wave), bool)
    for first, last in voiced_runs(flags):
        owned[max(0, first * hop - hop // 2): min(len(wave), last * hop - hop // 2)] = True
    voiced = wave[owned]
    power = float(np.mean(voiced ** 2))
    result = {"voiced_seconds": float(owned.sum() / rate), "f0_hz": float(np.exp(np.mean(np.log(f0[flags])))),
              "crest": float(np.abs(voiced).max() / np.sqrt(power)),
              "kurtosis": float(np.mean(voiced ** 4) / power ** 2)}

    # Zero-phase band filters, so the envelope keeps the timing of the waveform.
    envelopes = {}
    for low, high in BANDS:
        sos = butter(4, (low, min(high, 0.49 * rate)), btype="bandpass", fs=rate, output="sos")
        envelopes[f"{low}-{high}"] = np.abs(hilbert(sosfiltfilt(sos, wave)))

    edge_frames = int(round(EDGE_SECONDS / HOP_SECONDS))
    rows = {"all": [], "edge": [], "interior": []}
    for first, last in voiced_runs(flags):
        for frame in range(first, last):
            half = int(round(PERIODS * rate / f0[frame] / 2))
            centre = frame * hop
            # The window must lie inside the voiced run, so an unvoiced
            # consonant next to it never enters the measure.
            if centre - half < first * hop - hop // 2 or centre + half > last * hop - hop // 2 or centre + half > len(wave):
                continue
            segment = wave[centre - half: centre + half]
            energy = float(np.mean(segment ** 2))
            if energy <= 1e-10:
                continue
            row = {"local_crest_db": 20 * np.log10(np.abs(segment).max() / np.sqrt(energy)),
                   "local_kurtosis": float(np.mean(segment ** 4) / energy ** 2)}
            for name, envelope in envelopes.items():
                piece = envelope[centre - half: centre + half]
                # Peak over mean of the band envelope: 1 for a steady tone, large
                # when the band's partials line up into one burst per period.
                row[f"envelope_peak_{name}"] = float(piece.max() / max(piece.mean(), 1e-12))
            where = "edge" if min(frame - first, last - 1 - frame) < edge_frames else "interior"
            rows["all"].append(row)
            rows[where].append(row)
    if not rows["all"]:
        return None
    for where, values in rows.items():
        suffix = "" if where == "all" else f"@{where}"
        for key in rows["all"][0]:
            result[key + suffix] = float(np.mean([v[key] for v in values])) if values else float("nan")
        result["frames" + suffix] = len(values)
    return result


def paired(reference, generated, pitch=None):
    """Paired statistics of generated minus reference over utterances."""
    from scipy import stats
    reference, generated = np.asarray(reference), np.asarray(generated)
    keep = np.isfinite(reference) & np.isfinite(generated)
    reference, generated = reference[keep], generated[keep]
    difference = generated - reference
    n = len(difference)
    if n < 3:
        return {"n": n}
    mean, deviation = float(difference.mean()), float(difference.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * deviation / np.sqrt(n))
    result = {"n": n, "reference": float(reference.mean()), "generated": float(generated.mean()), "difference": mean,
              "relative": mean / float(reference.mean()), "ci95": [mean - half, mean + half],
              "effect_size_dz": mean / deviation if deviation > 0 else float("nan"),
              "t_p": float(stats.ttest_rel(generated, reference).pvalue),
              "wilcoxon_p": float(stats.wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0,
              "share_above_teacher": float(np.mean(difference > 0))}
    if pitch is not None:
        # What is left of the difference when the model's pitch equals the
        # teacher's: the intercept of difference against the pitch gap in semitones.
        gap = np.asarray(pitch)[keep]
        fit = stats.linregress(gap, difference)
        half = float(stats.t.ppf(0.975, n - 2) * fit.intercept_stderr)
        result["at_equal_pitch"] = {"difference": float(fit.intercept), "ci95": [fit.intercept - half, fit.intercept + half],
                                    "per_semitone": float(fit.slope), "mean_pitch_gap_semitones": float(gap.mean())}
    return result


def find_pairs(directory, generated_suffix):
    pairs = []
    for reference in sorted(directory.glob("*-reference.wav")):
        stem = reference.name[: -len("-reference.wav")]
        generated = directory / f"{stem}{generated_suffix}.wav"
        if generated.exists():
            pairs.append({"id": stem, "voice": stem.rsplit("_", 1)[1], "reference": reference, "generated": generated})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="directory of <id>_<voice>-reference.wav and -generated.wav")
    parser.add_argument("--generated", type=Path, help="take the generated half from this directory instead")
    parser.add_argument("--generated-suffix", default="-generated")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pairs = find_pairs(args.pairs, args.generated_suffix)
    if args.generated:
        pairs = [{**p, "generated": args.generated / p["generated"].name} for p in pairs]
        pairs = [p for p in pairs if p["generated"].exists()]
    if not pairs:
        parser.error("no reference/generated pairs found")
    files = [p[kind] for p in pairs for kind in ("reference", "generated")]
    with ProcessPoolExecutor(args.workers) as pool:
        measured = dict(zip(files, pool.map(measure, files, chunksize=2)))

    rows = []
    for p in pairs:
        reference, generated = measured[p["reference"]], measured[p["generated"]]
        if reference is None or generated is None:
            continue
        rows.append({"id": p["id"], "voice": p["voice"], "reference": reference, "generated": generated})

    keys = [k for k in rows[0]["reference"] if not k.startswith(("frames", "voiced_seconds"))]
    summary = {}
    for voice in sorted({r["voice"] for r in rows}):
        mine = [r for r in rows if r["voice"] == voice]
        gap = [12 * np.log2(r["generated"]["f0_hz"] / r["reference"]["f0_hz"]) for r in mine]
        summary[voice] = {key: paired([r["reference"][key] for r in mine], [r["generated"][key] for r in mine],
                                      None if key == "f0_hz" else gap) for key in keys}

    for voice, values in summary.items():
        print(f"\n{voice}  (generated minus its own teacher, n={values['crest']['n']} utterances)")
        print(f"  {'measure':30}{'teacher':>9}{'model':>9}{'diff':>9}{'rel':>8}  {'95% CI':>19}{'dz':>7}{'t p':>10}{'wilcoxon':>10}{'above':>7}{'eq-pitch':>10}")
        for key in keys:
            v = values[key]
            if v["n"] < 3:
                continue
            mark = "*" if key in PRIMARY else " "
            equal = v.get("at_equal_pitch", {}).get("difference", float("nan"))
            print(f" {mark}{key:30}{v['reference']:9.3f}{v['generated']:9.3f}{v['difference']:+9.3f}{v['relative']:+8.1%}"
                  f"  [{v['ci95'][0]:+8.3f},{v['ci95'][1]:+8.3f}]{v['effect_size_dz']:+7.2f}{v['t_p']:10.2g}{v['wilcoxon_p']:10.2g}"
                  f"{v['share_above_teacher']:7.0%}{equal:+10.3f}")
    print("\n* fixed in advance. 'above' is the share of utterances where the model exceeds its teacher;")
    print("  'eq-pitch' is the difference left when the model's pitch equals the teacher's.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"pairs": str(args.pairs), "generated": str(args.generated or args.pairs),
                                           "summary": summary, "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
