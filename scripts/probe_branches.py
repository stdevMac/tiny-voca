#!/usr/bin/env python3
"""Measure how much energy the noise branch carries, in the model's own units.

The record concludes that "the noise branch carries almost no energy; the
artifact is entirely in the harmonic path", on RMS values of renders that
`probe_variants.py` peak-normalises before writing, so the variants are not on
one scale. Here nothing is normalised or clipped. Every sentence of the report's
validation and test groups is rendered three times through `infer._voice`'s
call (baseline, `harmonic_only`, `noise_only`; same seed, so the same noise
draw), and the two branches are checked to add up to the baseline.

The quantity is 10*log10(E_noise / E_harmonic), energies summed over the
samples of a frame class first, then one ratio per utterance. Classes come from
the model's own voicing decision (`prosody[:, 1] > 0`, one value per 256-sample
frame): voiced, unvoiced, and within voiced runs the first and last 4 frames
(edge, about 43 ms) against the rest (interior). Bands (300-1000 ... 8000-11500 Hz)
use a 1024-point Hann STFT with hop 256 whose window is centred on the middle of
each acoustic frame; band energies are summed over the frames of a class.
A run shorter than 8 frames is all edge. Utterances lacking a class are dropped
from that statistic.

Statistics are per voice only: the voices differ in pitch and every such measure
is confounded by it. Edge versus interior is a paired difference per utterance.

    TINY_VOCA_LANGUAGE=es .venv/bin/python scripts/probe_branches.py \
        --output ~/tiny-voca-artifacts/phase-dispersion/branches.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

EDGE = 4
HOP = 256
FFT = 1024
BANDS = ((300, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 11500))
ROOT = Path.home() / "tiny-voca-artifacts"
CLASSES = ("voiced", "unvoiced", "edge", "interior")


def voiced_runs(flags):
    """[(first, last+1)] frame ranges of consecutive voiced frames."""
    padded = np.concatenate(([False], flags, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[0::2], change[1::2]))


def frame_classes(flags):
    edge = np.zeros(len(flags), bool)
    for first, last in voiced_runs(flags):
        edge[first:first + EDGE] = edge[max(first, last - EDGE):last] = True
    return {"voiced": flags, "unvoiced": ~flags, "edge": flags & edge, "interior": flags & ~edge}


def band_power(wave, frames, rate):
    """[frames, bands] power of each STFT frame, the window centred on the middle of acoustic frame i."""
    x = torch.from_numpy(wave).double()
    # centre=False frame i starts at i*HOP - (FFT/2 - HOP/2), so left padding of that size centres it on the frame
    x = torch.nn.functional.pad(x, (FFT // 2 - HOP // 2, FFT))
    spec = torch.stft(x, FFT, HOP, window=torch.hann_window(FFT, dtype=torch.double), center=False, return_complex=True)
    power = (spec.abs() ** 2)[:, :frames].numpy()
    freq = np.fft.rfftfreq(FFT, 1 / rate)
    return np.stack([power[(freq >= lo) & (freq < hi)].sum(0) for lo, hi in BANDS], axis=1)


def ratio(noise, harmonic):
    return float(10 * np.log10(noise / harmonic)) if noise > 0 and harmonic > 0 else float("nan")


def measure(model, text, voice, rate):
    from tiny_voca.text import phonemize
    phones = phonemize(text)
    device = next(model.parameters()).device
    args = (torch.tensor([phones.ids], device=device), torch.tensor([phones.stress], device=device),
            torch.tensor([len(phones)], device=device), torch.tensor([model.config.speakers.index(voice)], device=device))
    out = {}
    for variant in (None, "harmonic_only", "noise_only"):
        # Pulses alone, as when this was measured: the harmonic branch is then the pulse train and nothing else.
        audio, lengths, details = model.synthesize(*args, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, variant=variant,
                                                   noise_above=None)
        out[variant] = audio[0, :int(lengths[0])].double().numpy()
    gap = float(np.abs(out["harmonic_only"] + out["noise_only"] - out[None]).max())
    scale = float(np.abs(out[None]).max())
    flags = (details["prosody"][0][:, 1] > 0).numpy()[:len(out[None]) // HOP]
    frames = len(flags)
    cut = frames * HOP
    h, n = out["harmonic_only"][:cut], out["noise_only"][:cut]
    classes = frame_classes(flags)
    row = {"seconds": len(out[None]) / rate, "frames": {k: int(v.sum()) for k, v in classes.items()},
           "max_abs_baseline": scale, "max_abs_sum_error": gap,
           "rms_harmonic": float(np.sqrt(np.mean(h ** 2))), "rms_noise": float(np.sqrt(np.mean(n ** 2)))}
    eh, en = (np.square(v).reshape(frames, HOP).sum(1) for v in (h, n))
    ph, pn = band_power(h, frames, rate), band_power(n, frames, rate)
    for name, mask in classes.items():
        row[f"db_{name}"] = ratio(en[mask].sum(), eh[mask].sum()) if mask.any() else float("nan")
        for b, (lo, hi) in enumerate(BANDS):
            if name != "unvoiced":
                row[f"db_{name}@{lo}-{hi}"] = ratio(pn[mask, b].sum(), ph[mask, b].sum()) if mask.any() else float("nan")
    return row, gap, scale


def interval(values):
    from scipy import stats
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {"n": n}
    mean, half = float(values.mean()), float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "ci95": [mean - half, mean + half]}


def paired(edge, interior):
    from scipy import stats
    edge, interior = np.asarray(edge, float), np.asarray(interior, float)
    keep = np.isfinite(edge) & np.isfinite(interior)
    edge, interior = edge[keep], interior[keep]
    result = interval(edge - interior)
    if result["n"] < 3:
        return result
    d = edge - interior
    result.update({"edge": float(edge.mean()), "interior": float(interior.mean()),
                   "t_p": float(stats.ttest_rel(edge, interior).pvalue),
                   "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                   "share_edge_above": float(np.mean(d > 0))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "es-1-anneal/runs/es-1-anneal/inference.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "es-1-anneal/eval/anneal-es-1-anneal/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "phase-dispersion/branches.json")
    parser.add_argument("--groups", nargs="+", default=["validation", "test"])
    parser.add_argument("--tolerance", type=float, default=1e-5, help="largest allowed |harmonic+noise-baseline| relative to the baseline peak")
    args = parser.parse_args()

    from tiny_voca.infer import load_model, match_frontend
    from tiny_voca.text import FRONTEND_VARIABLE
    if FRONTEND_VARIABLE not in os.environ:
        match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    rate = model.config.sample_rate
    samples = [s for s in json.loads(args.report.read_text())["samples"] if s["group"] in args.groups]

    rows = []
    for s in samples:
        row, gap, scale = measure(model, s["text"], s["voice"], rate)
        if gap > args.tolerance * scale:
            raise SystemExit(f"SANITY CHECK FAILED on {s['id']}: max|harmonic+noise-baseline|={gap:.3g}, baseline peak {scale:.3g}")
        rows.append({"id": s["id"], "group": s["group"], "voice": s["voice"], "text": s["text"], **row})
    worst = max(r["max_abs_sum_error"] / r["max_abs_baseline"] for r in rows)
    print(f"sanity: harmonic_only + noise_only == baseline on all {len(rows)} renders, worst relative error {worst:.2g}")

    keys = [k for k in rows[0] if k.startswith("db_")]
    summary = {}
    for voice in sorted({r["voice"] for r in rows}):
        mine = [r for r in rows if r["voice"] == voice]
        summary[voice] = {"n": len(mine), "levels": {k: interval([r[k] for r in mine]) for k in keys},
                          "edge_minus_interior": {"broadband": paired([r["db_edge"] for r in mine], [r["db_interior"] for r in mine]),
                                                  **{f"{lo}-{hi}": paired([r[f"db_edge@{lo}-{hi}"] for r in mine], [r[f"db_interior@{lo}-{hi}"] for r in mine])
                                                     for lo, hi in BANDS}}}

    for voice, v in summary.items():
        print(f"\n{voice}  noise-to-harmonic energy ratio, dB (n={v['n']} utterances, mean [95% CI])")
        for k, s in v["levels"].items():
            if s["n"] >= 3:
                print(f"  {k[3:]:22}{s['mean']:+8.1f}  [{s['ci95'][0]:+7.1f},{s['ci95'][1]:+7.1f}]  n={s['n']}")
        print(f"  edge minus interior, paired (dB)\n  {'':12}{'edge':>8}{'interior':>9}{'diff':>8}  {'95% CI':>17}{'t p':>10}{'wilcoxon':>10}{'edge>int':>9}{'n':>4}")
        for k, s in v["edge_minus_interior"].items():
            if s["n"] >= 3:
                print(f"  {k:12}{s['edge']:8.1f}{s['interior']:9.1f}{s['mean']:+8.2f}  [{s['ci95'][0]:+7.2f},{s['ci95'][1]:+7.2f}]"
                      f"{s['t_p']:10.2g}{s['wilcoxon_p']:10.2g}{s['share_edge_above']:9.0%}{s['n']:4d}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "report": str(args.report), "edge_frames": EDGE,
                                           "stft": {"fft": FFT, "hop": HOP, "window": "hann", "centred_on_frame_middle": True},
                                           "bands_hz": BANDS, "worst_relative_sum_error": worst, "summary": summary, "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
