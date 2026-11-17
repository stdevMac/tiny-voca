#!/usr/bin/env python3
"""Sweep the anticausal half of the cepstral filter, and expose the difference.

The harmonic branch filters a pulse train with a complex cepstrum. The causal
half can only respond to what has already happened; the anticausal half reaches
forward, so it is the only part of the filter that can put energy *before* a
pulse. That is pre-echo by construction, and it is the one mechanism that
matches "an echo at the edges of words".

The model was trained with it, so removing it changes the timbre. This renders a
sweep instead of a switch, level-matched to the baseline, so the ear can find
the point where the smear goes without the level confound. It also writes the
literal difference between the baseline and the minimum-phase render, which is
the anticausal component on its own.

    python scripts/probe_anticausal.py --checkpoint data/runs/es-1-anneal/inference.pt \
        --output data/eval/anticausal
"""

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.infer import load_model
from tiny_voca.text import phonemize

SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)
TEXTS = (
    ("statement", "El tren de las ocho y media sale del andén número cuatro."),
    ("question", "¿Cree que el comité aprobará la propuesta antes del verano?"),
)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch-range", type=float, default=1.5)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    rate = model.config.sample_rate
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, text in TEXTS:
        phonemes = phonemize(text)
        tokens = torch.tensor([phonemes.ids], device=args.device)
        stress = torch.tensor([phonemes.stress], device=args.device)
        length = torch.tensor([len(phonemes)], device=args.device)
        for voice, label in enumerate(model.config.speakers):
            speaker = torch.tensor([voice], device=args.device)
            rendered = {}
            for scale in SCALES:
                with torch.inference_mode():
                    wave, lengths, _ = model.synthesize(tokens, stress, length, speaker, speed=1.0, seed=args.seed,
                                                        pitch_range=args.pitch_range, anticausal_scale=scale,
                                                        noise_above=None)   # pulses alone, as when this was measured
                rendered[scale] = wave[0, : int(lengths[0])].float().cpu().numpy()
            # Level-match every render to the baseline, so what is compared is the
            # timbre and not the 11 dB the anticausal half contributes.
            reference = rms(rendered[1.0])
            for scale in SCALES:
                audio = rendered[scale] * (reference / rms(rendered[scale]))
                file = f"{name}-{label}-anti{int(scale * 100):03d}.wav"
                sf.write(args.output / file, np.clip(audio, -1, 1) * 0.9, rate, subtype="PCM_16")
                rows.append({"text": name, "voice": label, "anticausal_scale": scale, "file": file})
            # The anticausal component on its own: baseline minus minimum phase.
            residual = rendered[1.0] * (reference / rms(rendered[1.0])) - rendered[0.0] * (reference / rms(rendered[0.0]))
            residual = residual / max(np.abs(residual).max(), 1e-6) * 0.9
            file = f"{name}-{label}-difference.wav"
            sf.write(args.output / file, residual, rate, subtype="PCM_16")
            rows.append({"text": name, "voice": label, "anticausal_scale": "difference", "file": file})
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    columns = [f"anti {int(s * 100)}%" for s in SCALES] + ["difference"]
    page = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>anticausal cepstrum sweep</title>",
            "<style>body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1200px;margin:2rem auto;color:#1a1a1a}"
            "table{border-collapse:collapse;width:100%}th,td{padding:.45rem .3rem;border-bottom:1px solid #e6e6e6;text-align:left}"
            "th{font-size:.72rem;text-transform:uppercase;color:#777}audio{width:100%;height:30px}"
            "td.n{font-weight:600;width:12%}p{color:#555}</style>",
            "<h1>Anticausal cepstrum sweep — level matched</h1>",
            "<p>The anticausal half of the filter is the only part that can put energy before a pulse. "
            "100% is what ships; 0% is minimum phase. The last column is the difference between them: "
            "the anticausal component on its own.</p>",
            "<table><thead><tr><th>Sentence</th><th>Voice</th>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead><tbody>"]
    for name, _ in TEXTS:
        for label in model.config.speakers:
            cells = "".join(f"<td><audio controls preload='none' src='{name}-{label}-anti{int(s * 100):03d}.wav'></audio></td>"
                            for s in SCALES)
            cells += f"<td><audio controls preload='none' src='{name}-{label}-difference.wav'></audio></td>"
            page.append(f"<tr><td class='n'>{html.escape(name)}</td><td>{label}</td>{cells}</tr>")
    page.append("</tbody></table></html>")
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"wrote {len(rows)} clips and {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
