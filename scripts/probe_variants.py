#!/usr/bin/env python3
"""Render the same sentences with one part of the synthesis head at a time.

Every measurement in this repository says the model tracks its teacher: HNR,
spectral contrast, flux, mel distance, shimmer, durations. The listener says
otherwise, at the edges of words, worse on the low voice. When the metrics and
the ears disagree, the ears are the instrument and the job is to build the
comparison that lets them answer a question the metrics cannot pose.

So this renders each sentence four ways, changing only the head:

    baseline        what ships
    harmonic_only   the pulse train and its envelope, no noise branch
    noise_only      the noise branch alone
    minimum_phase   the anticausal half of the cepstrum zeroed, so the filter
                    cannot smear energy backwards into the silence before a word

If the roughness lives in one branch, `harmonic_only` or `noise_only` says which.
If it is pre-echo, `minimum_phase` removes it.

    python scripts/probe_variants.py --checkpoint data/runs/es-1-anneal/inference.pt \
        --output data/eval/variants
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

VARIANTS = ("baseline", "harmonic_only", "noise_only", "minimum_phase")
TEXTS = (
    ("statement", "El tren de las ocho y media sale del andén número cuatro."),
    ("question", "¿Cree que el comité aprobará la propuesta antes del verano?"),
    ("paragraph", "Ayer por la tarde fuimos al mercado del barrio. Compramos tomates, cebollas y un poco de queso."),
)


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
        tokens, stress = torch.tensor([phonemes.ids], device=args.device), torch.tensor([phonemes.stress], device=args.device)
        length = torch.tensor([len(phonemes)], device=args.device)
        for voice, label in enumerate(model.config.speakers):
            speaker = torch.tensor([voice], device=args.device)
            for variant in VARIANTS:
                with torch.inference_mode():
                    wave, lengths, _ = model.synthesize(tokens, stress, length, speaker, speed=1.0, seed=args.seed,
                                                        pitch_range=args.pitch_range,
                                                        variant=None if variant == "baseline" else variant,
                                                        noise_above=None)   # pulses alone, as when this was measured
                audio = wave[0, : int(lengths[0])].float().cpu().numpy()
                audio = audio / max(np.abs(audio).max(), 1e-6) * 0.95
                file = f"{name}-{label}-{variant}.wav"
                sf.write(args.output / file, audio, rate, subtype="PCM_16")
                rows.append({"text": name, "voice": label, "variant": variant, "file": file})
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    page = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>synthesis head variants</title>",
            "<style>body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:2rem auto;color:#1a1a1a}"
            "table{border-collapse:collapse;width:100%}th,td{padding:.45rem .35rem;border-bottom:1px solid #e6e6e6;text-align:left}"
            "th{font-size:.75rem;text-transform:uppercase;color:#777}audio{width:100%;height:32px}"
            "td.n{font-weight:600;width:16%}p{color:#555}</style>",
            "<h1>Synthesis head variants — same sentence, same seed</h1>",
            "<p>Only the head changes. <b>harmonic_only</b> and <b>noise_only</b> say which branch carries the artifact; "
            "<b>minimum_phase</b> removes the anticausal cepstrum, i.e. any energy smeared backwards before a word starts.</p>",
            "<table><thead><tr><th>Sentence</th><th>Voice</th>" + "".join(f"<th>{v}</th>" for v in VARIANTS) + "</tr></thead><tbody>"]
    for name, _ in TEXTS:
        for label in model.config.speakers:
            page.append(f"<tr><td class='n'>{html.escape(name)}</td><td>{label}</td>" + "".join(
                f"<td><audio controls preload='none' src='{html.escape(f'{name}-{label}-{v}.wav')}'></audio></td>" for v in VARIANTS) + "</tr>")
    page.append("</tbody></table></html>")
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"wrote {len(rows)} clips and {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
