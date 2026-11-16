"""Speak a fixed set of Spanish texts, so the deliverable can be heard.

Every other check in this repository measures something narrow: word error rate
on held-out sentences, the pitch slope of a question, the distance between two
voices. None of them answers the question a user actually has, which is whether
the model reads ordinary Spanish — a paragraph, a price, a date, a list — in
both voices, without falling apart on the parts the corpus never contained.

This writes that set to a directory with a manifest, so the audio can be played
in order and the text checked against it:

    python scripts/demo_es.py --checkpoint data/runs/es-1/inference.pt \
        --output data/eval/es-1/demo

Long text goes through ``speak_text``, which reads sentence by sentence and
joins the audio, so this also exercises the path a real caller uses rather than
the one-sentence path the evaluation uses.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.infer import load_model, speak_text

# One entry per thing a Spanish voice has to be able to do. The register is
# mixed on purpose: the corpus's questions are conversational, so a formal
# question here tests the case the corpus is thinnest on.
TEXTS = (
    ("paragraph", "Ayer por la tarde fuimos al mercado del barrio. Compramos tomates, "
                  "cebollas y un poco de queso manchego. Luego volvimos a casa caminando "
                  "por la orilla del río, porque hacía muy buena tarde."),
    ("numbers", "El paquete pesa 1.234,56 gramos y cuesta 27,90 euros con el envío incluido."),
    ("date", "La reunión será el 3 de mayo de 2026 a las 17:45 en la sala número 12."),
    ("list", "Para la excursión necesitas: agua, protector solar, una gorra y calzado cómodo."),
    ("question_yesno", "¿Cree que el comité aprobará la propuesta antes del verano?"),
    ("question_wh", "¿Dónde dejaste las llaves del coche esta mañana?"),
    ("exclamation", "¡Menuda sorpresa nos dio tu hermano con aquella noticia!"),
    ("statement", "El tren de las ocho y media sale del andén número cuatro."),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    rate = model.config.sample_rate
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, text in TEXTS:
        for voice in model.config.speakers:
            with torch.no_grad():
                wave, spoken = speak_text(model, text, voice, seed=args.seed)
            samples = np.asarray(wave.detach().cpu(), dtype=np.float32)
            path = args.output / f"{name}-{voice}.wav"
            sf.write(path, samples, rate, subtype="PCM_16")
            rows.append({"kind": name, "voice": voice, "audio": path.name, "text": text,
                         "seconds": round(float(samples.size) / rate, 3),
                         "utterances": len(spoken),
                         "peak": round(float(np.abs(samples).max()), 4)})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    total = sum(row["seconds"] for row in rows)
    print(json.dumps({"clips": len(rows), "seconds": round(total, 2), "output": str(args.output)}))


if __name__ == "__main__":
    main()
