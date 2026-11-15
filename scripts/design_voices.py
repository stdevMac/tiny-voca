"""Design the two Spanish voices and write their reference clips. GPU only.

The tiny model learns whatever the teacher says, so the voices are chosen once,
here, and then reused for every clip. This script runs Qwen3-TTS VoiceDesign
from text descriptions, keeps the best take, and writes:

    <root>/references/F2.wav, <root>/references/M2.wav   24 kHz mono reference clips
    <root>/voices.json                                   what generate_audio.py reads

The reference text must be exactly what the clip says: the cloner uses it to
align the prompt. It is written into voices.json, so nothing has to be
transcribed by hand.

    python scripts/design_voices.py --root data/voices-es --config voices/es.json

Re-running with the same seeds and descriptions reproduces the same clips.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON: {name: {instruct, text, seed}}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--takes", type=int, default=3, help="Clips per voice; the loudest clean one is kept")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    from tiny_voca.languages import profile as language

    config = json.loads(args.config.read_text(encoding="utf-8"))
    references = args.root / "references"
    references.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map=args.device, dtype=torch.bfloat16, attn_implementation="sdpa")
    voices = {}
    for name, spec in config.items():
        best = None
        for take in range(args.takes):
            torch.manual_seed(spec["seed"] + take)
            waves, rate = model.generate_voice_design(text=spec["text"], language=language.TTS_LANGUAGE,
                                                      instruct=spec["instruct"])
            wave = np.asarray(waves[0], dtype=np.float32)
            peak = float(np.abs(wave).max()) if wave.size else 0.0
            if not np.isfinite(wave).all() or peak < 1e-4:
                continue
            # Prefer the take that is neither clipped nor nearly silent.
            score = -abs(peak - 0.7)
            if best is None or score > best[0]:
                best = (score, wave * (0.95 / peak), take)
        if best is None:
            raise SystemExit(f"no usable clip for voice {name}")
        path = references / f"{name}.wav"
        sf.write(path, best[1], rate, subtype="PCM_16")
        voices[name] = {"text": spec["text"], "instruct": spec["instruct"], "seed": spec["seed"], "take": best[2],
                        "reference_audio": f"references/{name}.wav"}
        print(json.dumps({"voice": name, "path": str(path), "seconds": round(len(best[1]) / rate, 3), "take": best[2]}))
    (args.root / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
