"""Ask the aligner which phone inventory a recording actually matches.

The front end is a rule system; the teacher is a neural voice. Where they
disagree — lleísmo vs yeísmo, distinción vs seseo, a lenited stop — the CTC
alignment gets weaker and the durations get worse. This measures it instead of
guessing: align the same recording under the current inventory and under a
variant, and compare the per-phone probability.

    # is the teacher lleísta ([ʎ]) or yeísta ([ʝ])?
    python scripts/probe_phones.py --audio calle.wav --text "La lluvia caía sobre la calle." --map ʎ:ʝ

    # is it distinción ([θ]) or seseo ([s])?
    python scripts/probe_phones.py --audio zaragoza.wav --text "Zaragoza y Cádiz." --map θ:s

The variant that scores higher is the one the recording was made with. `--map`
can be repeated; each variant is tried on its own against the unmodified
baseline.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tiny_voca.align import AlignmentSettings, PhoneRecognizer, align_utterance
from tiny_voca.config import Config
from tiny_voca.text import SYMBOL_TO_ID, Phonemized, phonemize


def remap(phones, mapping):
    symbols = tuple(mapping.get(symbol, symbol) for symbol in phones.symbols)
    return Phonemized(symbols, tuple(SYMBOL_TO_ID[s] for s in symbols), phones.stress, phones.unknown)


def score(recognizer, phones, wave, voiced, config, settings):
    try:
        durations, qc = align_utterance(recognizer, phones, wave, voiced, config, settings)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"token_probability_mean": qc["token_probability_mean"], "weak_phone_fraction": qc["weak_phone_fraction"],
            "max_phone_frames": qc["max_phone_frames"], "phones_per_second": qc["phones_per_second"],
            "durations": durations}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--map", action="append", default=[], metavar="FROM:TO",
                        help="Variant to test against the baseline, for example ʎ:ʝ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    import soundfile as sf
    config = Config()
    wave, rate = sf.read(args.audio, dtype="float32")
    if rate != config.sample_rate:
        raise SystemExit(f"expected {config.sample_rate} Hz, got {rate}")
    frames = len(wave) // config.hop
    wave = wave[:frames * config.hop]
    # The pitch tracker's voicing is only used for the QC probe; a crude energy
    # gate is enough to keep the alignment itself honest.
    from tiny_voca.align import frame_energy_db, silence_mask
    level = frame_energy_db(wave, config.hop)
    silent, _ = silence_mask(level, AlignmentSettings())
    voiced = (~silent).astype(np.float32)
    recognizer = PhoneRecognizer(args.device)
    base = phonemize(args.text)
    results = {"text": args.text, "baseline": score(recognizer, base, wave, voiced, config, AlignmentSettings()),
               "variants": {}}
    for specification in args.map:
        source, _, target = specification.partition(":")
        if not source or not target:
            raise SystemExit(f"--map wants FROM:TO, got {specification!r}")
        variant = remap(base, {source: target})
        results["variants"][specification] = score(recognizer, variant, wave, voiced, config, AlignmentSettings())
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"text      {args.text}")
        print(f"baseline  {results['baseline'].get('token_probability_mean')} "
              f"(weak {results['baseline'].get('weak_phone_fraction')})")
        for name, value in results["variants"].items():
            print(f"{name:9} {value.get('token_probability_mean')} (weak {value.get('weak_phone_fraction')})")
        print("\nhigher token_probability_mean is the better match")


if __name__ == "__main__":
    main()
