#!/usr/bin/env python3
"""Disperse the phases of the excitation at inference, measure it, and let the ear judge.

The waveform head excites its filter with a pulse train: every harmonic at phase
zero, each period's energy in one instant. `probe_peakiness.py` shows the result
on the low voice: M2's voiced waveform is peakier than its own teacher's in
every one of 40 utterances, F2's is not. This probe gives harmonic k a phase
offset instead (`harmonic_phases` in `tiny_voca.model`), which leaves every
magnitude where it was and moves only the shape of the period. Nothing is
retrained; the weights are the shipped ones.

A setting is `beta=<x>`, `power=<x>`, `scatter=<x>` or several joined by commas:

    beta     -beta * k**2. Harmonic k is delayed by beta*k/pi of a period, so
             the pulse becomes a sweep. One period of spread is beta = pi/K for
             K harmonics: 0.0245 at 88 Hz (M2), 0.068 at 244 Hz (F2). A negative
             beta sweeps the other way. (-beta * k, without the square, is only
             a delay and changes nothing.)
    power    the exponent, 2 unless given. Nearer 1 the delay grows per octave
             rather than per harmonic, so the low harmonics get their share.
    scatter  a fixed pseudo-random offset within ±scatter*pi per harmonic,
             seeded per utterance.

`measure` renders the evaluation sentences with each setting and compares the
waveform's peakiness with the teacher's, per voice:

    ./scripts/probe_dispersion.py measure --checkpoint ~/tiny-voca-artifacts/es-1-anneal/runs/es-1-anneal/inference.pt \
        --evaluation ~/tiny-voca-artifacts/es-1-anneal/eval/anneal-es-1-anneal \
        --settings beta=0.006 beta=0.012 scatter=1 --output ~/tiny-voca-artifacts/phase-dispersion/measure.json

`page` renders a listening set, level matched, and builds the blind page:

    ./scripts/probe_dispersion.py page --checkpoint ... --evaluation ... --rows 69cf0694be230d63_M2 tren_M2 tren_F2 \
        --settings beta=0.006 beta=0.012 scatter=1 --pitch-range 1.5 --output ~/tiny-voca-artifacts/phase-dispersion/listen
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.evaluate import pcm16
from tiny_voca.infer import load_model
from tiny_voca.model import harmonic_phases
from tiny_voca.text import phonemize

from probe_peakiness import PRIMARY, measure, paired, pitch_track, shape

SHOWN = (*PRIMARY, "local_kurtosis", "envelope_peak_300-1000", "envelope_peak_1000-2000", "envelope_peak_2000-4000",
         "envelope_peak_4000-8000", "local_crest_db@edge", "local_crest_db@interior")
NOVEL = {
    "tren": "El tren de las ocho y media sale del andén número cuatro.",
    "comite": "¿Cree que el comité aprobará la propuesta antes del verano?",
}


def parse(setting):
    """'beta=0.4,power=1.3' -> {'beta': 0.4, 'scatter': 0.0, 'power': 1.3}."""
    values = {"beta": 0.0, "scatter": 0.0, "power": 2.0}
    for part in setting.split(","):
        key, _, value = part.partition("=")
        if key not in values:
            raise ValueError(f"unknown setting {part!r}; use beta=<x>, power=<x> and/or scatter=<x>")
        values[key] = float(value)
    return values


def delay(values, harmonic):
    """Delay of a harmonic in periods: minus the slope of the phase, over 2 pi."""
    return values["beta"] * values["power"] * harmonic ** (values["power"] - 1) / (2 * np.pi)


def describe(setting, config, hertz):
    """What a setting does to a voice at this pitch, in periods of spread."""
    values = parse(setting)
    parts = []
    if values["beta"]:
        top = config.harmonic_limit * config.sample_rate / hertz
        whole = abs(delay(values, top) - delay(values, 1))
        middle = abs(delay(values, 4000 / hertz) - delay(values, 1000 / hertz))
        parts.append(f"pulse swept over {whole:.2f} of a period ({middle:.2f} between 1 and 4 kHz), "
                     f"{'high harmonics last' if values['beta'] > 0 else 'high harmonics first'}")
    if values["scatter"]:
        parts.append(f"random phase within ±{values['scatter']:g}π per harmonic")
    return "; ".join(parts) or "pulse train"


class Renderer:
    def __init__(self, checkpoint, seed, pitch_range):
        self.model = load_model(checkpoint, "cpu")
        self.config = self.model.config
        self.seed, self.pitch_range = seed, pitch_range
        self.count = int(self.config.harmonic_limit * self.config.sample_rate / self.config.f0_min) + 2

    def __call__(self, text, voice, setting=None, key=""):
        """One utterance as float64 samples, exactly as `tiny_voca.infer.speak`
        makes it when `setting` is None. The scatter is seeded by `key`, so an
        utterance keeps its phases wherever it is rendered."""
        offsets = None
        if setting is not None:
            offsets = harmonic_phases(self.count, **parse(setting), seed=zlib.crc32(key.encode()))
        phones = phonemize(text)
        speaker = torch.tensor([self.config.speakers.index(voice)])
        wave, lengths, _ = self.model.synthesize(torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)]),
                                                 speaker, 1.0, 0.0, self.seed, self.pitch_range, harmonic_phase=offsets,
                                                 noise_above=None)   # pulses alone, as when this was measured
        return wave[0, :int(lengths[0])].double().numpy()


def samples_of(evaluation):
    report = json.loads((evaluation / "report.json").read_text(encoding="utf-8"))
    return report, [s for s in report["samples"] if s["group"] != "novel"]


def pitch_of(wave, rate):
    """Typical pitch of a render in Hz."""
    f0, flags = pitch_track(wave - wave.mean(), rate)
    return float(np.exp(np.mean(np.log(f0[flags])))) if flags.any() else 150.0


def track_file(path):
    wave, rate = sf.read(path, dtype="float64")
    return pitch_track(wave - wave.mean(), rate)


def measure_command(args):
    render = Renderer(args.checkpoint, args.seed, args.pitch_range)
    rate = render.config.sample_rate
    report, samples = samples_of(args.evaluation)
    if args.limit:
        samples = [s for voice in render.config.speakers for s in [x for x in samples if x["voice"] == voice][:args.limit]]
    audio = args.evaluation / "audio"

    # The plain render must be the audio the evaluation wrote, or this probe is
    # not measuring the model that was judged. It is never bit-exact: the
    # evaluation ran on a GPU, and the oscillator integrates the pitch, so float
    # differences grow with time and harmonic number (measured: 60 dB down).
    rendered, worst = {}, np.inf
    for s in samples:
        rendered[s["id"]] = {"pulse": render(s["text"], s["voice"])}
        stored, _ = sf.read(audio / s["generated"], dtype="float64")
        mine = pcm16(rendered[s["id"]]["pulse"])[0]
        if len(stored) != len(mine):
            raise SystemExit(f"{s['id']}: the render is {len(mine)} samples and the evaluation's file is {len(stored)}")
        worst = min(worst, float(10 * np.log10(np.sum(stored ** 2) / max(np.sum((stored - mine) ** 2), 1e-20))))
    print(f"plain render against the evaluation's audio: the difference is at least {worst:.1f} dB below the signal")
    if worst < 40:
        raise SystemExit("the plain render does not reproduce the evaluation's audio")

    with ProcessPoolExecutor(args.workers) as pool:
        references = list(pool.map(measure, [audio / s["reference"] for s in samples], chunksize=2))
        tracks = list(pool.map(track_file, [audio / s["generated"] for s in samples], chunksize=2))
    for number, setting in enumerate(args.settings):
        for s in samples:
            rendered[s["id"]][setting] = render(s["text"], s["voice"], setting, s["id"])
        print(f"rendered {setting} ({number + 1}/{len(args.settings)})", flush=True)

    rows = []
    for s, reference, track in zip(samples, references, tracks):
        measures = {name: shape(wave, rate, *track) for name, wave in rendered[s["id"]].items()}
        levels = {name: float(np.sqrt(np.mean(wave ** 2))) for name, wave in rendered[s["id"]].items()}
        if reference is None or any(m is None for m in measures.values()):
            continue
        rows.append({"id": s["id"], "voice": s["voice"], "reference": reference, "generated": measures, "rms": levels})

    summary = {}
    for voice in render.config.speakers:
        mine = [r for r in rows if r["voice"] == voice]
        gap = [12 * np.log2(r["generated"]["pulse"]["f0_hz"] / r["reference"]["f0_hz"]) for r in mine]
        hertz = float(np.exp(np.mean([np.log(r["generated"]["pulse"]["f0_hz"]) for r in mine])))
        summary[voice] = {}
        print(f"\n{voice}  model minus its own teacher, mean over {len(mine)} utterances [95% CI]; teacher value in the first row")
        print(f"  {'setting':24}" + "".join(f"{key.replace('envelope_peak_', 'env ').replace('local_', 'loc '):>17}" for key in SHOWN) + "   level")
        print(f"  {'teacher':24}" + "".join(f"{np.mean([r['reference'][key] for r in mine]):17.3f}" for key in SHOWN))
        for name in ("pulse", *args.settings):
            summary[voice][name] = {key: paired([r["reference"][key] for r in mine], [r["generated"][name][key] for r in mine], gap)
                                    for key in SHOWN}
            level = float(np.mean([20 * np.log10(r["rms"][name] / r["rms"]["pulse"]) for r in mine]))
            summary[voice][name]["level_db"] = level
            summary[voice][name]["description"] = "pulse train (ships)" if name == "pulse" else describe(name, render.config, hertz)
            cells = "".join(f"{v['difference']:+8.3f}{'*' if v['ci95'][0] > 0 or v['ci95'][1] < 0 else ' '}{v['share_above_teacher']:5.0%}   "
                            for v in (summary[voice][name][key] for key in SHOWN))
            print(f"  {name:24}{cells}{level:+6.2f} dB")
    print("\neach cell: difference from the teacher, * when its 95% interval excludes zero, then the share of utterances above the teacher.")
    print("level: the render's RMS against the pulse train's.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "step": report.get("step"), "settings": args.settings,
                                           "summary": summary, "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.output}")


def page_command(args):
    from listening_page import build
    render = Renderer(args.checkpoint, args.seed, args.pitch_range)
    rate = render.config.sample_rate
    _, samples = samples_of(args.evaluation)
    known = {s["id"]: s for s in samples}
    jobs = []
    for row in args.rows:
        name, _, voice = row.rpartition("_")
        if row in known:
            jobs.append({"row": row, "voice": known[row]["voice"], "text": known[row]["text"],
                         "teacher": args.evaluation / "audio" / known[row]["reference"]})
        elif name in NOVEL and voice in render.config.speakers:
            jobs.append({"row": row, "voice": voice, "text": NOVEL[name], "teacher": None})
        else:
            raise SystemExit(f"{row} is neither an evaluation sentence nor one of {', '.join(NOVEL)} followed by _<voice>")

    args.output.mkdir(parents=True, exist_ok=True)
    clips = []
    for job in jobs:
        pulse = render(job["text"], job["voice"])
        level = float(np.sqrt(np.mean(pulse ** 2)))
        hertz = pitch_of(pulse, rate)
        row = [("shipped", "shipped", "what ships today: the pulse train", pulse),
               ("candidate", "shipped-copy", "identical to what ships today (the control)", pulse)]
        for setting in args.settings:
            wave = render(job["text"], job["voice"], setting, job["row"])
            # Matched in level to the shipped render, so loudness cannot decide.
            row.append(("candidate", setting, describe(setting, render.config, hertz), wave * (level / np.sqrt(np.mean(wave ** 2)))))
        if job["teacher"] is not None:
            teacher, _ = sf.read(job["teacher"], dtype="float64")
            row.insert(0, ("teacher", "teacher", "the teacher's recording", teacher * (level / np.sqrt(np.mean(teacher ** 2)))))
        # One gain for the whole row, set by its loudest sample: nothing clips and
        # the level match survives. Peaky renders would otherwise clip first.
        gain = 0.9 / max(float(np.abs(wave).max()) for _, _, _, wave in row)
        for role, variant, description, wave in row:
            # Candidate files carry no hint of what they are.
            name = role if role != "candidate" else hashlib.sha256(f"{job['row']} {variant} {args.seed}".encode()).hexdigest()[:10]
            file = f"{job['row']}-{name}.wav"
            sf.write(args.output / file, wave * gain, rate, subtype="PCM_16")
            clips.append({"row": job["row"], "voice": job["voice"], "text": job["text"], "role": role, "variant": variant,
                          "description": description, "file": file})
        print(f"rendered {job['row']} ({hertz:.0f} Hz, row gain {20 * np.log10(gain):+.1f} dB)", flush=True)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips), encoding="utf-8")
    intro = ("<p>Each row is one sentence. <b>Ships today</b> is the current model; <b>Teacher</b>, where there is one, is the recording "
             "the model learned from. The lettered clips are the same sentence with only the phase of the excitation changed, in a "
             "different order in every row, and one of them is secretly identical to <b>Ships today</b>.</p>"
             "<p>Listen for the roughness or faint echo at the start and end of words. For each letter, say whether it is worse, the "
             "same, or better than <b>Ships today</b>. If the voice changes in some other way, good or bad, say so in the row's note. "
             "Headphones, and one volume setting for the whole page.</p>")
    page = build(args.output, "Phase dispersion — blind listening", intro, clips, seed=args.seed)
    print(f"wrote {len(clips)} clips and {page}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, command in (("measure", measure_command), ("page", page_command)):
        sub = commands.add_parser(name)
        sub.set_defaults(run=command)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--evaluation", type=Path, required=True, help="a tiny_voca.evaluate output: report.json and audio/")
        sub.add_argument("--settings", nargs="+", required=True)
        sub.add_argument("--output", type=Path, required=name == "page")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--pitch-range", type=float, default=1.0)
        if name == "measure":
            sub.add_argument("--limit", type=int, help="only the first N sentences of each voice")
            sub.add_argument("--workers", type=int, default=8)
        else:
            sub.add_argument("--rows", nargs="+", required=True,
                             help=f"in page order: evaluation sentence ids, which bring their teacher recording, "
                                  f"or an unseen sentence as <name>_<voice> ({', '.join(NOVEL)})")
    args = parser.parse_args()
    for setting in args.settings:
        parse(setting)
    args.run(args)


if __name__ == "__main__":
    main()
