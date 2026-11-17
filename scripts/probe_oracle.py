#!/usr/bin/env python3
"""Is the word-edge artifact in the waveform head, or in what predicts its parameters?

The head (`SourceFilter`) turns 134 numbers per frame into audio: the cepstra
and gains of a filtered pulse train and of filtered noise. The model predicts
those numbers. Here they are instead fitted, by gradient descent, to one
teacher recording at a time, with the pitch the recording was labelled with in
the oscillator. Nothing is trained: the weights are the shipped ones and are
never updated, and a fit is thrown away after its one recording.

Every render is on the teacher's own timing, so it is compared with the
recording frame by frame:

    forced   the shipped model given the recording's durations, with the
             recording's pitch in the oscillator; the decoder is conditioned on
             the model's own predicted pitch, voicing and energy. This is the
             state the model was last trained in.
    oracle   the head alone, its per-frame parameters fitted to the recording
             on the training's spectral loss (2 * STFT + 10 * mel), starting
             from what the model predicts.
    critics  the same fit with the training's own critics (frozen) adding their
             feature-matching distance, balanced against the spectral loss the
             way training does it (`gan_ratio`). The adversarial term is left
             out: against critics that cannot answer back it only finds what
             fools them.

If the oracle is free of the artifact, the head can make clean audio and the
fault is in what the model predicts (labels, conditioning, objective). If the
oracle has it too, the head cannot represent the recording there and no amount
of training on this head will.

`fit` renders and measures; `summary` prints the tables again; `page` builds a
blind listening page from fitted recordings.

    TINY_VOCA_LANGUAGE=es ./scripts/probe_oracle.py fit --per-voice 32 --output ~/tiny-voca-artifacts/oracle
    TINY_VOCA_LANGUAGE=es ./scripts/probe_oracle.py page --output ~/tiny-voca-artifacts/oracle \
        --rows 69cf0694be230d63_M2 b79abf5cc6784891_M2 69cf0694be230d63_F2
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_peakiness import pitch_track, shape
from probe_pitch_labels import adjacent_periods, word_layout

ROOT = Path.home() / "tiny-voca-artifacts"
EDGE = 4                       # frames; 43 ms
PEAKINESS = ("crest", "kurtosis", "local_crest_db", "local_crest_db@edge", "local_crest_db@interior")
CLASSES = ("word interior", "word edge in phrase", "word edge at pause", "pause beside a word", "pause elsewhere",
           "vowel, pitch tracked", "vowel, no pitch")
VARIANTS = {"forced": "the shipped model on the teacher's timing and pitch",
            "oracle": "the head alone, fitted to the recording (spectral loss)",
            "critics": "the head alone, fitted to the recording (spectral loss and the critics' features)"}
_state = {}


class Held(nn.Module):
    """Stands in for one of the head's linear layers and returns fitted values."""

    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(value.clone())

    def forward(self, hidden):
        return self.value


def recording(row, audio, config):
    """A teacher recording as `AnnotatedDataset` serves it with `voiced_vowels`,
    from the audio file instead of the feature cache."""
    from tiny_voca.data import VOWEL_IDS, continuous_pitch
    from tiny_voca.prepare import extract

    wave, rate = sf.read(audio / f"{row['id']}.flac", dtype="float32")
    stored = extract(wave, config)
    if rate != config.sample_rate or stored["prosody"].shape[0] != row["frames"]:
        raise ValueError(f"{row['id']}: audio does not match the annotation")
    pitch, trusted = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
    vowel = torch.repeat_interleave(torch.tensor([t in VOWEL_IDS for t in row["tokens"]]), torch.tensor(row["durations"]))
    voiced = torch.maximum(stored["prosody"][:, 1], vowel.to(stored["prosody"].dtype))
    prosody = torch.stack((pitch, voiced, (stored["prosody"][:, 2] - config.energy_mean) / config.energy_scale), dim=-1)
    return {"wave": stored["wave"], "prosody": prosody, "tracked": stored["prosody"][:, 1] > 0.5, "vowel": vowel}


@torch.no_grad()
def forced(model, row, prosody, own=True):
    """The head's inputs for the recording's durations: acoustic states and
    frame count. `own` conditions the decoder on the model's predicted prosody,
    as the end of training did; otherwise on the recording's."""
    c = model.config
    tokens, stress = torch.tensor([row["tokens"]]), torch.tensor([row["stress"]])
    speakers = torch.tensor([c.speakers.index(row["speaker"])])
    hidden, token_mask = model.encode(tokens, stress, torch.tensor([len(row["tokens"])]), speakers)
    frames, frame_mask, lengths = model.expand(hidden, torch.tensor([row["durations"]]), token_mask, speakers)
    prosody_hidden, prediction = model.predict_prosody(frames, frame_mask)
    condition = model.hard_prosody(prediction) if own else prosody[None]
    return model.decode(prosody_hidden, condition, frame_mask), lengths


def seeded_noise(samples, seed=0):
    """The noise `TinyVoca.synthesize` gives the first row of a batch."""
    return torch.randn(samples, generator=torch.Generator().manual_seed(seed))[None]


def fit(model, acoustic, f0, lengths, wave, steps, rate, critics=None, ratio=0.7, crop=8192, crops=4, seed=0):
    """A copy of the head whose per-frame parameters were fitted to `wave`.
    Noise is drawn afresh every step, so the fit cannot lean on one draw, and
    the learning rate falls to zero, so the last steps do not wander."""
    from tiny_voca.losses import SpectralLoss, generator_gan_loss

    c = model.config
    head = copy.deepcopy(model.vocoder)
    with torch.no_grad():
        head.harmonic, head.noise = Held(model.vocoder.harmonic(acoustic)), Held(model.vocoder.noise(acoustic))
    spectral = SpectralLoss(c.sample_rate)
    stand_in = torch.zeros(1, acoustic.shape[1], 1)
    optimizer = torch.optim.Adam([head.harmonic.value, head.noise.value], lr=rate, betas=(0.9, 0.99))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(seed + 1)
    samples = int(lengths) * c.hop
    history = []
    for step in range(steps):
        render = head(stand_in, f0, lengths, noise=torch.randn(wave.shape, generator=generator))
        stft, mel = spectral(render, wave, lengths * c.hop)
        loss = reconstruction = 2 * stft + 10 * mel
        if critics is not None:
            size = min(crop, samples)
            start = (torch.rand(crops, generator=generator) * (samples - size + 1)).long()
            index = (start[:, None] + torch.arange(size)).flatten()[None]
            fake, real = render.gather(1, index).view(-1, size), wave.gather(1, index).view(-1, size)
            with torch.no_grad():
                real_features = critics(real)
            _, matching = generator_gan_loss(real_features, critics(fake))
            ours = torch.autograd.grad(reconstruction, render, retain_graph=True)[0].gather(1, index)
            theirs = torch.autograd.grad(matching, render, retain_graph=True)[0].gather(1, index)
            loss = reconstruction + (ratio * ours.norm() / theirs.norm().clamp_min(1e-12)).detach() * matching
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, stft.item(), mel.item()))
    return head.requires_grad_(False), stand_in, history


def frame_classes(row, tracked, vowel):
    from tiny_voca.text import PHONE_IDS

    owner, since, until, pause_before, pause_after = word_layout(row["tokens"], row["durations"], PHONE_IDS)
    word = since >= 0
    initial, final = word & (since < EDGE), word & (until < EDGE)
    at_pause = (initial & pause_before) | (final & pause_after)
    # A pause frame is beside a word when a word starts or ends within EDGE frames of it.
    near = np.zeros(len(word), bool)
    for shift in range(1, EDGE + 1):
        near[:-shift] |= word[shift:]
        near[shift:] |= word[:-shift]
    return {"word interior": word & ~initial & ~final, "word edge in phrase": (initial | final) & ~at_pause,
            "word edge at pause": at_pause, "pause beside a word": ~word & near, "pause elsewhere": ~word & ~near,
            "vowel, pitch tracked": vowel & tracked, "vowel, no pitch": vowel & ~tracked}


def log_mel(wave, config, bank, window):
    from tiny_voca.losses import magnitude
    return torch.log(magnitude(wave, config.n_fft, config.hop, window).transpose(1, 2) @ bank.T + 1e-5)[0]   # [frames + 1, mels]


def frame_energy(wave, hop, frames):
    """Energy of the hop of samples centred on each frame."""
    padded = np.pad(wave.astype(np.float64), (hop // 2, hop))
    return (padded[:frames * hop].reshape(frames, hop) ** 2).sum(1)


def measure(row, teacher, renders, branches, classes, config, tracked_hz):
    """Per frame class: distance to the recording, level against it, and how
    periodic each waveform is. `branches` holds (harmonic only, noise only)
    renders for the variants that have them."""
    from tiny_voca.losses import mel_filterbank

    frames, rate = row["frames"], config.sample_rate
    bank, window = mel_filterbank(rate, config.n_fft, config.n_mels), torch.hann_window(config.n_fft)
    reference = log_mel(torch.from_numpy(teacher)[None], config, bank, window)[:frames]
    energy = {"teacher": frame_energy(teacher, config.hop, frames)}
    typical = float(np.exp(np.median(np.log(tracked_hz)))) if len(tracked_hz) else 150.0
    period = int(round(rate / typical))
    periodic = {"teacher": adjacent_periods(teacher, rate, config.hop, frames, typical / 1.8, typical * 1.8, period)[0]}
    f0, flags = pitch_track(teacher.astype(np.float64) - teacher.mean(), rate)
    result = {"id": row["id"], "voice": row["speaker"], "typical_hz": typical,
              "frames": {name: int(member.sum()) for name, member in classes.items()},
              "peakiness": {"teacher": shape(teacher.astype(np.float64), rate, f0, flags)}, "variants": {}}
    for name, wave in renders.items():
        distance = (log_mel(torch.from_numpy(wave)[None], config, bank, window)[:frames] - reference).abs().mean(1).numpy()
        energy[name] = frame_energy(wave, config.hop, frames)
        periodic[name] = adjacent_periods(wave, rate, config.hop, frames, typical / 1.8, typical * 1.8, period)[0]
        entry = {}
        for label, member in classes.items():
            if not member.any():
                continue
            entry[label] = {"mel": float(distance[member].mean()),
                            "level_db": float(10 * np.log10((energy[name][member].sum() + 1e-12) / (energy["teacher"][member].sum() + 1e-12))),
                            "periodicity": float(periodic[name][member].mean()),
                            "teacher_periodicity": float(periodic["teacher"][member].mean())}
            if name in branches:
                harmonic, noise = (frame_energy(b, config.hop, frames) for b in branches[name])
                entry[label]["noise_to_harmonic_db"] = float(10 * np.log10((noise[member].sum() + 1e-12) / (harmonic[member].sum() + 1e-12)))
        result["variants"][name] = entry
        result["peakiness"][name] = shape(wave.astype(np.float64), rate, f0, flags)
    return result


def work(task):
    row, args = task
    torch.set_num_threads(1)
    from tiny_voca.infer import load_model
    from tiny_voca.losses import Discriminators

    if "model" not in _state:
        _state["model"] = load_model(args.checkpoint)
        if "critics" in args.variants:
            critics = Discriminators()
            critics.load_state_dict(torch.load(args.critics, map_location="cpu", weights_only=True)["critics"])
            _state["critics"] = critics.eval().requires_grad_(False)
    model = _state["model"]
    c = model.config
    target = args.output / "rows" / f"{row['id']}.json"
    if target.exists() and not args.again:
        return json.loads(target.read_text())
    started = time.time()
    item = recording(row, args.audio, c)
    wave = item["wave"][None]
    f0 = model.frequency(item["prosody"][None, :, 0])
    acoustic, lengths = forced(model, row, item["prosody"])
    noise = seeded_noise(wave.shape[-1], args.seed)
    heads = {"forced": (model.vocoder, acoustic, None)}
    for variant in args.variants:
        heads[variant] = fit(model, acoustic, f0, lengths, wave, args.steps, args.rate, _state.get("critics") if variant == "critics" else None,
                             seed=args.seed)
    renders, branches, histories = {}, {}, {}
    with torch.no_grad():
        for name, (head, hidden, history) in heads.items():
            renders[name] = head(hidden, f0, lengths, noise=noise)[0].numpy()
            branches[name] = tuple(head(hidden, f0, lengths, noise=noise, variant=v)[0].numpy() for v in ("harmonic_only", "noise_only"))
            if history:
                histories[name] = history
    teacher = item["wave"].numpy()
    classes = frame_classes(row, item["tracked"].numpy(), item["vowel"].numpy())
    result = measure(row, teacher, renders, branches, classes, c, f0[0].numpy()[item["tracked"].numpy()])
    result.update(text=row["text"], history=histories, seconds=time.time() - started)
    (args.output / "audio").mkdir(parents=True, exist_ok=True)
    for name, audio in {"teacher": teacher, **renders}.items():
        sf.write(args.output / "audio" / f"{row['id']}-{name}.wav", audio, c.sample_rate, subtype="FLOAT")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def interval(values):
    from scipy import stats
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if len(values) < 3:
        return len(values), float("nan"), float("nan")
    return len(values), float(values.mean()), float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def summarize(results):
    from scipy import stats
    for voice in sorted({r["voice"] for r in results}):
        mine = [r for r in results if r["voice"] == voice]
        variants = list(mine[0]["variants"])
        print(f"\n{voice}: {len(mine)} recordings, typical pitch {np.exp(np.mean([np.log(r['typical_hz']) for r in mine])):.0f} Hz")
        for key, title, digits in (("mel", "distance to the recording (mean |log-mel difference|, per frame)", 3),
                                   ("level_db", "level against the recording (dB)", 1),
                                   ("periodicity", "periodicity (correlation of one period with the next; 'teacher' is the recording)", 3),
                                   ("noise_to_harmonic_db", "noise branch over harmonic branch (dB)", 1)):
            print(f"  {title}")
            columns = (["teacher"] if key == "periodicity" else []) + variants
            print(f"    {'frames':24}{'share':>7}" + "".join(f"{name:>18}" for name in columns)
                  + "".join(f"{name + ' - forced':>26}" for name in variants[1:]))
            total = sum(sum(r["frames"][label] for label in CLASSES[:5]) for r in mine)
            for label in CLASSES:
                cells = []
                for name in columns:
                    values = [r["variants"][variants[0]].get(label, {}).get("teacher_periodicity") if name == "teacher"
                              else r["variants"][name].get(label, {}).get(key) for r in mine]
                    n, mean, half = interval(values)
                    cells.append(f"{mean:11.{digits}f} ±{half:5.{digits}f}")
                for name in variants[1:]:
                    pairs = [(r["variants"][name][label][key], r["variants"]["forced"][label][key]) for r in mine
                             if label in r["variants"][name] and key in r["variants"][name][label]]
                    difference = np.array([a - b for a, b in pairs])
                    n, mean, half = interval(difference)
                    p = stats.wilcoxon(difference).pvalue if n >= 3 and np.any(difference != 0) else float("nan")
                    cells.append(f"{mean:+9.{digits}f} ±{half:5.{digits}f} p={p:<7.2g}")
                print(f"    {label:24}{sum(r['frames'][label] for r in mine) / total:7.1%}" + "".join(cells))
        print("  peakiness of the voiced waveform, on the recording's pitch track (difference from the recording, paired)")
        for key in PEAKINESS:
            cells = []
            for name in variants:
                pairs = [(r["peakiness"][name][key], r["peakiness"]["teacher"][key]) for r in mine
                         if r["peakiness"].get(name) and r["peakiness"]["teacher"]]
                difference = np.array([a - b for a, b in pairs])
                n, mean, half = interval(difference)
                cells.append(f"{name}: {mean:+.2f} ±{half:.2f} ({np.mean(difference > 0):.0%} above)")
            n, mean, half = interval([r["peakiness"]["teacher"][key] for r in mine if r["peakiness"]["teacher"]])
            print(f"    {key:26} recording {mean:6.2f}   " + "   ".join(cells))
        fits = [r["history"][name] for r in mine for name in r["history"]]
        if fits:
            for name in variants[1:]:
                first = np.mean([2 * r["history"][name][0][1] + 10 * r["history"][name][0][2] for r in mine])
                last = np.mean([2 * r["history"][name][-1][1] + 10 * r["history"][name][-1][2] for r in mine])
                print(f"  {name}: spectral loss {first:.2f} at the model's own parameters, {last:.2f} after the fit")


def fit_command(args):
    from tiny_voca.text import frontend_fingerprint
    rows = []
    for split in args.splits:
        rows += [json.loads(line) for line in (args.annotations / f"{split}.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    if any(row["frontend"] != frontend_fingerprint() for row in rows):
        raise SystemExit("the annotations were made with another front end; set TINY_VOCA_LANGUAGE to match")
    known = {row["id"]: row for row in rows}
    chosen = [known[name] for name in args.rows]
    for voice in sorted({row["speaker"] for row in rows}):
        mine = sorted((row for row in rows if row["speaker"] == voice and row["id"] not in args.rows), key=lambda row: row["id"])
        random.Random(args.seed).shuffle(mine)
        have = sum(row["speaker"] == voice for row in chosen)
        chosen += mine[:max(0, args.per_voice - have)]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(work, [(row, args) for row in chosen]):
            results.append(result)
            print(f"{len(results):3d}/{len(chosen)} {result['id']} {result['seconds']:.0f}s", flush=True)
    summarize(results)


def summary_command(args):
    summarize([json.loads(path.read_text()) for path in sorted((args.output / "rows").glob("*.json"))])


def page_command(args):
    from listening_page import build
    from tiny_voca.infer import load_model
    from tiny_voca.text import phonemize

    model = load_model(args.checkpoint)
    c = model.config
    directory = args.output / "listen"
    directory.mkdir(parents=True, exist_ok=True)
    clips = []
    for name in args.rows:
        result = json.loads((args.output / "rows" / f"{name}.json").read_text())
        phones = phonemize(result["text"])
        wave, lengths, _ = model.synthesize(torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)]),
                                            torch.tensor([c.speakers.index(result["voice"])]), 1.0, 0.0, args.seed, args.pitch_range,
                                            noise_above=None)   # pulses alone: what the fitted head is compared with
        shipped = wave[0, :int(lengths[0])].double().numpy()
        level = float(np.sqrt(np.mean(shipped ** 2)))
        row = [("shipped", "shipped", "what ships today, speaking the sentence on its own", shipped),
               ("candidate", "shipped-copy", "identical to what ships today (the control)", shipped)]
        for variant in ("teacher", *args.variants):
            audio, _ = sf.read(args.output / "audio" / f"{name}-{variant}.wav", dtype="float64")
            # Matched in level to the shipped render, so loudness cannot decide.
            audio = audio * (level / np.sqrt(np.mean(audio ** 2)))
            row.append(("teacher", "teacher", "the teacher's recording", audio) if variant == "teacher"
                       else ("candidate", variant, VARIANTS[variant], audio))
        gain = 0.9 / max(float(np.abs(audio).max()) for _, _, _, audio in row)
        for role, variant, description, audio in row:
            label = role if role != "candidate" else hashlib.sha256(f"{name} {variant} {args.seed}".encode()).hexdigest()[:10]
            sf.write(directory / f"{name}-{label}.wav", audio * gain, c.sample_rate, subtype="PCM_16")
            clips.append({"row": name, "voice": result["voice"], "text": result["text"], "role": role, "variant": variant,
                          "description": description, "file": f"{name}-{label}.wav"})
        print(f"rendered {name} (row gain {20 * np.log10(gain):+.1f} dB)", flush=True)
    (directory / "manifest.jsonl").write_text("".join(json.dumps(clip, ensure_ascii=False) + "\n" for clip in clips), encoding="utf-8")
    intro = ("<p>Each row is one sentence the teacher recorded. <b>Teacher</b> is that recording and <b>Ships today</b> is the current "
             "model saying the sentence on its own. The lettered clips all follow the teacher's timing and melody, so they sound like "
             "the teacher's reading and not like <b>Ships today</b>'s; they are in a different order in every row, and one of them is "
             "secretly identical to <b>Ships today</b>.</p>"
             "<p>Listen only for the roughness or faint echo at the start and end of words. For each letter, say whether it has "
             "<b>more</b> of it (worse), the <b>same</b>, or <b>less</b> (better) than <b>Ships today</b>. A clip as clean as the "
             "teacher is \"better\". If a clip is wrong in some other way (hiss, buzz, a metallic tone), say so in the row's note; "
             "that matters too. Headphones, and one volume setting for the whole page.</p>")
    page = build(directory, args.title, intro, clips, seed=args.seed)
    print(f"wrote {len(clips)} clips and {page}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, command in (("fit", fit_command), ("summary", summary_command), ("page", page_command)):
        sub = commands.add_parser(name)
        sub.set_defaults(run=command)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--checkpoint", type=Path, default=ROOT / "es-1-anneal/runs/es-1-anneal/inference.pt")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--variants", nargs="+", default=["oracle"], choices=[v for v in VARIANTS if v != "forced"] if name == "fit" else list(VARIANTS))
        sub.add_argument("--rows", nargs="*", default=[], help="recordings to include by id (fit) or to show, in page order (page)")
        if name == "fit":
            sub.add_argument("--annotations", type=Path, default=ROOT / "es-1/annotations-es")
            sub.add_argument("--audio", type=Path, default=ROOT / "es-1/audio-es/audio", help="directory of <id>.flac teacher recordings")
            sub.add_argument("--critics", type=Path, default=ROOT / "es-1-anneal/runs/es-1-anneal/last.pt",
                             help="training checkpoint holding the critics, for the 'critics' variant")
            sub.add_argument("--splits", nargs="+", default=["validation", "test"])
            sub.add_argument("--per-voice", type=int, default=32)
            sub.add_argument("--steps", type=int, default=400)
            sub.add_argument("--rate", type=float, default=0.02)
            sub.add_argument("--workers", type=int, default=8)
            sub.add_argument("--again", action="store_true", help="refit recordings that already have a result")
        if name == "page":
            sub.add_argument("--pitch-range", type=float, default=1.5)
            sub.add_argument("--title", default="Head or model — blind listening")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
