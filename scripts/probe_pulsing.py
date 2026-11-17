#!/usr/bin/env python3
"""Does the model make noise-like sounds out of pitch pulses?

A voiceless consonant is noise, and the head has a noise branch for it. Nothing
stops the model from rendering it through the pulse train instead, and the
oscillator never stops. The high band then arrives in bursts at the pitch: for
a voice near 90 Hz that is a rattle, for one near 240 Hz much less of one.

Every render is the shipped model on the teacher's timing, with the teacher's
pitch in the oscillator (the state `probe_oracle.py` calls `forced`), so it is
compared with the recording frame by frame, per phone class and per voice:

    harmonic share  how much of the model's energy in a band comes from the
                    pulse-train branch. The model only: a recording has no
                    branches.
    pulsing         how much of the 2-8 kHz envelope repeats from one period of
                    the oscillator to the next: low for noise, towards 1 for
                    pulses. Recording, model, and the fitted head of
                    `probe_oracle.py` where it has one (`fitted`; `critics`
                    is its fit with the critics' features). The same measure at
                    1.37 times the period is the cross-check: noise scores the
                    same at both, and pulses at the pitch score low there.
    mel             distance to the recording, to see what a setting costs.
    low band        the first four harmonics, in the frames the model calls
                    unvoiced: how loud, and how much of it repeats at the
                    pitch. A voiced consonant the model calls unvoiced still
                    hums there, and noise in place of that hum is a rumble.

`--settings` adds renders with noise in place of pulses in the harmonic
branch's excitation (`noise_above` in `SourceFilter`), each given as
<unvoiced>,<voiced>: the number of harmonics of the pitch that stay pulses in
the frames the model calls unvoiced and in those it calls voiced. `2,inf`
keeps the fundamental as pulses where the model says unvoiced, turns the rest
to noise there, and leaves voiced frames alone; `2,33` also puts noise above
the 33rd harmonic of voiced frames, 3 kHz for a voice at 90 Hz and 8.3 kHz for
one at 250 Hz. `0,inf` is noise all the way down in unvoiced frames. A frame is
voiced where `synthesize` would say so: the model's own flag, or a vowel.

Only a voice against its own recordings means anything here.

    TINY_VOCA_LANGUAGE=es ./scripts/probe_pulsing.py --output ~/tiny-voca-artifacts/pulsing --settings 0,inf 2,inf 2,33

`--page` renders the settings as the model speaks on its own, matched in level,
and writes a blind listening page for the recordings named by `--rows`:

    TINY_VOCA_LANGUAGE=es ./scripts/probe_pulsing.py --output ~/tiny-voca-artifacts/pulsing --settings 2,inf 2,33 \
        --page ~/tiny-voca-artifacts/pulsing/listen --rows 69cf0694be230d63_M2 69cf0694be230d63_F2
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_oracle import ROOT, interval, log_mel, recording

BANDS = ((0, 12000), (300, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 11500))
HIGH = (2000, 8000)
CLASSES = ("voiceless fricative", "voiceless stop", "voiced consonant", "vowel", "vowel, pitch tracked", "vowel, no pitch", "pause")
OFF_PITCH = 1.37
LOW = (0.5, 4.5)        # the low band, in harmonics of the utterance's typical pitch
_state = {}


def phone_classes(row):
    from tiny_voca.text import CONSONANTS, FRICATIVES, SYMBOL_TO_ID, VOICELESS, VOWELS

    def owned(symbols):
        return np.isin(owner, [SYMBOL_TO_ID[s] for s in symbols])
    owner = np.repeat(np.asarray(row["tokens"]), row["durations"])
    fricative, voiceless, consonant, vowel = owned(VOICELESS & FRICATIVES), owned(VOICELESS), owned(CONSONANTS), owned(VOWELS)
    return {"voiceless fricative": fricative, "voiceless stop": voiceless & ~fricative, "voiced consonant": consonant & ~voiceless,
            "vowel": vowel, "pause": ~consonant & ~vowel}, vowel


def band_energy(wave, rate, frames, fft=1024, hop=256):
    """Energy per frame in each of BANDS: [frames, bands]."""
    power = torch.stft(torch.from_numpy(wave).float(), fft, hop, window=torch.hann_window(fft), return_complex=True).abs().double().numpy() ** 2
    hertz = np.arange(fft // 2 + 1) * rate / fft
    return np.stack([power[(hertz >= low) & (hertz < high)].sum(0)[:frames] for low, high in BANDS], axis=1)


def envelope(wave, rate, smooth=1000.0):
    """Envelope of the high band, keeping modulation up to `smooth` Hz."""
    from scipy import signal
    narrow = signal.sosfiltfilt(signal.butter(6, HIGH, btype="bandpass", fs=rate, output="sos"), wave.astype(np.float64))
    return signal.sosfiltfilt(signal.butter(4, smooth, fs=rate, output="sos"), np.abs(signal.hilbert(narrow)))


def low_band(wave, rate, hertz, hop):
    """The band of the first four harmonics of a pitch of `hertz`, and its energy per frame."""
    from scipy import signal
    narrow = signal.sosfiltfilt(signal.butter(4, (LOW[0] * hertz, LOW[1] * hertz), btype="bandpass", fs=rate, output="sos"), wave.astype(np.float64))
    return narrow, np.convolve(narrow ** 2, np.ones(2 * hop) / (2 * hop), "same")[::hop]


def weighted(values, weights):
    keep = np.isfinite(values) & (weights > 0)
    return float((values[keep] * weights[keep]).sum() / weights[keep].sum()) if keep.any() else None


def pulsing(outline, periods, hop, stretch=1.0, slack=0.05):
    """Per frame: the correlation of a signal (the high band's envelope, or the
    low band itself) over one period with the period after it, the pair centred
    on the frame, at the best lag within `slack` of the period. `periods` are
    the oscillator's, in samples."""
    reach = int(np.ceil(periods.max() * stretch * (1 + slack))) + 2
    padded = np.pad(outline, (reach, 2 * reach))
    scores = np.full(len(periods), np.nan)
    for frame, period in enumerate(periods * stretch):
        window = int(round(period))
        lags = np.arange(int(np.floor(period * (1 - slack))), int(np.ceil(period * (1 + slack))) + 1)
        start = frame * hop + reach - (window + lags) // 2
        first = padded[start[:, None] + np.arange(window)]
        second = padded[(start + lags)[:, None] + np.arange(window)]
        first, second = first - first.mean(1, keepdims=True), second - second.mean(1, keepdims=True)
        power = (first ** 2).sum(1) * (second ** 2).sum(1)
        if (power > 0).all():       # digital silence has no envelope to compare
            scores[frame] = ((first * second).sum(1) / np.sqrt(power)).max()
    return scores


def work(task):
    row, args = task
    torch.set_num_threads(1)
    from tiny_voca.infer import load_model
    from tiny_voca.losses import mel_filterbank

    if "model" not in _state:
        _state["model"] = load_model(args.checkpoint)
    model = _state["model"]
    c = model.config
    item = recording(row, args.audio, c)
    f0 = model.frequency(item["prosody"][None, :, 0])
    classes, vowel = phone_classes(row)
    tracked = item["tracked"].numpy()
    classes.update({"vowel, pitch tracked": vowel & tracked, "vowel, no pitch": vowel & ~tracked})
    with torch.no_grad():
        tokens, stress = torch.tensor([row["tokens"]]), torch.tensor([row["stress"]])
        speakers = torch.tensor([c.speakers.index(row["speaker"])])
        hidden, token_mask = model.encode(tokens, stress, torch.tensor([len(row["tokens"])]), speakers)
        frames, frame_mask, lengths = model.expand(hidden, torch.tensor([row["durations"]]), token_mask, speakers)
        prosody_hidden, prediction = model.predict_prosody(frames, frame_mask)
        acoustic = model.decode(prosody_hidden, model.hard_prosody(prediction), frame_mask)
        # The two streams `synthesize` draws for the first row of a batch.
        stream = torch.Generator().manual_seed(args.seed)
        noise, breath = (torch.randn(item["wave"].shape[-1], generator=stream)[None] for _ in range(2))
        full, harmonic, hiss = (model.vocoder(acoustic, f0, lengths, noise=noise, variant=v)[0].numpy()
                                for v in (None, "harmonic_only", "noise_only"))
        flagged = (prediction[0, :, 1] > 0).numpy()
        unvoiced = ~(flagged | vowel)
        voiced = torch.from_numpy(~unvoiced)[None]
        waves = {"recording": item["wave"].numpy(), "model": full}
        for setting in args.settings:
            low, high = (float(value) for value in setting.split(","))
            waves[setting] = model.vocoder(acoustic, f0, lengths, noise=noise, breath=breath,
                                           noise_above=torch.where(voiced, high, low))[0].numpy()
    for name, variant in (("fitted", "oracle"), ("critics", "critics")):
        render = args.oracle / "audio" / f"{row['id']}-{variant}.wav"
        if render.exists():
            waves[name] = sf.read(render, dtype="float32")[0]
    count, rate = row["frames"], c.sample_rate
    periods = rate / f0[0].numpy().astype(np.float64)
    energy = {name: band_energy(wave, rate, count) for name, wave in {**waves, "harmonic": harmonic, "hiss": hiss}.items()}
    high = [i for i, band in enumerate(BANDS) if HIGH[0] <= band[0] and band[1] <= HIGH[1]]
    bank, window = mel_filterbank(rate, c.n_fft, c.n_mels), torch.hann_window(c.n_fft)
    mel = {name: log_mel(torch.from_numpy(wave)[None], c, bank, window)[:count] for name, wave in waves.items()}
    scores, typical = {}, float(np.median(f0[0].numpy()))
    for name, wave in waves.items():
        outline = envelope(wave, rate)
        narrow, strength = low_band(wave, rate, typical, c.hop)
        scores[name] = (pulsing(outline, periods, c.hop), pulsing(outline, periods, c.hop, OFF_PITCH),
                        (mel[name] - mel["recording"]).abs().mean(1).numpy(), pulsing(narrow, periods, c.hop), strength[:count])
    result = {"id": row["id"], "voice": row["speaker"], "hertz": float(np.exp(np.mean(np.log(f0[0].numpy()[tracked])))) if tracked.any() else None,
              "classes": {}}
    whole = energy["recording"][:, 0].mean()
    for label, member in classes.items():
        if not member.any():
            continue
        entry = {"frames": int(member.sum()), "tracked": float(tracked[member].mean()), "flagged": float(flagged[member].mean()),
                 "share": [float(energy["harmonic"][member, band].sum() / (energy["harmonic"][member, band].sum() + energy["hiss"][member, band].sum() + 1e-20))
                           for band in range(len(BANDS))],
                 # The recording's high band here against its own level over the whole utterance: is it loud enough to matter?
                 "presence_db": float(10 * np.log10(energy["recording"][member][:, high].sum(1).mean() / whole + 1e-12))}
        quiet = member & unvoiced           # where the setting for unvoiced frames acts
        for name in waves:
            on, off, distance, tone, strength = scores[name]
            entry[name] = {"pulsing": float(np.nanmean(on[member])) if np.isfinite(on[member]).any() else None,
                           "off_pitch": float(np.nanmean(off[member])) if np.isfinite(off[member]).any() else None,
                           "mel": float(distance[member].mean()),
                           "level_db": float(10 * np.log10((energy[name][member][:, high].sum() + 1e-20) / (energy["recording"][member][:, high].sum() + 1e-20))),
                           # The low band against the recording's over the whole utterance, and how much of its energy is a tone at the pitch.
                           "low_db": float(10 * np.log10(strength[quiet].mean() / scores["recording"][4].mean() + 1e-12)) if quiet.any() else None,
                           "low_tone": weighted(tone[quiet], strength[quiet])}
        result["classes"][label] = entry
    return result


def difference(mine, label, name, key):
    """Paired `name` minus recording over the utterances that have both."""
    from scipy import stats
    pairs = [(r["classes"][label][name][key], r["classes"][label]["recording"][key]) for r in mine
             if label in r["classes"] and name in r["classes"][label]
             and r["classes"][label][name][key] is not None and r["classes"][label]["recording"][key] is not None]
    gaps = np.array([a - b for a, b in pairs])
    n, mean, half = interval(gaps)
    p = stats.wilcoxon(gaps).pvalue if n >= 3 and np.any(gaps != 0) else float("nan")
    return f"{mean:+.3f} ±{half:.3f} p={p:<8.2g} {np.mean(gaps > 0) if n else float('nan'):4.0%} above"


def summarize(results):
    for voice in sorted({r["voice"] for r in results}):
        mine = [r for r in results if r["voice"] == voice]
        names = [name for name in mine[0]["classes"]["vowel"] if isinstance(mine[0]["classes"]["vowel"][name], dict)]
        print(f"\n{voice}: {len(mine)} recordings, {np.exp(np.mean([np.log(r['hertz']) for r in mine if r['hertz']])):.0f} Hz")
        total = sum(entry["frames"] for r in mine for label, entry in r["classes"].items() if "," not in label)

        def mean(label, pick):
            return interval([pick(r["classes"][label]) for r in mine if label in r["classes"]])[1]
        print("  frames, who calls them voiced, and the recording's 2-8 kHz level there against its whole-utterance level")
        print(f"    {'':22}{'share':>8}{'recording':>11}{'model':>8}{'level dB':>10}")
        for label in CLASSES:
            frames = sum(r["classes"][label]["frames"] for r in mine if label in r["classes"])
            print(f"    {label:22}{frames / total:8.1%}{mean(label, lambda e: e['tracked']):11.1%}{mean(label, lambda e: e['flagged']):8.1%}"
                  f"{mean(label, lambda e: e['presence_db']):10.1f}")
        print("  pulse-train branch's share of the model's energy, by band (Hz)")
        print(f"    {'':22}" + "".join(f"{f'{low}-{high}':>12}" for low, high in BANDS))
        for label in CLASSES:
            print(f"    {label:22}" + "".join(f"{mean(label, lambda e, band=band: e['share'][band]):12.1%}" for band in range(len(BANDS))))
        for key, title in (("pulsing", "pulsing: how much of the 2-8 kHz envelope repeats at the oscillator's period (low noise, 1 pulses)"),
                           ("off_pitch", f"cross-check: the same at {OFF_PITCH} times the period (noise scores as above, pulses low)"),
                           ("mel", "distance to the recording (mean |log-mel difference|, per frame)"),
                           ("level_db", "2-8 kHz level against the recording (dB)"),
                           ("low_db", "low band (the first four harmonics) where the model says unvoiced: level against the recording's whole-utterance low band (dB)"),
                           ("low_tone", "low band where the model says unvoiced: how much of its energy repeats at the oscillator's period (noise low, a hum 1)")):
            if key not in mine[0]["classes"]["pause"]["recording"]:
                continue                    # a run from before the low band was measured
            labels = [label for label in CLASSES if "vowel" not in label] if key.startswith("low") else CLASSES
            print(f"  {title}")
            print(f"    {'':22}" + "".join(f"{name:>11}" for name in names))
            for label in labels:
                print(f"    {label:22}" + "".join(f"{mean(label, lambda e, name=name: (e.get(name) or {}).get(key)):11.3f}" for name in names))
            if key in ("pulsing", "off_pitch", "low_tone"):
                print("    against the recording, paired over recordings")
                for label in labels:
                    for name in names[1:]:
                        print(f"    {label:22}{name:>11}   {difference(mine, label, name, key)}")


def cutoffs(setting):
    low, _, high = setting.partition(",")
    return float(low), float(high)


def describe(setting):
    def side(harmonics, where):
        if harmonics == np.inf:
            return f"pulses {where}"
        return f"noise {where}" if harmonics == 0 else f"noise above harmonic {harmonics:g} {where}"
    low, high = cutoffs(setting)
    return f"excitation of the pitched branch: {side(low, 'where the model says unvoiced')}, {side(high, 'where it says voiced')}"


def page(args, rows):
    """The settings as the model speaks on its own, against what ships."""
    from listening_page import build
    from tiny_voca.infer import load_model
    from tiny_voca.text import phonemize

    model = load_model(args.checkpoint)
    c = model.config
    args.page.mkdir(parents=True, exist_ok=True)
    clips = []
    for name in args.rows:
        row = rows[name]
        phones = phonemize(row["text"])
        inputs = (torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)]),
                  torch.tensor([c.speakers.index(row["speaker"])]), 1.0, 0.0, args.seed)

        def render(setting=None):
            with torch.no_grad():
                wave, lengths, _ = model.synthesize(*inputs, noise_above=setting and cutoffs(setting))
            return wave[0, :int(lengths[0])].double().numpy()
        shipped = render()
        level = float(np.sqrt(np.mean(shipped ** 2)))
        teacher, _ = sf.read(args.audio / f"{name}.flac", dtype="float64")
        group = [("teacher", "teacher", "the teacher's recording", teacher),
                 ("shipped", "shipped", "what ships today: pulses throughout", shipped),
                 ("candidate", "shipped-copy", "identical to what ships today (the control)", shipped)]
        group += [("candidate", setting, describe(setting), render(setting)) for setting in args.settings]
        # Matched in level to the shipped render, so loudness cannot decide; then
        # one gain for the row, set by its loudest sample, so nothing clips.
        group = [(role, variant, description, wave * (level / np.sqrt(np.mean(wave ** 2)))) for role, variant, description, wave in group]
        gain = 0.9 / max(float(np.abs(wave).max()) for _, _, _, wave in group)
        for role, variant, description, wave in group:
            # Candidate files carry no hint of what they are.
            label = role if role != "candidate" else hashlib.sha256(f"{name} {variant} {args.seed}".encode()).hexdigest()[:10]
            sf.write(args.page / f"{name}-{label}.wav", wave * gain, c.sample_rate, subtype="PCM_16")
            clips.append({"row": name, "voice": row["speaker"], "text": row["text"], "role": role, "variant": variant,
                          "description": description, "file": f"{name}-{label}.wav"})
        print(f"rendered {name} (row gain {20 * np.log10(gain):+.1f} dB)", flush=True)
    (args.page / "manifest.jsonl").write_text("".join(json.dumps(clip, ensure_ascii=False) + "\n" for clip in clips), encoding="utf-8")
    intro = ("<p>Each row is one sentence. <b>Teacher</b> is the recording the model learned from and <b>Ships today</b> is the "
             "current model. The lettered clips are the same sentence from the same model with one thing changed in how the sound "
             "is made; they are in a different order in every row, and one of them is secretly identical to <b>Ships today</b>.</p>"
             "<p>Listen for the roughness or faint echo at the start and end of words. For each letter, say whether it has "
             "<b>more</b> of it (worse), the <b>same</b>, or <b>less</b> (better) than <b>Ships today</b>. If a clip is wrong in "
             "some other way (hiss, breathiness, a whisper where there should be voice), say so in the row's note; that matters "
             "as much. Headphones, and one volume setting for the whole page.</p>")
    print(f"wrote {len(clips)} clips and {build(args.page, args.title, intro, clips, seed=args.seed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle",
                        help="a `probe_oracle.py fit` output: its recordings are measured, and its fitted renders with them")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "es-1-anneal/runs/es-1-anneal/inference.pt")
    parser.add_argument("--annotations", type=Path, default=ROOT / "es-1/annotations-es")
    parser.add_argument("--audio", type=Path, default=ROOT / "es-1/audio-es/audio", help="directory of <id>.flac teacher recordings")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--settings", nargs="*", default=[], help="noise excitation to render as well: <unvoiced>,<voiced> harmonics kept as pulses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summary", action="store_true", help="print the tables of an earlier run again")
    parser.add_argument("--page", type=Path, help="write a blind listening page of the settings here instead of measuring")
    parser.add_argument("--rows", nargs="+", default=[], help="recordings on the page, by id and in page order")
    parser.add_argument("--title", default="Noise in place of pulses — blind listening")
    args = parser.parse_args()
    for setting in args.settings:
        cutoffs(setting)
    target = args.output / "rows.json"
    if args.summary:
        return summarize(json.loads(target.read_text()))
    rows = []
    for split in args.splits:
        rows += [json.loads(line) for line in (args.annotations / f"{split}.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    if args.page:
        return page(args, {row["id"]: row for row in rows})
    wanted = {path.stem for path in (args.oracle / "rows").glob("*.json")}
    rows = sorted((row for row in rows if row["id"] in wanted), key=lambda row: row["id"])
    if len(rows) != len(wanted):
        raise SystemExit(f"{len(wanted) - len(rows)} of the fitted recordings are not in the annotations")
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(work, [(row, args) for row in rows]):
            results.append(result)
            print(f"{len(results):3d}/{len(rows)} {result['id']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    summarize(results)


if __name__ == "__main__":
    main()
