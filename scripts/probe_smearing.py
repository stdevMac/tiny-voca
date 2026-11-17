"""Test whether the synthesis head's window is what smears word edges.

The vocoder is not neural: `SourceFilter` shapes a band-limited pulse train with
a cepstral envelope applied through a 512-sample Hann STFT (21 ms) at a 128
sample hop. Every gain change is therefore cross-faded over 21 ms, and the pulse
train runs through silence, so voiced energy rings into the boundary either side
of it. A listener hears that as a timbre at the start and end of each word.

This compares three variants of the same weights, same sentence, same seed:

  baseline   the shipped head
  short      the same filter with a 256-sample window zero-padded to 512
  short+gate the same, and the pulse train is muted where the model says the
             frame is unvoiced, with a 2 ms ramp

Nothing is retrained: the cepstral envelope does not depend on the FFT size, so
the existing weights apply to all three. Writes WAVs next to a JSON of the
measurements so the difference can be heard and read.

    python scripts/probe_smearing.py --checkpoint <inference.pt> --output /tmp/smear
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.infer import load_model
from tiny_voca.model import cepstrum_response, filtered, pulse_train
from tiny_voca.text import phonemize

TEXT = "Por eso las empresas debían adoptar este nuevo sistema."


def filtered_short(source, response, fft, hop, window):
    """`filtered` with a shorter analysis window, zero-padded to the same FFT."""
    win = torch.hann_window(window.shape[-1] // 2, device=source.device, dtype=source.dtype)
    spectrum = torch.stft(source, fft, hop, win_length=win.shape[-1], window=win,
                          center=True, pad_mode="constant", return_complex=True)
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, win_length=win.shape[-1],
                       window=win, center=True, length=source.shape[-1])


def vocoder_variant(model, acoustic, f0, lengths, noise, voiced, short=False, gate=False):
    """A copy of `SourceFilter.forward` with the two switches."""
    c = model.config
    v = model.vocoder
    frames = acoustic.shape[1]
    samples = frames * c.hop
    audible = torch.ones(samples, device=acoustic.device).expand(len(lengths), -1)
    for row, n in enumerate(lengths):
        audible[row, int(n) * c.hop:] = 0
    last = (lengths - 1).clamp_min(0)[:, None]

    h = v.harmonic(acoustic)
    split = c.cep_causal + c.cep_anti
    sub = v._subframes(frames, lengths, v.harmonic_steps)
    here, following = sub // v.harmonic_steps, torch.minimum((sub + 1) // v.harmonic_steps, last)
    causal = 0.5 * (v._take(h[..., :c.cep_causal], here) + v._take(h[..., :c.cep_causal], following)) * v.causal_lifter
    anti = 0.5 * (v._take(h[..., c.cep_causal:split], here) + v._take(h[..., c.cep_causal:split], following)) * v.anti_lifter
    causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
    harmonic_response = cepstrum_response(causal, anti, c.synth_fft)

    n = v.noise(acoustic)
    sub = v._subframes(frames, lengths, v.noise_steps)
    shape = v._take(n[..., :c.noise_cep], sub // v.noise_steps) * v.noise_lifter
    shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
    noise_response = cepstrum_response(shape, None, c.noise_fft)

    f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
    pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, None) * audible
    if gate:
        # Mute the oscillator where the model says the frame is unvoiced, ramped
        # over 2 ms so the mute itself does not click.
        gate_wave = F.interpolate(voiced[:, None].float(), size=samples, mode="linear", align_corners=False)[:, 0]
        ramp = max(1, int(0.002 * c.sample_rate))
        gate_wave = F.avg_pool1d(F.pad(gate_wave[:, None], (ramp, ramp), mode="replicate"), 2 * ramp + 1, stride=1)[:, 0]
        pulses = pulses * gate_wave
    if noise is None:
        noise = torch.randn(acoustic.shape[0], samples, device=acoustic.device)

    shape_fn = filtered_short if short else filtered
    wave = (v._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, v.synth_window) if not short else
            filtered_short(F.pad(pulses, (0, c.synth_fft // 2)), harmonic_response, c.synth_fft, c.synth_hop, v.synth_window)[:, :samples])
    wave = wave + (filtered_short(F.pad(noise * audible, (0, c.noise_fft // 2)), noise_response, c.noise_fft, c.noise_hop, v.noise_window)[:, :samples]
                   if short else v._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, v.noise_window))
    return wave * audible


def leak(wave, sample_rate, hop):
    """Energy at the pitch harmonics in the quietest frames, in dB below speech.

    A smeared boundary leaves voiced energy where the frame is meant to be
    silent; that energy is pitched, so it hides from spectral flatness.
    """
    import librosa
    w = np.asarray(wave, dtype=np.float32)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=hop))**2
    energy = S.sum(axis=0)
    quiet = energy <= np.percentile(energy, 15)
    loud = energy >= np.percentile(energy, 60)
    return float(10*np.log10(energy[quiet].mean() / energy[loud].mean() + 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)
    c = model.config
    phones = phonemize(TEXT)
    tokens = torch.tensor([phones.ids], device=args.device)
    stress = torch.tensor([phones.stress], device=args.device)
    speaker = torch.tensor([0], device=args.device)
    lengths = torch.tensor([len(phones)], device=args.device)

    with torch.inference_mode():
        hidden, mask = model.encode(tokens, stress, lengths, speaker)
        durations = model.predict_durations(hidden, mask, tokens, 1.0)
        frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, speaker)
        prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
        ends = durations.cumsum(1)
        pos = torch.arange(frames.shape[1], device=args.device)[None].expand(len(lengths), -1).contiguous()
        owner = torch.searchsorted(ends, pos, right=True).clamp(max=tokens.shape[1] - 1)
        vowel = model.vowel_tokens[tokens].gather(1, owner) & frame_mask
        prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                             prosody[..., 2:]), dim=-1)
        prosody = model.widen_pitch(prosody, frame_mask, 1.5)
        acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
        f0 = model.frequency(prosody[..., 0])
        voiced = (prosody[..., 1] > 0).float() * frame_mask.float()
        noise = torch.randn(1, int(frame_lengths[0]) * c.hop, device=args.device,
                            generator=torch.Generator().manual_seed(args.seed))

        results = {}
        for name, kwargs in (("baseline", {}), ("short", {"short": True}), ("short+gate", {"short": True, "gate": True})):
            wave = vocoder_variant(model, acoustic, f0, frame_lengths, noise, voiced, **kwargs)
            w = wave[0, :int(frame_lengths[0]) * c.hop].detach().cpu().numpy()
            peak = float(np.abs(w).max()) or 1.0
            sf.write(args.output / f"{name}.wav", w / peak * 0.95, c.sample_rate, subtype="PCM_16")
            results[name] = {"quiet_to_speech_db": round(leak(w, c.sample_rate, c.hop), 1)}
            print(json.dumps({"variant": name, **results[name]}), flush=True)

    (args.output / "measurements.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
