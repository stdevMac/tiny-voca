"""tiny-voca v2: feed-forward transformer with a pitch-driven source-filter head.

All weights start random. Tensor conventions: hidden [B,T,C], frame features
[B,T], waveform [B,S]. Acoustic frame i is centred on sample ``i * hop``.

    phoneme + stress + voice -> transformer encoder -> duration predictor
        -> repeat tokens by duration (supervised durations while training)
        -> frame transformer A -> pitch / voicing / energy
        -> frame transformer B -> filter parameters
        -> harmonic pulse train (from pitch) and noise, each shaped by its
           time-varying filter in the STFT domain -> 24 kHz waveform

The waveform head has no neural upsampling network. The oscillator supplies the
periodicity that a tiny decoder cannot render reliably on its own; the network
only has to predict slowly varying spectral envelopes (complex cepstra), which
is what keeps the voice from sounding buzzy or unstable at this size.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config
from .text import LANGUAGE, PHONE_IDS, SYMBOL_TO_ID, VOWELS


class UtteranceTooLong(ValueError):
    """The utterance needs more frames than ``max_frames``: split the input."""


def length_mask(lengths, size=None):
    size = int(lengths.max()) if size is None else size
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def rotary_table(size, width, device):
    inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width))
    angle = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inverse[None]
    return angle.cos(), angle.sin()


def rotate(x, table):
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Self-attention with rotary (relative) positions: no absolute position
    table, so utterances longer than those seen in training stay in range."""

    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, mask, table):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotate(q, table), rotate(k, table), v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.channels)
        self.attention = Attention(c.channels, c.heads, c.dropout)
        self.norm2 = nn.LayerNorm(c.channels)
        self.depthwise = nn.Conv1d(c.channels, c.channels, c.kernel, padding=c.kernel // 2, groups=c.channels)
        self.expand = nn.Linear(c.channels, c.feedforward)
        self.project = nn.Linear(c.feedforward, c.channels)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, mask, table):
        keep = mask[..., None]
        x = (x + self.dropout(self.attention(self.norm1(x), mask, table))) * keep
        # Mask before the convolution so padded positions cannot leak back in.
        y = self.depthwise((self.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
        y = self.project(self.dropout(F.gelu(self.expand(y))))
        return (x + self.dropout(y)) * keep


class Stack(nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.blocks = nn.ModuleList(Block(c) for _ in range(layers))
        self.norm = nn.LayerNorm(c.channels)
        self.width = c.channels // c.heads

    def forward(self, x, mask):
        table = rotary_table(x.shape[1], self.width, x.device)
        for block in self.blocks:
            x = block(x, mask, table)
        return self.norm(x) * mask[..., None]


class Predictor(nn.Module):
    def __init__(self, c, outputs):
        super().__init__()
        p = c.predictor_channels
        self.conv1 = nn.Conv1d(c.channels, p, 3, padding=1)
        self.conv2 = nn.Conv1d(p, p, 3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(p), nn.LayerNorm(p)
        self.drop = nn.Dropout(c.dropout)
        self.out = nn.Linear(p, outputs)

    def forward(self, x, mask):
        x = self.conv1((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm1(F.gelu(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm2(F.gelu(x)))
        return self.out(x) * mask[..., None]


def harmonic_phases(count, beta=0.0, scatter=0.0, seed=0, power=2.0):
    """Phase offsets [count] for harmonics 1..count, for ``pulse_train``.

    A pulse train has every harmonic at phase zero, so each period's energy sits
    in one instant. ``-beta * k**2`` bends the phase: harmonic k is delayed by
    ``beta * k / pi`` of a period and the pulse becomes a sweep. With K harmonics,
    ``beta = pi / K`` spreads it over exactly one period (Schroeder's low-peak
    phases); a negative beta sweeps the other way. A term linear in k would do
    nothing: ``cos(k*x - b*k)`` is ``cos(k*(x - b))``, the same pulse later.

    ``power`` is the exponent in ``-beta * k**power``, and the delay of harmonic
    k is ``beta * power * k**(power - 1) / (2 * pi)`` of a period. At 2 the delay
    grows by the same amount per harmonic, so nearly all of the spread lands on
    the high ones; nearer 1 it grows by the same amount per octave, which is how
    the ear's own bands are spaced.

    ``scatter`` adds a fixed pseudo-random offset within ``±scatter * pi`` to
    each harmonic, the same in every period of the utterance."""
    k = torch.arange(1, count + 1, dtype=torch.float64)
    offsets = -beta * k ** power
    if scatter:
        generator = torch.Generator().manual_seed(seed + 104729)
        offsets = offsets + scatter * math.pi * (2 * torch.rand(count, generator=generator, dtype=torch.float64) - 1)
    return offsets


def pulse_train(f0, sample_rate, hop, limit=0.47, phase=None, harmonic_phase=None):
    """Band-limited pulse train [B, T*hop] with unit power for f0 [B, T] in Hz.

    Frame values sit at samples i*hop and are interpolated in the log domain.
    The sum of all harmonics below ``limit * sample_rate`` has the closed form
    sin((K+1/2)x) / (2 sin(x/2)) - 1/2; the top harmonic fades in and out so the
    signal stays continuous while the pitch glides. Float64 keeps the phase exact.

    ``harmonic_phase`` [K] or [B, K] gives harmonic k its own phase offset (see
    ``harmonic_phases``). The harmonics are then summed one by one with the same
    amplitudes as the closed form, so the magnitude spectrum is unchanged and
    only the shape of the period moves. None keeps the closed form.
    """
    frames = f0.shape[1]
    knots = torch.log(f0.double().clamp_min(1.0))
    knots = torch.cat((knots, knots[:, -1:]), dim=1)
    position = torch.arange(frames * hop, device=f0.device, dtype=torch.float64) / hop
    low = position.floor().long()
    weight = position - low
    frequency = torch.exp(knots[:, low] * (1 - weight) + knots[:, low + 1] * weight)
    theta = 2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1)
    if phase is not None:
        theta = theta + phase.double()[:, None]
    theta = torch.remainder(theta, 2 * math.pi)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    if harmonic_phase is not None:
        count = int(whole.max()) + 1
        offsets = harmonic_phase.to(theta)
        if offsets.shape[-1] < count:
            raise ValueError(f"harmonic_phase covers {offsets.shape[-1]} harmonics and this pitch needs {count}")
        wave = torch.zeros_like(theta)
        for k in range(1, count + 1):
            # Harmonic k is whole while it is under the limit and fades as it
            # crosses it: the closed form's amplitudes, written out.
            wave = wave + (harmonics - (k - 1)).clamp(0.0, 1.0) * torch.cos(k * theta + offsets[..., k - 1:k])
        return (wave * torch.sqrt(2.0 / harmonics)).float()
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def filtered(source, response, fft, hop, window):
    """Time-varying filtering: response [B, frames, bins] multiplies the STFT."""
    spectrum = torch.stft(source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, window=window, center=True, length=source.shape[-1])


def cepstrum_response(causal, anticausal, fft):
    """Complex frequency response exp(FFT(cepstrum)). Causal coefficients give a
    minimum-phase filter; anticausal ones add a maximum-phase part (glottal pulse)."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., :causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1]:] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return torch.polar(torch.exp(spectrum.real.clamp(-14, 6)), spectrum.imag)


# What inference renders from noise instead of pulses: the harmonics of the pitch
# above this number, in the frames the model calls unvoiced and in those it calls
# voiced (see ``SourceFilter.forward``). The weights were trained with pulses
# alone. At a 90 Hz pitch the pulses are 11 ms apart, and breath, frication and
# silence rendered with them beat at the pitch; a listener heard that as roughness
# at the edges of words and preferred this setting in every row of a blind page
# (docs/word-edge-artifact.md, section 11). Spanish only: nobody has listened to
# the English model with it, so English keeps pulses alone, as it shipped.
NOISE_ABOVE = (2.0, 33.0) if LANGUAGE == "es" else None


class SourceFilter(nn.Module):
    """Harmonic-plus-noise synthesis controlled per frame by two linear heads."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.harmonic_steps, self.noise_steps = c.hop // c.synth_hop, c.hop // c.noise_hop
        self.harmonic = nn.Linear(c.channels, c.cep_causal + c.cep_anti + self.harmonic_steps)
        self.noise = nn.Linear(c.channels, c.noise_cep + self.noise_steps)
        for head, level in ((self.harmonic, -1.5), (self.noise, -2.5)):   # start quiet and spectrally flat
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[0] = level

        def lifter(size, first):
            scale = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4)
            if first == 0:
                scale[0] = 2.0      # coefficient 0 is the log gain and needs range
            return scale
        self.register_buffer("causal_lifter", lifter(c.cep_causal, 0), persistent=False)
        self.register_buffer("anti_lifter", lifter(c.cep_anti, 1), persistent=False)
        self.register_buffer("noise_lifter", lifter(c.noise_cep, 0), persistent=False)
        self.register_buffer("synth_window", torch.hann_window(c.synth_fft), persistent=False)
        self.register_buffer("noise_window", torch.hann_window(c.noise_fft), persistent=False)

    @staticmethod
    def _take(values, index):
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))

    def _subframes(self, frames, lengths, steps):
        """Sub-frame j uses frame j // steps; indices past an utterance's end
        repeat its last sub-frame so padding never changes real audio. Two extra
        sub-frames cover the half window of zeros appended to the source."""
        sub = torch.arange(steps * frames + 3, device=lengths.device)[None].expand(len(lengths), -1)
        return torch.minimum(sub, (steps * lengths - 1).clamp_min(0)[:, None])

    @staticmethod
    def _shape(source, response, fft, hop, window):
        # Half a window of trailing zeros: the frames around an utterance's end are
        # then the same whether it stands alone or is padded inside a batch.
        return filtered(F.pad(source, (0, fft // 2)), response, fft, hop, window)[:, :source.shape[-1]]

    def _noise_share(self, harmonics, f0, here, following):
        """How the harmonic branch's excitation is split between pulses and noise,
        per sub-frame and bin: the share of the power that is noise, and the band
        the noise may fill. `harmonics` [B, T] counts the harmonics of the pitch
        that stay pulses; the share rises from 0 to 1 over the octave centred on
        the last of them, so 0 is all noise and infinity all pulses. The noise
        stays where the pulse train has harmonics, from half the pitch up to the
        band limit: the filter was never trained on anything else."""
        c = self.config
        hertz = torch.arange(c.synth_fft // 2 + 1, device=f0.device) * (c.sample_rate / c.synth_fft)
        octaves = torch.log2(hertz.clamp_min(1e-3) / (harmonics.float() * f0).clamp_min(1e-6)[..., None]).clamp(-0.5, 0.5)
        share = 0.5 * (1 + torch.sin(math.pi * octaves))
        band = (hertz >= 0.5 * f0[..., None]) & (hertz <= c.harmonic_limit * c.sample_rate)
        share, band = (0.5 * (self._take(value, here) + self._take(value, following)) for value in (share, band.float()))
        return share, band

    def forward(self, hidden, f0, lengths, noise=None, phase=None, variant=None, anticausal_scale=1.0, harmonic_phase=None,
                noise_above=None, breath=None):
        """``noise_above`` [B, T] is a harmonic number: above that harmonic of the
        frame's pitch the harmonic branch is driven by noise (``breath``
        [B, samples], drawn here if absent) in place of pulses,
        with the same power in every band and through the same filter. None is
        the pulse train alone, which is what the weights were trained with and
        what training still uses; ``TinyVoca.synthesize`` passes ``NOISE_ABOVE``."""
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]
            # Harmonic branch: envelope interpolated between frames, one gain per sub-frame.
            h = self.harmonic(hidden)
            split = c.cep_causal + c.cep_anti
            sub = self._subframes(frames, lengths, self.harmonic_steps)
            here, following = sub // self.harmonic_steps, torch.minimum((sub + 1) // self.harmonic_steps, last)
            causal = 0.5 * (self._take(h[..., :c.cep_causal], here) + self._take(h[..., :c.cep_causal], following)) * self.causal_lifter
            anti = 0.5 * (self._take(h[..., c.cep_causal:split], here) + self._take(h[..., c.cep_causal:split], following)) * self.anti_lifter
            # The anticausal half of the cepstrum is the only part of the filter that
            # can put energy before a pulse, i.e. the only part that can pre-echo. It
            # is scaled rather than switched so a partial setting can keep the timbre
            # it was trained for while removing the smear.
            if anticausal_scale != 1.0:
                anti = anti * anticausal_scale
            if variant == "minimum_phase":
                anti = torch.zeros_like(anti)
            causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
            harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
            # Noise branch: one envelope per frame, four gains per frame for sharp consonants.
            n = self.noise(hidden)
            sub = self._subframes(frames, lengths, self.noise_steps)
            shape = self._take(n[..., :c.noise_cep], sub // self.noise_steps) * self.noise_lifter
            shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
            noise_response = cepstrum_response(shape, None, c.noise_fft)
            # Hold the last real pitch value through padding: the pitch glide inside an
            # utterance's final frame must not depend on what the batch was padded with.
            f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
            # Diagnosing the word-edge artifact: the pulse train puts every harmonic
            # at phase zero. `harmonic_phase` disperses them without touching the
            # magnitudes; None is the pulse train the weights were trained with.
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase, harmonic_phase) * audible
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            # Diagnosing the word-edge artifact: render one branch at a time, so a
            # listener can say which of the two carries it.
            if variant == "harmonic_only":
                noise = torch.zeros_like(noise)
            elif variant == "noise_only":
                pulses = torch.zeros_like(pulses)
            if noise_above is None:
                harmonic = self._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window)
            else:
                # A 90 Hz pulse train can only put energy near its pulses, so whatever
                # the model renders with it arrives in bursts, noise-like sounds
                # included. Here those bands are excited by noise instead. A pulse
                # train of unit power spreads it evenly up to the band limit; white
                # noise of power 0.5 / limit has the same density.
                share, band = self._noise_share(noise_above, f0, here, following)
                if breath is None:
                    breath = torch.randn(hidden.shape[0], samples, device=hidden.device)
                if variant == "noise_only":
                    breath = torch.zeros_like(breath)
                harmonic = (self._shape(pulses, harmonic_response * torch.sqrt(1 - share), c.synth_fft, c.synth_hop, self.synth_window)
                            + self._shape(breath * audible * math.sqrt(0.5 / c.harmonic_limit), harmonic_response * torch.sqrt(share) * band,
                                          c.synth_fft, c.synth_hop, self.synth_window))
            wave = harmonic + self._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window)
            return wave * audible


class TinyVoca(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        self.stress = nn.Embedding(3, c.channels)
        self.voices = nn.Embedding(len(c.speakers), c.channels)
        self.encoder = Stack(c, c.encoder_layers)
        # Outputs per token: log(1 + frames) and, for pause-capable tokens, a
        # "has frames at all" logit. A pause is absent more often than present, so
        # one regression would average both cases into a pause that is too short.
        self.duration = Predictor(c, 2)
        self.frame_voices = nn.Embedding(len(c.speakers), c.channels)
        self.frame_position = nn.Linear(3, c.channels)
        self.prosody_stack = Stack(c, c.prosody_layers)
        # Outputs: continuous log(F0/reference), voiced logit, normalized log RMS energy.
        self.prosody_head = nn.Linear(c.channels, 3)
        self.prosody_projection = nn.Linear(3, c.channels)
        self.acoustic_stack = Stack(c, c.acoustic_layers)
        self.vocoder = SourceFilter(c)
        phones = torch.zeros(c.vocab_size, dtype=torch.bool)
        phones[sorted(PHONE_IDS)] = True
        self.register_buffer("phone_tokens", phones, persistent=False)
        vowels = torch.zeros(c.vocab_size, dtype=torch.bool)
        vowels[[SYMBOL_TO_ID[v] for v in VOWELS]] = True
        self.register_buffer("vowel_tokens", vowels, persistent=False)

    def encode(self, tokens, stress, lengths, speakers):
        if tokens.ndim != 2 or stress.shape != tokens.shape or lengths.shape != (tokens.shape[0],) or speakers.shape != lengths.shape:
            raise ValueError("expected tokens[B,N], stress[B,N], lengths[B], speakers[B]")
        if tokens.shape[1] > self.config.max_tokens or bool((lengths < 1).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("invalid token length")
        mask = length_mask(lengths, tokens.shape[1])
        x = (self.tokens(tokens) + self.stress(stress) + self.voices(speakers)[:, None]) * mask[..., None]
        return self.encoder(x, mask), mask

    def expand(self, hidden, durations, token_mask, speakers):
        """Repeat token states by integer durations (zero is allowed for pause
        tokens) and add each frame's position inside its token."""
        if durations.shape != token_mask.shape or bool((durations < 0).any()) or bool((durations[~token_mask] != 0).any()):
            raise ValueError("invalid durations")
        lengths = durations.sum(1)
        if bool((lengths < 1).any()):
            raise ValueError("utterance has no frames")
        if bool((lengths > self.config.max_frames).any()):
            raise UtteranceTooLong("utterance exceeds max_frames; split the input")
        size = int(lengths.max())
        ends = durations.cumsum(1)
        frame = torch.arange(size, device=hidden.device)[None].expand(hidden.shape[0], size).contiguous()
        index = torch.searchsorted(ends, frame, right=True).clamp(max=durations.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        mask = frame < lengths[:, None]
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + self.frame_position(features) + self.frame_voices(speakers)[:, None]
        return expanded * mask[..., None], mask, lengths

    def predict_prosody(self, frames, frame_mask):
        hidden = self.prosody_stack(frames, frame_mask)
        return hidden, self.prosody_head(hidden) * frame_mask[..., None]

    @staticmethod
    def hard_prosody(prediction):
        """What the decoder is conditioned on when it is not teacher-forced."""
        return torch.stack((prediction[..., 0].clamp(-2.0, 1.5), (prediction[..., 1] > 0).to(prediction.dtype),
                            prediction[..., 2].clamp(-4.5, 3.0)), dim=-1)

    def decode(self, prosody_hidden, condition, frame_mask):
        return self.acoustic_stack((prosody_hidden + self.prosody_projection(condition)) * frame_mask[..., None], frame_mask)

    def frequency(self, log_f0):
        c = self.config
        return (c.pitch_reference_hz * torch.exp(log_f0.float())).clamp(c.f0_min, c.f0_max)

    def predict_durations(self, hidden, mask, tokens, speed=1.0):
        output = self.duration(hidden, mask).float()
        frames = torch.round(torch.expm1(output[..., 0].clamp(0, math.log1p(400))) / speed).long().clamp_min(1)
        # Every phoneme is audible for at least one frame. A pause-capable token
        # (edge, punctuation, word boundary) lasts its predicted length when the
        # presence logit says there is a pause, and vanishes otherwise.
        present = self.phone_tokens[tokens] | (output[..., 1] > 0)
        return torch.where(present, frames, torch.zeros_like(frames)) * mask

    @staticmethod
    def widen_pitch(prosody, frame_mask, pitch_range):
        """Scale the predicted pitch movement around each utterance's voiced mean.
        Regression predicts a flatter contour than the recordings; values above 1
        restore range (1.0 leaves the prediction untouched)."""
        if pitch_range == 1.0:
            return prosody
        voiced = ((prosody[..., 1] > 0) & frame_mask).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    @staticmethod
    def pitch_wobble(shape, semitones, seed, device):
        """Small random pitch movement for the oscillator only (log-F0 units).
        Recorded pitch has cycle-to-cycle irregularity that a predicted contour
        lacks; two thirds of the wobble is slow drift, one third frame-level."""
        generator = torch.Generator().manual_seed(seed + 7919)
        fast = torch.randn(shape, generator=generator)
        slow = F.avg_pool1d(torch.randn(shape[0], 1, shape[1] + 16, generator=generator), 17, stride=1)[:, 0]
        slow = slow / slow.std().clamp_min(1e-6)
        return ((2 * slow + fast) / 5 ** 0.5 * semitones * math.log(2) / 12).to(device)

    @torch.inference_mode()
    def synthesize(self, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, variant=None, anticausal_scale=1.0, harmonic_phase=None, noise_above=NOISE_ABOVE):
        """Free-running synthesis: predicted durations, pitch, voicing and energy.
        ``harmonic_phase`` (see ``harmonic_phases``) disperses the excitation.
        ``noise_above`` is a pair of harmonic numbers, for the frames the model
        calls unvoiced and for those it calls voiced: above that harmonic of the
        pitch the harmonic branch is driven by noise instead of pulses (0 is all
        noise, ``math.inf`` all pulses). None is pulses alone, which is how the
        weights were trained and what shipped before ``NOISE_ABOVE``."""
        if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
            raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")
        hidden, mask = self.encode(tokens, stress, lengths, speakers)
        durations = self.predict_durations(hidden, mask, tokens, speed)
        frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers)
        prosody_hidden, prosody = self.predict_prosody(frames, frame_mask)
        if voiced_vowels:
            # The pitch tracker missed creaky vowels in the recordings, and the
            # voicing predictor learned to drop whole vowels (heard as a whispered
            # word). A vowel is voiced by definition, so its frames are kept voiced.
            ends = durations.cumsum(1)
            frame = torch.arange(frames.shape[1], device=tokens.device)[None].expand(len(lengths), -1).contiguous()
            owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
            vowel = self.vowel_tokens[tokens].gather(1, owner) & frame_mask
            prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                                 prosody[..., 2:]), dim=-1)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        acoustic = self.decode(prosody_hidden, self.hard_prosody(prosody), frame_mask)
        # One noise stream per row, drawn at the row's own length, so batching or
        # padding never changes a row's audio.
        streams = [torch.Generator().manual_seed(seed + row) for row in range(len(frame_lengths))]
        noise = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=stream)
                                           for stream, n in zip(streams, frame_lengths)], batch_first=True)
        cutoff = breath = None
        if noise_above is not None:
            unvoiced, voiced = noise_above
            cutoff = torch.where(prosody[..., 1] > 0, float(voiced), float(unvoiced))
            # Drawn after each row's noise from the same stream, so the noise branch hears what it always did.
            breath = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=stream)
                                                for stream, n in zip(streams, frame_lengths)], batch_first=True).to(hidden.device)
        pitch = prosody[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        wave = self.vocoder(acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device),
                            variant=variant, anticausal_scale=anticausal_scale, harmonic_phase=harmonic_phase,
                            noise_above=cutoff, breath=breath)
        return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask}
