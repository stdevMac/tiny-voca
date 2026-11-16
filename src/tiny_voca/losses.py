import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def magnitude(wave, fft, hop, window):
    spectrum = torch.stft(wave.float(), fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    # sqrt(re^2 + im^2 + eps): finite gradient at exact zeros (padding, silence).
    return torch.view_as_real(spectrum).square().sum(-1).add(1e-9).sqrt()


def mel_filterbank(sample_rate, fft, n_mels):
    """Slaney-scale triangular filters with area normalisation, [n_mels, bins]."""
    step, knee, slope = 200.0 / 3, 1000.0, math.log(6.4) / 27

    def to_hz(mel):
        return torch.where(mel < knee / step, mel * step, knee * torch.exp((mel - knee / step) * slope))
    top = knee / step + math.log(sample_rate / 2 / knee) / slope if sample_rate / 2 > knee else sample_rate / 2 / step
    points = to_hz(torch.linspace(0, top, n_mels + 2, dtype=torch.float64))
    bins = torch.linspace(0, sample_rate / 2, fft // 2 + 1, dtype=torch.float64)
    rising = (bins[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - bins[None]) / (points[2:] - points[1:-1])[:, None]
    weights = torch.minimum(rising, falling).clamp_min(0) * (2.0 / (points[2:] - points[:-2]))[:, None]
    return weights.float()


class SpectralLoss(nn.Module):
    """Multi-resolution STFT and log-mel distances over whole utterances.

    Frames whose centre lies beyond an utterance's length are masked, so padded
    batches give the same value as the utterances would alone. No parameters.
    """

    def __init__(self, sample_rate, resolutions=((512, 128), (1024, 256), (2048, 512)), mel=(1024, 256, 80)):
        super().__init__()
        self.resolutions, self.mel = tuple(resolutions), tuple(mel)
        for fft in {fft for fft, _ in self.resolutions} | {self.mel[0]}:
            self.register_buffer(f"window_{fft}", torch.hann_window(fft), persistent=False)
        self.register_buffer("filters", mel_filterbank(sample_rate, self.mel[0], self.mel[2]), persistent=False)

    def log_mel(self, wave):
        fft, hop, _ = self.mel
        return torch.log((self.filters @ magnitude(wave, fft, hop, getattr(self, f"window_{fft}"))).clamp_min(1e-5))

    def forward(self, predicted, target, lengths):
        total = predicted.new_zeros((), dtype=torch.float32)
        for fft, hop in self.resolutions:
            window = getattr(self, f"window_{fft}")
            p, t = magnitude(predicted, fft, hop, window), magnitude(target, fft, hop, window)
            mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
            logarithmic = ((p.clamp_min(1e-5).log() - t.clamp_min(1e-5).log()).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
            # Spectral convergence per utterance; the floor keeps silent input finite.
            convergence = (((p - t).square() * mask).sum((1, 2)) + 1e-9).sqrt() / (t.square() * mask).sum((1, 2)).sqrt().clamp_min(1.0)
            total = total + logarithmic + convergence.mean()
        hop = self.mel[1]
        p, t = self.log_mel(predicted), self.log_mel(target)
        mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
        mel = ((p - t).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
        return total / len(self.resolutions), mel


class HarmonicLoss(nn.Module):
    """Penalise the share of spectral energy that falls between the harmonics.

    The multi-resolution STFT and mel distances are satisfied by a spectrum that
    is right on average, and that includes one whose partials are smeared into
    the gaps between them: the error is spread thinly over many bins, so a
    per-bin magnitude distance barely sees it. A listener does, as roughness at
    the edges of words, and it shows up as lost periodicity (HNR) against the
    recording the model was taught from.

    For every voiced frame this compares the inter-harmonic share of the
    prediction's spectrum with the recording's own share and penalises the log
    ratio. It is scale free, so it cannot be satisfied by turning the audio up
    or down, and a prediction already as clean as its teacher costs nothing,
    which keeps it from fighting the reconstruction loss on clean material.

    ``fft``/``hop`` default to the acoustic frame rate, so one spectrum frame is
    one acoustic frame and the per-frame F0 lines up without resampling.
    """

    def __init__(self, sample_rate, fft=1024, hop=256, ceiling=8000.0, tolerance=1.5, floor=0.02, limit=20.0):
        super().__init__()
        self.fft, self.hop, self.ceiling, self.floor, self.limit = fft, hop, ceiling, floor, limit
        # Half-width of a harmonic in Hz. One and a half bins covers the leakage
        # of a Hann window without reaching halfway to the next partial.
        self.tolerance = tolerance * sample_rate / fft
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.register_buffer("frequencies", torch.fft.rfftfreq(fft, 1 / sample_rate), persistent=False)

    def share(self, wave, f0):
        """Energy between the partials over energy on them, [B, frames]."""
        spectrum = magnitude(wave, self.fft, self.hop, self.window)
        frames = min(f0.shape[1], spectrum.shape[-1])
        energy = spectrum[..., :frames].transpose(1, 2)                       # [B, frames, bins]
        f0 = f0[:, :frames].clamp_min(1.0)[..., None]                         # [B, frames, 1]
        multiple = self.frequencies[None, None] / f0
        distance = (multiple - multiple.round()).abs() * f0                   # Hz to the nearest partial
        band = self.frequencies[None, None] <= self.ceiling
        harmonic = (distance <= self.tolerance) & band
        on = (energy * harmonic).sum(-1)
        between = (energy * (band & ~harmonic)).sum(-1)
        # Both ends are clamped: a frame with almost no harmonic energy would
        # otherwise return an unbounded ratio and dominate the batch.
        return (between / on.clamp_min(1e-6)).clamp(self.floor, self.limit)

    def forward(self, predicted, target, f0, mask):
        predicted, target = predicted[:, :mask.shape[1] * self.hop], target[:, :mask.shape[1] * self.hop]
        error = (self.share(predicted, f0).log() - self.share(target, f0).log()).abs()
        return masked_mean(error, mask)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=(1, 32, 96, 192, 384)):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0))) for a, b in zip(channels[:-1], channels[1:]))
        self.output = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wave):
        wave = F.pad(wave, (0, (-wave.shape[-1]) % self.period))
        x = wave.reshape(wave.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, fft, hop, channels=32):
        super().__init__()
        self.fft, self.hop = fft, hop
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.output = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, wave):
        x = magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]   # [B,1,frames,bins]
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class Discriminators(nn.Module):
    """Training-only waveform critics. Never included in an inference export."""

    def __init__(self, periods=(2, 3, 5, 7, 11), resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.networks = nn.ModuleList([PeriodDiscriminator(p) for p in periods]
                                      + [SpectrogramDiscriminator(fft, hop) for fft, hop in resolutions])

    def forward(self, wave):
        return [network(wave) for network in self.networks]


def discriminator_loss(real, fake):
    return sum((r-1).square().mean() + f.square().mean()
               for (r, _), (f, _) in zip(real, fake)) / len(real)


def generator_gan_loss(real, fake):
    adversarial = sum((f-1).square().mean() for f, _ in fake) / len(fake)
    matching = sum(F.l1_loss(f, r.detach())
                   for (_, rs), (_, fs) in zip(real, fake) for r, f in zip(rs, fs))
    matching = matching / sum(len(fs) for _, fs in fake)
    return adversarial, matching
