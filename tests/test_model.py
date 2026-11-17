import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

from tiny_voca.config import CHECKPOINT_FORMAT, Config
from tiny_voca.data import collate
from tiny_voca.infer import load_model
from tiny_voca.inspect import parameter_report
from tiny_voca.losses import Discriminators
from tiny_voca.model import NOISE_ABOVE, SourceFilter, TinyVoca, cepstrum_response, filtered, harmonic_phases, pulse_train
from tiny_voca.train import (WEIGHTS, TrainingSystem, atomic_save, export_model, learning_rate, load_compatible, optimize_step,
                             teacher_probability)
from helpers import SYMBOLS, sample, small_config, write_annotations

RECIPE = {"learning_rate": 1e-3, "warmup": 2, "lr_half_life": 1000.0, "grad_clip": 10.0, "gan_start": 1, "crop_samples": 4096, "crops": 1,
          "teacher_hold": 1, "teacher_ramp": 1, "teacher_final": 0.0, "weights": WEIGHTS, "gan_ratio": 0.7, "gan_ramp": 2,
          "feature_matching": 30.0, "gan_scale_max": 2000.0, "branched_from": None}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)

    def test_default_model_is_under_five_million_parameters(self):
        report = parameter_report(Config())
        self.assertEqual(sum(report["inference_parts"].values()), report["inference_parameters"])
        self.assertLess(report["inference_parameters"], 5_000_000)
        self.assertEqual(report["inference_parameters"], sum(p.numel() for p in TinyVoca().parameters()))
        self.assertGreater(report["inference_parts"]["frame transformer B"], report["inference_parts"]["source-filter waveform head"])

    def test_pulse_train_has_the_requested_pitch_unit_power_and_band_limit(self):
        rate, hop = 24000, 256
        for hertz in (110.0, 220.0):
            wave = pulse_train(torch.full((1, 200), hertz), rate, hop)[0].double()
            self.assertEqual(wave.numel(), 200 * hop)
            self.assertAlmostEqual(float(wave.square().mean()), 1.0, delta=0.02)
            spectrum = torch.fft.rfft(wave * torch.hann_window(wave.numel(), dtype=torch.float64)).abs()
            frequency = torch.fft.rfftfreq(wave.numel(), 1 / rate)
            self.assertAlmostEqual(float(frequency[spectrum[:int(1.5 * hertz * wave.numel() / rate)].argmax()]), hertz, delta=1.0)
            self.assertLess(float(spectrum[frequency > 0.48 * rate].max() / spectrum.max()), 1e-3)     # nothing above the limit
            harmonic = spectrum[(frequency - 10 * hertz).abs().argmin()]
            self.assertAlmostEqual(float(harmonic / spectrum[(frequency - hertz).abs().argmin()]), 1.0, delta=0.05)  # flat source
        glide = pulse_train(torch.linspace(90, 300, 100)[None], rate, hop)
        self.assertTrue(torch.isfinite(glide).all())
        self.assertLess(float(glide.abs().max()), 40)
        shifted = pulse_train(torch.full((1, 20), 150.0), rate, hop, phase=torch.tensor([1.0]))
        self.assertFalse(torch.allclose(shifted, pulse_train(torch.full((1, 20), 150.0), rate, hop)))

    def test_harmonic_phases_move_the_shape_of_the_period_and_nothing_else(self):
        rate, hop = 24000, 256
        glide = torch.cat((torch.linspace(70, 140, 60), torch.linspace(140, 300, 60)))[None]
        count = int(0.47 * rate / 70) + 2
        # No offsets: the harmonics summed one by one are the closed form.
        self.assertTrue(torch.allclose(pulse_train(glide, rate, hop, harmonic_phase=torch.zeros(count)), pulse_train(glide, rate, hop), atol=2e-5))
        # A phase linear in the harmonic number is the same pulse train, later.
        late = pulse_train(glide, rate, hop, harmonic_phase=harmonic_phases(count, beta=0.7, power=1.0))
        self.assertTrue(torch.allclose(late, pulse_train(glide, rate, hop, phase=torch.tensor([-0.7])), atol=2e-5))
        # Curved or scattered phases keep every harmonic's magnitude and the power,
        # and take the peak out of the period. 96 Hz is 250 samples a period.
        steady = torch.full((1, 250), 96.0)
        pulses = pulse_train(steady, rate, hop)[0, 16000:48000].double()
        partials = int(0.47 * rate / 96)
        for offsets in (harmonic_phases(count, beta=math.pi / partials), harmonic_phases(count, beta=0.7, power=1.5),
                        harmonic_phases(count, scatter=1.0, seed=3)):
            spread = pulse_train(steady, rate, hop, harmonic_phase=offsets)[0, 16000:48000].double()
            self.assertTrue(torch.allclose(torch.fft.rfft(spread).abs(), torch.fft.rfft(pulses).abs(), atol=1e-2))
            self.assertAlmostEqual(float(spread.square().mean()), float(pulses.square().mean()), delta=1e-4)
            self.assertLess(float(spread.abs().max()), 0.4 * float(pulses.abs().max()))
        self.assertFalse(torch.equal(harmonic_phases(count, scatter=1.0, seed=3), harmonic_phases(count, scatter=1.0, seed=4)))
        with self.assertRaises(ValueError):
            pulse_train(steady, rate, hop, harmonic_phase=torch.zeros(8))

    def test_stft_filtering_is_transparent_for_a_flat_response(self):
        source = torch.randn(2, 4096)
        window = torch.hann_window(512)
        frames = 4096 // 128 + 1
        flat = torch.ones(2, frames, 257, dtype=torch.complex64)
        torch.testing.assert_close(filtered(source, flat, 512, 128, window), source, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(filtered(source, 0.5 * flat, 512, 128, window), 0.5 * source, atol=1e-4, rtol=1e-4)
        with self.assertRaises(ValueError):
            filtered(source, flat[:, :-1], 512, 128, window)

    def test_cepstrum_gives_gain_and_minimum_phase(self):
        zero = cepstrum_response(torch.zeros(1, 8), torch.zeros(1, 2), 64)
        torch.testing.assert_close(zero, torch.ones(1, 33, dtype=torch.complex64))
        gain = cepstrum_response(torch.tensor([[math.log(0.5), 0, 0, 0]]), None, 64)
        torch.testing.assert_close(gain.abs(), torch.full((1, 33), 0.5), atol=1e-6, rtol=1e-6)
        # A causal cepstrum is a minimum-phase filter: its impulse response is causal and front-loaded.
        response = cepstrum_response(torch.tensor([[0.0, 0.8, -0.3, 0.1]]), None, 64)
        impulse = torch.fft.irfft(response, n=64)[0]
        self.assertGreater(float(impulse[:8].square().sum() / impulse.square().sum()), 0.999)

    def test_expand_skips_zero_duration_tokens_and_marks_positions(self):
        model = TinyVoca(small_config()).eval()
        hidden = torch.arange(4, dtype=torch.float32)[None, :, None].expand(1, 4, 32).contiguous()
        durations = torch.tensor([[2, 0, 3, 1]])
        with torch.no_grad():
            model.frame_position.weight.zero_()
            model.frame_position.bias.zero_()
            model.frame_voices.weight.zero_()
            frames, mask, lengths = model.expand(hidden, durations, torch.ones(1, 4, dtype=torch.bool), torch.tensor([0]))
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual(frames[0, :, 0].tolist(), [0, 0, 2, 2, 2, 3])        # token 1 has no frame
        with self.assertRaises(ValueError):
            model.expand(hidden, torch.tensor([[1, -1, 1, 1]]), torch.ones(1, 4, dtype=torch.bool), torch.tensor([0]))
        with self.assertRaises(ValueError):
            model.expand(hidden, torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool), torch.tensor([0]))

    def test_padding_does_not_change_the_short_example(self):
        c = small_config()
        model = TinyVoca(c).eval()
        with torch.no_grad():        # make the heads non-trivial so the test has teeth
            for head in (model.vocoder.harmonic, model.vocoder.noise):
                head.weight.normal_(std=0.05)
            model.duration.out.weight.zero_()            # three frames per token: rows differ in length
            model.duration.out.bias.fill_(math.log1p(3))
        a, b = sample(c), sample(c, voice=1, durations=(5, 6, 7, 6, 12, 3, 6, 14, 7, 6, 2, 8))
        tokens = torch.stack((a["tokens"], b["tokens"]))
        stress = torch.stack((a["stress"], b["stress"]))
        lengths = torch.tensor([7, len(SYMBOLS)])     # first row uses only 7 tokens: genuine token padding
        tokens[0, 7:], stress[0, 7:] = 0, 0
        wave, samples, details = model.synthesize(tokens, stress, lengths, torch.tensor([0, 1]))
        alone, n, _ = model.synthesize(tokens[:1, :7], stress[:1, :7], lengths[:1], torch.tensor([0]))
        self.assertEqual(int(n[0]), int(samples[0]))
        self.assertLess(int(samples[0]), int(samples[1]))
        # Float32 rounding through exp() and two STFT round trips reaches ~1e-4; the
        # padding bugs this guards against (pitch glide into padding, shared noise
        # stream, missing end frames) each moved samples by more than 1e-2.
        torch.testing.assert_close(wave[0, :int(n[0])], alone[0], atol=5e-4, rtol=1e-3)
        self.assertGreater(float(alone.abs().max()), 0.01)
        self.assertEqual(float(wave[0, int(n[0]):].abs().max()), 0.0)
        # Every phoneme is audible; pause tokens may get zero frames.
        phones = model.phone_tokens[tokens[1]]
        self.assertTrue(bool((details["durations"][1][phones] >= 1).all()))
        for speed in (0, float("nan"), 5):
            with self.assertRaises(ValueError):
                model.synthesize(tokens, stress, lengths, torch.tensor([0, 1]), speed)

    def test_source_filter_is_differentiable_and_respects_lengths(self):
        c = small_config()
        head = SourceFilter(c)
        hidden = torch.randn(2, 12, c.channels, requires_grad=True)
        with torch.no_grad():
            head.harmonic.weight.normal_(std=0.05)
            head.noise.weight.normal_(std=0.05)
        wave = head(hidden, torch.full((2, 12), 180.0), torch.tensor([12, 7]))
        self.assertEqual(wave.shape, (2, 12 * c.hop))
        self.assertEqual(float(wave[1, 7 * c.hop:].abs().max()), 0.0)
        wave.square().mean().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad[1, :7].abs().sum()), 0)
        self.assertEqual(float(hidden.grad[1, 7:].abs().sum()), 0.0)        # padded frames cannot influence audio

    def test_noise_excitation_keeps_the_spectrum_and_loses_the_pitch(self):
        c = small_config()
        head = SourceFilter(c)
        with torch.no_grad():
            head.harmonic.bias[1:4] = torch.tensor([1.0, -0.6, 0.4])        # a sloping envelope, so the filter matters
        frames, period = 300, 240                                           # 100 Hz
        hidden, pitch, silence = torch.zeros(1, frames, c.channels), torch.full((1, frames), 100.0), torch.zeros(1, frames * c.hop)
        breath = torch.randn(1, frames * c.hop, generator=torch.Generator().manual_seed(1))

        def render(cutoff=None, length=frames, **options):                  # the harmonic branch alone
            with torch.no_grad():
                return head(hidden, pitch, torch.tensor([length]), noise=silence, noise_above=cutoff, breath=breath, **options)
        pulses = render()
        # A cut-off no frequency reaches leaves the pulse train alone.
        self.assertTrue(torch.allclose(render(torch.full((1, frames), math.inf)), pulses, atol=1e-7))
        hiss = render(torch.zeros(1, frames))

        def bands(wave):       # power per kHz, from 1 to 11
            power = torch.stft(wave[0], 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().square().mean(1)
            return power[86:].unfold(0, 85, 85).sum(1)[:10]
        self.assertLess(float((10 * torch.log10(bands(hiss) / bands(pulses))).abs().max()), 0.5)
        self.assertAlmostEqual(float(hiss.square().mean() / pulses.square().mean()), 1.0, delta=0.05)

        def repeats(wave):     # correlation of each period with the next
            body = wave[0, 4 * period:-4 * period]
            return float(torch.corrcoef(torch.stack((body[:-period], body[period:])))[0, 1])
        self.assertGreater(repeats(pulses), 0.99)
        self.assertLess(abs(repeats(hiss)), 0.2)
        # Noise above the twentieth harmonic, 2 kHz, in the first half only: the
        # second half is the pulse train, and so is the first below the cut-off.
        middle = frames // 2 * c.hop
        mixed = render(torch.cat((torch.full((1, frames // 2), 20.0), torch.full((1, frames - frames // 2), math.inf)), dim=1))
        self.assertTrue(torch.allclose(mixed[:, middle + 4 * c.hop:], pulses[:, middle + 4 * c.hop:], atol=1e-6))
        self.assertLess(float((10 * torch.log10(bands(mixed[:, :middle]) / bands(pulses[:, :middle]))).abs().max()), 0.7)
        low, reference = (torch.stft(wave[0, :middle], 2048, 512, window=torch.hann_window(2048), return_complex=True)[:100]      # below 1.2 kHz
                          for wave in (mixed, pulses))
        self.assertTrue(torch.allclose(low, reference, atol=0.02 * float(reference.abs().max())))
        # The noise that replaces pulses belongs to the harmonic branch, and it respects lengths.
        self.assertEqual(float(render(torch.zeros(1, frames), variant="noise_only").abs().max()), 0.0)
        self.assertEqual(float(render(torch.zeros(1, frames), length=100)[0, 100 * c.hop:].abs().max()), 0.0)

    def test_noise_excitation_is_the_default_and_can_be_switched_off(self):
        torch.manual_seed(0)
        model = TinyVoca(small_config()).eval()
        with torch.no_grad():
            model.vocoder.harmonic.weight.normal_(std=0.05)
        tokens = torch.tensor([[1, 20, 50, 27, 51, 2]])
        inputs = (tokens, torch.zeros_like(tokens), torch.tensor([6]), torch.tensor([0]))
        with torch.no_grad():
            plain, lengths, _ = model.synthesize(*inputs, noise_above=None)         # what the weights were trained with
            default, _, _ = model.synthesize(*inputs)
            named, _, _ = model.synthesize(*inputs, noise_above=NOISE_ABOVE)
            same, _, _ = model.synthesize(*inputs, noise_above=(math.inf, math.inf))
            changed, _, _ = model.synthesize(*inputs, noise_above=(0.0, 0.0))
            again, _, _ = model.synthesize(*inputs, noise_above=(0.0, 0.0))
        self.assertTrue(torch.equal(default, named))
        if NOISE_ABOVE is None:                                             # English: pulses alone, as shipped
            self.assertTrue(torch.equal(default, plain))
        else:
            self.assertEqual(NOISE_ABOVE, (2.0, 33.0))
            self.assertFalse(torch.allclose(default, plain, atol=1e-4))
        self.assertTrue(torch.allclose(same, plain, atol=1e-7))
        self.assertFalse(torch.allclose(changed, plain, atol=1e-4))
        self.assertTrue(torch.equal(changed, again))                        # seeded like the rest

    def test_both_training_phases_update_every_inference_module(self):
        c = small_config()
        system, critics = TrainingSystem(c), Discriminators(periods=(2, 3), resolutions=((256, 64),))
        optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
        d_optimizer = torch.optim.AdamW(critics.parameters(), lr=1e-3)
        batch = collate([sample(c), sample(c, voice=1, durations=(5, 6, 7, 6, 12, 3, 6, 14, 7, 6, 0, 8))])
        before = {name: value.detach().clone() for name, value in system.model.named_parameters()}
        for step in (0, 1, 2):
            metrics = optimize_step(system, critics, optimizer, d_optimizer, batch, step, RECIPE)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()), metrics)
        self.assertIn("adversarial", metrics)
        self.assertGreater(metrics["gan_scale"], 0)
        unchanged = [name for name, value in system.model.named_parameters() if torch.equal(value, before[name])]
        self.assertEqual(unchanged, [])
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in system.parameters()))

    def test_adversarial_gradient_is_balanced_against_reconstruction(self):
        # Regression for the "GAN phase that does nothing": with fixed weights the adversarial
        # gradient at the waveform was under 1% of the spectral one on the real run.
        c = small_config()
        torch.manual_seed(5)
        system, critics = TrainingSystem(c), Discriminators(periods=(2, 3), resolutions=((256, 64),))
        batch = collate([sample(c), sample(c, voice=1)])
        loss, _, auxiliary = system.objective(batch, 1.0, random_phase=False)
        from tiny_voca.losses import generator_gan_loss
        from tiny_voca.train import crops
        wave = auxiliary["wave"]
        fake, real, index = crops(wave, batch["wave"], batch["frame_lengths"] * c.hop, 4096, 1)
        self.assertEqual(index.shape, (2, 4096))
        torch.testing.assert_close(fake, wave.gather(1, index))
        adversarial, matching = generator_gan_loss(critics(real), critics(fake))
        gan = adversarial + 30.0 * matching
        reconstruction = torch.autograd.grad(auxiliary["reconstruction"], wave, retain_graph=True)[0].gather(1, index)
        raw = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
        scale = 0.7 * reconstruction.norm() / raw.norm()
        balanced = torch.autograd.grad(scale.detach() * gan, wave)[0].gather(1, index)
        self.assertAlmostEqual(float(balanced.norm() / reconstruction.norm()), 0.7, places=4)

    def test_pause_tokens_vanish_unless_predicted_present_and_pitch_range_scales_movement(self):
        model = TinyVoca(small_config()).eval()
        tokens = sample(small_config())["tokens"][None]
        mask = torch.ones_like(tokens, dtype=torch.bool)
        hidden = torch.zeros(1, tokens.shape[1], 32)
        phones = model.phone_tokens[tokens[0]]
        with torch.no_grad():
            model.duration.out.weight.zero_()
            model.duration.out.bias.copy_(torch.tensor([math.log1p(5), -3.0]))       # 5 frames, "no pause"
            absent = model.predict_durations(hidden, mask, tokens)[0]
            model.duration.out.bias.copy_(torch.tensor([math.log1p(5), 3.0]))        # 5 frames, "pause"
            present = model.predict_durations(hidden, mask, tokens)[0]
            model.duration.out.bias.copy_(torch.tensor([-2.0, -3.0]))                # shorter than a frame
            floor = model.predict_durations(hidden, mask, tokens)[0]
        self.assertEqual(absent[phones].tolist(), [5] * int(phones.sum()))
        self.assertEqual(absent[~phones].tolist(), [0] * int((~phones).sum()))
        self.assertEqual(present.tolist(), [5] * tokens.shape[1])
        self.assertEqual(floor[phones].tolist(), [1] * int(phones.sum()))            # a phoneme is never dropped
        self.assertEqual(model.predict_durations(hidden, mask, tokens, speed=2.5)[0][phones].tolist(), [1] * int(phones.sum()))
        prosody = torch.tensor([[[0.0, 1.0, 0.0], [0.4, 1.0, 0.0], [9.0, -1.0, 0.0], [-0.4, 1.0, 0.0]]])
        frame_mask = torch.ones(1, 4, dtype=torch.bool)
        wide = TinyVoca.widen_pitch(prosody, frame_mask, 1.5)
        torch.testing.assert_close(wide[0, [0, 1, 3], 0], torch.tensor([0.0, 0.6, -0.6]))      # around the voiced mean (0)
        torch.testing.assert_close(wide[..., 1:], prosody[..., 1:])
        self.assertIs(TinyVoca.widen_pitch(prosody, frame_mask, 1.0), prosody)

    def test_branching_keeps_old_weights_when_a_layer_grows(self):
        old = {"weight": torch.ones(1, 3), "bias": torch.full((1,), 2.0), "gone": torch.zeros(1)}
        layer = torch.nn.Linear(3, 2)
        fresh = layer.weight.detach().clone()
        report = load_compatible(layer, old)
        self.assertEqual(report, {"grown": ["weight", "bias"], "new": [], "dropped": ["gone"]})
        torch.testing.assert_close(layer.weight[0], torch.ones(3))
        torch.testing.assert_close(layer.weight[1], fresh[1])                      # the new row keeps its initialisation
        self.assertEqual(float(layer.bias[0]), 2.0)
        with self.assertRaisesRegex(ValueError, "changed shape"):
            load_compatible(torch.nn.Linear(2, 2), {"weight": torch.ones(2, 3), "bias": torch.ones(2)})

    def test_teacher_schedule_and_learning_rate(self):
        self.assertEqual(teacher_probability(0, 100, 200, 0.5), 1.0)
        self.assertAlmostEqual(teacher_probability(200, 100, 200, 0.5), 0.75)
        self.assertEqual(teacher_probability(10_000, 100, 200, 0.5), 0.5)
        self.assertAlmostEqual(learning_rate(0, 1e-3, 10, 1000), 1e-4)
        self.assertAlmostEqual(learning_rate(1000, 1e-3, 10, 1000), 5e-4)
        self.assertAlmostEqual(learning_rate(1000, 1e-3, 10, 1000, start=1000), 5e-5)     # a branched run warms up again
        c = small_config()
        system = TrainingSystem(c).eval()
        batch = collate([sample(c), sample(c, voice=1)])
        with torch.no_grad():
            teacher = system.objective(batch, 1.0, random_phase=False)[1]
            free = system.objective(batch, 0.0, random_phase=False)[1]
        self.assertEqual(float(teacher["teacher_fraction"]), 1.0)
        self.assertEqual(float(free["teacher_fraction"]), 0.0)
        torch.testing.assert_close(teacher["pitch"], free["pitch"])       # predictors do not depend on the conditioning choice
        # The pitch-spread term compares per-utterance standard deviations over trusted voiced frames.
        from tiny_voca.train import masked_std
        values = torch.tensor([[1.0, 3.0, 100.0, 5.0], [2.0, 2.0, 2.0, 2.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
        torch.testing.assert_close(masked_std(values, mask), torch.tensor([(8 / 3) ** 0.5, 1e-4]), atol=1e-5, rtol=1e-4)
        self.assertGreater(float(teacher["pitch_spread"]), 0)
        self.assertTrue(0 <= float(teacher["pitch_spread_ratio"]) < 5)

    def test_objective_rejects_durations_that_do_not_cover_the_frames(self):
        c = small_config()
        system = TrainingSystem(c).eval()
        item = sample(c)
        item["durations"] = item["durations"].clone()
        item["durations"][4] -= 1
        with self.assertRaisesRegex(ValueError, "frame count"), torch.no_grad():
            system.objective(collate([item]), 1.0)

    def test_export_roundtrip_and_explicit_rejection_of_old_checkpoints(self):
        model = TinyVoca(small_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            export_model(model, path, 7, "annotation")
            restored = load_model(path)
            self.assertEqual(set(model.state_dict()), set(restored.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(any("mel_head" in key or "critic" in key for key in restored.state_dict()))
            state = torch.load(path, weights_only=True)
            self.assertEqual(state["format_version"], CHECKPOINT_FORMAT)
            for change, message in (({"format_version": 3}, "byte-token"),
                                    ({"frontend": {**state["frontend"], "fingerprint": "other"}}, "front end")):
                atomic_save({**state, **change}, path)
                with self.assertRaisesRegex(ValueError, message):
                    load_model(path)

    def test_training_cli_resumes_same_as_uninterrupted_run(self):
        c = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(c.to_dict()))
            train = write_annotations(root, c, "train")
            validation = write_annotations(root, c, "validation")
            base = [sys.executable, "-m", "tiny_voca.train", "--train", str(train), "--validation", str(validation),
                    "--config", str(root / "config.json"), "--device", "cpu", "--batch-frames", "200", "--max-batch", "2",
                    "--save-every", "1", "--gan-start", "1", "--warmup", "2", "--teacher-hold", "1", "--teacher-ramp", "1",
                    "--crop-samples", "4096", "--crops", "1", "--validation-size", "2", "--max-seconds", "300", "--keep-every", "2"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

            def run(arguments, expect=0):
                result = subprocess.run(base + arguments, capture_output=True, text=True, env=env, timeout=600)
                self.assertEqual(result.returncode != 0, expect != 0, result.stdout + result.stderr)
                return result
            run(["--output", str(root / "resumed"), "--steps", "1"])
            run(["--output", str(root / "resumed"), "--steps", "2", "--resume", str(root / "resumed/last.pt")])
            run(["--output", str(root / "continuous"), "--steps", "2"])
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            continuous = torch.load(root / "continuous/last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(resumed["format_version"], CHECKPOINT_FORMAT)
            self.assertEqual(torch.load(root / "resumed/step-00000001.pt", weights_only=True)["step"], 1)
            self.assertTrue((root / "resumed/step-00000002.pt").is_file())
            for key, value in resumed["system"].items():
                # Resuming must reproduce the run. Bit-exactness is not something
                # PyTorch guarantees across processes on every CPU backend (some
                # runs come back one float32 ULP off), so the check is a tight
                # numerical tolerance rather than equality.
                torch.testing.assert_close(value, continuous["system"][key], rtol=1e-6, atol=1e-8)
            record = json.loads((root / "resumed/metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("validation_predicted_mel", record)
            self.assertIn("steps_per_second", record)
            self.assertEqual(load_model(root / "resumed/inference.pt").config, c)
            # A deadline that has already passed refuses to start; a stale format refuses to resume.
            (root / "budget.json").write_text(json.dumps({"deadline": 1.0}))
            failed = run(["--output", str(root / "late"), "--steps", "1", "--deadline-file", str(root / "budget.json")], expect=1)
            self.assertIn("deadline", failed.stderr)
            atomic_save({**resumed, "format_version": 3}, root / "old.pt")
            failed = run(["--output", str(root / "resumed"), "--steps", "3", "--resume", str(root / "old.pt")], expect=1)
            self.assertIn("cannot be resumed", failed.stderr)
            # A changed recipe cannot be resumed, but it can be branched into a new run, which then resumes normally.
            changed = ["--gan-ratio", "0.4", "--weights", '{"mel": 5}']
            failed = run(["--output", str(root / "resumed"), "--steps", "3", "--resume", str(root / "resumed/last.pt")] + changed, expect=1)
            self.assertIn("--init-from", failed.stderr)
            run(["--output", str(root / "branch"), "--steps", "3", "--init-from", str(root / "resumed/last.pt")] + changed)
            branch = torch.load(root / "branch/last.pt", weights_only=True)
            self.assertEqual(branch["step"], 3)
            self.assertEqual(branch["recipe"]["branched_from"]["step"], 2)
            self.assertEqual(branch["recipe"]["weights"]["mel"], 5.0)
            # Same architecture, source already in its adversarial phase: critics and optimizer moments carry over.
            self.assertEqual(branch["recipe"]["branched_from"]["loaded"],
                             {"grown": [], "new": [], "dropped": [], "critics": True, "optimizer": True})
            run(["--output", str(root / "branch"), "--steps", "4", "--resume", str(root / "branch/last.pt")] + changed)
            self.assertEqual(torch.load(root / "branch/last.pt", weights_only=True)["step"], 4)


if __name__ == "__main__":
    unittest.main()
