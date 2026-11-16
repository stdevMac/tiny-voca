# The word-edge artifact

Everything known about the one defect that survives every measurement: the
model sounds rough, or faintly echoed, at the start and end of words. It is
worse on the male voice. The teacher recordings are clean to the same ear.

This file is the record of what has been tried. Read it before proposing
anything, because most of the obvious ideas have been tried and did not work.

---

## 1. The system

`tiny-voca`, forked and retrained for Spanish (es-ES). Two voices, `F2` and
`M2`. 24 kHz. 4,534,411 inference parameters.

The waveform head is a **source-filter vocoder**, not a neural vocoder:

| part | setting |
|---|---|
| excitation | band-limited pulse train at the per-frame F0, phase-continuous, harmonics to `0.47 × Nyquist` |
| spectral envelope | complex cepstrum, 80 causal + 16 anticausal coefficients |
| envelope application | STFT filtering, `synth_fft=512`, `synth_hop=128` → **21.3 ms window, 46.9 Hz bins, 5.33 ms hop** |
| envelope update | interpolated between acoustic frames (acoustic hop 256 = 93.75 frames/s), averaged `here`/`following` per sub-frame |
| noise branch | white noise × its own cepstral envelope, `noise_fft=256`, `noise_hop=64`, 32 coefficients |

Trained on 22.68 h / 19,238 utterances of Qwen3-TTS teacher audio.

F2's F0 is ~244 Hz (period 4.1 ms). M2's F0 is ~88 Hz (period 11.4 ms).
The synthesis window holds **5.2 periods of F2 but only 1.9 periods of M2.**

## 2. The shipped model is not broken

Step 142,000 (`~/tiny-voca-artifacts/es-1-anneal/runs/es-1-anneal/inference.pt`):

- validation WER 2.00%, test WER 1.08%, novel sentences 0.0% (28 clips)
- punctuation probe: 100% of pairs rise more as a question
- yes/no intonation: 87.5% of the teacher's rise recovered
- voices: F2 244.2 Hz, M2 90.4 Hz, separation 17.20 st (92% of reference)

So this is a timbre artifact on an otherwise accurate model, not a failure.

## 3. The complaint, in the listener's words

> "row 1 is clean, and row 2 has it, however, on the female voice is better"
> "Female is usable, and pretty decent, but not the male, it still have the same issues"

Row 1 was the teacher's own recording of the same sentence; row 2 was the model.
This has been consistent across several rounds and several different A/B pages.

**Treat the listener as the instrument.** The measurements below have been wrong
twice while the listener was right.

## 4. Measurements, model vs its own teacher

All on paired `<id>_<voice>-reference.wav` / `-generated.wav` from
`~/tiny-voca-artifacts/es-1-harmonic/data/eval/harmonic-check/audio`
(40 validation + 40 test utterances, both voices).

| metric | F2 ref → gen | M2 ref → gen | verdict |
|---|---|---|---|
| HNR, 30 utts/voice | 12.61 → 10.93 dB (**−1.68**) | 16.98 → 16.90 dB (**−0.08**) | M2 has **no** periodicity deficit |
| spectral contrast | 16.75 → 17.33 | — | tracks teacher |
| mel std | 5.51 → 5.18 | — | tracks teacher |
| spectral flux | 19.24 → 18.20 | — | tracks teacher |
| spectral flatness | 0.0485 → 0.0370 | — | tracks teacher |
| boundary tail (voiced→unvoiced) | model −6.0, teacher −6.4 (English model −8.7) | — | small |
| pre-onset ring | model −14.4, teacher −14.5 dB | — | none |
| fricative/vowel duration ratio | model 0.71, teacher 0.76 | — | small |
| roughness, 20–300 Hz modulation share | gen **smoother** in all 4 bands | gen **+0.31 dB rougher** in 80–400 Hz, smoother elsewhere | weak, M2-specific |
| shimmer (cycle amplitude) | 0.0925 → 0.0936 | 0.1630 → 0.1680 | +1% / +3% relative |
| jitter (cycle period) | 0.0090 → 0.0096 | 0.0123 → 0.0118 | none |
| cycle-to-cycle timbre distance | 0.9300 → 0.9325 | 0.8934 → 0.9039 | +0.003 / +0.010 |
| **crest factor (voiced)** | 5.54 → 5.32 | **6.16 → 7.17** | **M2 +16%** |
| **kurtosis (voiced)** | 5.53 → 5.40 | **7.15 → 10.63** | **M2 +49%** |

The last two rows are the only strong, voice-specific, correctly-signed
difference found so far. See §7.

## 5. Experiments that failed

Every one of these was run. Do not repeat them.

**Inference-only, on the shipped weights:**

1. Shorter synthesis window — **worse**.
2. Longer 43 ms window (`synth_fft=1024`, `synth_hop=256`) — **worse** (train/test
   mismatch: the model learned the 512-point cepstrum).
3. Sub-frame gain smoothing — **no change**.
4. Gating the pulse train by voicing — 1.6 dB, **marginal**.
5. ONNX export fidelity — PyTorch HNR 10.5 dB vs ONNX 10.4 dB, **faithful**, not the cause.
6. Raising M2's pitch +3/+6/+9/+12 st — HNR got **monotonically worse**.
7. Branch isolation (`scripts/probe_variants.py`): `harmonic_only` rms 0.1468 vs
   baseline 0.1471 vs `noise_only` 0.0626. **The noise branch carries almost no
   energy; the artifact is entirely in the harmonic path.**
8. Minimum phase / anticausal sweep (`scripts/probe_anticausal.py`): zeroing the
   16 anticausal cepstral coefficients drops the level **11 dB**, so it is a main
   part of the envelope, not a leak. A level-matched sweep at
   100/75/50/25/0% anticausal was rejected by the listener: **"not good at all"**.

**Training:**

9. **Inter-harmonic loss fine-tune** (`scripts/harmonic_es.sh`,
   `HarmonicLoss` in `src/tiny_voca/losses.py`): penalise the share of energy
   between harmonics relative to the recording's own share, per voiced frame.
   20,000 steps from step 142,000, weight 4.0, LR 3.1e-5 → 1.6e-5, on a 5090.
   Result: **no gain**. HNR F2 −1.68 → −1.85 dB, M2 −0.08 → −0.39 dB, and it
   degraded monotonically from the first checkpoint (145k) onward. Validation WER
   2.00% → 2.77%, test WER unchanged, mel unchanged. **Not shipped.**
   Artifacts: `~/tiny-voca-artifacts/es-1-harmonic/`.

## 6. Measurements that were wrong

Recorded so they are not repeated:

- **"M2 loses 2.2 dB of periodicity."** From a **single sentence**. With 30
  utterances the deficit is −0.08 dB. This wrong number started the whole
  harmonic-loss direction.
- **"16 dB frame-rate artifact at 93.75 Hz."** M2's 88 Hz fundamental was
  leaking into the 88–100 Hz measurement band. F2-only the difference is 5.7 dB
  at −48 dB, inaudible.
- **"The 85.3 ms / 11.7 Hz cepstral comb is the model."** It is in the teacher
  too — it is the analysis window.
- **"M2's dense harmonics exceed the cepstral envelope's resolution."** Tested by
  raising the pitch; HNR got worse, killing the hypothesis.

## 7. The open hypothesis: impulse-train excitation

The head excites the filter with a **pulse train**, which means every harmonic
is phase-locked to the same instant. Real speech is not: the glottal source and
the vocal tract disperse the partials. A phase-locked excitation is the textbook
"buzzy" source-filter sound.

Why this fits everything:

- **It gets worse as F0 falls.** M2 has ~64 harmonics below 5.6 kHz, all
  aligned; F2 has ~23. More aligned partials, more buzz.
- **It is invisible to every metric above.** They are magnitude or
  periodicity measures; the magnitude spectrum of a pulse train is correct and
  its periodicity is maximal. Only the *phase* is wrong.
- **It is not jitter or shimmer.** The buzz is deterministic, so those measures
  stay flat — which is exactly what was observed.
- **It is not fixable by envelope work.** Hence eight failed attempts.

The supporting measurement is the crest factor / kurtosis table in §4: the male
voice's generated audio is 16% more peaky and 49% more kurtotic than its own
teacher, while the female voice is slightly *less* peaky. That is the signature
of an excitation closer to a raw impulse than the teacher's.

**Not yet tried:** replacing or dispersing the excitation.

Candidate fixes, cheapest first:

1. **Phase dispersion at inference.** Give harmonic `k` a phase offset
   `φ_k = −β·k` (or a fixed pseudo-random phase per harmonic, seeded per
   utterance). Same magnitude spectrum, no phase locking, no retraining. This is
   the single cheapest test and it is the one to run first.
2. **A glottal pulse instead of an impulse.** Rosenberg or LF pulse shape with
   an open-quotient parameter, which is what the teacher's source actually is.
3. **A small neural vocoder** (HiFi-GAN or similar) on the existing mel/cepstral
   features. This is the real answer if 1 and 2 do not do it, and it is what
   round 2 should budget for.

## 8. Reproducing

Scripts, all in `scripts/`:

| script | what it measures |
|---|---|
| `probe_periodicity.py` | HNR against the teacher, per voice |
| `probe_variants.py` | branch isolation: baseline / harmonic_only / noise_only / minimum_phase |
| `probe_anticausal.py` | anticausal cepstrum sweep, level matched, plus the difference signal |
| `probe_smearing.py` | synthesis window and pulse-gating variants |
| `probe_punctuation.py`, `probe_intonation.py`, `probe_voices.py` | the round-1 acceptance battery |
| `harmonic_es.sh` | the inter-harmonic fine-tune (failed; kept for the record) |
| `prepare_features.sh`, `upload_run.sh` | rebuild features on a fresh host; push a run to one |

Artifacts, all local:

- `~/tiny-voca-artifacts/es-1/` — the run: corpus, 2.4 GB teacher FLAC, annotations, checkpoints
- `~/tiny-voca-artifacts/es-1-anneal/` — **the shipped model, step 142,000**
- `~/tiny-voca-artifacts/es-1-harmonic/` — the failed fine-tune, its eval and its A/B pages
- `~/tiny-voca-artifacts/variants/`, `~/tiny-voca-artifacts/anticausal/` — listening pages

A GPU host is not running. Features rebuild from the teacher FLAC in ~10 min on
a many-core host (`prepare_features.sh`); the annotations do not need rebuilding.
A 5090 on Runpod secure cloud costs $1.39/h and trains this model at ~6.1 steps/s
at the anneal's recipe (`batch_frames=9000`, `max_batch=32`).

## 9. Ground rules for the next session

- **The listener outranks the metrics.** Twice the numbers said "fine" and the
  ear said otherwise, and the ear was right both times.
- **Do not retrain to test a hypothesis you can test at inference.** Eight
  inference experiments cost minutes; the one training experiment cost an hour
  and answered nothing.
- **Do not trust a single sentence.** The −2.2 dB that started this was one clip.
  Use ≥30 utterances per voice for any model-vs-teacher claim.
- **Never compare F2 with M2.** HNR, crest factor and shimmer are all confounded
  by F0. Only model-vs-its-own-teacher is valid.
