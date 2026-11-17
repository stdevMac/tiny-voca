# The word-edge artifact

Everything known about the one defect that survives every measurement: the
model sounds rough, or faintly echoed, at the start and end of words. It is
worse on the male voice. The teacher recordings are clean to the same ear.

This file is the record of what has been tried. Read it before proposing
anything, because most of the obvious ideas have been tried and did not work.

**Status, 2026-11-17.** One change has been preferred by the listener: at
inference, part of the pulse train is replaced by noise (§11, setting `2,33`).
On the male voice it was rated better in four rows of four, blind, with the
hidden copy of the shipped render rated the same in all four. On a second page,
twelve new sentences run from the files that would ship, nothing was rated
worse and the preference held, more weakly. It is the default of
`TinyVoca.synthesize` and of the ONNX export, and the round-1 acceptance
battery passes with it: word error rate, durations and pitch are unchanged,
and it runs about a tenth slower.
The weights are the shipped ones (es-1-anneal, step 142,000); nothing was
retrained, nothing is committed, and the app still points at the old file.
"Better" is not "as clean as the teacher": what is still open is at the end
of §11.

---

## 1. The system

`tiny-voca`, forked and retrained for Spanish (es-ES). Two voices, `F2` and
`M2`. 24 kHz. 4,534,411 inference parameters.

The waveform head is a **source-filter vocoder**, not a neural vocoder:

| part | setting |
|---|---|
| excitation | band-limited pulse train at the per-frame F0, phase-continuous, harmonics to `0.47 × Nyquist`. Since §11, at inference only: above harmonic 33 of the pitch in voiced frames, and above harmonic 2 in unvoiced ones, the same filter is driven by noise of the same power |
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

Two things about this table, found later:

- The audio it measures is the **failed fine-tune's** (step 162,000), not the
  shipped model's. On the shipped model (step 142,000, 40 utterances per voice,
  `scripts/probe_peakiness.py`) the last two rows read: M2 crest 6.21 → 7.02
  (+0.81, 95% CI +0.57 to +1.05), kurtosis 7.41 → 9.86 (+2.44, CI +1.92 to
  +2.97), and the crest factor of each period against its own two periods
  +0.86 dB (CI +0.75 to +0.98, above the teacher in 40 of 40 utterances); F2
  crest −0.12, kurtosis +0.15, local crest +0.02 dB, none distinguishable from
  zero. So the male gap is real and robust. It is not what the listener hears;
  see experiment 10.
- The three word-edge rows (boundary tail, pre-onset ring, duration ratio) have
  no M2 value. The complaint is about M2 at word edges, so those rows say
  nothing about it.

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
   *Later: this does not hold where the complaint is.* Those rms values are of
   clips each normalised to its own peak, over whole utterances. Per frame class
   (`scripts/probe_branches.py`, 40 utterances per voice), noise against
   harmonic in M2's voiced frames is −21.2 dB overall but −2.0 dB at 4–8 kHz and
   +3.5 dB at 8–11.5 kHz, and it is 6.6 dB higher in the four frames at each
   word edge than inside the word (F2: −24.0 dB, −5.4 dB, +0.7 dB, edges
   +6.9 dB). The noise branch is not ruled out.
8. Minimum phase / anticausal sweep (`scripts/probe_anticausal.py`): zeroing the
   16 anticausal cepstral coefficients drops the level **11 dB**, so it is a main
   part of the envelope, not a leak. A level-matched sweep at
   100/75/50/25/0% anticausal was rejected by the listener: **"not good at all"**.
   *Later:* scaling those coefficients changes the envelope's magnitude as well
   as its phase, so this sweep does not say what phase alone does. Experiment 10
   does.
10. **Phase dispersion of the excitation** (`harmonic_phases` and the
   `harmonic_phase` argument in `src/tiny_voca/model.py`,
   `scripts/probe_dispersion.py`). Harmonic `k` gets the phase `−β·k^p` (a
   sweep inside each period) or a fixed pseudo-random phase; the magnitude
   spectrum of the excitation is unchanged. It does what §7 asked of it to the
   numbers: on 40 utterances, M2's local crest against the teacher goes from
   +0.87 dB (pulse) to +0.17 (`β=0.0245`), −0.56 (`β=0.05`), −1.26 (`β=0.1`)
   and −1.64 dB (random phases), and kurtosis from +2.44 to −1.95. A blind page
   of six sentences (four M2, two F2) with six settings and a hidden copy of the
   shipped render found **no winner**. Ratings as better/same/worse than the
   shipped render: hidden copy 1/5/0, random ±0.25π 3/3/0, random ±π 2/3/1,
   `β=0.0245` 0/5/1, `β=0.05` 0/3/2, `β=0.1` 0/3/2, `β=5.4, p=1.2` 1/3/2. The
   hidden copy was itself called better once, so one or two "better" in six is
   the floor. The listener's summary: "more or less". **Not shipped; the default
   is unchanged.** Page: `~/tiny-voca-artifacts/phase-dispersion/listen/`.

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

## 7. The impulse-train hypothesis (tested: the gap is real, the cause is not this)

*Outcome, experiment 10: the peakiness gap below is robust (§4), and dispersing
the phases removes it, and the listener hears no improvement. A waveform that
is too peaky is a true description of M2's output and is not the word-edge
artifact. One correction to the text below: `φ_k = −β·k` does nothing at all,
because `cos(k·x − β·k)` is `cos(k·(x − β))`, the same pulse train delayed; the
phase has to bend with `k` (`−β·k²`) or be random. The section is kept as it
was written.*

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

## 8. Where the teacher's pitch is missing

`scripts/probe_pitch_labels.py`, 120 recordings per voice. The training labels
come from a pitch tracker with a 43 ms frame; where it finds no pitch on a
vowel the label is interpolated and the oscillator runs on regardless. The
question was whether those frames sit at word edges.

| vowel frames with no tracked pitch | F2 | M2 |
|---|---|---|
| all | 3.4% | 19.6% |
| inside words | 1.8% | 19.4% |
| first or last 4 frames of a word, inside a phrase | 1.9% / 4.9% | 9.1% / 10.6% |
| first 4 frames of a word after a pause | 38.7% | 80.5% |
| last 4 frames of a word before a pause | 46.9% | 89.9% |

- They are **not** at word edges. For M2 the edges inside a phrase are better
  tracked than word interiors (paired, −9.5 and −8.6 points, CIs clear of zero).
- They **are** at phrase edges, for both voices and almost always for M2.
- Most are not periodic at all: of M2's untracked vowel frames, 19% have one
  period that correlates with the next at 0.8 or more (tracked frames: 83%). A
  tracker with an 85 ms frame recovers 48% of them. So this is mostly the male
  teacher's own creaky, irregular voice, a fifth of his vowel frames, and only
  partly a tracker that is too short for 90 Hz.

What that means for the head: a strictly periodic oscillator cannot produce
irregular phonation, so on those frames the model has no correct output
available to it. Whether that is what the listener hears is not established.

## 9. The English model has it too

The listener reports the same disturbance on the male voice of the upstream
English demo, "perhaps a little bit lower". If so the artifact belongs to the
architecture or its objective, not to the Spanish data, labels or recipe.

## 10. What the head can do when it is fitted to one recording

`scripts/probe_oracle.py fit`, 32 recordings per voice. The model's weights are
left alone. For one recording at a time, the per-frame parameters the head
receives (both cepstra and the gains) are optimised directly against the
teacher's audio with the training spectral loss and nothing else, 400 steps from
the model's own prediction, on the teacher's timing and pitch. It is a
diagnostic of the head and the loss, not a model anyone could ship.

| | F2 | M2 |
|---|---|---|
| spectral loss, model's parameters → fitted | 8.41 → 2.90 | 7.42 → 2.53 |
| mel distance to the recording, word interior | 0.477 → 0.132 | 0.408 → 0.100 |
| mel distance, word edge inside a phrase | 0.525 → 0.140 | 0.440 → 0.101 |
| mel distance, word edge at a pause | 0.597 → 0.187 | 0.478 → 0.126 |
| noise branch against harmonic branch, word interior (dB) | −23.3 → −26.5 | −18.7 → −22.8 |
| noise against harmonic, pause beside a word (dB) | −5.0 → −6.7 | −19.9 → −31.5 |
| local crest against the recording (dB), model → fitted | −0.05 → −0.16 | +0.70 ±0.13 → +0.52 ±0.12 |

- The head can sit three to four times closer to the recording's spectrum than
  the model puts it, at word edges as much as inside words. By the spectral
  measures the envelope is not the limit.
- Free to choose, the spectral loss moves energy **out of the noise branch and
  into the pulse train**, in every frame class and for both voices.
- What survives the fit for M2 is everything this file is about: the waveform
  stays peakier than the recording, and the high band keeps arriving in bursts
  at the pitch (§11: voiceless stops 0.575 fitted, 0.544 model, 0.137
  recording). The spectral loss does not see it.
- The listener's verdict on the fitted renders is at the end of this section:
  not better.

**The same fit with the training's critics** (`--variants critics`: their
feature-matching distance added, critics frozen, no adversarial term). Six
recordings, four of them M2, so a pointer and not a finding: it is far below
the 30-recording rule, and no paired test on four can pass.

| M2, 4 recordings: 2–8 kHz pulsing (§11) | recording | model | fitted | fitted with critics |
|---|---|---|---|---|
| voiceless stop | 0.093 | 0.514 | 0.561 | 0.411 |
| pause | 0.222 | 0.829 | 0.590 | 0.558 |
| voiced consonant | 0.528 | 0.821 | 0.767 | 0.681 |
| vowel | 0.722 | 0.875 | 0.854 | 0.787 |

The critics pull the right way everywhere and undo the move into the pulse
train (noise against harmonic, word interior: −23.0 dB fitted, −19.4 with
critics, −18.1 model). The waveform ends less peaky than the recording (local
crest −0.70 dB against it, where the model is +0.63). They do not bring stops
or pauses near the recording, and the mel distance is worse than the plain
fit's (0.156 against 0.098 inside words). On these four recordings the
inference switch of §11 gets much closer (`2,33`: stops 0.132, pauses 0.217,
vowels 0.693).

**Verdict (the listener, blind, 2026-11-16): fitting the head does not remove
it.** `~/tiny-voca-artifacts/oracle-critics/listen/`, the four M2 recordings
above, every candidate on the teacher's timing and pitch, level-matched, 16
ratings against the shipped render and no notes:

| candidate | better | same | worse |
|---|---|---|---|
| hidden copy of the shipped render | 1 | 3 | 0 |
| the model's own parameters on the teacher's timing and pitch | 0 | 3 | 1 |
| fitted | 0 | 3 | 1 |
| fitted with critics | 1 | 3 | 0 |

The fitted head with the critics scored exactly what the hidden copy scored,
and the plain fit one row below it. A head handed the best parameters this loss
can find for one recording still sounds like the shipped model at the edges of
words, so more training against the same objective is not expected to fix it.
(The fine-tune of §5, experiment 9, added a different loss term and is not
explained by this; it is only one more result pointing the same way.) Four
sentences and one listener: enough to stop expecting a win from this
direction, not a proof.

## 11. Noise rendered as pitch pulses (the listener prefers the fix; it is the default)

**The hypothesis.** A voiceless consonant, a burst and the floor of a pause are
noise. The head has a noise branch for them, but nothing makes the model use
it, and the oscillator never stops. Between 300 Hz and 4 kHz, 96% or more of
M2's energy comes from the pulse-train branch even in voiceless fricatives and
pauses (88% or more in voiceless stops). Sound that should be noise then
arrives as one burst per pitch period: 4 ms apart for F2, 11 ms apart for M2.

**The measure** (`scripts/probe_pulsing.py`, 32 recordings per voice, the
shipped model on the teacher's timing and pitch). Band-pass 2–8 kHz, take the
envelope, and per frame correlate one oscillator period of it with the next.
Noise scores low, a pulse train towards 1. Means per phone class; `fitted` is
the head of §10; the last three columns are the settings described below.

| M2 (90 Hz) | recording | model | fitted | `2,inf` | `2,33` | `2,22` |
|---|---|---|---|---|---|---|
| voiceless fricative | 0.289 | 0.356 | 0.412 | 0.295 | 0.279 | 0.274 |
| voiceless stop | 0.137 | 0.544 | 0.575 | 0.200 | 0.149 | 0.117 |
| voiced consonant | 0.536 | 0.831 | 0.816 | 0.763 | 0.579 | 0.284 |
| vowel | 0.689 | 0.850 | 0.831 | 0.842 | 0.653 | 0.322 |
| vowel, pitch tracked | 0.775 | 0.883 | 0.870 | 0.879 | 0.699 | 0.325 |
| vowel, no pitch (§8) | 0.312 | 0.675 | 0.628 | 0.645 | 0.439 | 0.306 |
| pause | 0.229 | 0.802 | 0.577 | 0.228 | 0.229 | 0.229 |

| F2 (251 Hz) | recording | model | fitted | `2,inf` | `2,33` | `2,22` |
|---|---|---|---|---|---|---|
| voiceless fricative | 0.267 | 0.281 | 0.319 | 0.195 | 0.191 | 0.164 |
| voiceless stop | 0.207 | 0.297 | 0.375 | 0.234 | 0.233 | 0.220 |
| voiced consonant | 0.635 | 0.603 | 0.662 | 0.603 | 0.601 | 0.574 |
| vowel | 0.757 | 0.741 | 0.785 | 0.740 | 0.736 | 0.705 |
| vowel, pitch tracked | 0.783 | 0.760 | 0.809 | 0.759 | 0.756 | 0.724 |
| vowel, no pitch (§8) | 0.267 | 0.368 | 0.309 | 0.354 | 0.341 | 0.337 |
| pause | 0.170 | 0.268 | 0.219 | 0.057 | 0.056 | 0.056 |

- **M2's model is above its teacher in every class**, in all 32 recordings for
  stops, voiced consonants, vowels and pauses (paired: stops +0.407 ±0.044,
  pauses +0.574 ±0.025, voiced consonants +0.296 ±0.023, vowels +0.161 ±0.024).
  **F2's is not**: vowels and voiced consonants are within 0.04 of the teacher
  and on the other side of it; stops are +0.090 ±0.056 and pauses +0.098 ±0.072,
  the pauses at −52 dB. This is the first measurement in this file that
  separates the voices the way the listener does, at the sounds found at word
  edges.
- **The fitted head is as high as the model.** The spectral loss does not
  penalise it; §10 shows the loss prefers the pulse train.
- What it does not show: the measure rises by construction when the excitation
  is locked to the oscillator, so it says what the sound is made of, not what is
  heard. That bursts 11 ms apart are heard and bursts 4 ms apart are not is an
  argument, not a measurement. Experiment 10 also moved its number to the
  teacher's and the listener preferred nothing.

**The switch.** `noise_above` in `SourceFilter.forward` and
`TinyVoca.synthesize` (`src/tiny_voca/model.py`). Inference only; None is the
pulse train alone and the audio is then exactly what it was (tested). It is a
harmonic number per frame: above that harmonic of the frame's pitch, the
harmonic branch's filter is driven by white noise of the pulse train's spectral
density in place of the pulses, crossfaded over one octave, power kept.
`synthesize` takes two numbers, for the frames the model calls unvoiced and for
those it calls voiced. A setting is written `<unvoiced>,<voiced>`; `inf` is all
pulses.

- *Harmonics, not Hz.* A first sweep in Hz (`run-hertz.log`) put M2's vowels on
  the teacher's value with noise above about 3 kHz in voiced frames, while any
  cut-off below 6 kHz moved F2, which already matches, away from its teacher.
  33 harmonics is 3 kHz at 90 Hz and 8.3 kHz at 250 Hz.
- *Unvoiced frames, `2`:* M2's pauses land on the teacher (0.228 against 0.229),
  stops go from 0.544 to 0.200 (teacher 0.137), fricatives match. F2's
  fricatives and pauses end a little below the teacher (the pauses are at
  −52 dB).
- *Voiced frames, `33`:* M2's vowels 0.850 → 0.653 (teacher 0.689), voiced
  consonants 0.831 → 0.579 (0.536); F2 unchanged. `22` is far past the teacher
  on purpose, to learn whether the ear follows the number at all.
- *Not reached:* the vowel frames with no tracked pitch stay above the teacher
  at any setting that leaves the other vowels alone. Noise is not irregular
  phonation.

**Why the fundamental stays in unvoiced frames** (`low-band/run.log`). In the
band of the first four harmonics, the recording often hums where the model says
unvoiced. Share of that band's energy that repeats at the pitch, M2:

| frames the model calls unvoiced | recording | model | `0,inf` | `2,inf` |
|---|---|---|---|---|
| voiceless fricative | 0.637 | 0.795 | 0.279 | 0.777 |
| voiceless stop | 0.247 | 0.399 | 0.197 | 0.320 |
| voiced consonant | 0.399 | 0.694 | 0.207 | 0.582 |
| pause | 0.214 | 0.806 | 0.185 | 0.478 |

All noise (`0`) turns that hum into a rumble: in free-running renders the
60–300 Hz level strayed from the shipped render by −17.6 to +6.8 dB within
24 ms; with `2` it stays within −3.9 to +2.8 dB. `2` sits between the shipped
model and the recording in every class. The 2–8 kHz table is the same for both.
The price: what is left in M2's pauses, 46 dB under the speech as in the
recording, still hums more than the recording does (0.478 against 0.214).

**What it costs on the old metrics.** The mel distance to the recording rises
where noise replaces pulses: M2 voiceless fricatives 0.396 → 0.445, stops
0.620 → 0.658, pauses 0.769 → 0.799; `33` adds about 0.01 in voiced frames.
The 2–8 kHz level does not move (within 0.15 dB). A random excitation raises
the expected spectral error against any one recording, which is the pressure
that put these sounds in the pulse train to begin with.

**The page.** `~/tiny-voca-artifacts/pulsing/listen/`: six sentences (four M2,
two F2), a hidden copy of the shipped render and `2,inf`, `2,33`, `2,22`,
level-matched. `listen-unvoiced-0/` is a first build with `0` in unvoiced
frames, set aside before anyone listened.

**Verdict (the listener, blind, 2026-11-16): `2,33` is better on M2.** 24
ratings against the shipped render, no notes, nothing rated worse anywhere:

| | M2 (4 rows): better / same / worse | F2 (2 rows): better / same / worse |
|---|---|---|
| hidden copy of the shipped render | 0 / 4 / 0 | 1 / 1 / 0 |
| `2,inf` | 2 / 2 / 0 | 2 / 0 / 0 |
| `2,33` | 4 / 0 / 0 | 1 / 1 / 0 |
| `2,22` | 3 / 1 / 0 | 1 / 1 / 0 |

- On M2 the control is clean (the hidden copy was called the same four times
  of four) and the ratings follow the measure: noise in the unvoiced frames
  alone helps in half the rows, adding it above harmonic 33 in the voiced ones
  helps in all four, and going past the teacher (`22`) is not better than `33`.
  This is the first change in this file that the ear preferred.
- On F2 the page says nothing either way: the hidden copy was itself called
  better once in two rows, so "better" there is within the listener's noise.
  What it does say is that nothing was called worse.
- Four M2 sentences and one listener. It is a preference over the shipped
  render, not a statement that the roughness is gone.

**It is the default.** `model.NOISE_ABOVE = (2.0, 33.0)`. `TinyVoca.synthesize`,
`infer.speak` and `speak_text`, `tiny_voca.evaluate` and the ONNX export
(`tiny_voca.export`) use it; `noise_above=None`, or `--pulses-only` on the
command line, is the pulse train alone, bit for bit what shipped before.
`SourceFilter.forward` still defaults to None, so training is untouched and the
weights are the shipped ones: nothing was retrained. The exported graph was
checked against PyTorch on the shipped weights with `scripts/check_onnx.py`
(10 utterances, the same noise fed to both): the largest difference is 0.0049
of the peak, against 0.0048 for the pulses-only export, so the new branch adds
no error of its own.

**The check before release (the listener, blind, 2026-11-16): nothing worse,
nothing new, and the preference holds, more weakly.**
`~/tiny-voca-artifacts/noise-default/listen/`: twelve sentences nobody had
heard (six M2, six F2), each run from the two ONNX files, the release that
shipped and the new one, level-matched. The graphs draw their own noise, so the
control is a second run of the old file, not a copy. 24 ratings, no notes:

| | M2 (6 rows): better / same / worse | F2 (6 rows): better / same / worse |
|---|---|---|
| the old release, run a second time | 2 / 4 / 0 | 4 / 2 / 0 |
| the new release | 4 / 2 / 0 | 6 / 0 / 0 |

- Nothing was called worse and no note reports hiss, breath or a lisp. That is
  the release question, asked of the files that ship, and it is answered.
- A "better" on this page means little on its own: the old release run twice
  was called better than itself in six rows of twelve (four of six on F2). On
  the first page the control was the identical file and M2 called it the same
  four times of four, so either the redrawn noise is audible or two unknown
  clips after a known one invite a "better"; the page cannot tell which. Only
  the rows where the two clips were rated differently carry information.
- Row by row, the new release was rated above the control in five rows (M2:
  final-03, 04, 06; F2: final-09, 11), below it in one (M2 final-01) and level
  with it in six. On M2 alone: three above, one below, two level.
- With the first page, M2 stands at seven sentences above the control, one
  below and two level, out of ten. The direction has not changed; the size is
  small enough that the listener cannot always pick the new clip out.

**The acceptance battery (2026-11-17): passed. Word error rate, durations and
pitch are unchanged; the figures that moved are a pitch tracker following fewer
of M2's frames.**
`noise-default/battery.sh` runs the round-1 battery of §2 on this machine's CPU
twice: with `--pulses-only`, which is the release that shipped, and with the
default. `battery-noasr.sh` is the same without the recognizer (`eval-noasr/`).
`compare.py` sets both beside the run recorded on the GPU host when the model
shipped (`compare.txt`). Both are the shipped weights.

*The control.* Pulses alone on this machine is the recorded run again: the mel
distance is equal in 40 sentences of 40 per voice, the lengths and levels in 54
of 54, the intonation and voices probes to the second decimal, and two runs
here wrote byte-identical audio (108 clips of 108). Whatever separates the two
local runs is the switch and nothing else.

*Word error rate: no change* (large-v3-turbo, 40 validation, 40 test and 28
new sentences).

| | recorded, GPU host | pulses alone, here | default, here |
|---|---|---|---|
| validation | 2.00% | 2.54% | 2.00% |
| validation, F2 / M2 | 1.33% / 2.68% | 2.41% / 2.68% | 1.33% / 2.68% |
| test | 1.08% | 1.79% | 1.79% |
| test, F2 / M2 | 0.95% / 1.20% | 0.95% / 2.62% | 0.95% / 2.62% |
| new sentences | 0.0% | 0.0% | 0.0% |
| the recordings, validation / test | 1.41% / 2.09% | 1.20% / 2.09% | 1.20% / 2.09% |

Of the 108 sentences, 106 are transcribed word for word the same from both
renders, all 54 of M2 among them. The two that differ are F2, both in
validation, and both are spellings: "caserío" became "cacerío" (one error more)
and "2x3" became "dos por tres" (three fewer). That is all of validation's
2.54 → 2.00, and it is not an improvement. What this shows is that the
recognizer still understands every sentence it understood; it could not show a
small change in quality either way.

*The recognizer is repeatable on one machine and not across two.* Here it
transcribed the 80 recordings identically in both runs. Against the GPU host,
on audio the control shows to be the same, three generated transcripts score
differently ("dos por tres" / "2x3", "caserío" / "cacerío", "Ainhoa" /
"A Inoa") and so does one recording's: 0.5 to 0.7 points on a 40-sentence set
with no change in the audio. §2's figures are the GPU host's. The baseline for
anything transcribed here is the middle column.

*What the switch cannot touch did not move.* Durations, pauses and predicted
pitch are identical in all 108 sentences (`duration_ratio`, `pause_ratio`,
`generated_pitch_std_semitones`, `generated_voiced_fraction`, the length of
every clip). The level is the same (RMS equal; M2's peak 0.006 lower).

*The mel distance rose, as "what it costs" above said it would:* F2 +0.006
(higher in 34 sentences of 40), M2 +0.030 (40 of 40); validation 0.548 → 0.565,
test 0.542 → 0.560. A random excitation raises it against any one recording, so
this number cannot say better or worse here; it is recorded so that nobody
reads it as a regression later, or as the reason to switch the noise off.

*Voices and intonation.*

| | recorded | pulses alone, here | default, here |
|---|---|---|---|
| F2 median pitch | 244.2 Hz | 244.2 | 243.5 |
| M2 median pitch | 90.4 Hz | 90.4 | 90.9 (the reference is 90.9) |
| separation | 17.20 st, 92% of the reference | 17.20, 92% | 17.06, 91% |
| yes/no questions: mean slope, share rising | +3.52 st/s, 13 of 14 | +3.52, 13 of 14 | +3.33, 13 of 14 |
| yes/no rise recovered from the teacher | 87.5% | 87.5% | 86.6% |
| wh- questions (3 clips): rise recovered | 55.8% | 55.8% | 55.4% |
| statements, mean slope (40) | −4.26 | −4.26 | −4.37 |

The model's pitch is the same in both renders. These figures are read off the
audio by a pitch tracker, which follows fewer frames of the new one; the
punctuation probe below is where that was taken apart.

*The punctuation probe: 24 pairs of 24 rise more as a question, as shipped.*
The mean rise is 5.41 st/s (5.37 with pulses here, 5.42 recorded). One of its
side figures moved: questions whose slope is above zero, 17 of 24 → 14 of 24
(F2 10 → 9, M2 7 → 5: three M2 questions and one F2 crossed downwards, one M2
upwards), and M2's smallest rise fell from +1.98 to +0.91 st/s. Four checks,
because a question that stops rising would be a regression the listener was
not asked about:

1. *The pitch did not move* (`slope_check.py`). On the frames pYIN voices in
   both renders the median gap is 0.0 cents, with no octave jumps, and the
   probe's line refitted on those frames agrees between the renders (median
   gap F2 0.03 st/s, M2 0.06; 1 clip of 24 on M2 beyond 0.5).
2. *The window moved.* The probe fits the last 35% of whatever pYIN calls
   voiced. On M2 pYIN voices 6,695 frames where it voiced 8,229, and the end
   of that span comes 149 ms earlier on average (more than 30 ms in 18 clips of
   24); on F2 it moves 4 ms. Frames that lost their track are 19% of the M2
   question windows and 26% of the M2 statement ones, 1–2% on F2
   (`tail_level.py`).
3. *What those frames are, by the model's own labels* (`tail_frames.py`): of
   the 641 lost in M2's windows, 326 are pauses and the silence after the
   sentence (all but one more than 20 dB under the speech), 157 are voiceless
   consonants the model does not voice, mostly `s`, `θ`, `k`, and 54 more are
   other consonant frames. Those are the frames this section is about: sound with no pitch that
   the pulse train gave one, which the tracker then followed. 104 are vowel
   frames, 7% of the vowel frames in the windows.
4. *The vowels are the same sound* (`vowel_edges.py`: the harmonic branch
   alone, both ways, in a band that is all noise in an unvoiced frame and all
   pulses in a voiced one, read back as the share of the excitation that is
   noise). Frames the model does not voice: 0.99. Vowel frames: median 0.000,
   mean 0.003 on M2 and 0.002 on F2. The synthesis window carries the
   neighbour's noise into a vowel only in the frame that touches an unvoiced
   one, and there in 9% of them on M2 (17% on F2) by more than a quarter;
   two frames away it is nil, and no vowel of any length is left without clean
   frames. The vowel frames pYIN dropped (81 of the model's frames, of 1,280 in the
   windows) are not the ones that changed (noise share median 0.000; 6% above
   a quarter). They sit beside the unvoiced
   frames: 69% within three frames of one, against 20% of the vowel frames it
   kept, and three frames is half of pYIN's 64 ms window. The other 31% are
   further in and this does not explain them (the tracker's smoothing, or the
   noise above harmonic 33; not separated): 25 frames.

So the side figure fell because the probe now fits a shorter, different
stretch of M2's sentence ending, not because the endings changed. It also
means "above zero" was partly reading pitch off pulses in consonants and
silence before, and the two renders' figures are not the same measurement.
What is not covered by any of this: the listener heard one M2 question on the
final page (final-04, rated above the control).

*Speed: about a tenth slower.* `timing.py` runs the two exports in turn on the
same input (30 rounds per cell; three sentence lengths, both voices). The new
one takes 1.09 to 1.14 times as long in onnxruntime, with one thread and with
four, and 1.09 to 1.10 in PyTorch. Its fastest runs are 36 to 77 times real
time, against 38 to 88. The machine was busy with other work throughout (load
average 25 falling to 9, on 8 cores), so the ratio is the result and the
absolute figures are what a busy laptop does. A first run under heavier load
gave 1.06 to 1.16 for the exports (`timing-first.log`).

*Tests and the app.* The repository's tests pass on the working tree (96, 6
skipped). The new export has the inputs and outputs of the old one, and
`app_es/server.py` runs it unchanged when `TINY_VOCA_ES_MODEL` and
`TINY_VOCA_ES_CONFIG` point at it: both voices through `/api/tts` and
`/api/tts/stream`. The app's default path is still the file that shipped.

**What is still open.**

- *How much evidence there is.* One listener, ten M2 sentences and eight F2
  over two pages. On M2 the new render was rated above the control in seven,
  level with it in two and below it in one; nothing on either page was called
  worse than the shipped render. That supports "not worse, and usually a
  little better on M2". It does not support a size, and on F2 it supports only
  "not worse".
- *The sentence that went the other way.* M2 final-01, "Seis científicos suizos
  se sientan siempre en la misma silla.", was rated below the second run of
  the old file. One row, on a page whose control was over-rated throughout,
  but it is the sentence made of the sound the switch changes most (`s` is the
  commonest frame it turns to noise). If anyone reports a hiss or a lisp,
  listen there first.
- *It is not the teacher.* Every rating since the switch is against the shipped
  render; no page has put the new render beside the recording, which is the
  comparison the complaint in §3 was made on. By this section's measure M2's
  fricatives, stops, pauses and voiced consonants are now within 0.05 of the
  teacher and the tracked vowels 0.08 under it. Three things are known not to
  be there: the vowel frames where the teacher's voice
  is irregular (§8, a fifth of M2's vowel frames: 0.439 against 0.312, and no
  setting reaches them), the hum left in M2's pauses (0.478 against 0.214),
  and anything the ear hears that this measure does not, which §7 and §10 are
  two examples of.
- *The weights have never heard this excitation.* The filter was trained under
  pulses and is now driven by noise in bands it shaped for pulses. Training
  with the mixed excitation is the next experiment the rules allow (a variant
  has been preferred) and it has not been run. §10 is the reason not to expect
  much from more training against the same loss; the reason it could differ is
  that the excitation, not the loss, would change. It costs about an hour of a
  GPU, then this battery and one short page.
- *Numbers that changed meaning.* The mel distance to the recording is higher
  by construction. The punctuation probe's "above zero" figure is not the same
  measurement on the two renders. The recognizer differs between machines on
  identical audio, so a word error rate is comparable only with a run on the
  same machine.
- *The English model* (§9) has the artifact and the same head. The switch has
  not been tried there.
- *Release mechanics.* Nothing here is committed. `run-es.sh` and
  `app_es/server.py` still default to `tiny-voca-es-anneal.onnx`; the new
  export is `noise-default/onnx/tiny-voca-es-noise.onnx` with its `.json`, and
  the shipped file was not overwritten.

## 12. Reproducing

Scripts, all in `scripts/`:

| script | what it measures |
|---|---|
| `probe_periodicity.py` | HNR against the teacher, per voice |
| `probe_variants.py` | branch isolation: baseline / harmonic_only / noise_only / minimum_phase |
| `probe_anticausal.py` | anticausal cepstrum sweep, level matched, plus the difference signal |
| `probe_smearing.py` | synthesis window and pulse-gating variants |
| `probe_peakiness.py` | crest factor and kurtosis of voiced speech against the teacher, paired |
| `probe_dispersion.py` | phase dispersion of the excitation: measurements and the blind page |
| `probe_branches.py` | noise branch against harmonic branch, by frame class and band |
| `probe_pitch_labels.py` | where the teacher's pitch labels are missing, by position in the word |
| `probe_oracle.py` | the head fitted to one teacher recording at a time: what the head can do against what the model predicts (§10) |
| `probe_pulsing.py` | how much of the 2–8 kHz band arrives in bursts at the pitch, by phone class; noise excitation and its blind page (§11) |
| `listening_page.py` | the blind listening page the probes build |
| `probe_punctuation.py`, `probe_intonation.py`, `probe_voices.py` | the round-1 acceptance battery, with `python -m tiny_voca.evaluate` |
| `harmonic_es.sh` | the inter-harmonic fine-tune (failed; kept for the record) |
| `prepare_features.sh`, `upload_run.sh` | rebuild features on a fresh host; push a run to one |

Since §11 became the default, `synthesize` renders with `model.NOISE_ABOVE`
unless told otherwise. Everything in §2 to §10 was measured with pulses alone:
`probe_variants.py`, `probe_anticausal.py`, `probe_dispersion.py`,
`probe_branches.py` and `probe_oracle.py` pass `noise_above=None` themselves,
`probe_pulsing.py` names its settings, and `tiny_voca.evaluate`,
`probe_punctuation.py` and `tiny_voca.export` take `--pulses-only`.
`probe_periodicity.py` and `probe_smearing.py` were left alone: the first now
measures the default, the second builds its own copy of the head from pulses.

Artifacts, all local:

- `~/tiny-voca-artifacts/es-1/` — the run: corpus, 2.4 GB teacher FLAC, annotations, checkpoints
- `~/tiny-voca-artifacts/es-1-anneal/` — **the shipped model, step 142,000**
- `~/tiny-voca-artifacts/es-1-harmonic/` — the failed fine-tune, its eval and its A/B pages
- `~/tiny-voca-artifacts/variants/`, `~/tiny-voca-artifacts/anticausal/` — listening pages
- `~/tiny-voca-artifacts/phase-dispersion/` — experiment 10: measurements and its page
- `~/tiny-voca-artifacts/pitch-labels/` — §8
- `~/tiny-voca-artifacts/oracle/` — §10: 64 fitted recordings, their audio and measurements; `~/tiny-voca-artifacts/oracle-critics/` is the six refitted with the critics (`fit.log`, `pulsing.log`)
- `~/tiny-voca-artifacts/noise-default/` — §11, the default: `battery.sh` and `battery.log` (the acceptance battery, pulses alone and the default, in `eval/`), `battery-noasr.sh`, `battery-noasr.log` and `eval-noasr/` (the same without the recognizer), `compare.py` (the three runs side by side; `compare.txt` and `compare-noasr.txt` are its output), `slope_check.py`, `tail_level.py`, `tail_frames.py` and `vowel_edges.py` with their logs (why the punctuation probe's side figure moved on M2), `timing.py`, `timing.log` and `timing-first.log` (the two exports timed in turn), `features-es/` (validation and test features rebuilt on this machine), `onnx/` (the exports and their check against PyTorch), `final_page.py` and `listen/` (the page for the check before release: twelve new sentences, the release that shipped against the new one, both run from their ONNX files, with a second run of the old file as the control)
- `~/tiny-voca-artifacts/pulsing/` — §11: measurements (`run.log`; `low-band/run.log` compares what is kept as pulses in unvoiced frames; `run-unvoiced-0.log` and `run-hertz.log` are earlier sweeps, the second with cut-offs in Hz) and the page in `listen/` (`listen-unvoiced-0/` is the build set aside)

A GPU host is not running. Features rebuild from the teacher FLAC in ~10 min on
a many-core host (`prepare_features.sh`); the annotations do not need rebuilding.
The battery does not need a host: the validation and test features (1,230
recordings) rebuild on a laptop in a few minutes with `tiny_voca.prepare`, and
`tiny_voca.evaluate --features` reads them from wherever they are.
A 5090 on Runpod secure cloud costs $1.39/h and trains this model at ~6.1 steps/s
at the anneal's recipe (`batch_frames=9000`, `max_batch=32`).

## 13. Ground rules for the next session

- **The listener outranks the metrics.** Twice the numbers said "fine" and the
  ear said otherwise, and the ear was right both times.
- **Do not retrain to test a hypothesis you can test at inference.** Eight
  inference experiments cost minutes; the one training experiment cost an hour
  and answered nothing.
- **Do not trust a single sentence.** The −2.2 dB that started this was one clip.
  Use ≥30 utterances per voice for any model-vs-teacher claim.
- **Never compare F2 with M2.** HNR, crest factor and shimmer are all confounded
  by F0. Only model-vs-its-own-teacher is valid.
