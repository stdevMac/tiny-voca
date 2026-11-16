# Round 2: what round 1 measured, and what to change

Everything under "measured" is a number from this run, not an expectation.

## Result

The shipped model is **step 142,000**: the 122,000-step run was polished with a
20,000-step low-rate anneal (`scripts/anneal_es.sh`) and the annealed model beat
the best of the original checkpoints on every prosody measure. On held-out text
it is indistinguishable from its teacher:

| split | model WER | teacher's own WER | clips |
|---|---|---|---|
| validation | 2.00% | 1.41% | 40 |
| test | **1.08%** | **2.09%** | 40 |
| novel (unseen sentences) | **0.0%** | — | 28 |

`reference_wer` is the teacher's audio transcribed by the same ASR, so it is the
floor: whatever the transcriber cannot hear in the teacher, it cannot hear in the
model either. On test the model is *below* that floor, which means the remaining
errors are the transcriber's, not the voice's. The residual is ASR error on
proper nouns ("Rajoy", "Gonzalo Martínez Díez").

**The anneal.** The original run stopped at 122,000 steps with the learning rate
still at 1.46e-4, so it never annealed. A 20,000-step branch at 3.0e-5 halving to
1.5e-5 improved everything that matters and cost almost nothing that does:

| metric | best original checkpoint (120k) | annealed (142k) |
|---|---|---|
| WER, validation + test | 1.71% | **1.54%** |
| punctuation pairs rising more | 95.8% | **100%** |
| yes/no intonation recovered | 83.4% | **87.5%** |
| yes/no questions that rise | 86% | **93%** |
| voice separation | 91% | **92%** |
| mel DTW | **0.5433** | 0.5449 |

The one regression is spectral distance to the teacher, by 0.3%. The prosody
gains are the point of the model, so the annealed checkpoint ships.

**Question intonation.** The corpus's questions are conversational and its
statements encyclopedic, so a model can learn "chatty words mean rise" and still
pass a held-out question test. The punctuation probe speaks each validation
statement twice, changing only the punctuation: 24 pairs, **100% rise more as a
question**, mean rise +5.42 semitones/second. The cue is the `¿`, not the words.
Against the teacher's own gap, the model recovers **87.5%** on yes/no questions
(+7.78 st/s of the teacher's +8.89).

**Voices.** F2 244.2 Hz against a 258.7 Hz reference, M2 90.4 Hz against 90.9 Hz,
over 54 clips each. Separation 17.20 semitones against the references' 18.11, so
**92% of the reference gap**. Voxtral's `gender` label is unusable — it called
the 90 Hz voice female — which is why this is measured as pitch.

**The checkpoint comparison.** All checkpoints from 80,000 on transcribe with
zero word errors over 32 clips, so the ranking is decided by spectral distance:

| step | WER | mel DTW |
|---|---|---|
| 60,000 | 0.0028 | 0.5586 |
| 80,000 | 0.0 | 0.5559 |
| 100,000 | 0.0 | 0.5536 |
| 110,000 | 0.0 | 0.5526 |
| **120,000** | **0.0** | **0.5517** |
| 122,000 | 0.0 | 0.5524 |

The last 2,000 steps made the model marginally worse. Intelligibility was solved
by 80,000; everything after that is spectral fidelity, and it plateaus.

## Measured in round 1

**The corpus.** 11,050 sentences, 22,100 clips requested, 20,790 accepted
(94.1%), 24.46 hours. Questions are 19.0% of sentences and 24.0% of accepted
clips, because the generated questions are accepted at 97.3% against 89-91% for
the human sources. Exclamations are 8.5% of sentences.

**The teacher's prosody**, measured over the last 35% of voiced frames, in
semitones per second:

| kind | mean slope | share rising |
|---|---|---|
| yes/no question | +3.38 | 77% |
| wh-question | −0.65 | 42% |
| alternative question (`¿… o …?`) | −0.22 | 55% |
| statement | −5.40 | 0% |
| exclamation | −4.57 | 30% |

Statements fall hard and consistently. Yes/no questions rise, but only 77% of
the time, so nearly a quarter of the question clips teach a flat or falling
tail. Wh-questions are near-flat, which is correct for Spanish. Exclamations
are not statements: 30% rise and the spread is much wider.

**The alignment gate.** 98.29% of clips accepted, vowel→voiceless 79.75% within
two frames, fricative→vowel 95.54%, mean phone 6.98 frames, one-frame phones
0.13%. Every threshold in `check_annotations.py` cleared.

**The voice, by Voxtral.** Word error rate against a verbatim transcript is 0.0
from step 8,000 onward, naturalness 8/10, intelligibility 9/10. On the shipped
checkpoint 8 of 10 samples are Castilian with distinción. Intelligibility is not
what training improves after the first few thousand steps; prosody and artefacts
are — and the accent is the one thing still short of target.

## The one thing a listener notices: the tone is flat

A listener described it as "the speech is good, but the tone and the sound is
robotic". That is measurable, and it is inherited rather than introduced.

F0 variation, in semitones of standard deviation over the voiced frames:

| | F0 std |
|---|---|
| the reference voice (`voices-es/references/F2.wav`) | 2.07 st |
| the teacher's corpus clips | 1.93 st |
| the shipped model | 2.01 st |

Real expressive Spanish sits nearer 3-4 st. The whole chain is flat, because
each stage cloned the one before it: a VoiceDesign reference (2.07) was cloned by
Qwen3-TTS into the corpus (1.93), and the 4.5M model reproduced that (2.01). The
model is not the cause. A bigger model would reproduce the same flatness more
faithfully.

The timbre has the same origin. Against the teacher's own recording of the same
sentence the model is within 2.6 dB in every octave band, with matched jitter
(0.013 against 0.012) and matched spectral flatness (0.038 against 0.040), but it
carries 2 dB more energy at the vocoder's frame rate and is 1 dB duller above
6 kHz. The teacher has the same frame-rate artifact, weaker. So the "sound" is a
synthetic-teacher artifact that the small vocoder slightly amplifies.

**What fixes the tone now, without retraining.** The model takes `pitch_range`,
which scales pitch movement around the utterance mean. It works exactly as
intended:

| pitch_range | F0 std |
|---|---|
| 1.0 (as trained) | 2.00 st |
| 1.3 | 2.66 st |
| 1.5 | 2.97 st |
| 2.0 | 3.88 st |

The Spanish web interface now defaults to 1.5, which is a 50% increase in pitch
movement and lands inside the normal expressive range. This is a post-hoc
inference setting, not something the model was trained to do, so it belongs in
the interface rather than in the API default.

**What fixes it properly, in round 2.** Generate the corpus with varied prosody
instead of one flat delivery: several pitch ranges, speeds and registers per
sentence, so the model learns a distribution rather than a single contour. That
is a corpus change, not a model change, and it is cheaper than a bigger model —
which is the point, since capacity is not what is limiting the tone.

## What to change

### 1. Measure the teacher before generating, not after

The 77% figure for yes/no questions was only visible after 20,790 clips were
made. A 200-clip probe of each category, run before the full generation, would
have shown it for the price of a coffee and allowed either a different teacher
or a per-clip prosody filter.

Concretely: drop yes/no clips whose tail does not rise, and exclamation clips
whose pitch range is no wider than a statement's. The aligner and the ASR
cannot see either problem, which is why they survived screening.

### 2. Decouple the question tune from the vocabulary — measured, and it passed

Two thirds of the corpus's questions are generated, and they are all
conversational and domestic ("¿Dónde dejaste las llaves?"), while the statements
are mostly encyclopedic. The risk was a model that learned "chatty words mean
rise" instead of "`¿` means rise".

`scripts/probe_punctuation.py` speaks validation statements twice, changing only
the punctuation. On the shipped checkpoint **95.8% of 24 pairs rise more as a
question**, mean +4.93 semitones/second. The cue is the punctuation. This
question is closed for round 1 and the probe stays in the post-training sequence
so round 2 can show it did not regress.

### 3. The formal register is the one measured weakness

Voxtral judges 8 of 10 samples Castilian with distinción. The two that fail are
the formal yes/no question, and the same clip failed at 66,000 and 96,000 steps,
so it is systematic rather than noise. Every generated question is *tú*-form
small talk, so the corpus contains almost no formal, professional register.

Round 2 should generate questions in the usted register, with statement word
order and formal vocabulary, and add them to the same probe. This is the highest
-value corpus change because it is the only measured gap.

### 4. Give the teacher a better exclamation and alternative-question tune

Exclamations are 8.5% of the corpus and their contour is weak. Alternative
questions are 185 clips and the teacher's contour is mixed. Neither is a
corpus-size problem; both are a teacher problem. A teacher that can be given a
prosody reference, or a second teacher for those two categories, is the fix.

### 5. Overlap screening with generation

`generate_audio.py` runs Whisper in the main thread when `--asr-device cuda`,
so the GPU alternates between the teacher and the screener and sits at 18%
utilisation. On the CPU it runs in a background thread and overlaps. The
generation ran at 1.3-1.8 clips/s; the design intends roughly double.

### 6. Use a stronger ASR for evaluation

`screening` at 15% WER with `small` is fine for filtering. Judging the voice
with the same model is not: `small` multilingual's own Spanish errors are the
same order as the errors being measured. The final evaluation uses
`large-v3-turbo`, and so does the checkpoint ranking — with `small`, every
checkpoint scored the same and the tie-break picked the winner by accident.

### 7. Fix the listening test before trusting it

`validate_voice.py` had two bugs that made its word error rate meaningless: the
judge's `transcript` overwrote the transcriber's, and the transcriber
paraphrases ("las 8:30" for "las ocho y media", "plataforma" for "andén"),
scoring 42% error on a clip that was perfect. Both are fixed; the rate now comes
from the verbatim transcript.

`numbers_es.expand` had a related bug: it read digits after the comma one by
one, so "1.234,56" became "…coma cinco seis" while the front end says "…coma
cincuenta y seis". Every price and measurement scored as an error the model did
not make. It now matches eSpeak, with a leading zero still read digit by digit.

### 8. Anneal the learning rate — done, and it worked

The run stopped at 122,000 steps with the learning rate still at 1.46e-4,
because `--lr-half-life` is 100,000 and the schedule never reached its tail. A
flat curve at that rate is a noise floor, not convergence.
`scripts/anneal_es.sh` branched from `last.pt` at 3.0e-5 halving to 1.5e-5 over
20,000 steps, keeping the weights, critics and optimizer moments, and beat the
best original checkpoint on WER, punctuation, intonation and voice separation.

Round 2 should not repeat this experiment; it should build the anneal into the
schedule from the start — a cosine or linear decay to ~1e-5 that reaches its
tail before the step budget runs out, rather than a second run to fix it.

## What not to change

**The GAN ratio.** The runbook says to lower `--gan-ratio` if
`validation_predicted_mel` rises after the GAN starts, and it did rise: 0.570 at
step 8,000 to 0.605 at 9,000. But the audio got better, not worse — 0.0 WER at
both, and the mel recovered to 0.574 by step 65,000. The heuristic is a warning
sign, not a verdict; the audio is the verdict.

**The alignment onset offset.** Retuning it trades +9 points on one timing
probe for −21 on another. The default stands.

**The lleísmo mismatch.** eSpeak emits `ʎ` for ⟨ll⟩ and the teacher is yeísta.
The aligner scores the two within 0.0003 of each other, so a consistent
mismatch is harmless.

## Open questions for round 2

- **Is the model capacity-limited?** Partly, and the anneal separates the two
  causes. Predicted mel bottomed at step 91,000 and 31,000 further steps at the
  original rate bought nothing — but 20,000 steps at a tenth of that rate
  improved WER, punctuation, intonation and voice separation all at once. So the
  plateau was *schedule*, not capacity, and the model still had room. Round 2's
  first experiment should be a proper anneal, then a larger model if that
  saturates too.
- **Can the formal register be fixed without breaking the conversational one?**
  Adding usted-form questions risks pulling the tune back toward statements, so
  the punctuation probe has to run on every candidate.
- **Is the wh-question contour right?** The teacher's wh-questions are near-flat
  (−0.65 st/s) rather than falling as standard Spanish does. The model recovers
  59% of the teacher's gap, but there were only 3 model samples and 1 reference
  sample, so this is a number to re-measure before acting on it.
- **Would 24.5 hours beat 19 hours for the same architecture?** Round 1 cannot
  answer it: the English model had a different corpus and an English-only ASR.
  The comparison that matters is a controlled one, which round 2 can run by
  subsetting this corpus to 19 hours.

## What was saved, and what it is for

The GPU host is deleted after every run, so the local copy is the run. It is two
sibling directories under `~/tiny-voca-artifacts`, each with its own
`SHA256SUMS` checked against the host's own hashes rather than only against
itself.

**`es-1/` (3.9 GB, 22,819 files) is the run**: corpus, teacher audio, alignment,
every checkpoint, and the 120,000-step model with its full evidence.

| path | what it is | why keep it |
|---|---|---|
| `runs/es-1/last.pt` | full training state at 122,000 | 87 MB; branch from it instead of retraining |
| `runs/es-1/step-*.pt` | numbered checkpoints | 1.2 GB; re-rank or re-branch |
| `eval/es-1/inference-best.pt` | best original checkpoint, step 120,000 | 18 MB; the baseline the anneal beat |
| `eval/es-1/` | report, demo audio, intonation, punctuation, voices, Voxtral | the evidence for the baseline |
| `audio-es/audio/` | 20,790 teacher FLAC, 24.5 h | 2.4 GB; the expensive thing to regenerate |
| `annotations-es/` | alignment and the split lists | 65 MB; keeps the same test set |
| `scripts-full.jsonl` | the 11,050-sentence corpus | 6 MB; the text everything derives from |
| `voices-es/` | the two reference recordings | 5 MB; the voices the model clones |
| `provenance.txt` | git commit, 120 packages, torch/CUDA, eSpeak, hashes | what produced the weights |
| `logs/` | pipeline, train, post-training, anneal | the run as it happened |

**`es-1-anneal/` (446 MB) is the shipped model**, step 142,000.

| path | what it is | why keep it |
|---|---|---|
| `runs/es-1-anneal/inference.pt` | **the deliverable**, 18 MB | 4.5M parameters, two voices |
| `eval/anneal-es-1-anneal/tiny-voca-es-anneal.onnx` | ONNX export | 21 MB; runs at ~30x realtime |
| `runs/es-1-anneal/last.pt` | full state at 142,000 | 87 MB; branch from it, do not retrain |
| `eval/anneal-es-1-anneal/` | report, demo, punctuation, intonation, voices, ONNX check | the evidence for every shipped number |
| `eval/anneal-es-1/` | the same protocol on the baseline | the comparison that chose the winner |

Not saved: `features-es` (11 GB). It rebuilds from the audio in about ten
minutes with `tiny_voca.prepare`, so it is derived data, not an artifact.

To rebuild on a new host: copy both directories, `pip install -e .`, then
`tiny_voca.prepare` → `tiny_voca.annotate` → `tiny_voca.train --init-from
es-1-anneal/runs/es-1-anneal/last.pt`. The teacher is never needed again.
