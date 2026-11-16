# Round 2: what round 1 measured, and what to change

Written while round 1 trains. Everything under "measured" is a number from this
run, not an expectation. Update the result section when the run finishes.

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

**The voice, by Voxtral.** At step 8,000, 36,000 and 66,000 the word error rate
against a verbatim transcript is 0.0, naturalness 8/10, intelligibility 9/10,
Castilian with distinción on 11 of 12 samples. Intelligibility is not what
training improves after the first few thousand steps; prosody and artefacts are.

## What to change

### 1. Measure the teacher before generating, not after

The 77% figure for yes/no questions was only visible after 20,790 clips were
made. A 200-clip probe of each category, run before the full generation, would
have shown it for the price of a coffee and allowed either a different teacher
or a per-clip prosody filter.

Concretely: drop yes/no clips whose tail does not rise, and exclamation clips
whose pitch range is no wider than a statement's. The aligner and the ASR
cannot see either problem, which is why they survived screening.

### 2. Decouple the question tune from the vocabulary

Two thirds of the corpus's questions are now generated, and they are all
conversational and domestic ("¿Dónde dejaste las llaves?"). The statements are
mostly encyclopedic. A 4.5M model can learn "chatty words mean rise" instead of
"`¿` means rise".

`scripts/probe_punctuation.py` tests exactly this: it speaks validation
statements twice, changing only the punctuation. Run it on round 1's checkpoint
first. If the share of pairs that rise more as a question is low, round 2 needs
formal questions — usted-form, professional register, statement word order.

### 3. Give the teacher a better exclamation and alternative-question tune

Exclamations are 8.5% of the corpus and their contour is weak. Alternative
questions are 185 clips and the teacher's contour is mixed. Neither is a
corpus-size problem; both are a teacher problem. A teacher that can be given a
prosody reference, or a second teacher for those two categories, is the fix.

### 4. Overlap screening with generation

`generate_audio.py` runs Whisper in the main thread when `--asr-device cuda`,
so the GPU alternates between the teacher and the screener and sits at 18%
utilisation. On the CPU it runs in a background thread and overlaps. The
generation ran at 1.3-1.8 clips/s; the design intends roughly double.

### 5. Use a stronger ASR for evaluation

`screening` at 15% WER with `small` is fine for filtering. Judging the voice
with the same model is not: `small` multilingual's own Spanish errors are the
same order as the errors being measured. The final evaluation uses
`large-v3-turbo`.

### 6. Fix the listening test before trusting it

`validate_voice.py` had two bugs that made its word error rate meaningless: the
judge's `transcript` overwrote the transcriber's, and the transcriber
paraphrases ("las 8:30" for "las ocho y media", "plataforma" for "andén"),
scoring 42% error on a clip that was perfect. Both are fixed; the rate now comes
from the verbatim transcript.

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

- Does the model raise pitch on a `¿` it has never seen in that vocabulary?
  Round 1's punctuation probe answers this.
- Is 24.5 hours better than 19 for a 4.5M model, or is it capacity-limited?
  Compare round 1's final WER against the English model's 0.6-0.9%.
- The corpus is now 24.5 hours against the English 19, so the step count was
  raised from 104,000 to 122,000 to keep the number of passes over the data
  equal. If the later checkpoints are no better than 100,000, that is evidence
  the model is capacity-limited and round 2 should grow the model instead.
