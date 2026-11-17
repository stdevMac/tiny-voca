---
license: cc-by-4.0
language:
- es
task_categories:
- text-to-speech
pretty_name: tiny-voca-es
size_categories:
- 10K<n<100K
tags:
- synthetic
- spanish
- es-ES
- audiofolder
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train/**
  - split: validation
    path: data/validation/**
  - split: test
    path: data/test/**
---
# tiny-voca-es dataset

A synthetic Castilian Spanish (es-ES) speech corpus with two voices, F2 (female)
and M2 (male), made to train the tiny-voca-es text-to-speech model. Nearly every
sentence is read by both voices.

It is the Spanish counterpart of the
[tiny-voca dataset](https://huggingface.co/datasets/n4ze3m/tiny-voca) by Muhammed
Nazeem (n4ze3m), made with the same method.

| | |
|---|---|
| Clips | 20,790 |
| Duration | 24.46 hours |
| Sentences | 10,583 distinct texts (11,050 were requested) |
| Voices | F2 (10,389 clips), M2 (10,401 clips) |
| Audio | 24 kHz mono, 16-bit FLAC |
| Language | Spanish, Castilian (es-ES) |
| License | CC BY 4.0 (see [License](#license)) |

- Model: https://huggingface.co/marcosmaceo/tiny-voca-es
- Code: https://github.com/stdevMac/tiny-voca

## Usage

```python
from datasets import load_dataset

data = load_dataset("marcosmaceo/tiny-voca-es")   # audiofolder: train, validation, test

row = data["train"][0]
print(row["voice_id"], row["text"])
audio = row["audio"]["array"]
```

## Contents

| Split | Clips | Hours | F2 | M2 |
|---|---:|---:|---:|---:|
| train | 19,560 | 22.98 | 9,774 | 9,786 |
| validation | 675 | 0.81 | 338 | 337 |
| test | 555 | 0.67 | 277 | 278 |

Splits are made by sentence: both readings of a sentence are always in the same
split, and no sentence is in two splits. The validation and test sets (360 and
300 requested sentences) contain only sentences from the Common Voice files and
the project templates; every LLM-written question and exclamation is in train.

Repository layout:

| Path | Contents |
|---|---|
| `data/<split>/<voice>/<id>.flac` | The accepted clips, one folder per voice (F2, M2) |
| `data/<split>/metadata.jsonl` | One row per clip, the columns below |
| `annotations/` | Phoneme tokens and durations, see [Annotations](#annotations) |
| `text/scripts-full.jsonl` | All 11,050 requested sentences, with source and split |
| `voices/` | `voices.json` (the voice descriptions, texts and seeds), `es-ES.json` (the same descriptions as used by the code), `references/F2.wav` and `M2.wav` (the voice prompts) |
| `generation/manifest.jsonl` | The generator's full record for the accepted clips (its `audio` paths are relative to the generator's output folder; use `data/<split>/<voice>/<id>.flac` here) |
| `generation/review.jsonl` | The 1,310 clips that failed screening twice; their audio is not included |
| `generation/progress.json`, `generation/provenance.txt` | End state of the generation run; software versions and front-end fingerprint of the host that made the data |
| `SHA256SUMS` | Checksums of every file in this repository except this README |

Text categories in the accepted clips:

| Category | Clips | Origin | License of the text |
|---|---:|---|---|
| `reading` | 10,112 | Mozilla Common Voice, Wikipedia sentences (`common_voice_wiki`) | CC0 |
| `conversation` | 4,977 | Mozilla Common Voice, sentence collector (`common_voice_sentence_collector`) | CC0 |
| `question` | 2,857 | Written for this project with an LLM (`mistral_questions`) | project-own |
| `pronunciation` | 1,658 | Templates written for this project: numbers, dates, money, names, lists (`tiny_voca_templates`) | project-own |
| `exclamation` | 1,186 | Written for this project with an LLM (`mistral_exclamations`) | project-own |

The questions and exclamations were added because the Common Voice files held
too few: Spanish marks a yes/no question with intonation alone, and the corpus
without them was 7% questions. Numbers are written out in words in all text, so
the corpus has almost no digits.

## How it was made

1. The two voices were created from Spanish text descriptions with Qwen3-TTS
   1.7B VoiceDesign, asking for Castilian speakers with the s/z distinction.
   They are not recordings or imitations of real people. The reference clips
   are in `voices/references/`, and the descriptions and seeds are in
   `voices/voices.json`.
2. Each sentence was synthesized in both voices with Qwen3-TTS 1.7B Base
   (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`), using the reference clips as the voice
   prompt.
3. Each clip was transcribed with Whisper small. Clips above 15% word error rate,
   or with implausible duration, speaking rate or silence, were generated a
   second time (532 accepted clips are second attempts). Clips that failed twice
   were left out: 1,310 of 22,100. All 1,310 are listed in
   `generation/review.jsonl` with the reason `asr_mismatch`.
4. Accepted clips were peak-normalized to 0.95 and stored as 16-bit FLAC.

The generation scripts are in the code repository (`scripts/generate_audio.py`,
`scripts/build_text_es.py`, `scripts/build_questions_es.py`). The automatic
screening catches wrong or missing words. It does not measure naturalness, and
the corpus has not been checked by listeners clip by clip.

## Columns

| Column | Description |
|---|---|
| `file_name` | Path of the clip in the split folder, `<voice>/<id>.flac` |
| `id` | Clip ID, `<text_id>_<voice_id>` |
| `text_id`, `voice_id` | Sentence and voice (`F2` or `M2`) |
| `text` | The sentence that was read |
| `category`, `origin` | Kind of sentence and where it came from |
| `source_id`, `source_license`, `source_url` | Source attribution |
| `duration_seconds` | Audio length |
| `asr_transcript`, `normalized_wer` | Whisper transcript and its normalized word error rate against `text` |
| `attempt`, `seed` | Generation attempt (1 or 2) and random seed |
| `teacher_model`, `reference_audio` | Model and voice prompt used |

## Annotations

`annotations/` holds the phoneme-level annotations used to train the model:
phonemes, stress, and the duration of each phoneme in 10.7 ms frames (256
samples at 24 kHz), from forced alignment with
`facebook/wav2vec2-lv-60-espeak-cv-ft`. The method is the one described in the
upstream project's
[annotations-v3.md](https://github.com/stdevMac/tiny-voca/blob/main/docs/annotations-v3.md).
Phonemes come from eSpeak NG 1.52.0, voice `es`, accent es-ES: 61 symbols, 42
phones, front-end fingerprint
`54572700c27c1563f4a58567cd3ec1b8b26e14b77ec670bf912b612b9e7623fa`. Annotation
fingerprint: `6fe19bd24d1e90fa13c6f9f91abe62ed56b1bfe5aac58a2786254464b0185f06`.

| Split | Accepted | Rejected | Hours accepted |
|---|---:|---:|---:|
| train | 19,238 | 322 | 22.68 |
| validation | 656 | 19 | 0.80 |
| test | 541 | 14 | 0.66 |

Rejected rows are listed in the `*.rejected.jsonl` files: weak alignment,
vowels the pitch tracker found unvoiced, low mean confidence, or an overlong
phone. `summary.json` has the quantiles and the timing probe;
`annotations.json` has the identity of the run (aligner revision, settings).
The `path` field in a row points into the acoustic feature directory of the
training host and is not usable here; match rows to audio by `id`.

As in the upstream corpus, no hand-labelled boundaries exist; accuracy is
estimated from the voicing probe in `summary.json` and from whether a model
trained on these durations speaks intelligibly.

## Limitations

- The speech is synthetic. It carries the pronunciation habits and errors of the
  model that generated it, including a flat pitch (about 1.9 semitones of
  standard deviation) and yes/no questions that rise in only 77% of clips.
- Two voices, one accent (Castilian, with distinción), read speech only.
- The teacher and eSpeak NG disagree on some words (heteronyms, foreign names);
  the annotation token is eSpeak's and the audio is the teacher's.
- Some Wikipedia sentences are fragments or lists of proper names.
- The validation and test sets are small (675 and 555 clips).

## License

CC BY 4.0 for the audio, metadata and annotations. Please credit
"tiny-voca-es dataset" (https://huggingface.co/datasets/marcosmaceo/tiny-voca-es)
and the upstream tiny-voca dataset by Muhammed Nazeem. See `LICENSE.md` for the
third-party material: the sentences are CC0 (Mozilla Common Voice) or written
for this project, and the audio was generated with Qwen3-TTS (Apache-2.0).

```bibtex
@misc{marcosmaceo2026tinyvocaesdata,
  author = {marcosmaceo},
  title  = {tiny-voca-es dataset: synthetic two-voice Castilian Spanish speech},
  year   = {2026},
  url    = {https://huggingface.co/datasets/marcosmaceo/tiny-voca-es},
  note   = {Spanish counterpart of the tiny-voca dataset by Muhammed Nazeem, https://huggingface.co/datasets/n4ze3m/tiny-voca}
}
```
