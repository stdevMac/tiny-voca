---
license: apache-2.0
language:
- es
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- spanish
- es-ES
- onnx
- source-filter
- synthetic-data
datasets:
- marcosmaceo/tiny-voca-es
---
# tiny-voca-es

tiny-voca-es is a small Spanish (Castilian, es-ES) text-to-speech model with
two voices, F2 (female) and M2 (male). The complete model, including the
waveform generator, has 4,534,411 parameters and was trained from scratch on
about 22.7 hours of synthetic speech. It runs faster than real time on a laptop
CPU (36 to 77 times real time in ONNX Runtime, see [Speed](#speed)).

It is a Spanish fork of [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) by
Muhammed Nazeem (n4ze3m). The architecture, the training recipe and the
waveform generator are his; this release changes the text front end, the
phoneme inventory, the voices and the data. See [Credit](#credit).

<audio controls src="https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/F2-1.wav"></audio>

- Code: https://github.com/stdevMac/tiny-voca
- Live demo: https://stdevmac.github.io/tiny-voca/
- Dataset: https://huggingface.co/datasets/marcosmaceo/tiny-voca-es
- Upstream English model: https://huggingface.co/n4ze3m/tiny-voca

| | |
|---|---|
| Parameters | 4,534,411 |
| Voices | F2, M2 (Castilian Spanish, with the distinction between s and z) |
| Output | 24 kHz mono |
| Input | Text, converted to phonemes with eSpeak NG 1.52.0 |
| Pretrained weights in the model | None |
| Training step | 142,000 |
| License | Apache-2.0 (weights and code); eSpeak NG is GPL-3.0, see [Phonemizer](#phonemizer) |

## Files

| File | Size | Contents |
|---|---:|---|
| `tiny-voca-es.pt` | 18.2 MB | PyTorch inference checkpoint, float32, with its configuration |
| `tiny-voca-es.onnx` | 21.4 MB | ONNX export, float32, opset 17, with the mixed excitation described below |
| `tiny-voca-es.json` | 1 KB | Phoneme vocabulary, voices, front-end fingerprint, excitation setting of the ONNX file |
| `samples/` | | The audio on this page |
| `SHA256SUMS` | | Checksums of all of the above |

SHA256:

- `tiny-voca-es.pt`: `89c323d517ceb4279cdc39e1f61ae790b721362c0d2fb20fc497289e45e445c9`
- `tiny-voca-es.onnx`: `b4a9ca3becb9c7e6138c96ca8f7dceae760ccccf9305fbea9a6990667ff23a6c`
- `tiny-voca-es.json`: `bec46d85ccf80f746f7052c8532c9e74d04c00763831620c47f5442b52afc05f`

The `.pt` and the `.onnx` hold the same weights (step 142,000). The `.onnx`
renders with the default excitation (see [Excitation](#excitation)); the `.pt`
does too when it is run with the repository's code.

## Samples

Rendered from `tiny-voca-es.pt` with the default settings. None of these
sentences is in the training data.

| Text | Audio |
|---|---|
| Hola, ¿cómo estás? Esto es tiny voca, una voz pequeña en español. (F2) | <audio controls src="https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/F2-1.wav"></audio> |
| Zaragoza está a tres horas de Madrid en tren. (F2) | <audio controls src="https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/F2-2.wav"></audio> |
| Buenos días. ¿Quieres un café con leche? (M2) | <audio controls src="https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/M2-1.wav"></audio> |
| ¡Qué sorpresa! No esperaba verte por aquí. (M2) | <audio controls src="https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/M2-2.wav"></audio> |

## Usage

The model needs the code in the GitHub repository (the text front end lives
there) and eSpeak NG **1.52.0** installed as a program on the `PATH`.

```bash
# eSpeak NG 1.52.0, for example: brew install espeak-ng   (check: espeak-ng --version)
git clone https://github.com/stdevMac/tiny-voca && cd tiny-voca
pip install -e ".[export]"       # installs torch too
hf download marcosmaceo/tiny-voca-es --local-dir tiny-voca-es
```

### The repository's command line

```bash
TINY_VOCA_LANGUAGE=es python -m tiny_voca.infer \
  --checkpoint tiny-voca-es/tiny-voca-es.pt \
  --voice F2 \
  --text "Hola, ¿cómo estás? Esto es tiny voca en español." \
  --output hola.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text`. |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Added to log F0. 0.058 is about one semitone. |
| `--pitch-range` | 1.0 | Scales pitch movement around the utterance mean. |

Text of any length works. The command reads it one sentence at a time and joins
the audio. A text that ends without a mark gets a full stop. The saved audio
never peaks above 0.99.

### ONNX

The graph takes phoneme ids and stress values and returns the waveform. Text is
converted to ids outside the graph with the repository's `tiny_voca.text`, which
calls eSpeak NG; the ids are the positions in `symbols` in
`tiny-voca-es.json`. The noise signal is drawn inside the graph.

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tiny_voca.text import phonemize   # runs eSpeak NG 1.52.0; Spanish is the default

phones = phonemize("Hola, ¿cómo estás? Esto es tiny voca.")
one = lambda value: np.array([value], dtype=np.float32)

session = ort.InferenceSession("tiny-voca-es/tiny-voca-es.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = F2, 1 = M2
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
})
audio = audio[0]
audio = audio * min(1.0, 0.99 / np.abs(audio).max())    # the model can peak above 1.0
sf.write("hola.wav", audio, 24000)
```

| Input | Type | Shape | Meaning |
|---|---|---|---|
| `tokens` | int64 | [1, N] | Phoneme ids, including the start and end tokens |
| `stress` | int64 | [1, N] | 0 none, 1 primary, 2 secondary |
| `voice` | int64 | [1] | 0 = F2, 1 = M2 |
| `speed` | float32 | [1] | 1.0 as predicted; larger is faster |
| `pitch_shift` | float32 | [1] | Added to log F0 |
| `pitch_range` | float32 | [1] | Scales pitch movement |

| Output | Type | Shape | Meaning |
|---|---|---|---|
| `audio` | float32 | [1, samples] | Waveform at 24 kHz |
| `durations` | int64 | [1, N] | Length of each token in frames of 256 samples |

The model reads one utterance per call, up to 512 tokens and 32 seconds. Split
longer text into sentences (`tiny_voca.text.sentences` does this) and end each
utterance with a mark such as a full stop. The audio can peak slightly above 1.0
(1.02 in one evaluation clip), so scale it down before saving 16-bit samples, as
above.

## Voices

| Voice | Designed as |
|---|---|
| F2 | A young woman from Madrid, about thirty, clear and warm, medium pitch |
| M2 | An adult man from Valladolid, about forty-five, calm, medium-low pitch |

Both were designed from Spanish text descriptions with Qwen3-TTS VoiceDesign,
asking for Castilian pronunciation with the s/z distinction (the descriptions
and seeds are in `voices.json` in the dataset). They are not recordings or
imitations of real people. Measured on 54 clips each, the median pitch is
243.5 Hz for F2 (reference 258.7 Hz) and 90.9 Hz for M2 (reference 90.9 Hz).

## Phonemizer

The model was trained on phonemes from **eSpeak NG 1.52.0**, voice `es`
(accent `es-ES`), and needs the same front end at inference. A different
eSpeak version can write different phonemes for the same word, which the model
has never seen.

- The configuration records the front-end fingerprint
  `54572700c27c1563f4a58567cd3ec1b8b26e14b77ec670bf912b612b9e7623fa`. It
  covers the phoneme inventory (61 symbols), the voice and the fallback rules.
  The repository's code compares it with the checkpoint and refuses to run on a
  mismatch.
- Check your install with `espeak-ng --version`; it must say 1.52.0.
- eSpeak NG is licensed GPL-3.0. This model repository does not contain or
  link to it: the Python code runs it as a separate program and reads its
  output, and no eSpeak NG code or data is part of the weights. The
  [browser demo](https://stdevmac.github.io/tiny-voca/) does ship eSpeak NG
  1.52.0 compiled to WebAssembly, under GPL-3.0, with its licence and the
  script that builds it from the upstream `1.52.0` tag
  (`scripts/build_espeak_wasm.sh` in the GitHub repository). If you ship the
  weights together with eSpeak NG you are shipping a GPL-3.0 program, with
  the obligations that come with it.
- Numbers, dates and money are read by eSpeak NG from context ("4.090 euros"
  becomes "cuatro mil noventa euros"). Slashes in dates are read literally, so
  write dates out.

## Excitation

The waveform is made by a source-filter synthesizer: a band-limited pulse train
at the predicted pitch and a noise signal, each shaped by a time-varying
filter that the network predicts once per frame (256 samples, 10.7 ms).

The released weights were trained with pulses alone as the harmonic source.
At inference the default replaces part of that pulse train with noise of the
same power: above harmonic 33 of the pitch in voiced frames and above harmonic 2
in unvoiced frames (`model.NOISE_ABOVE = (2, 33)` in the code; `noise_above` in
`tiny-voca-es.json`). Voiceless consonants, bursts and pauses are noise, and
the trained head renders them partly as pitch pulses, which is audible as
roughness at the start and end of words, more on M2 than on F2.

The setting was chosen by blind listening, not by a metric. On M2 a listener
rated the new render better in four rows of four, with a hidden copy of the
first release rated the same as itself; on a second page of twelve new sentences
nothing was rated worse than the first release (one M2 row was rated below a
second copy of it) and the preference held, more weakly. That is one listener and about twenty sentences: it supports "not worse,
and usually a little better on M2", and for F2 only "not worse". The weights
were not retrained for this and have not been tried with the mixed excitation in
training.

The first release's render is available: `--pulses-only` in `python -m tiny_voca.export`
and `python -m tiny_voca.evaluate` produces it, and `noise_above=None` does the same in
`tiny_voca.infer.speak`. (The `tiny_voca.infer` command line has no switch for it.)
The default costs about a tenth more compute (see [Speed](#speed)) and raises
the mel distance to the training recordings (validation 0.548 to 0.565, test
0.542 to 0.560), which is expected when a random signal replaces a regular one
and is not a quality measurement.

## Model

Text is converted to phonemes with a stress value for each. A transformer
encoder (4 blocks) reads the phonemes, a duration predictor gives each one a
length in frames, and two frame-level transformers (2 and 5 blocks) predict
pitch, voicing and energy and then the parameters of the waveform generator.
All blocks use 192 channels, 4 heads and rotary position embeddings. The
waveform generator is not a neural vocoder; it has the structure of the English
model's, which has 25,862 parameters. Spanish changes only the phoneme
inventory (42 phones, 61 symbols with the special and punctuation symbols, so
the embedding table is smaller) and the data.

## Speed

ONNX Runtime on a busy laptop CPU, with one and with four threads, the
fastest runs over three sentence lengths and both voices: 36 to 77 times real
time with the default excitation, against 38 to 88 for pulses alone. The machine
was running other work, so treat the ratio (1.09 to 1.14 times as long with the
default) as the result and the absolute figures as a lower bound.

## Training

| | |
|---|---|
| Data | [marcosmaceo/tiny-voca-es](https://huggingface.co/datasets/marcosmaceo/tiny-voca-es): 19,238 aligned training clips, 22.7 hours, two voices (the dataset has 20,790 clips, 24.5 hours, in all splits) |
| Steps | 142,000: 122,000 steps, then a 20,000-step anneal at a low learning rate (3.0e-5 halving to 1.5e-5) |
| Hardware | One RTX 4090; the anneal took 80 minutes |
| Losses | Multi-resolution spectral losses; adversarial loss with multi-period and multi-resolution discriminators after step 8,000 |

The training audio is synthetic. Both voices were created from text
descriptions with Qwen3-TTS VoiceDesign, and the recordings were generated with
Qwen3-TTS 1.7B Base cloning the two reference clips. No recording of a real
speaker is involved. Phoneme durations come from forced alignment with
`facebook/wav2vec2-lv-60-espeak-cv-ft`, which was used only to label the
training data. No pretrained weights are part of the model.

The sentences come from the Mozilla Common Voice Spanish sentence files (CC0),
plus questions and exclamations written for this project with a language model
(Mistral Large) and templates written for this project. The dataset card has
the details.

## Evaluation

Word error rate of synthesized speech, transcribed with faster-whisper
large-v3-turbo (Spanish). The teacher's recordings of the same sentences are
scored the same way for reference. Rendered on a laptop CPU, voice defaults,
seed 0.

| Sentences | Clips | WER, default render | WER, `--pulses-only` | WER, recordings |
|---|---:|---:|---:|---:|
| Validation | 40 | 2.00% | 2.54% | 1.20% |
| Test | 40 | 1.79% | 1.79% | 2.09% |
| New sentences, not in the dataset | 28 | 0.0% | 0.0% | |

These sets are small and the scores are a sanity check, not a benchmark. Of the
108 sentences, 106 are transcribed word for word the same by both renders; the
validation difference is one spelling that went each way ("caserío" read as
"cacerío", and "2x3" read as "dos por tres"). Recognizer output differs between
machines on identical audio by 0.5 to 0.7 points on a set of 40, so compare a
word error rate only with one measured on the same machine. When the model was
first evaluated on the GPU host that trained it, the pulses-only render scored
2.00% on validation and 1.08% on test.

Other checks, on the same files:

- Durations, pauses and predicted pitch are identical between the default and
  pulses-only renders in all 108 sentences.
- Question intonation: speaking each of 24 validation statements twice,
  changing only the punctuation, the question rose more in 24 of 24 pairs
  (the cue is the opening `¿`, not the words). On yes/no questions the model
  recovers 86.6% of the teacher's rise (87.5% with pulses only).
- Accent: an audio language model (Voxtral, used for checking only) judged 8 of
  10 samples of the shipped checkpoint (rendered before the mixed excitation) as Castilian with the s/z distinction.
  Treat it as indicative.

No formal listening test has been run.

## Limitations

- Spanish of Spain (es-ES) only. The front end has rules for Rioplatense
  (`es-AR`) and Cuban (`es-CU`) pronunciation, but they are approximations that
  no native speaker has checked, and this model was trained on es-ES phonemes.
- Two fixed voices, no voice cloning.
- The training data is itself synthetic, so the model cannot be more natural
  than the model that generated it, and it inherits the teacher's habits. The
  prosody is flat: the pitch varies by 2.0 semitones (standard deviation),
  matching the teacher, where expressive Spanish is nearer 3 to 4. The
  `--pitch-range` option widens it without retraining (1.5 gives about 3.0
  semitones); it is an inference setting the model was not trained for.
- Word starts and ends can sound rough or faintly echoed, more on M2. The
  mixed excitation reduces it; it is not as clean as the teacher recordings.
  If you hear a hiss or a lisp, it is most likely on a word made of `s`.
- The same text is always read the same way; there is no sampling of prosody.
- Yes/no questions in the teacher rise only 77% of the time, so some questions
  end flat.
- Proper names, foreign words and unusual spellings are often mispronounced;
  numbers, abbreviations and dates depend on eSpeak NG.
- It needs eSpeak NG 1.52.0 exactly and the repository's text code; it is not a
  self-contained ONNX file.
- No streaming. Up to 512 tokens (32 seconds) per utterance.

## License

The weights and code are released under Apache-2.0. eSpeak NG (GPL-3.0) is
required at inference. It is not distributed in this repository, where it is
used as a separate program; the browser demo distributes a WebAssembly build of
it under GPL-3.0. The training data's text is CC0 or written for this project; see the
dataset card for the dataset's license. Qwen3-TTS (Apache-2.0) generated the
training audio, `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0) labelled it
and Whisper (MIT) screened and scored it; none of them is part of the model.

## Credit

This is a fork of tiny-voca by Muhammed Nazeem
([github.com/n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca),
[huggingface.co/n4ze3m/tiny-voca](https://huggingface.co/n4ze3m/tiny-voca),
Apache-2.0). The model, the training recipe, the source-filter waveform
generator and the annotation method are from that project. The design in turn
builds on FastSpeech 2, DDSP and the neural homomorphic vocoder, and HiFi-GAN
and UnivNet; no code or weights from those projects are used. If you use this
model, please cite the original project as well.

## Citation

```bibtex
@software{marcosmaceo2026tinyvocaes,
  author = {marcosmaceo},
  title  = {tiny-voca-es: a 4.5M-parameter two-voice Spanish text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/marcosmaceo/tiny-voca-es},
  note   = {Spanish fork of tiny-voca by Muhammed Nazeem, https://github.com/n4ze3m/tiny-voca}
}

@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
