# tiny-voca-es

A small Spanish (Spain) text-to-speech model. It has two voices, 4.5M
parameters, and runs on a CPU or in the browser.

It is a fork of [tiny-voca](https://github.com/n4ze3m/tiny-voca) by Muhammed
Nazeem. The model, the vocoder and the training recipe are his. This fork
changes the text front end, the phoneme inventory and the voices, and trains a
new model on Spanish speech.

[Demo](https://stdevmac.github.io/tiny-voca/) ·
[Model](https://huggingface.co/marcosmaceo/tiny-voca-es) ·
[Dataset](https://huggingface.co/datasets/marcosmaceo/tiny-voca-es) ·
[Code](https://github.com/stdevMac/tiny-voca) ·
[Upstream (English)](https://github.com/n4ze3m/tiny-voca)

| | |
|---|---|
| Parameters | 4,534,411, waveform generator included |
| Size | 18 MB (PyTorch), 21 MB (ONNX) |
| Voices | F2 (female) and M2 (male), Castilian Spanish with distinción |
| Output | 24 kHz mono |
| Phonemizer | eSpeak NG 1.52.0, `es` voice, 42 phones |
| Training data | 22.7 hours of synthetic speech from Qwen3-TTS (24.5 hours in the published dataset, splits included) |
| License | Apache-2.0 (code and weights) |

## Listen

- **In the browser:** <https://stdevmac.github.io/tiny-voca/>. Everything runs
  on your machine: the model is downloaded from Hugging Face, eSpeak NG 1.52.0
  runs as WebAssembly, and the model runs in ONNX Runtime Web. Nothing you type
  is sent to a server.
- **Samples:** the [`samples/`](https://huggingface.co/marcosmaceo/tiny-voca-es/tree/main/samples)
  folder of the model repository, for example
  [F2](https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/F2-1.wav)
  and [M2](https://huggingface.co/marcosmaceo/tiny-voca-es/resolve/main/samples/M2-1.wav).

## Run it locally

You need Python 3.10 or newer and eSpeak NG **1.52.0** on `PATH`.

```bash
git clone https://github.com/stdevMac/tiny-voca.git
cd tiny-voca
brew install espeak-ng            # macOS; check that `espeak-ng --version` says 1.52.0
./run-es.sh
```

`./run-es.sh` starts a local web page at <http://127.0.0.1:8000>. The first run
creates `app_es/.venv` with the few packages the page needs (no PyTorch). If
there are no local model files, it downloads `tiny-voca-es.onnx` and
`tiny-voca-es.json`, about 21 MB, from the
[model repository](https://huggingface.co/marcosmaceo/tiny-voca-es) into
`weights/es/`. The model runs on the CPU with ONNX Runtime. See
[`app_es/README.md`](app_es/README.md) for its options and its HTTP API.

The version of eSpeak NG matters. A different version produces slightly
different phonemes, and the server refuses to start when the phonemes do not
match the ones the model was trained on. Ubuntu 24.04 ships 1.51; build 1.52.0
from source as [`scripts/host_setup.sh`](scripts/host_setup.sh) does.

## Use it from Python

Install the package with PyTorch, and download the weights:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[export]" huggingface_hub
.venv/bin/hf download marcosmaceo/tiny-voca-es --local-dir weights/es
```

`make venv` installs everything, including the training and evaluation
dependencies. It is only needed to train or evaluate.

### Command line

```bash
.venv/bin/python -m tiny_voca.infer \
  --checkpoint weights/es/tiny-voca-es.pt \
  --voice F2 \
  --text "Hola, ¿cómo estás? Esto es tiny voca en español." \
  --output hola.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a UTF-8 file instead of `--text` |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate; larger is faster |
| `--pitch-shift` | 0 | Added to log F0; 0.058 is about one semitone |
| `--pitch-range` | 1.0 | Scales pitch movement. The local app uses 1.5, which sounds less flat |
| `--device` | `cpu` | Torch device |

Text of any length works. It is read one sentence at a time and the audio is
joined. A text that ends without a mark gets a full stop. Numbers, dates and
money are read by eSpeak from context: `Cuesta 4.090 euros.` is read *cuesta
cuatro mil noventa euros*. A sentence with nothing to read is skipped. The
saved audio never peaks above 0.99.

The language is chosen with `TINY_VOCA_LANGUAGE`, which is `es` by default in
this fork. `TINY_VOCA_ACCENT` is `es-ES`; `es-AR` and `es-CU` exist but have not
been checked by a native speaker, and this model was trained on `es-ES` only.

### Python

```python
import soundfile as sf
from tiny_voca.infer import load_model, speak_text

model = load_model("weights/es/tiny-voca-es.pt")
wave, spoken = speak_text(model, "Buenos días. ¿Qué tal has dormido?", "M2")
sf.write("buenos-dias.wav", wave.numpy(), model.config.sample_rate)
```

### ONNX

The graph takes phoneme ids and stress values and returns the waveform. Text is
turned into ids outside the graph, with the same front end as above.

```python
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tiny_voca.text import frontend_fingerprint, phonemize

config = json.load(open("weights/es/tiny-voca-es.json"))
assert frontend_fingerprint() == config["frontend_fingerprint"], "wrong eSpeak NG version"
session = ort.InferenceSession("weights/es/tiny-voca-es.onnx")

phones = phonemize("¿Cree que el comité aprobará la propuesta antes del verano?")
one = lambda value: np.array([value], dtype=np.float32)
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([config["voices"].index("M2")], dtype=np.int64),
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
})
audio = audio[0]
audio = audio * min(1.0, 0.99 / np.abs(audio).max())    # the model can peak above 1.0
sf.write("pregunta.wav", audio, config["sample_rate"])
```

One call reads one utterance, up to 512 tokens and 3,000 frames (32 seconds).
Split longer text into sentences first; `tiny_voca.text.sentences` does it. The
graph draws its own noise, so two runs of the same text are not bit-identical.

To export the file yourself:

```bash
.venv/bin/python -m tiny_voca.export --checkpoint weights/es/tiny-voca-es.pt --output tiny-voca-es.onnx
```

This writes `tiny-voca-es.onnx` and `tiny-voca-es.json`.

## How good it is

Word error rate of the model's speech, transcribed by Whisper large-v3-turbo
(through faster-whisper). The teacher's recordings of the same sentences are
scored the same way, as a floor.

| Sentences | Clips | WER, tiny-voca-es | WER, teacher recordings |
|---|---:|---:|---:|
| Validation | 40 | 2.00% | 1.41% |
| Test | 40 | 1.08% | 2.09% |
| New sentences, not in the dataset | 28 | 0.0% | |

These sets are small. The scores say the speech is intelligible; they are not
a benchmark. Most of the remaining errors are proper nouns and spellings
("caserío" heard as "cacerío"). The same recognizer gives scores that differ by
0.5 to 0.7 points on identical audio on two machines, so compare only runs from
the same machine.

These figures were measured on the GPU host with pulses-only excitation, the
release before the current default. The current default (see *Model* below) was
measured on a laptop against pulses-only on the same laptop: validation 2.00%
against 2.54%, test 1.79% against 1.79%, new sentences 0.0% for both. 106 of
the 108 transcripts are identical, and the two that differ are spellings.

Other checks, from [`docs/round2.md`](docs/round2.md) and
[`docs/word-edge-artifact.md`](docs/word-edge-artifact.md):

- Questions: the same sentence read with `¿…?` rises more than with a full stop
  in 24 pairs of 24.
- Voices: median pitch 244 Hz (F2) and 91 Hz (M2), against 259 Hz and 91 Hz for
  the reference recordings.
- Speed: in ONNX Runtime the fastest runs were 36 to 77 times real time on a
  busy laptop. In the local app, eSpeak adds 120 to 530 ms per sentence.
- No formal listening test has been run. One listener has compared versions
  blind, on short pages.

## Model

The architecture is upstream's: a phoneme encoder, explicit duration, pitch,
voicing and energy prediction (as in FastSpeech 2), and a source-filter
synthesizer instead of a neural vocoder. A pulse train at the predicted pitch
and a noise signal are each shaped by a filter whose cepstrum the network
predicts once per frame. [`docs/model.md`](docs/model.md) describes it, with the
English model's numbers.

One thing differs at inference in Spanish. Above harmonic 33 of the pitch in
voiced frames, and above harmonic 2 in unvoiced frames, the pulse train's filter
is driven by noise of the same power instead of pulses (`model.NOISE_ABOVE`).
This reduced a roughness at the edges of words on the male voice, and a blind
listener preferred it. The weights are unchanged; only the excitation at
inference is. The English model keeps pulses alone. The reasons are in
[`docs/spanish.md`](docs/spanish.md#excitation-at-inference) and
[`docs/word-edge-artifact.md`](docs/word-edge-artifact.md), section 11.

What is Spanish about it:

| | |
|---|---|
| Phonemes | 42 phones, 61 symbols, from eSpeak NG `es` |
| Aligner | `facebook/wav2vec2-lv-60-espeak-cv-ft`, which already covers every Spanish phone |
| Voices | F2 and M2, designed with Qwen3-TTS VoiceDesign from Spanish descriptions |
| Teacher | Qwen3-TTS 1.7B Base, cloning the two reference clips |
| Accent | `es-ES`; `es-AR` and `es-CU` front-end rules exist but are unvalidated |

The model never sees language, only phonemes, and the forced aligner is already
multilingual with an eSpeak label set. So the work was the front end. The
design is in [`docs/spanish.md`](docs/spanish.md).

## Data and training

The training audio is synthetic. No recording of a real speaker is used.

- **Text:** 11,050 sentences. 8,100 come from Mozilla Common Voice (CC0): 5,400
  from its Wikipedia sentences and 2,700 from its sentence collector. 900 are
  templates written for this project (numbers, dates, money, names). 2,050 are
  questions and exclamations generated with Mistral Large for this project.
- **Audio:** each sentence read by both voices with Qwen3-TTS 1.7B Base. 20,790
  clips (24.5 hours) passed screening by Whisper at 15% word error rate.
- **Alignment:** phoneme durations from forced alignment with
  `facebook/wav2vec2-lv-60-espeak-cv-ft`. 19,238 utterances (22.7 hours) are in
  the training split.
- **Training:** 122,000 steps on one RTX 4090, then a 20,000-step anneal at a
  low learning rate. The released model is step 142,000.

The dataset is at
[marcosmaceo/tiny-voca-es](https://huggingface.co/datasets/marcosmaceo/tiny-voca-es).
Its card lists what it contains and its license.

To reproduce the run, follow [`docs/training-es.md`](docs/training-es.md). The
short version:

```bash
make venv test
make pilot                         # the whole chain on macOS voices, no GPU, about 1 hour
# on a GPU host:
make scripts voices audio features annotate train evaluate export
```

What the first run measured, and what to change next time, is in
[`docs/round2.md`](docs/round2.md).

## Tests

```bash
make test                          # Spanish: 96 tests, 6 skipped
TINY_VOCA_LANGUAGE=en make test    # English: 96 tests, 18 skipped
```

## Limitations

- Spanish from Spain only, two fixed voices, no voice cloning.
- The delivery is somewhat flat. The flatness comes from the synthetic teacher,
  not the model; `--pitch-range 1.5` helps. See [`docs/round2.md`](docs/round2.md).
- A slight roughness remains at the edges of words, more on the male voice. It
  is reduced, not gone. [`docs/word-edge-artifact.md`](docs/word-edge-artifact.md)
  records everything tried.
- The model cannot be more natural than the model that generated its data.
- Formal questions (the *usted* register) are the weakest measured case: the
  corpus has few of them.
- Foreign names and unusual spellings are often mispronounced.
- The front end needs eSpeak NG 1.52.0 exactly. There is no Spanish front end
  without eSpeak yet, so `TINY_VOCA_FRONTEND=own` does not work for Spanish.
- No streaming inside one sentence. The local app and the demo play each
  sentence as soon as it is ready.

## Licenses

- **Code and weights:** Apache-2.0, as upstream. See [`LICENSE`](LICENSE).
- **eSpeak NG:** GPL-3.0. It is not part of this repository or of the weights.
  The Python code runs the `espeak-ng` program as a separate process. The
  browser demo loads a WebAssembly build of eSpeak NG 1.52.0, which stays under
  GPL-3.0. It is built from the 1.52.0 source, plus a small driver, by
  [`scripts/build_espeak_wasm.sh`](scripts/build_espeak_wasm.sh) and served
  with eSpeak NG's license (`LICENSE-espeak-ng.txt`). The model was trained on
  eSpeak's phoneme output; [`docs/spanish.md`](docs/spanish.md#licensing)
  discusses what that means.
- **Training text:** Mozilla Common Voice sentences (CC0) and sentences written
  or generated for this project. Upstream's English data also used LibriTTS-R
  transcripts (CC BY 4.0); this dataset does not.
- **Tools used to make the data, not part of the model:** Qwen3-TTS
  (Apache-2.0), `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0), Whisper
  and faster-whisper (MIT), Mistral Large (sentence generation).

## The English model

The English tiny-voca is upstream's work: weights, dataset and demo.

- Code and instructions: <https://github.com/n4ze3m/tiny-voca>
- Weights: <https://huggingface.co/n4ze3m/tiny-voca>
- Dataset: <https://huggingface.co/datasets/n4ze3m/tiny-voca>
- Model description: [`docs/model.md`](docs/model.md)

This fork keeps the English front end byte for byte. With
`TINY_VOCA_LANGUAGE=en`, English checkpoints still load. `./run.sh` starts a
local page for the English model ([`app/README.md`](app/README.md)), and `web/`
is upstream's English browser demo. Upstream's README also lists other
languages and ports built on tiny-voca by the community.

## Layout

```
src/tiny_voca/
  languages/english.py     English inventory and text rules (unchanged behaviour)
  languages/spanish.py     Spanish inventory, accents, fallbacks
  text.py                  the front end; picks a profile at import
  numbers_es.py            Spanish number reading, for WER and corpus building
  model.py train.py ...    upstream's model and training code
site/                      the Spanish browser demo (GitHub Pages)
app_es/                    local Spanish web page (./run-es.sh)
app/                       local English web page (./run.sh)
web/                       upstream's English browser demo
scripts/
  run_pipeline.sh          pilot or full run, per-stage logs, budget, stop on failure
  host_setup.sh            Ubuntu 24.04 provisioning (eSpeak 1.52 from source)
  preflight.py             checks the host before it spends money
  build_text_es.py         Spanish sentence list from CC0 corpora + templates
  design_voices.py         Qwen3-TTS VoiceDesign -> the two reference clips
  generate_audio.py        teacher TTS -> audio + manifest (GPU)
  generate_audio_local.py  macOS `say` stand-in, for smoke tests only
  anneal_es.sh             the low-rate anneal that produced the release
  check_annotations.py     go/no-go gate on the alignment probe
  probe_*.py               the measurements behind docs/round2.md and docs/word-edge-artifact.md
voices/es-ES.json          the two voice descriptions
docs/spanish.md            front-end design
docs/training-es.md        the runbook
docs/round2.md             what the run measured
docs/word-edge-artifact.md the roughness at word edges, and the fix
docs/model.md              the English model
```

## Credits

Muhammed Nazeem wrote tiny-voca: the model, the source-filter vocoder, the
training recipe, the dataset design and the forced-alignment approach. This
fork only changes what is language-specific. Please cite his work:

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```

Also used: eSpeak NG, Qwen3-TTS, Mozilla Common Voice,
`facebook/wav2vec2-lv-60-espeak-cv-ft`, faster-whisper and ONNX Runtime.
