# tiny-voca-es

A Spanish fork of [tiny-voca](https://github.com/n4ze3m/tiny-voca): a
4.5M-parameter, 18 MB text-to-speech model that runs on a CPU. The model and the
training recipe are unchanged; the text front end, the phoneme inventory and the
voices are Spanish.

**Status: the Spanish front end is done and tested. No Spanish checkpoint has
been trained yet** — that is the next step, and it needs a GPU.
[`docs/training-es.md`](docs/training-es.md) is the runbook.

## Quick start

```bash
make venv
make test                      # 89 tests, Spanish by default
make pilot                     # whole chain on macOS voices, no GPU, ~1 h
```

Then, on a GPU host, `make scripts voices audio features annotate train evaluate
export`. Every stage is one target; see the runbook.

## Use it

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca-es.pt \
  --voice F2 \
  --text "Hola, ¿cómo estás? Esto es tiny voca en español." \
  --output hola.wav
```

Long text is read one sentence at a time and joined. Numbers, dates and money
are read from context by the phonemizer: `4.090 euros` becomes *cuatro mil
noventa euros*. A sentence with nothing to read is skipped.

## What is Spanish about it

| | |
|---|---|
| Phonemes | 42 phones, 61 symbols, from eSpeak NG `es` |
| Aligner | `facebook/wav2vec2-lv-60-espeak-cv-ft` already covers every Spanish phone |
| Voices | F2 and M2, designed with Qwen3-TTS VoiceDesign |
| Accent | `es-ES` today; `es-AR` and `es-CU` are wired but unvalidated |
| Teacher | Qwen3-TTS 1.7B Base, cloned from the two reference clips |

The whole design is in [`docs/spanish.md`](docs/spanish.md).

Two properties make this cheap to do: the model never sees language, only
phonemes, and the forced aligner used to label durations is already multilingual
with an eSpeak label set. So the only real work is the front end — and the
inventory it produces is a subset of labels the aligner already knows.

## Language switch

```bash
export TINY_VOCA_LANGUAGE=es     # default here
export TINY_VOCA_ACCENT=es-ES    # or es-AR, es-CU
```

`TINY_VOCA_LANGUAGE=en` restores the original English front end exactly, with
its original fingerprint, so English checkpoints still load. The choice is made
at import because the inventory is part of the model configuration.

## Layout

```
src/tiny_voca/
  languages/english.py     English inventory and text rules (unchanged behaviour)
  languages/spanish.py     Spanish inventory, accents, fallbacks
  text.py                  the front end; picks a profile at import
  numbers_es.py            Spanish number reading, for WER and corpus building
  model.py train.py ...    unchanged from upstream
scripts/
  run_pipeline.sh          pilot or full run, per-stage logs, budget, stop on failure
  host_setup.sh            Ubuntu 24.04 provisioning (eSpeak 1.52 from source)
  preflight.py             checks the host before it spends money
  build_text_es.py         Spanish sentence list from CC0 corpora + templates
  design_voices.py         Qwen3-TTS VoiceDesign -> the two reference clips
  generate_audio.py        teacher TTS -> audio + manifest (GPU)
  generate_audio_local.py  macOS `say` stand-in, for smoke tests only
  split_manifest.py        one JSONL per split, which prepare expects
  check_annotations.py     go/no-go gate on the alignment probe
  validate_voice.py        Voxtral listening test: WER, accent, distinción, naturalness
  probe_phones.py          asks the aligner which phone inventory a clip matches
  espeak_probe.py          behavioural fingerprint of the phonemizer
voices/es-ES.json          the two voice descriptions
docs/spanish.md            front-end design
docs/training-es.md        the end-to-end runbook and budget
```

## Limits

* **No Spanish checkpoint yet.** Until one exists, `infer` has nothing to load.
* `es-AR` and `es-CU` rules are approximations; validate with a native speaker.
* The browser demo (`web/`) is still the English one.
* eSpeak NG is GPL-3.0. A model trained on its phonemes inherits that question
  if you plan to ship it; [`docs/spanish.md`](docs/spanish.md) explains the way
  out (a permissive Spanish G2P behind the existing `FRONTENDS` switch).

## Upstream

Everything here builds on Muhammed Nazeem's
[tiny-voca](https://github.com/n4ze3m/tiny-voca) — model, recipe, dataset design
and the forced-alignment approach. The English model, weights and dataset are
his work; this fork only changes what is language-specific.

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```

## License

Apache-2.0 for code and weights, as upstream. The Spanish sentence sources are
CC0 (Mozilla Common Voice) and templates written for this project. See
[`docs/spanish.md`](docs/spanish.md) for the eSpeak licensing note.
