# Spanish in tiny-voca

This fork makes tiny-voca speak Spanish. The model, the training recipe and the
waveform head are unchanged: only the text front end, the phoneme inventory and
the voice identities are language-specific. Everything else — the source-filter
vocoder, the duration/prosody predictors, the GAN schedule — is reused as is.

## What is language-specific, and what is not

| Piece | Language-specific? | Notes |
|---|---|---|
| `tiny_voca/model.py` | no | 4.5M parameters, identical structure |
| `tiny_voca/losses.py`, `train.py`, `data.py` | no | |
| `tiny_voca/text.py` | **yes** | inventory, eSpeak voice, text rules |
| `tiny_voca/languages/` | **yes** | one profile per language |
| `tiny_voca/config.py` | partly | `vocab_size` follows the active inventory |
| `tiny_voca/align.py` | no | the aligner already covers Spanish |
| `web/` | **yes** | upstream's English browser demo, unchanged |
| `site/` | **yes** | the Spanish browser demo: eSpeak NG 1.52.0 as WebAssembly, ONNX Runtime Web |

Select the language once per process:

```bash
export TINY_VOCA_LANGUAGE=es      # default in this fork
export TINY_VOCA_ACCENT=es-ES     # es-ES (default), es-AR, es-CU
```

The choice is made at import, because the inventory is part of the model
configuration and every module reads `SYMBOLS` at import time. `en` restores
the original English front end byte for byte: its fingerprint is still
`a6f1cf32…`, so released English checkpoints keep loading.

## Phoneme inventory

42 phones, 61 symbols including the five specials and fourteen punctuation
marks. It is the union of what eSpeak emits for Spanish over a 25,000-sentence
sample of the Common Voice Spanish Wikipedia corpus, minus foreign-language
artefacts, plus `ʒ` for Rioplatense *sheísmo*.

| Kind | Symbols |
|---|---|
| Plosives | `p b t d k ɡ` |
| Affricates | `tʃ dʒ ts` |
| Fricatives | `f θ s x h ʃ ʒ` |
| Nasals | `m n ɲ ŋ` |
| Liquids | `l ʎ ɾ r` |
| Approximants | `j w` |
| Allophones | `ð β ɣ ʝ` (lenited /b d g ʝ/), `pː` |
| Vowels | `a e i o u ɛ` |
| Diphthongs | `aɪ aʊ eɪ eʊ oɪ` |

Two properties matter:

1. **Every symbol is also a label of the forced aligner.**
   `facebook/wav2vec2-lv-60-espeak-cv-ft` has 392 eSpeak phoneme labels across
   60 languages, and all 42 Spanish phones are among them. One aligned label is
   therefore one token, with no mapping layer. This is why the annotation step
   needed no changes at all.
2. **The order is append-only.** Changing it changes `SYMBOLS`, which changes
   the front-end fingerprint, which invalidates every checkpoint. Add new
   symbols at the end.

## Accents

Accents differ in the front end, not in the inventory, so one checkpoint can
serve all three later.

| Code | Name | eSpeak voice | Rules |
|---|---|---|---|
| `es-ES` | Castilian (Spain) | `es` | distinción (`θ` for ⟨z⟩, ⟨ce⟩, ⟨ci⟩), lleísmo (`ʎ`) |
| `es-AR` | Rioplatense (Argentina) | `es` | seseo (`θ → s`), `sheísmo` (`ʝ ʎ → ʃ`) |
| `es-CU` | Cuban | `es` | seseo, yeísmo (`ʎ → ʝ`), syllable-final `/s/ → h`, `/d/` drops in `-ado`/`-ada` |

All three start from the Castilian voice and are rewritten by rules. `es-419`
was the obvious choice for the Latin American ones, but it renders ⟨ll⟩ as `j j`
and ⟨y⟩ as `ʝ` inconsistently, which makes sheísmo a guessing game. One voice
plus explicit rules is predictable, and it keeps a single eSpeak version in the
pipeline.

**Only `es-ES` has been checked against a native ear.** The other two are
rule-based approximations; validate them with a speaker before training on
them. The rules live in `src/tiny_voca/languages/spanish.py` and are applied
after eSpeak, so they are deterministic and identical at annotation and
inference time.

## Text handling

`normalize_text` and `chunks` are shared, with the language profile supplying
the abbreviation list. Spanish specifics:

* `¿` and `¡` are tokens, so they carry intonation and pauses like `?` and `!`.
* `«»` are normalised to `"`, as `“”` already were.
* `3,14` is a decimal and `1.000` is a thousand: a comma or a full stop between
  digits stays inside the text piece and is read by eSpeak, not split off.
* Abbreviations: `sr sra srta dr dra prof profa ud uds etc ej pág núm av tel
  dpto gral cnel aprox izq dcha`.
* Single-letter words (`a`, `e`, `o`, `u`, `y`) are real words, not OCR noise.

Numbers, dates and money are read by eSpeak from context, which is the point of
using it. Measured on the corpus:

```
"Cuesta 4.090 euros."          -> cuatro mil noventa euros
"El 12/09/2026 a las 18:30."   -> ... (slashes are read literally: avoid them)
"Vivo en el 3º piso, 2ª."      -> tercero ... segunda
"El 25% de 1.500 personas."    -> veinticinco por ciento de mil quinientas
```

`tiny_voca/numbers_es.py` is a separate, small number reader. The phonemizer
does not use it; it exists so word-error-rate scoring can compare "4090" with
"cuatro mil noventa", and so the corpus builder can keep digits out of the
training text.

## Excitation at inference

The Spanish model speaks with a mixed excitation; the English one does not.
The synthesis head has a harmonic branch, driven by a pulse train at the
predicted pitch, and a noise branch. The model barely uses the noise branch,
so breath, frication and the floor of a pause come out of the pulse train as
one burst per pitch period. At M2's 90 Hz those bursts are 11 ms apart, and a
listener heard them as roughness at the edges of words.

At inference, the harmonic branch's filter is therefore driven by white noise
of the same power instead of pulses above a harmonic of the frame's pitch:
above harmonic 2 in frames the model calls unvoiced, above harmonic 33 in
voiced ones, crossfaded over one octave. This is `model.NOISE_ABOVE = (2.0,
33.0)`, and it is the default of `TinyVoca.synthesize`, `infer.speak` and
`speak_text`, `tiny_voca.evaluate` and the ONNX export. Thirty-three
harmonics is 3 kHz for M2 and 8.3 kHz for F2, so F2's voiced frames are almost
untouched.

- Training is unchanged (`SourceFilter.forward` defaults to pulses alone) and
  the weights are the ones trained with pulses; nothing was retrained.
- In a blind comparison against pulses alone, the listener preferred it on M2
  and found nothing worse on either voice. Word error rate, durations and pitch
  did not change; synthesis takes about a tenth longer.
- `--pulses-only` in `tiny_voca.evaluate` and `tiny_voca.export`, or
  `noise_above=None` in Python, gives the pulse train alone, bit for bit the
  earlier release.
- English keeps pulses alone (`NOISE_ABOVE` is None when
  `TINY_VOCA_LANGUAGE=en`): nobody has listened to the English model with it.

The measurements, the listening pages and what is still open are in
[word-edge-artifact.md](word-edge-artifact.md), section 11.

## Voices

Two voices, F2 and M2, designed once with Qwen3-TTS VoiceDesign from Spanish
descriptions (`voices/es-ES.json`) and then cloned for every clip. The
descriptions ask for a Castilian speaker with distinción, because the `es-ES`
front end produces `θ` and the teacher must pronounce it.

## Licensing

eSpeak NG is GPL-3.0. The English project avoided it for its released model by
writing `dizione`, a permissive CMUdict-based front end. There is no equivalent
Spanish G2P in this fork, so the Spanish model is trained on, and speaks with,
eSpeak's phonemes.

How the release handles it: the code and the weights are Apache-2.0, and
eSpeak NG is not part of either. The Python code runs the `espeak-ng` program as
a separate process, which the user installs. The browser demo loads a
WebAssembly build of eSpeak NG 1.52.0, built from its source by
`scripts/build_espeak_wasm.sh` and served with eSpeak NG's own license. The
original project also shipped weights trained on eSpeak phonemes under
Apache-2.0.

The way to remove eSpeak entirely is a permissive Spanish front end behind the
same `FRONTENDS` switch (`text.py` already supports two). Candidates: `gruut`
(MIT, has Spanish), or a rule-based Spanish G2P written for the project. The
inventory is fixed, so swapping the front end only changes the fingerprint, not
the model; whether the model then needs retraining depends on how closely the
new front end matches eSpeak's output.

## Not done yet

* `web/` is still upstream's English demo. The Spanish demo is a separate site
  in `site/`, which ships eSpeak NG as WebAssembly instead of a new G2P.
* `scripts/compare_espeak.py` compares eSpeak with dizione and is English-only.
* No Spanish `own` front end, so `TINY_VOCA_FRONTEND=own` raises for `es`.
