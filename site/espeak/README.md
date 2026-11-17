# eSpeak NG 1.52.0 for the browser (Spanish)

This folder is [eSpeak NG](https://github.com/espeak-ng/espeak-ng) compiled to
WebAssembly, with only the data Spanish needs. The demo uses it in a Web Worker
to turn text into the phonemes the tiny-voca model was trained on. It must
produce exactly the same output as the native program the model was trained
with.

- Upstream: https://github.com/espeak-ng/espeak-ng, tag `1.52.0`,
  commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`
- Toolchain: emscripten 4.0.10 (Docker image `emscripten/emsdk:4.0.10`, pinned
  by digest in the build script)

## Files

| File | What it is |
| --- | --- |
| `espeak.js` | The public ES module API (hand-written) |
| `espeak-ng-core.mjs` | emscripten JS glue (generated) |
| `espeak-ng-core.wasm` | libespeak-ng plus a small driver (generated) |
| `espeak-ng-data.pack` | gzip of `phontab`, `phonindex`, `phondata`, `intonations`, `es_dict` and `lang/roa/es`, plus the dictionaries and lang files of the languages a Spanish voice can switch to (see below), all compiled from the same 1.52.0 tree (generated) |
| `LICENSE-espeak-ng.txt` | GPL-3.0, copied from the upstream `COPYING` |

The generated data files are byte-identical to the ones Homebrew's
`espeak-ng` 1.52.0 installs, except `ko_dict`, whose rule section is ordered
differently (most likely because the dictionary compiler sorts with the C
library's `qsort`, which orders equal keys differently on glibc and macOS). Korean output still matches.

### Why other languages' dictionaries are included

When Spanish text contains something `es_dict` cannot read, eSpeak switches
language for that word and marks it in the output, e.g. `−3` (U+2212 minus)
gives `(en)_m_ˈaɪ_n_ə_s_(es) t_ɾ_ˈe_s` and Cyrillic, Arabic or Hebrew letters
are named in English. Greek, Armenian, Devanagari, Bengali, Gurmukhi,
Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala, Georgian and Hangul are
read with their own language (the `alphabets[]` table in
`src/libespeak-ng/tr_languages.c`). Without those dictionaries the output
differs from the native program for such text, so the pack carries `en`, `el`,
`hy`, `hi`, `bn`, `pa`, `gu`, `ta`, `te`, `kn`, `ml`, `si`, `ka` and `ko`
(`SWITCH_LANGS` in the build script). English and Greek add about 130 KB gzip;
the other twelve about 20 KB each and only matter for text in those scripts.

## API

```js
import { createEspeak } from "./espeak/espeak.js";

const espeak = await createEspeak();   // optional baseUrl: URL, URL string, or a path in Node
espeak.version;                        // "1.52.0"
espeak.phonemize("Hola mundo");        // "ˈo_l_a m_ˈu_n_d_o\n"
```

`phonemize(text)` returns exactly what

    espeak-ng -q --ipa --sep=_ -v es --stdin

prints on stdout when its standard input is `text + "\n"`, including the line
breaks: the CLI writes one line per clause (a clause ends at `.`, `?`, `!`,
`,`, `;`, `:` and so on), so `"¿Qué tal? Bien."` becomes
`"k_ˈe t_ˈa_l\nb_j_ˈe_n\n"`. It works in a browser page, in a module Worker
and in Node. `phonemize` is synchronous; `createEspeak` downloads about
0.95 MB (gzip) once.

The WASM driver does not use `espeak_TextToPhonemes`. It replays what the CLI's
`main()` does: the same initialisation calls (synchronous output, voice `es`),
the phoneme trace with IPA and separator `_`, the same `espeak_Synth` flags,
and the same handling of standard input (the CLI overwrites the last byte it
read, the newline, with NUL). The trace is written to a memory stream instead
of stdout.

## Rebuild

    scripts/build_espeak_wasm.sh

It needs Docker. It clones the tag, checks the commit hash, builds eSpeak NG
natively to compile the data, builds the library with emscripten, and writes
the generated files above into this folder. `espeak.js` and this README are
not generated.

Check it against the native program (needs `espeak-ng` 1.52.0 on the PATH and
the repo's Python environment):

    node site/test/espeak_golden.mjs

## License

eSpeak NG is licensed under the GNU General Public License, version 3 or later
(see `LICENSE-espeak-ng.txt`). The corresponding source of the files in this
folder is the upstream repository at tag `1.52.0` (commit
`4870adfa25b1a32b4361592f1be8a40337c58d6c`) plus `scripts/build_espeak_wasm.sh`
in this repository, which contains the driver source (`tv_espeak.c`) and every
build step. `espeak.js` is part of that corresponding source.
