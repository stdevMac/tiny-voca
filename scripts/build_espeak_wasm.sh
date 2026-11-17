#!/usr/bin/env bash
# Build eSpeak NG 1.52.0 as WebAssembly for the browser demo.
#
#   scripts/build_espeak_wasm.sh            # writes site/espeak/
#
# Everything runs inside a pinned emscripten/emsdk Docker image:
#   1. clone espeak-ng at tag 1.52.0 and check the commit hash;
#   2. build it natively and compile the data from that same tree
#      (phontab, phonindex, phondata, intonations from phsource, es_dict from
#      dictsource), exactly as the upstream Makefile does;
#   3. build libespeak-ng with emcc and link a tiny driver (tv_espeak.c below)
#      that replays what the espeak-ng CLI does for
#        espeak-ng -q --ipa --sep=_ -v es --stdin
#      (same initialisation calls, same espeak_Synth flags, same "drop the last
#      byte of stdin" step), writing the phoneme trace to a memory stream
#      instead of stdout;
#   4. pack only the data Spanish needs into one gzip file.
#
# Outputs (site/espeak/): espeak-ng-core.mjs (emscripten glue),
# espeak-ng-core.wasm, espeak-ng-data.pack, LICENSE-espeak-ng.txt.
# site/espeak/espeak.js (the public API) and README.md are hand-written and
# are not touched by this script.
set -euo pipefail

ESPEAK_REPO=https://github.com/espeak-ng/espeak-ng.git
ESPEAK_TAG=1.52.0
ESPEAK_COMMIT=4870adfa25b1a32b4361592f1be8a40337c58d6c
EMSDK_VERSION=4.0.10
# Languages a Spanish voice can switch to in the middle of a text, so their
# dictionaries and lang files must ship too (the switch shows up as
# "(en)...(es)" in the output). English names letters and symbols es_dict
# lacks (Cyrillic, Arabic, Hebrew letters, U+2212 minus...). The others come
# from the alphabets[] table in src/libespeak-ng/tr_languages.c: Greek, Armenian,
# Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam,
# Sinhala, Georgian and Hangul text is read with that language. Dropping one
# only changes output for text containing that script (about 20 KB gzip each).
SWITCH_LANGS="en el hy hi bn pa gu ta te kn ml si ka ko"
# emscripten/emsdk publishes the amd64 image as <version> and the arm64 image
# as <version>-arm64; both are pinned by digest. emcc output does not depend on
# which one runs it.
case "$(uname -m)" in
  arm64|aarch64) EMSDK_IMAGE="emscripten/emsdk:${EMSDK_VERSION}-arm64@sha256:46566098d0e12a11aed071a4aaa20e6a626ab3d011a246846ae2423a21b659d9" ;;
  *)             EMSDK_IMAGE="emscripten/emsdk:${EMSDK_VERSION}@sha256:90b757eb11fa9a0e3ce4d2d9f76d932a56018e4accc37b5a28b2783751e60eb7" ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/site/espeak"
mkdir -p "${OUT}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/espeak-wasm.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# ---------------------------------------------------------------- driver (C)
cat > "${WORK}/tv_espeak.c" <<'EOF'
/* Replays espeak-ng.c main() for: -q --ipa --sep=_ -v <voice> --stdin
 * Initialisation happens once (tv_init); every tv_phonemize call is one
 * "process run": the input bytes are what the CLI would read from stdin. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <emscripten/emscripten.h>

static char voice_name[64];

/* espeak-ng.c SynthCallback with quiet=true: discard audio, keep going. */
static int synth_callback(short *wav, int numsamples, espeak_EVENT *events)
{
	(void)wav; (void)numsamples; (void)events;
	return 0;
}

static int set_voice(void)
{
	espeak_ng_STATUS result = espeak_ng_SetVoiceByName(voice_name);
	if (result != ENS_OK) {
		espeak_VOICE voice_select;
		memset(&voice_select, 0, sizeof(voice_select));
		voice_select.languages = voice_name;
		result = espeak_ng_SetVoiceByProperties(&voice_select);
	}
	return result == ENS_OK ? 0 : (int)result;
}

EMSCRIPTEN_KEEPALIVE int tv_init(const char *data_path, const char *voice)
{
	snprintf(voice_name, sizeof(voice_name), "%s", voice);
	espeak_ng_InitializePath(data_path);
	espeak_ng_ERROR_CONTEXT context = NULL;
	espeak_ng_STATUS result = espeak_ng_Initialize(&context);
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, context);
		espeak_ng_ClearErrorContext(&context);
		return (int)result;
	}
	result = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, NULL);
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, NULL);
		return (int)result;
	}
	espeak_SetSynthCallback(synth_callback);
	return set_voice();
}

EMSCRIPTEN_KEEPALIVE const char *tv_version(void)
{
	return espeak_Info(NULL);
}

/* input: the bytes the CLI would read from stdin (not NUL terminated).
 * Returns a malloc'd NUL-terminated copy of what the CLI prints to stdout;
 * the caller frees it. *out_len receives its length. */
EMSCRIPTEN_KEEPALIVE char *tv_phonemize(const char *input, int input_len, int *out_len)
{
	char *out = NULL;
	size_t size = 0;
	FILE *f_phonemes_out = open_memstream(&out, &size);
	if (f_phonemes_out == NULL)
		return NULL;

	/* --ipa sets espeakPHONEMES_IPA; --sep=_ sets espeakPHONEMES_SHOW and the
	 * separator '_' in bits 8-23. */
	espeak_SetPhonemeTrace(espeakPHONEMES_IPA | espeakPHONEMES_SHOW | ('_' << 8), f_phonemes_out);

	/* espeak-ng.c "bulk input on stdin": read every byte, then overwrite the
	 * last one with NUL and synthesise with size ix+1. */
	int ix = input_len;
	if (ix > 0) {
		char *p_text = (char *)malloc((size_t)ix + 1);
		if (p_text != NULL) {
			memcpy(p_text, input, (size_t)ix);
			p_text[ix] = 0;
			p_text[ix-1] = 0;
			int synth_flags = espeakCHARS_AUTO | espeakPHONEMES | espeakENDPAUSE;
			espeak_Synth(p_text, (size_t)ix + 1, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
			free(p_text);
		}
	}
	espeak_ng_Synchronize();

	espeak_SetPhonemeTrace(0, stderr);
	fclose(f_phonemes_out);
	*out_len = (int)size;
	return out;
}
EOF

# -------------------------------------------------------- in-container build
UID_GID="$(id -u):$(id -g)"
cat > "${WORK}/build.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends autoconf automake libtool >/dev/null

git clone -q --depth 1 --branch "${ESPEAK_TAG}" "${ESPEAK_REPO}" /work/src
got="\$(git -C /work/src rev-parse HEAD)"
if [ "\${got}" != "${ESPEAK_COMMIT}" ]; then
  echo "espeak-ng tag ${ESPEAK_TAG} is \${got}, expected ${ESPEAK_COMMIT}" >&2
  exit 1
fi
echo "espeak-ng ${ESPEAK_TAG} at \${got}"

CONFIG_OPTS="--without-pcaudiolib --without-sonic --without-speechplayer --without-mbrola --without-async --with-klatt"

# Native build: only to compile the data from this source tree.
cp -a /work/src /work/native
cd /work/native
./autogen.sh >/dev/null 2>&1
./configure \${CONFIG_OPTS} >/dev/null
nice -n 10 make -j"\$(nproc)" src/espeak-ng >/dev/null
nice -n 10 make phsource/phonemes.stamp >/dev/null
for lang in es ${SWITCH_LANGS}; do
  nice -n 10 make "espeak-ng-data/\${lang}_dict" >/dev/null
done
ESPEAK_DATA_PATH=/work/native ./src/espeak-ng --version

# WebAssembly build of the library.
cp -a /work/src /work/wasm
cd /work/wasm
./autogen.sh >/dev/null 2>&1
# src/include/compat/wctype.h maps iswupper & co. to ucd_isupper & co. musl's
# <wchar.h> also declares iswupper(wint_t), and wint_t is int on wasm32 while
# ucd's codepoint_t is uint32_t, so the renamed declarations conflict. Reading
# the system <wchar.h> first, before the macros exist, avoids that; the ucd
# functions still replace the libc ones everywhere, as in the native build.
SYS_WCHAR="\$(dirname "\$(command -v emcc)")/cache/sysroot/include/wchar.h"
test -f "\${SYS_WCHAR}"
emconfigure ./configure --host=wasm32-unknown-emscripten --disable-shared --enable-static \
  CFLAGS="-O2 -include \${SYS_WCHAR}" \${CONFIG_OPTS} >/dev/null
nice -n 10 emmake make -j"\$(nproc)" src/libespeak-ng.la >/dev/null
nice -n 10 emcc -O2 -I src/include /build/tv_espeak.c src/.libs/libespeak-ng.a \\
  -o /work/out/espeak-ng-core.mjs \\
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEspeakCore \\
  -sENVIRONMENT=web,worker,node \\
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=16MB \\
  -sFORCE_FILESYSTEM=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 \\
  -sEXPORTED_FUNCTIONS=_tv_init,_tv_phonemize,_tv_version,_malloc,_free \\
  -sEXPORTED_RUNTIME_METHODS=FS,UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPU8,HEAP32

# Pack the data Spanish needs (es plus SWITCH_LANGS): one gzip stream of
#   "TVPK" | u32le header length | JSON [[path, size], ...] | file bytes
cd /work/native/espeak-ng-data
LANG_FILES="\$(for lang in es ${SWITCH_LANGS}; do echo "\${lang}_dict"; find lang -type f -name "\${lang}"; done)"
LANG_FILES="\${LANG_FILES}" python3 - <<'PY'
import gzip, json, os, struct
files = ["phontab", "phonindex", "phondata", "intonations"] + os.environ["LANG_FILES"].split()
blobs = [open(f, "rb").read() for f in files]
header = json.dumps([[f, len(b)] for f, b in zip(files, blobs)]).encode()
raw = b"TVPK" + struct.pack("<I", len(header)) + header + b"".join(blobs)
with open("/work/out/espeak-ng-data.pack", "wb") as out:
    out.write(gzip.compress(raw, compresslevel=9, mtime=0))
PY
cp /work/src/COPYING /work/out/LICENSE-espeak-ng.txt
chown -R "${UID_GID}" /work/out
EOF

mkdir -p "${WORK}/out"
nice -n 10 docker run --rm \
  -v "${WORK}:/build:ro" -v "${WORK}/out:/work/out" \
  "${EMSDK_IMAGE}" bash /build/build.sh

for f in espeak-ng-core.mjs espeak-ng-core.wasm espeak-ng-data.pack LICENSE-espeak-ng.txt; do
  cp "${WORK}/out/${f}" "${OUT}/${f}"
done
echo "wrote:"
( cd "${OUT}" && for f in *; do
    printf '  %-26s %8d bytes  %8d gzip\n' "${f}" "$(wc -c < "${f}")" "$(gzip -9c "${f}" | wc -c)"
  done )
