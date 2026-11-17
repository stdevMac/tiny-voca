// Spanish text front end for tiny-voca, in plain JavaScript (ES module, no
// dependencies; runs in a browser module Worker and in Node).
//
// Mirrors, line for line, src/tiny_voca/text.py: normalize_text, chunks,
// parse_espeak, canonical, tokenize, phonemize (and espeak_words, with the
// eSpeak call replaced by the `espeak` callback), plus the Spanish profile in
// src/tiny_voca/languages/spanish.py (SYMBOLS, PHONES, VOWELS, FALLBACK,
// ABBREVIATIONS, apply_accent and the accents; default accent es-ES).
//
// The token ids are the indices of SYMBOLS, which equal the "symbols" array of
// the ONNX config. site/test/frontend_golden.mjs compares this file with the
// Python front end on thousands of sentences: it must pass after ANY change to
// either side.
//
// Python Unicode semantics reproduced here: str.isalnum/isalpha (\p{L}, \p{N}),
// str.isdigit (an exact table, wider than \p{Nd}: superscripts, circled digits),
// str.isspace / str.split() / str.strip() / regex \s (the Python whitespace
// set, which has \x1c-\x1f and \x85 but not U+FEFF, unlike JS \s), regex
// [^\W\d_] (letters and non-decimal numerics), unicodedata.category == "Cf",
// unicodedata.combining (canonical combining class != 0, an exact table), and
// iteration by code point rather than by UTF-16 unit.

const SPECIALS = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"];
const [, BOS, EOS, SP, UNK] = SPECIALS;
const PUNCTUATION = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…", "¿", "¡"];
const CONSONANTS = ["p", "b", "t", "d", "k", "ɡ", "f", "θ", "s", "x", "h", "tʃ", "dʒ", "ts",
  "m", "n", "ɲ", "ŋ", "l", "ʎ", "ɾ", "r", "j", "w", "ð", "β", "ɣ", "ʝ", "ʃ", "ʒ", "pː"];
const VOWELS = ["a", "e", "i", "o", "u", "ɛ", "aɪ", "aʊ", "eɪ", "eʊ", "oɪ"];
const PHONES = [...CONSONANTS, ...VOWELS];
export const SYMBOLS = [...SPECIALS, ...PUNCTUATION, ...PHONES];
const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]));
const PHONE_SET = new Set(PHONES);
const VOWEL_SET = new Set(VOWELS);
const PUNCTUATION_CHARS = new Set(PUNCTUATION);
const SEPARATOR = "_";
const STRESS_MARKS = [["ˈ", 1], ["ˌ", 2]];

const FALLBACK = new Map(Object.entries({
  "c": ["k"], "ɟ": ["d"], "ʈ": ["t"], "ɳ": ["n"], "ʋ": ["b"], "ʔ": ["h"],
  "ɪ": ["i"], "ɫ": ["l"], "ɑː": ["a"], "ə": ["e"], "aː": ["a"], "iː": ["i"],
  "uː": ["u"], "oː": ["o"], "eː": ["e"], "bʰ": ["b"], "ʰχ": ["x"], "χ": ["x"],
  "ɔ": ["o"], "z": ["s"], "v": ["b"], "ɹ": ["r"], "ɻ": ["r"], "ʁ": ["r"],
  "ɬ": ["l"], "d̪": ["d"], "t̪": ["t"], "s̪": ["s"], "n̪": ["n"], "l̪": ["l"],
  "β̞": ["β"], "ð̞": ["ð"], "ɣ̞": ["ɣ"], "ʝ̞": ["ʝ"], "ɟʝ": ["ʝ"],
  "ʰ": [], "ʲ": [], "ˠ": [], "ˀ": [], "ː": [],
}));
const ABBREVIATIONS = new Set(["sr", "sra", "srta", "dr", "dra", "prof", "profa", "ud", "uds", "etc",
  "ej", "pág", "núm", "av", "tel", "dpto", "gral", "cnel", "aprox", "izq", "dcha"]);

const QUOTES = new Map(Object.entries({
  "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
  "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
  "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")",
}));
const LANGUAGE_TAG = /\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)/g;

// Python's whitespace (str.isspace): JS \s lacks \x1c-\x1f and \x85 and wrongly has U+FEFF.
const WS = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${WS}]+`, "gu");
const WS_DASH_WS = new RegExp(`-{2,}|[${WS}]-[${WS}]`, "gu");
const WS_EDGES = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");
const IS_ALNUM = /^[\p{L}\p{N}]$/u;
const IS_ALPHA = /^\p{L}$/u;
const IS_CF = /^\p{Cf}$/u;
// Python's [^\W\d_]+$ : \w minus decimal digits minus "_"; `$` also matches before a final "\n".
const WORD_BEFORE = /([\p{L}\p{No}\p{Nl}]+)(?=\n?$)/u;

const inRanges = (table, code) => {
  let lo = 0, hi = (table.length >> 1) - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (code < table[2 * mid]) hi = mid - 1;
    else if (code > table[2 * mid + 1]) lo = mid + 1;
    else return true;
  }
  return false;
};
// Flat [start, end, start, end, ...] tables extracted from Python 3.13 (Unicode 15.1).
const DIGIT = [
  0x30,0x39,0xb2,0xb3,0xb9,0xb9,0x660,0x669,0x6f0,0x6f9,0x7c0,0x7c9,0x966,0x96f,0x9e6,0x9ef,
  0xa66,0xa6f,0xae6,0xaef,0xb66,0xb6f,0xbe6,0xbef,0xc66,0xc6f,0xce6,0xcef,0xd66,0xd6f,0xde6,
  0xdef,0xe50,0xe59,0xed0,0xed9,0xf20,0xf29,0x1040,0x1049,0x1090,0x1099,0x1369,0x1371,
  0x17e0,0x17e9,0x1810,0x1819,0x1946,0x194f,0x19d0,0x19da,0x1a80,0x1a89,0x1a90,0x1a99,
  0x1b50,0x1b59,0x1bb0,0x1bb9,0x1c40,0x1c49,0x1c50,0x1c59,0x2070,0x2070,0x2074,0x2079,
  0x2080,0x2089,0x2460,0x2468,0x2474,0x247c,0x2488,0x2490,0x24ea,0x24ea,0x24f5,0x24fd,
  0x24ff,0x24ff,0x2776,0x277e,0x2780,0x2788,0x278a,0x2792,0xa620,0xa629,0xa8d0,0xa8d9,
  0xa900,0xa909,0xa9d0,0xa9d9,0xa9f0,0xa9f9,0xaa50,0xaa59,0xabf0,0xabf9,0xff10,0xff19,
  0x104a0,0x104a9,0x10a40,0x10a43,0x10d30,0x10d39,0x10e60,0x10e68,0x11052,0x1105a,0x11066,
  0x1106f,0x110f0,0x110f9,0x11136,0x1113f,0x111d0,0x111d9,0x112f0,0x112f9,0x11450,0x11459,
  0x114d0,0x114d9,0x11650,0x11659,0x116c0,0x116c9,0x11730,0x11739,0x118e0,0x118e9,0x11950,
  0x11959,0x11c50,0x11c59,0x11d50,0x11d59,0x11da0,0x11da9,0x11f50,0x11f59,0x16a60,0x16a69,
  0x16ac0,0x16ac9,0x16b50,0x16b59,0x1d7ce,0x1d7ff,0x1e140,0x1e149,0x1e2f0,0x1e2f9,0x1e4f0,
  0x1e4f9,0x1e950,0x1e959,0x1f100,0x1f10a,0x1fbf0,0x1fbf9,
];
const COMBINING = [
  0x300,0x34e,0x350,0x36f,0x483,0x487,0x591,0x5bd,0x5bf,0x5bf,0x5c1,0x5c2,0x5c4,0x5c5,0x5c7,
  0x5c7,0x610,0x61a,0x64b,0x65f,0x670,0x670,0x6d6,0x6dc,0x6df,0x6e4,0x6e7,0x6e8,0x6ea,0x6ed,
  0x711,0x711,0x730,0x74a,0x7eb,0x7f3,0x7fd,0x7fd,0x816,0x819,0x81b,0x823,0x825,0x827,0x829,
  0x82d,0x859,0x85b,0x898,0x89f,0x8ca,0x8e1,0x8e3,0x8ff,0x93c,0x93c,0x94d,0x94d,0x951,0x954,
  0x9bc,0x9bc,0x9cd,0x9cd,0x9fe,0x9fe,0xa3c,0xa3c,0xa4d,0xa4d,0xabc,0xabc,0xacd,0xacd,0xb3c,
  0xb3c,0xb4d,0xb4d,0xbcd,0xbcd,0xc3c,0xc3c,0xc4d,0xc4d,0xc55,0xc56,0xcbc,0xcbc,0xccd,0xccd,
  0xd3b,0xd3c,0xd4d,0xd4d,0xdca,0xdca,0xe38,0xe3a,0xe48,0xe4b,0xeb8,0xeba,0xec8,0xecb,0xf18,
  0xf19,0xf35,0xf35,0xf37,0xf37,0xf39,0xf39,0xf71,0xf72,0xf74,0xf74,0xf7a,0xf7d,0xf80,0xf80,
  0xf82,0xf84,0xf86,0xf87,0xfc6,0xfc6,0x1037,0x1037,0x1039,0x103a,0x108d,0x108d,0x135d,
  0x135f,0x1714,0x1715,0x1734,0x1734,0x17d2,0x17d2,0x17dd,0x17dd,0x18a9,0x18a9,0x1939,
  0x193b,0x1a17,0x1a18,0x1a60,0x1a60,0x1a75,0x1a7c,0x1a7f,0x1a7f,0x1ab0,0x1abd,0x1abf,
  0x1ace,0x1b34,0x1b34,0x1b44,0x1b44,0x1b6b,0x1b73,0x1baa,0x1bab,0x1be6,0x1be6,0x1bf2,
  0x1bf3,0x1c37,0x1c37,0x1cd0,0x1cd2,0x1cd4,0x1ce0,0x1ce2,0x1ce8,0x1ced,0x1ced,0x1cf4,
  0x1cf4,0x1cf8,0x1cf9,0x1dc0,0x1dff,0x20d0,0x20dc,0x20e1,0x20e1,0x20e5,0x20f0,0x2cef,
  0x2cf1,0x2d7f,0x2d7f,0x2de0,0x2dff,0x302a,0x302f,0x3099,0x309a,0xa66f,0xa66f,0xa674,
  0xa67d,0xa69e,0xa69f,0xa6f0,0xa6f1,0xa806,0xa806,0xa82c,0xa82c,0xa8c4,0xa8c4,0xa8e0,
  0xa8f1,0xa92b,0xa92d,0xa953,0xa953,0xa9b3,0xa9b3,0xa9c0,0xa9c0,0xaab0,0xaab0,0xaab2,
  0xaab4,0xaab7,0xaab8,0xaabe,0xaabf,0xaac1,0xaac1,0xaaf6,0xaaf6,0xabed,0xabed,0xfb1e,
  0xfb1e,0xfe20,0xfe2f,0x101fd,0x101fd,0x102e0,0x102e0,0x10376,0x1037a,0x10a0d,0x10a0d,
  0x10a0f,0x10a0f,0x10a38,0x10a3a,0x10a3f,0x10a3f,0x10ae5,0x10ae6,0x10d24,0x10d27,0x10eab,
  0x10eac,0x10efd,0x10eff,0x10f46,0x10f50,0x10f82,0x10f85,0x11046,0x11046,0x11070,0x11070,
  0x1107f,0x1107f,0x110b9,0x110ba,0x11100,0x11102,0x11133,0x11134,0x11173,0x11173,0x111c0,
  0x111c0,0x111ca,0x111ca,0x11235,0x11236,0x112e9,0x112ea,0x1133b,0x1133c,0x1134d,0x1134d,
  0x11366,0x1136c,0x11370,0x11374,0x11442,0x11442,0x11446,0x11446,0x1145e,0x1145e,0x114c2,
  0x114c3,0x115bf,0x115c0,0x1163f,0x1163f,0x116b6,0x116b7,0x1172b,0x1172b,0x11839,0x1183a,
  0x1193d,0x1193e,0x11943,0x11943,0x119e0,0x119e0,0x11a34,0x11a34,0x11a47,0x11a47,0x11a99,
  0x11a99,0x11c3f,0x11c3f,0x11d42,0x11d42,0x11d44,0x11d45,0x11d97,0x11d97,0x11f41,0x11f42,
  0x16af0,0x16af4,0x16b30,0x16b36,0x16ff0,0x16ff1,0x1bc9e,0x1bc9e,0x1d165,0x1d169,0x1d16d,
  0x1d172,0x1d17b,0x1d182,0x1d185,0x1d18b,0x1d1aa,0x1d1ad,0x1d242,0x1d244,0x1e000,0x1e006,
  0x1e008,0x1e018,0x1e01b,0x1e021,0x1e023,0x1e024,0x1e026,0x1e02a,0x1e08f,0x1e08f,0x1e130,
  0x1e136,0x1e2ae,0x1e2ae,0x1e2ec,0x1e2ef,0x1e4ec,0x1e4ef,0x1e8d0,0x1e8d6,0x1e944,0x1e94a,
];

const isAlnum = (c) => c !== "" && IS_ALNUM.test(c);
const isAlpha = (c) => c !== "" && IS_ALPHA.test(c);
const isDigit = (c) => c !== "" && inRanges(DIGIT, c.codePointAt(0));
const isCombining = (c) => inRanges(COMBINING, c.codePointAt(0));
const strip = (s) => s.replace(WS_EDGES, "");
const pySplit = (s) => s.split(WS_RUN).filter((x) => x !== "");

export class ValueError extends Error {
  constructor(message) { super(message); this.name = "ValueError"; }
}
/** The text has no word, number or letter to say: only marks, emoji and the like. */
export class NothingToPronounce extends ValueError {
  constructor(message) { super(message); this.name = "NothingToPronounce"; }
}

export function normalizeText(text) {
  let out = "";
  for (const ch of text.normalize("NFKC")) out += QUOTES.get(ch) ?? ch;
  out = out.replace(/\.{3,}/gu, "…");
  out = out.replace(WS_DASH_WS, "—");
  return strip(out.replace(WS_RUN, " "));
}

/** Returns an array of [isPunctuation, piece]. */
export function chunks(text) {
  const out = [];
  const cp = Array.from(text);
  let start = 0;
  for (let i = 0; i < cp.length; i++) {
    const char = cp[i];
    if (!PUNCTUATION_CHARS.has(char)) continue;
    const previous = i ? cp[i - 1] : "";
    const following = i + 1 < cp.length ? cp[i + 1] : "";
    if (char === "'" && isAlnum(previous) && isAlnum(following)) continue;
    if (".,:".includes(char) && isDigit(previous) && isDigit(following)) continue;
    if (char === ".") {
      const word = WORD_BEFORE.exec(cp.slice(0, i).join(""));
      if (word && ABBREVIATIONS.has(word[1].toLowerCase())) continue;
      if (isAlpha(previous) && isAlpha(following)) continue;
    }
    const piece = strip(cp.slice(start, i).join(""));
    if (piece) out.push([false, piece]);
    out.push([true, char]);
    start = i + 1;
  }
  const rest = strip(cp.slice(start).join(""));
  if (rest) out.push([false, rest]);
  return out;
}

/** eSpeak's raw stdout -> words, each an array of [phone, stress]. */
export function parseEspeak(output) {
  const words = [];
  for (const word of pySplit(output.replace(LANGUAGE_TAG, ""))) {
    const units = [];
    for (let unit of word.split(SEPARATOR)) {
      let stress = 0;
      for (const [mark, level] of STRESS_MARKS) {
        if (unit.includes(mark)) {
          stress = stress || level;
          unit = unit.split(mark).join("");
        }
      }
      let kept = "";
      for (const c of unit) if (!IS_CF.test(c) && c !== "(" && c !== ")") kept += c;
      if (kept) units.push([kept, stress]);
    }
    if (units.length) words.push(units);
  }
  return words;
}

/** Map an eSpeak phoneme to inventory phones; null when unknown. */
export function canonical(phone) {
  if (PHONE_SET.has(phone)) return [phone];
  if (FALLBACK.has(phone)) return FALLBACK.get(phone);
  let stripped = "";
  for (const c of phone.normalize("NFD")) if (!isCombining(c)) stripped += c;
  for (const candidate of [stripped, stripped.split("ː").join(""), stripped + "ː"]) {
    if (candidate !== phone && PHONE_SET.has(candidate)) return [candidate];
    if (candidate !== phone && FALLBACK.has(candidate)) return FALLBACK.get(candidate);
  }
  return null;
}

/** pieces: array of [isPunctuation, value]; a text piece's value is parseEspeak's word list. */
export function tokenize(pieces) {
  const symbols = [BOS], stress = [0], unknown = [];
  for (const [punctuation, value] of pieces) {
    if (punctuation) {
      if (symbols[symbols.length - 1] !== value) { // collapse repeats such as "!!"
        symbols.push(value);
        stress.push(0);
      }
      continue;
    }
    for (const word of value) {
      const last = symbols[symbols.length - 1];
      if (PHONE_SET.has(last) || last === UNK) {
        symbols.push(SP);
        stress.push(0);
      }
      for (const [phone, level] of word) {
        let mapped = canonical(phone);
        if (mapped === null) {
          unknown.push(phone);
          mapped = [UNK];
        }
        let carrier = mapped.findIndex((item) => VOWEL_SET.has(item));
        if (carrier < 0) carrier = 0;
        mapped.forEach((item, index) => {
          symbols.push(item);
          stress.push(index === carrier ? level : 0);
        });
      }
    }
  }
  symbols.push(EOS);
  stress.push(0);
  if (!symbols.some((s) => PHONE_SET.has(s) || s === UNK)) {
    throw new NothingToPronounce("the text has nothing to pronounce");
  }
  return { symbols, ids: symbols.map((s) => SYMBOL_TO_ID.get(s)), stress, unknown };
}

// Accents (languages/spanish.py). The released model uses the default, "es-ES",
// which leaves eSpeak's phones unchanged.
const seseo = (phone) => (phone === "θ" ? "s" : phone);
const sheismo = (words) => words.map((word) =>
  word.map(([phone, stress]) => [phone === "ʝ" || phone === "ʎ" ? "ʃ" : seseo(phone), stress]));
const cuban = (words) => words.map((word) => {
  const phones = word.map(([phone, stress]) => [phone === "ʎ" ? "ʝ" : seseo(phone), stress]);
  phones.forEach(([phone, stress], index) => {
    if (phone !== "s") return;
    const following = index + 1 < phones.length ? phones[index + 1][0] : null;
    if (following === null || (!VOWEL_SET.has(following) && following !== "j" && following !== "w")) {
      phones[index] = ["h", stress];
    }
  });
  const n = phones.length;
  if (n >= 3 && (phones[n - 2][0] === "d" || phones[n - 2][0] === "ð")
      && VOWEL_SET.has(phones[n - 3][0]) && VOWEL_SET.has(phones[n - 1][0])) {
    phones.splice(n - 2, 1);
  }
  return phones;
});
export const ACCENTS = { "es-ES": null, "es-AR": sheismo, "es-CU": cuban };
export const ESPEAK_VOICE = "es";

/**
 * text -> {symbols, ids, stress, unknown}. `espeak(piece)` must return the raw
 * stdout of `espeak-ng -q --ipa --sep=_ -v es --stdin` for `piece + "\n"`.
 * options.accent: "es-ES" (default), "es-AR" or "es-CU".
 */
export function phonemize(text, espeak, options = {}) {
  if (typeof text !== "string" || !strip(text)) throw new ValueError("text must not be empty");
  const accent = options.accent ?? "es-ES";
  if (!(accent in ACCENTS)) throw new ValueError(`accent must be one of ${Object.keys(ACCENTS)}, not ${accent}`);
  const transform = ACCENTS[accent];
  const pieces = [];
  for (const [punctuation, piece] of chunks(normalizeText(text))) {
    if (punctuation) { pieces.push([true, piece]); continue; }
    const words = parseEspeak(espeak(piece));
    pieces.push([false, transform ? transform(words) : words]);
  }
  return tokenize(pieces);
}
