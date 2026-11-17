// Splitting Spanish text into what the model speaks at once, in plain JavaScript.
// Mirrors sentences, finished and shorter in src/tiny_voca/text.py, with the
// Spanish SENTENCE_START and NOT_AN_END of src/tiny_voca/languages/spanish.py.
// site/test/sentences_golden.mjs compares the two; keep them alike.
//
// Python's regexes are Unicode-aware where JavaScript's are not: \s takes
// \x1c-\x1f and \x85 (and the Python side adds U+FEFF by hand), \b and \w
// count every letter and digit, not ASCII alone. Those classes are spelled out.

const WS = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff";
const PY_WS = WS.replace("\\ufeff", "");  // Python's str.strip and \s leave U+FEFF alone
const SENTENCE_START = "A-ZÁÉÍÓÚÑÜ0-9¿¡";
const NOT_AN_END = ["sr", "sra", "srta", "dr", "dra", "prof", "profa", "ud", "uds", "etc", "ej",
  "pág", "núm", "av", "tel", "dpto", "gral", "cnel", "aprox"];

const WORD = "[\\p{L}\\p{N}_]";
const SPACE = new RegExp(`[${WS}]+`, "gu");
const SENTENCE_END = new RegExp(`(?<=[.!?…]["')\\]]?)[${WS}]+(?=["'(\\[]?[${SENTENCE_START}])`, "u");
// \b before a word: the character before it is no word character.
const NOT_AN_END_RE = new RegExp(`(?:(?<!${WORD})(?:${NOT_AN_END.join("|")})|(?<!${WORD})[\\p{L}\\p{Nl}\\p{No}])\\.$`, "iu");
const CLAUSE_END = new RegExp(`(?<=[,;:—])[${PY_WS}]+`, "u");
const TEXT_END = new RegExp(`[${WS}]+$`, "u");
const CLOSING = "\"'”’)]»";
const ALNUM = /[\p{L}\p{N}]/u;

const strip = (text) => text.replace(new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "gu"), "");

/** The sentences of `text`, without a break after a title or an initial or inside a number. */
export function sentences(text) {
  const parts = [];
  let current = "";
  for (const piece of strip(text.replace(SPACE, " ")).split(SENTENCE_END)) {
    current = current ? `${current} ${piece}` : piece;
    if (!NOT_AN_END_RE.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => ALNUM.test(part));
}

/** `text` with a full stop added when it ends without a mark. */
export function finished(text) {
  const body = text.replace(TEXT_END, "");
  if (!body) return text;
  const words = body.split(SPACE);
  const last = words[words.length - 1];
  let marked;
  if (ALNUM.test(last)) {
    let end = last.length;
    while (end > 0 && CLOSING.includes(last[end - 1])) end -= 1;
    marked = end > 0 && ".!?…,;:".includes(last[end - 1]);
  } else {
    marked = /[.!?…]/u.test(last) || [",", ";", ":"].includes(last);
  }
  return marked ? text : body + ".";
}

/** `text` in smaller pieces: cut after commas and the like, else in two between words. [] for one word. */
export function shorter(text) {
  const pieces = text.split(CLAUSE_END);
  if (pieces.length > 1) return pieces;
  const words = text.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [words.slice(0, half).join(" "), words.slice(half).join(" ")];
}
