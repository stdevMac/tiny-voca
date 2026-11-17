// Golden test: the WebAssembly eSpeak NG must print exactly what the native
//   espeak-ng -q --ipa --sep=_ -v es --stdin
// prints for every text piece the Python front end would hand it.
//
//   node site/test/espeak_golden.mjs [--limit N] [--jobs N]
//
// Pieces come from tiny_voca.text.chunks(tiny_voca.text.normalize_text(s))
// (the non-punctuation ones) for every sentence in
// ~/tiny-voca-artifacts/es-1/scripts-full.jsonl, plus hand-written hard cases,
// which are also checked whole (unsplit). All pieces run through one
// long-lived WASM instance in file order, so state leaking between calls would
// show up as a difference. Exit code 0 only on 100% match.

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir, cpus } from "node:os";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createEspeak } from "../espeak/espeak.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SCRIPTS = path.join(homedir(), "tiny-voca-artifacts/es-1/scripts-full.jsonl");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const LIMIT = option("--limit", Infinity);
const JOBS = option("--jobs", Math.max(2, Math.min(8, cpus().length - 2)));

const HARD_CASES = [
  // numbers
  "3,14", "1.000.000", "1998", "12:30", "El 3,14 es pi.", "Pagó 1.000.000 de euros en 1998.",
  "Quedamos a las 12:30 o a las 7:05.", "0", "-5", "2,5 %", "1/2", "3.º", "1.º de mayo", "15€",
  "$20", "1e10", "007", "2024-05-01", "123456789012345", "IV", "Felipe VI", "siglo XXI",
  "+34 600 123 456", "10 km/h", "50 m²",
  // abbreviations
  "Sr. Pérez", "Dra. López", "Sra. Gómez y Srta. Ruiz", "EE. UU.", "etc.", "p. ej.", "a. C.",
  "Ud. y Uds.", "D. Quijote", "pág. 5", "nº 3", "ONU", "DNI", "UE", "S.A.", "Av. de la Constitución",
  // ¿¡ marks, quotes, ellipsis, dashes
  "¿Qué hora es?", "¡Qué bien!", "¿¡En serio!?", "«Hola», dijo.", "\"Hola\", dijo.", "“Hola”",
  "'hola'", "Pues… no sé.", "Pues... no sé.", "Bueno — dijo — vale.", "(entre paréntesis)",
  "[corchetes]", "uno; dos: tres", "a/b", "a & b", "#hashtag", "@usuario", "correo@ejemplo.com",
  "https://www.ejemplo.es/ruta?x=1", "50%", "*énfasis*", "_guion_bajo_",
  // accents, capitals, ñ/ü
  "ÁNGEL ÉXITO ÍNDICE ÓRBITA ÚLTIMO", "Él está aquí", "ÑANDÚ", "Ñoño", "pingüino", "PINGÜINO",
  "cigüeña", "año", "Año Nuevo", "ARGÜELLES", "camión CAMIÓN", "Ç ç", "à la carte", "naïve",
  // foreign words
  "WhatsApp", "Facebook", "Nueva York", "Instagram y TikTok", "Google Maps", "iPhone",
  "YouTube", "el software", "un e-mail", "OK", "wifi", "Wi-Fi", "pizza", "Shakespeare",
  "Schwarzenegger", "Tchaikovsky", "Mönchengladbach", "Beyoncé", "McDonald's", "Coca-Cola",
  // emoji and symbols
  "😀", "Hola 👋 qué tal", "❤️", "👍🏽", "🇪🇸", "Me encanta 🍕🍕🍕", "★", "→", "©", "™", "§", "∞",
  "日本語", "Привет", "Γειά", "​", " ", "á", "ﬁ",
  // other scripts: eSpeak switches to another language's dictionary for these
  "−3 °C", "α β γ", "Γειά σου", "Привет, мир", "שלום", "مرحبا", "Բարեւ", "नमस्ते", "নমস্কার",
  "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "નમસ્તે", "வணக்கம்", "నమస్కారం", "ನಮಸ್ಕಾರ", "നമസ്കാരം", "ආයුබෝවන්",
  "გამარჯობა", "안녕하세요", "감사합니다 친구", "こんにちは", "你好", "สวัสดี", "ሰላም", "⠓⠑⠇⠇⠕",
  "el ángulo α mide 30°", "Ω", "µ", "∑", "≠", "½", "¼", "²", "‰", "†", "•", "¶", "¥ £ ₩ ₹",
  // empty-ish
  "", " ", "  ", "\t", ".", ",", "¿", "?", "…", "-", "—", "'", "\"", "a", "y", "o", "e", "u", "Y",
  "ah", "eh", "mm", "hm", "x", "k", "ll", "rr",
  // multi-line / multi-clause (phonemize() accepts any text)
  "Uno.\nDos.", "línea uno\nlínea dos", "Hola. ¿Qué tal? Bien, gracias; ¿y tú? ¡Genial!",
];

function longInputs() {
  const sentence = "El rápido murciélago hindú comía feliz cardillo y kiwi, mientras la cigüeña tocaba el saxofón detrás del palenque de paja";
  return [
    Array(20).fill(sentence).join(" "),
    Array(60).fill(sentence).join(", "),
    Array(80).fill(sentence + ".").join(" "),
    "a".repeat(5000),
    Array(400).fill("palabra").join(" "),
    Array(300).fill("123").join(" "),
  ];
}

function pythonPieces(sentences) {
  const code = `
import json, sys
from tiny_voca import text
out = []
for s in json.load(sys.stdin):
    out.append([p for punct, p in text.chunks(text.normalize_text(s)) if not punct])
json.dump(out, sys.stdout, ensure_ascii=False)
`;
  const python = path.join(ROOT, ".venv/bin/python");
  const result = spawnSync(existsSync(python) ? python : "python3", ["-c", code], {
    cwd: ROOT, input: JSON.stringify(sentences), maxBuffer: 1 << 30,
    env: { ...process.env, PYTHONPATH: path.join(ROOT, "src"), TINY_VOCA_LANGUAGE: "es" },
  });
  if (result.status !== 0) throw new Error(`python splitter failed:\n${result.stderr}`);
  return JSON.parse(result.stdout.toString());
}

function nativeEspeak(text) {
  return new Promise((resolve, reject) => {
    const child = spawn("espeak-ng", ["-q", "--ipa", "--sep=_", "-v", "es", "--stdin"]);
    const chunks = [];
    child.stdout.on("data", (c) => chunks.push(c));
    child.stderr.resume();
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve(Buffer.concat(chunks).toString("utf8"))
      : reject(new Error(`espeak-ng exited ${code} for ${JSON.stringify(text)}`)));
    child.stdin.end(text + "\n");
  });
}

async function nativeAll(texts) {
  const results = new Array(texts.length);
  let next = 0;
  async function worker() {
    while (next < texts.length) {
      const i = next++;
      results[i] = await nativeEspeak(texts[i]);
    }
  }
  await Promise.all(Array.from({ length: JOBS }, worker));
  return results;
}

// ------------------------------------------------------------------ inputs
const nativeVersion = spawnSync("espeak-ng", ["--version"]).stdout.toString().trim();
let sentences = [];
if (existsSync(SCRIPTS)) {
  sentences = (await import("node:fs")).readFileSync(SCRIPTS, "utf8").split("\n")
    .filter((line) => line.trim()).map((line) => JSON.parse(line).text);
} else {
  console.warn(`missing ${SCRIPTS}; testing hard cases only`);
}
if (Number.isFinite(LIMIT)) {
  // deterministic sample: every k-th sentence
  const step = Math.max(1, Math.floor(sentences.length / LIMIT));
  sentences = sentences.filter((_, i) => i % step === 0).slice(0, LIMIT);
}
const hard = [...HARD_CASES, ...longInputs()];
const pieceLists = pythonPieces([...sentences, ...hard]);

const groups = [
  { name: "corpus pieces", texts: pieceLists.slice(0, sentences.length).flat() },
  { name: "hard-case pieces", texts: pieceLists.slice(sentences.length).flat() },
  { name: "hard cases whole", texts: hard },
];
// Run every hard case a second time at the end: same instance, later call.
groups.push({ name: "hard cases again", texts: hard });

// ------------------------------------------------------------------ compare
const t0 = performance.now();
const espeak = await createEspeak();
console.log(`wasm eSpeak NG ${espeak.version}; native: ${nativeVersion}`);

let failures = 0;
const failedSentences = new Set();
for (const group of groups) {
  const tStart = performance.now();
  const wasm = group.texts.map((t) => espeak.phonemize(t));
  const tWasm = performance.now() - tStart;
  const native = await nativeAll(group.texts);
  let match = 0;
  const diffs = [];
  group.texts.forEach((t, i) => {
    if (wasm[i] === native[i]) match++;
    else diffs.push({ input: t, wasm: wasm[i], native: native[i] });
  });
  failures += diffs.length;
  console.log(`${group.name}: ${match} / ${group.texts.length} exact`
    + ` (wasm ${(tWasm / Math.max(1, group.texts.length)).toFixed(2)} ms/piece)`);
  for (const d of diffs.slice(0, 10)) console.log("  DIFF", JSON.stringify(d));
  if (group.name === "corpus pieces" && diffs.length) {
    const bad = new Set(diffs.map((d) => d.input));
    pieceLists.slice(0, sentences.length).forEach((ps, i) => {
      if (ps.some((p) => bad.has(p))) failedSentences.add(i);
    });
  }
}
console.log(`corpus sentences: ${sentences.length - failedSentences.size} / ${sentences.length} with every piece exact`);
console.log(`${failures === 0 ? "PASS" : "FAIL"}: ${failures} difference(s); ${((performance.now() - t0) / 1000).toFixed(1)} s`);
process.exit(failures === 0 ? 0 : 1);
