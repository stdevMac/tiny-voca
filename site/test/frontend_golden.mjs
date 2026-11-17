// Golden test: site/frontend.js must produce exactly the tokens of the Python
// front end (src/tiny_voca/text.py). Run from the repo root:
//   node site/test/frontend_golden.mjs
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { phonemize, SYMBOLS } from "../frontend.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const home = os.homedir();
const CORPUS = process.env.TINY_VOCA_GOLDEN_CORPUS ?? path.join(home, "tiny-voca-artifacts/es-1/scripts-full.jsonl");
const CONFIGS = [path.join(root, "weights/es/tiny-voca-es.json"),
  path.join(home, "tiny-voca-artifacts/noise-default/onnx/tiny-voca-es-noise.json")];
const python = path.join(root, ".venv/bin/python");
const env = { ...process.env, PYTHONPATH: path.join(root, "src"), TINY_VOCA_LANGUAGE: "es" };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "frontend-golden-"));

// 1. The symbol inventory must equal the ONNX config's (when a config is present).
const CONFIG = CONFIGS.find((f) => fs.existsSync(f));
if (CONFIG) {
  const config = JSON.parse(fs.readFileSync(CONFIG, "utf8"));
  if (JSON.stringify(config.symbols) !== JSON.stringify(SYMBOLS)) {
    console.error(`SYMBOLS differ from the symbols of ${CONFIG}`);
    process.exit(1);
  }
  console.log(`SYMBOLS equal ${CONFIG} (${SYMBOLS.length} symbols)`);
} else console.log(`note: no ONNX config found (${CONFIGS.join(", ")}); symbol check against it skipped`);

// 2. Sentences. The corpus is optional; NOVEL_SENTENCES and the adversarial list always run.
let corpus = [];
if (fs.existsSync(CORPUS)) {
  corpus = fs.readFileSync(CORPUS, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).text);
} else console.log(`note: corpus ${CORPUS} not found (set TINY_VOCA_GOLDEN_CORPUS); corpus sentences skipped`);
const novel = JSON.parse(execFileSync(python, ["-c",
  "import json;from tiny_voca.languages.spanish import NOVEL_SENTENCES as N;print(json.dumps(N,ensure_ascii=False))"],
  { env, encoding: "utf8" }));
const nfd = (s) => s.normalize("NFD");
const adversarial = [
  // numbers
  "Mide 3,14 metros.", "Hay 1.000.000 de personas.", "Quedamos a las 12:30.", "Nació en 1998.", "Cuesta 2.500,75 euros.",
  "El 3,", "1,2,3,4", "10:00:30", "Son las 7.", "4.5", "3. Luego 4, 5: seis.", "Tiene ² y ③ y ٣٤ y १२.", "5,",
  // abbreviations and initialisms
  "El Sr. García y la Sra. López llegaron.", "La Dra. Ruiz habló.", "Prof. Pérez, Dpto. de Física, pág. 5, núm. 3, etc.",
  "Av. de América, tel. 5551234, aprox. 3 km.", "Gral. Cnel. Izq. Dcha. Ud. Uds. Srta. Profa.", "SR. PÉREZ", "Sr.Pérez",
  "Viven en E.E.U.U. desde entonces.", "p.ej. esto", "Es decir, i.e. así.", "a.m. y p.m.", "etc. Y luego", "Etc.",
  "Ud. y Uds. vendrán.", "Dr.", "Dr. ", "ej. 1", "Ñandú. Úrsula.", "Pág. 3.", "El señor Dr. Ñu.",
  // contractions and apostrophes
  "Don't worry, it's fine.", "l'amour d'Isabelle", "rock 'n' roll", "'hola'", "O'Brien", "5'6", "a'b'c", "ñ'ñ",
  // inverted marks
  "¿Qué hora es?", "¡Hola!", "¿¡Qué!?", "¡¡Hola!!", "¿Cómo estás? ¡Bien!", "?!", "¿", "¡Hola",
  // quotes
  '"comillas" «guillemets» ‘curly’ “doble” „bajas” ‚x‛ `back` ´acute´', "«Hola» dijo.", "Dijo: «no».",
  // ellipsis and dots
  "Bueno... no sé", "Bueno…no sé", "Bueno.... sí", "Bueno.. no", ". . .", "a...b", "...", "…",
  // dashes
  "Uno -- dos", "Uno - dos", "Uno – dos", "Uno — dos", "Uno ― dos", "uno-dos", "uno --- dos", "a - b - c", "- hola", "hola -", "a -b", "a- b", "a\t-\tb",
  // brackets
  "Hola (mundo) [x] {y}", "(hola)", "[]", "((()))", "a(b)c", "(es) (en) (ab-c)",
  // emoji only / nothing to pronounce
  "😀", "😀😀 🎉", "👍 hola 👍", "!!!", "...", "—", "¿?", "   ", "\t\n", "", ".", "?", "()", "123", "1", "a", "x",
  // whitespace
  "hola    mundo", "  hola  ", "hola\tmundo", "hola\nmundo\n\nfin", "hola\r\nmundo", "hola mundo", "hola mundo　fin",
  "hola​mundo", "hola﻿mundo", "﻿hola", "hola\x1cmundo", "hola\x1fmundo", "hola\x85mundo", "hola mundo fin",
  "hola\x0bmundo\x0c", "\x1c\x1d hola \x1e", " hola ", "hola‎", "a­ b", "co­operar",
  // fullwidth, compatibility
  "ＨＯＬＡ　ＭＵＮＤＯ！", "１２３，４５６", "３，１４", "Ｓｒ．Ｇａｒｃíａ", "ｈｏｌａ？", "ﬁn ﬂor", "ǆ ǈ", "№ 5", "℃ ™ ½", "㎏", "Ⅻ", "ⅷ", "①②③", "x²+y³", "H₂O", "1/2", "5%", "10€",
  // combining accents / NFD input
  nfd("Camión señor último pingüino"), nfd("¿Cómo estás? ¡Ñandú!"), nfd("Dr. Núñez"), "é", "ñ", "ä́",
  "́hola", "holá", "ñ̃", "über", "Å",
  // foreign words and scripts
  "Hello world, this is English.", "Je ne sais pas, c'est la vie.", "Straße über Äpfel", "Привет мир", "你好，世界", "こんにちは", "مرحبا", "שלום", "สวัสดี",
  "नमस्ते", "γειά σου", "Mozart y Beethoven en el Carnegie Hall", "WiFi, Bluetooth y USB", "McDonald's", "naïve café", "Zürich", "İstanbul", "ǅ",
  "𝐇𝐞𝐥𝐥𝐨", "𠀀 𠮷", "a😀b", "hola😀.", "😀.", "😀 1", "ABC", "ONU", "NASA y OTAN",
  // punctuation mixes
  "Hola, mundo; adiós: fin.", "Sí, sí, sí.", "No!!!", "¿¿Qué??", "a,b", "a.b", "a:b", "a;b", "a,.b", "hola,", ",hola", ".hola", "hola.", "hola. adiós.", "hola , adiós . fin",
  "x.y.z", "a.B", "A.b", "1.a", "a.1", "ñ.ú", "Ñ.Ú", "3.14", "3 .14", "3. 14", "3,A", "A,3", "a'", "'a", "1'2", "a'1", "1'a",
  "Texto muy largo ".repeat(40) + "fin.", "palabra ".repeat(200),
];
// Deterministic fuzz: random strings over a mixed alphabet, to stress the Unicode rules.
const alphabet = [..."abcdefghijklmnopqrstuvwxyzáéíóúüñABCXYZÁÑ0123456789 .,:;!?¿¡'\"()-—…«»‘’“”[]{}_/\\\t\n"],
  extras = [" ", "　", " ", "\x1c", "\x85", "﻿", "​", "́", "̃", "²", "③", "٣", "१", "Ⅷ", "½", "Ａ", "１", "，", "．", "！", "？",
    "😀", "你", "я", "ß", "ǆ", "ﬁ", "…", "...", "--", " - ", "é", "Dr.", "Sr.", "etc.", "p.ej.", "E.E.U.U.", "3,14", "1.000", "12:30", "d'a"];
let seed = 123456789;
const rand = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
const fuzz = [];
for (let i = 0; i < 2000; i++) {
  let s = "";
  const length = 1 + rand(40);
  for (let j = 0; j < length; j++) s += rand(5) === 0 ? extras[rand(extras.length)] : alphabet[rand(alphabet.length)];
  fuzz.push(s);
}
const sentences = [...corpus, ...novel, ...adversarial, ...fuzz];
console.log(`sentences: ${sentences.length} (corpus ${corpus.length}, novel ${novel.length}, adversarial ${adversarial.length}, fuzz ${fuzz.length})`);

// 3. Python reference.
const inFile = path.join(tmp, "in.json"), outFile = path.join(tmp, "out.json");
fs.writeFileSync(inFile, JSON.stringify(sentences));
execFileSync(python, [path.join(here, "frontend_reference.py"), inFile, outFile], { env, stdio: ["ignore", "inherit", "inherit"], maxBuffer: 1 << 28 });
const reference = JSON.parse(fs.readFileSync(outFile, "utf8"));
if (JSON.stringify(reference.symbols) !== JSON.stringify(SYMBOLS)) { console.error("SYMBOLS differ from Python's"); process.exit(1); }
const voice = reference.voice;

// 4. JS with the native eSpeak binary, cached per piece.
const cache = new Map();
const espeak = (piece) => {
  let out = cache.get(piece);
  if (out === undefined) {
    out = execFileSync("espeak-ng", ["-q", "--ipa", "--sep=_", "-v", voice, "--stdin"],
      { input: piece.replace(/\n+$/, "") + "\n", encoding: "utf8", maxBuffer: 1 << 26 });
    cache.set(piece, out);
  }
  return out;
};
const run = (text) => {
  try { return phonemize(text, espeak); } catch (error) { return { error: error.name }; }
};

// 5. Compare field by field.
let matched = 0;
const mismatches = [];
sentences.forEach((text, index) => {
  const want = reference.results[index], got = run(text);
  const bad = [];
  if (want.error || got.error) {
    if (want.error !== got.error) bad.push(["error", want.error ?? null, got.error ?? null]);
  } else {
    for (const field of ["symbols", "ids", "stress", "unknown"]) {
      if (JSON.stringify(want[field]) !== JSON.stringify(got[field])) bad.push([field, want[field], got[field]]);
    }
  }
  if (bad.length) mismatches.push({ index, text, bad }); else matched++;
});
console.log(`matched ${matched}/${sentences.length}`);
const errors = reference.results.filter((r) => r.error).reduce((m, r) => (m[r.error] = (m[r.error] ?? 0) + 1, m), {});
console.log(`python exceptions reproduced: ${JSON.stringify(errors)}`);
for (const { index, text, bad } of mismatches.slice(0, 20)) {
  console.log(`\nMISMATCH #${index} ${JSON.stringify(text)}`);
  for (const [field, want, got] of bad) console.log(`  ${field}\n    python: ${JSON.stringify(want)}\n    js:     ${JSON.stringify(got)}`);
}
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(mismatches.length ? 1 : 0);
