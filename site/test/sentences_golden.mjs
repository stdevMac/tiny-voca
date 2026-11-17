// Golden test: site/sentences.js must split text as sentences, finished and
// shorter in src/tiny_voca/text.py do. Run from the repo root:
//   node site/test/sentences_golden.mjs
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { finished, sentences, shorter } from "../sentences.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const python = process.env.PYTHON || path.join(root, ".venv/bin/python");
const env = { ...process.env, PYTHONPATH: path.join(root, "src"), TINY_VOCA_LANGUAGE: "es" };

const base = [
  "Hola, Sr. García. ¿Cómo está? Vive en la Av. Libertad 3. Bien",
  "El Dr. Pérez llegó. La Dra. Ruiz no.", "SR. PÉREZ. Hola.", "Prof. Ñúñez, pág. 5. Fin.",
  "Etc. Y luego", "J. R. R. Tolkien escribió. Luego", "Mide 3.5 metros. Cuesta 2.500,75 euros.",
  "Es la 1. Son las 2.", "¡Hola! ¿Qué tal?", "Dijo: «no». Luego se fue.", '"¡Vamos!" gritó. Y fue.',
  "(Nota.) Siguiente.", "[Fin.] Otro.", "Bueno... no sé. Quizá.", "Bueno… ¿sí?", "a.m. y p.m. Fin",
  "Hola 😀", "😀 😀", "see you :)", "Fin,", "Fin:", "Fin;", "Fin —", "Fin -", "¿Qué?»", "Hola.»",
  "", " ", "   hola   ", "hola mundo. Adiós", "a﻿b. C﻿d", "uno\u0085dos. Tres\u001cfin",
  "x　y. Z", "Hola.\n\nAdiós.\nFin", "Ñ. Á. É.", "²³. Hola", "Ⅻ. Siguiente", "ⅻ. siguiente",
  "uno, dos, tres, cuatro", "uno; dos: tres — cuatro", "una sola", "palabra", "a b c d e",
  "uno,dos", "uno,  dos", "uno,﻿dos", "_. Hola", "a_b. Hola", "ej. 1", "núm. 3. Ya", "NÚM. 4. Ya",
];
// Every pair and some triples, to cross the patterns.
const texts = [...base];
for (let i = 0; i < base.length; i += 3) for (let j = 1; j < base.length; j += 5) texts.push(base[i] + " " + base[j]);

const js = texts.map((t) => ({ sentences: sentences(t), finished: finished(t), shorter: shorter(t) }));
const py = JSON.parse(execFileSync(python, ["-c", `
import json, sys
from tiny_voca.text import finished, sentences, shorter
texts = json.load(sys.stdin)
print(json.dumps([{"sentences": sentences(t), "finished": finished(t), "shorter": shorter(t)} for t in texts], ensure_ascii=False))
`], { env, input: JSON.stringify(texts), encoding: "utf8" }));

let failures = 0;
texts.forEach((text, i) => {
  for (const key of ["sentences", "finished", "shorter"]) {
    if (JSON.stringify(js[i][key]) !== JSON.stringify(py[i][key])) {
      failures += 1;
      if (failures <= 20) console.error(`${key}(${JSON.stringify(text)})\n  js: ${JSON.stringify(js[i][key])}\n  py: ${JSON.stringify(py[i][key])}`);
    }
  }
});
console.log(`${texts.length} texts, ${failures} differences`);
process.exit(failures ? 1 : 0);
