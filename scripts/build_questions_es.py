"""Write extra Spanish questions for the corpus, with Mistral, and filter them hard.

The corpus builder aims for a quarter questions, but the CC0 sources ran out:
the conversational file holds 329 usable questions and all of them are already
in the corpus, and the wiki's question marks are broken fragments of quoted
titles ("A diferencia de sus anteriores discos Delirious?"). Without more
questions the corpus is 7% questions, and Spanish marks a yes/no question with
intonation alone, so that share is what the model would learn.

This asks a language model for natural Peninsular Spanish questions and keeps
only the ones that pass the same filter as the rest of the corpus: 5-30 words,
no digits, no shouting, and every phoneme known to the front end. Near
duplicates are dropped by a signature of the first and last words, because a
generator asked for 1,600 questions will otherwise return the same sentence
with a different noun.

    MISTRAL_API_KEY=... python scripts/build_questions_es.py \\
        --output data/questions-es.jsonl --count 1600 --existing data/scripts-full.jsonl

The output rows are train-split and carry the same schema as build_text_es.py,
so they can be appended to the corpus file directly.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.text import phonemize

OK = re.compile(r"^[¿¡A-ZÁÉÍÓÚÑÜ][^…]*[.?!]$")
BAD = re.compile(r"\b[A-ZÁÉÍÓÚÑÜ]{2,}\b|--|\s-\s|\b[bcdfghjklmnñpqrstvwxz]\b|\d")

PROMPT = """Escribe {count} preguntas en español de España, natural y conversacional, como las que una persona diría en voz alta.

Requisitos:
- Cada pregunta en su propia línea, sin numerar, sin comentarios.
- Entre 6 y 24 palabras.
- Empieza con el signo de apertura ¿ y termina con ?
- Sin cifras: escribe los números con letras.
- Sin abreviaturas, sin siglas, sin comillas, sin paréntesis, sin guiones.
- Nada de nombres de famosos ni de marcas.
- Varía mucho la estructura: preguntas de sí o no, preguntas con qué, quién, dónde, cuándo, cómo, cuánto, cuál, por qué; preguntas con coletilla como ¿verdad? o ¿no?; peticiones corteses con poder o querer; preguntas que empiezan por una aclaración como "Oye," o "Perdona,".
- Temas cotidianos y variados: casa, trabajo, transporte, comida, planes, salud, estudios, compras, tiempo, familia, ocio.
- Español de España: nada de ustedes, nada de acá, nada de carro ni computadora.
- No repitas la misma pregunta con otra palabra.

Devuelve solo las preguntas, una por línea."""


def usable(line):
    line = re.sub(r"\s+", " ", line.strip().replace("’", "'").replace("«", '"').replace("»", '"'))
    if not OK.match(line) or not 28 <= len(line) <= 170 or not 6 <= len(line.split()) <= 24:
        return None
    if BAD.search(line):
        return None
    if not line.endswith("?"):
        return None
    return line


def signature(text):
    """Rough shape of a question, to reject near-duplicates.

    The first and last content words are what a generator varies least: it will
    happily return "¿Dónde está la estación?" and "¿Dónde está el mercado?"
    forever. Keeping one per signature leaves the structures varied.
    """
    words = [w for w in re.findall(r"[a-záéíóúñü]+", text.lower()) if len(w) > 2]
    if len(words) < 4:
        return text.lower()
    return " ".join(words[:2] + words[-2:])


def ask(key, count, model, temperature, attempts=4):
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "user", "content": PROMPT.format(count=count)}]}).encode()
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request("https://api.mistral.ai/v1/chat/completions", data=body,
                                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            # Current models may return the text as a list of parts instead of a string.
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            return content.splitlines()
        except Exception as error:      # a dropped connection must not lose the round
            last = error
            time.sleep(2 ** attempt)
    raise SystemExit(f"the generator failed {attempts} times: {last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1600, help="Questions to keep after filtering")
    parser.add_argument("--existing", type=Path, help="Corpus JSONL whose texts must not be repeated")
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--per-call", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=40, help="Give up after this many calls")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("set MISTRAL_API_KEY")

    seen = set()
    if args.existing and args.existing.is_file():
        seen = {json.loads(line)["text"].lower() for line in args.existing.read_text(encoding="utf-8").splitlines() if line.strip()}
    # Resume: a timed-out run keeps everything it already wrote.
    rows = []
    if args.output.is_file():
        rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        seen |= {row["text"].lower() for row in rows}
    shapes = collections.Counter()
    for row in rows:
        shapes[signature(row["text"])] += 1
    kept, rejected = len(rows), collections.Counter()

    def flush():
        ordered = sorted(rows, key=lambda r: (r["character_count"], r["id"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")

    for round_number in range(args.rounds):
        if kept >= args.count:
            break
        for raw in ask(key, args.per_call, args.model, args.temperature):
            text = usable(raw)
            if text is None:
                rejected["filter"] += 1
                continue
            if text.lower() in seen:
                rejected["duplicate"] += 1
                continue
            shape = signature(text)
            # Two per shape is enough to teach the tune without letting one
            # sentence pattern dominate a tenth of the corpus.
            if shapes[shape] >= 2:
                rejected["shape"] += 1
                continue
            try:
                phones = phonemize(text)
            except Exception:
                rejected["frontend"] += 1
                continue
            if phones.unknown or len(phones) > 300:
                rejected["frontend"] += 1
                continue
            seen.add(text.lower())
            shapes[shape] += 1
            identifier = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append({"id": identifier, "text": text, "original_text": text, "category": "question",
                         "origin": "mistral_questions", "source_id": identifier,
                         "source_url": "https://github.com/n4ze3m/tiny-voca",
                         "source_license": "project-own", "group_id": "mistral_questions:" + identifier,
                         "split": "train", "llm_model": args.model, "llm_modified": False,
                         "word_count": len(text.split()), "character_count": len(text), "audio_status": "not_generated"})
            kept += 1
            if kept >= args.count:
                break
        flush()
        print(json.dumps({"round": round_number + 1, "kept": kept, "target": args.count,
                          "rejected": dict(rejected)}), flush=True)

    flush()
    words = [r["word_count"] for r in rows]
    print(json.dumps({"questions": len(rows), "median_words": sorted(words)[len(words) // 2],
                      "shapes": len(shapes), "rejected": dict(rejected)}))


if __name__ == "__main__":
    main()
