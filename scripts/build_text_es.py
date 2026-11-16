"""Build the Spanish script list for the teacher TTS. Cloud or laptop.

Sources, all free of third-party licence obligations:

* Mozilla Common Voice Spanish sentence files (CC0-1.0):
  ``sentence-collector.txt`` (conversational) and ``wiki.es.txt`` (reading),
  taken from the common-voice repository as local files.
* Template sentences written for this project (numbers, dates, money, names,
  questions, lists), released with the project.

Every script passes the front end without unknown phonemes and is assigned to
exactly one split by *sentence*, so both voices read the same text and a
sentence never appears in two splits. Numbers are written out in words: the
teacher reads digits inconsistently, and a wrong reading would poison both the
audio and the alignment.

    python scripts/build_text_es.py --collector sentence-collector.txt \\
        --wiki wiki.es.txt --output data/scripts-es.jsonl --count 9000
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.numbers_es import integer_words
from tiny_voca.text import phonemize

# A sentence we are willing to feed the teacher: a capital (or an opening mark)
# at the start, a real mark at the end, no shouting, no dashes, no stray letters.
# Single letters are legitimate Spanish words (a, e, o, u, y), so only the ones
# that cannot be a word are rejected.
OK = re.compile(r"^[¿¡A-ZÁÉÍÓÚÑÜ][^…]*[.?!]$")
BAD = re.compile(r"\b[A-ZÁÉÍÓÚÑÜ]{2,}\b|--|\s-\s|\b[bcdfghjklmnñpqrstvwxz]\b|\d")

FIRST = ("Lucía Mateo Sofía Diego Valentina Martín Paula Álvaro Carmen Javier Elena Pablo Marta Andrés Rocío Sergio "
         "Nuria Iván Clara Hugo Alba Bruno Irene Óscar Sara Adrián Noelia Rubén Laura Guillermo").split()
LAST = ("García Fernández González Rodríguez López Martínez Sánchez Pérez Gómez Ruiz Díaz Moreno Álvarez Romero "
        "Navarro Torres Domínguez Vázquez Ramos Gil Serrano Blanco Molina Castro").split()
PLACES = ["Madrid", "Barcelona", "Sevilla", "Valencia", "Bilbao", "Zaragoza", "Málaga", "Granada", "Toledo", "Salamanca",
          "la biblioteca", "el aeropuerto", "la farmacia", "la panadería", "la estación de tren", "el mercado"]
ITEMS = ["manzanas", "pilas", "cuadernos", "billetes", "mantas", "bocadillos", "paraguas", "velas", "sillas", "postales"]
MONTHS = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
DAYS = "lunes martes miércoles jueves viernes sábado domingo".split()
ORDINALS = ["primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo",
            "undécimo", "duodécimo", "décimo quinto", "vigésimo", "vigésimo tercero", "trigésimo"]
HOURS = "una dos tres cuatro cinco seis siete ocho nueve diez once doce".split()
MINUTES = ["en punto", "y cuarto", "y media", "menos cuarto", "y cinco", "y veinte"]


def clock(rng):
    return f"las {rng.choice(HOURS)} {rng.choice(MINUTES)}"


def templates(rng, count):
    name = lambda: rng.choice(FIRST)
    full = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    number = lambda low=1, high=999: integer_words(rng.randint(low, high))
    patterns = [
        lambda: f"Hola {name()}, ¡qué alegría verte! ¿Cómo ha ido tu día?",
        lambda: f"{name()}, ¿puedes llamar a {full()} antes de {clock(rng)}?",
        lambda: f"La reunión con {full()} se ha movido al {rng.choice(DAYS)} {rng.choice(ORDINALS)} de {rng.choice(MONTHS)}.",
        lambda: f"Cuesta {number()} euros y {number(1, 99)} céntimos, impuestos incluidos.",
        lambda: f"Necesitamos {number(2, 60)} {rng.choice(ITEMS)}, {number(2, 60)} {rng.choice(ITEMS)} y {number(2, 60)} {rng.choice(ITEMS)}.",
        lambda: f"¿El vuelo a {rng.choice(PLACES[:10])} sale a {clock(rng)} o se ha retrasado otra vez?",
        lambda: f"Me llamo {full()} y vivo a unos {number(2, 90)} kilómetros de {rng.choice(PLACES[:10])}.",
        lambda: f"¿De verdad {name()} terminó los {number(3, 40)} capítulos en {number(2, 12)} días?",
        lambda: f"Por favor, lleva {number(2, 20)} {rng.choice(ITEMS)} a {rng.choice(PLACES)} antes del {rng.choice(DAYS)}.",
        lambda: f"El {rng.choice(ORDINALS)} de {rng.choice(MONTHS)}, {name()} y {name()} van a {rng.choice(PLACES[:10])}.",
        lambda: f"Son {number(100, 999)} pasos, más o menos {number(2, 30)}.",
        lambda: f"¿Quién le dijo a {name()} que la tienda cierra a {clock(rng)}?",
        lambda: f"Primero paramos en {rng.choice(PLACES)}, luego en {rng.choice(PLACES)} y por último en {rng.choice(PLACES)}.",
        lambda: f"{full()} sacó un {number(40, 100)} por ciento, {number(2, 30)} puntos más que la última vez.",
        lambda: f"Espera, ¿dices que {name()} se fue hace {number(2, 50)} minutos?",
        lambda: f"Le pedí a {name()}, a {name()} y a {name()} que nos esperaran en {rng.choice(PLACES)}.",
        lambda: f"La receta necesita {number(2, 12)} tazas de harina, {number(2, 6)} huevos y una pizca de sal.",
        lambda: f"Sinceramente, {name()}, ¡no esperaba que viniera tanta gente!",
        lambda: f"¿Me recuerdas si {full()} prefiere el {rng.choice(DAYS)} o el {rng.choice(DAYS)}?",
        lambda: f"Llovió durante {number(2, 14)} días en {rng.choice(PLACES[:10])}, así que nos quedamos en casa leyendo.",
    ]
    out = set()
    while len(out) < count:
        out.add(rng.choice(patterns)())
    return sorted(out)


def usable(line):
    line = re.sub(r"\s+", " ", line.strip().replace("’", "'").replace("«", '"').replace("»", '"'))
    if not OK.match(line) or not 28 <= len(line) <= 170 or not 5 <= len(line.split()) <= 30:
        return None
    if BAD.search(line):
        return None
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collector", type=Path, required=True, help="Common Voice es sentence-collector.txt")
    parser.add_argument("--wiki", type=Path, required=True, help="Common Voice es wiki.es.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=9000, help="Total sentences (both voices read each one)")
    parser.add_argument("--validation", type=int, default=350)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--collector-share", type=float, default=0.30, help="Share of the total taken from the conversational file")
    parser.add_argument("--template-share", type=float, default=0.10, help="Share written from templates")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    seen = set()
    rows = []

    def take(lines, count, origin, url, category):
        pool = sorted({s for s in (usable(line) for line in lines) if s and s.lower() not in seen})
        rng.shuffle(pool)
        # A quarter questions and a tenth exclamations, the rest statements;
        # longer sentences first within each kind.
        quota = {"?": round(0.25 * count), "!": round(0.10 * count)}
        quota["."] = count - sum(quota.values())
        pool.sort(key=lambda s: len(s) < 45)
        taken = {"?": 0, "!": 0, ".": 0}
        kept = 0
        for text in pool:
            if kept == count:
                break
            if taken[text[-1]] >= quota[text[-1]]:
                continue
            taken[text[-1]] += 1
            try:
                phones = phonemize(text)
            except Exception:
                continue
            if phones.unknown or len(phones) > args.max_tokens:
                continue
            seen.add(text.lower())
            identifier = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append({"id": identifier, "text": text, "original_text": text, "category": category, "origin": origin,
                         "source_id": identifier, "source_url": url,
                         "source_license": "cc0-1.0" if origin.startswith("common_voice") else "project-own",
                         "group_id": origin + ":" + identifier, "split": "train", "llm_model": None, "llm_modified": False,
                         "word_count": len(text.split()), "character_count": len(text), "audio_status": "not_generated"})
            kept += 1
        return kept

    repo = "https://github.com/common-voice/common-voice/tree/main/server/data/es"
    templates_count = round(args.template_share * args.count)
    collector_count = round(args.collector_share * args.count)
    wiki_count = args.count - templates_count - collector_count
    counts = {
        "collector": take(args.collector.read_text(encoding="utf-8", errors="ignore").splitlines(), collector_count,
                          "common_voice_sentence_collector", repo, "conversation"),
        "wiki": take(rng.sample(args.wiki.read_text(encoding="utf-8", errors="ignore").splitlines(), 200000), wiki_count,
                     "common_voice_wiki", repo, "reading"),
        "templates": take(templates(rng, templates_count * 2), templates_count, "tiny_voca_templates",
                          "https://github.com/n4ze3m/tiny-voca", "pronunciation"),
    }

    # Split by sentence, both voices always together, validation and test first
    # so they are the same for every model trained on this corpus.
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        if index < args.test:
            row["split"] = "test"
        elif index < args.test + args.validation:
            row["split"] = "validation"
    rows.sort(key=lambda r: (r["character_count"], r["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    splits = {name: sum(r["split"] == name for r in rows) for name in ("train", "validation", "test")}
    print(json.dumps({"scripts": len(rows), **counts, "splits": splits,
                      "questions": sum(r["text"].endswith("?") for r in rows),
                      "exclamations": sum(r["text"].endswith("!") for r in rows),
                      "characters": sum(r["character_count"] for r in rows),
                      "median_characters": sorted(r["character_count"] for r in rows)[len(rows) // 2]}))


if __name__ == "__main__":
    main()
