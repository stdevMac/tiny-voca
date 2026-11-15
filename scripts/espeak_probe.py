"""Print a fingerprint of what the phonemizer actually does, not what version it is.

The front-end fingerprint in ``tiny_voca.text`` covers the inventory, the eSpeak
voice and the fallback table, but not the eSpeak build. The corpus is made on
the GPU host (Ubuntu ships espeak-ng 1.51) and inference happens on a laptop
(Homebrew ships 1.52), so a change in the program would silently produce
different phones at inference than at training.

This phonemizes a fixed set that covers every phone and every rule the Spanish
front end has, and prints a hash. Same hash on two machines means the same
phonemes for the same text.

    python scripts/espeak_probe.py            # this machine
    python scripts/espeak_probe.py --json     # machine-readable, for the runbook
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_voca.text import ACCENT, LANGUAGE, SYMBOLS, espeak_version, phonemize

# Every phone in the inventory, every accent rule, and the punctuation that
# carries pauses. Kept short: this runs on both machines and must be cheap.
PROBES = (
    "Zaragoza, Cádiz, Girona y Jerez.",
    "El niño comió piña y jamón en la mañana.",
    "La lluvia caía sobre la calle, y el perro ladraba.",
    "¿Cuántas personas van a venir a la reunión?",
    "¡Qué barbaridad! No me lo puedo creer.",
    "Los muchachos estaban cansados y se fueron.",
    "Cuesta cuatro mil noventa euros con impuesto.",
    "Vivo en el tercer piso, puerta segunda.",
    "El rey Juan Carlos y el teniente general.",
    "Ayer llovió muchísimo en Buenos Aires.",
    "Compró tres kilos de naranjas y dos de patatas.",
    "El arquitecto diseñó un edificio enorme.",
    "Había un taxi esperando en la esquina.",
    "Mi abuelo prepara un guiso de garbanzos.",
    "Los niños corrían por la playa mientras sus padres preparaban la cena.",
    "El tren de las ocho y media sale del andén cuatro.",
    "Aquella mañana el cielo estaba tan limpio que se veían las montañas.",
    "Necesito que me envíes el informe antes del viernes.",
    "El pequeño murciélago voló sobre el jardín.",
    "Quiero un café con leche y un croissant.",
    "El espectáculo empezó a las nueve y terminó tarde.",
    "Fue una experiencia extraordinaria, sinceramente.",
    "El juez dictó sentencia el jueves por la mañana.",
    "Compré un reloj antiguo en el mercado de antigüedades.",
    "La situación económica preocupa a mucha gente.",
    "El futbolista marcó un gol en el último minuto.",
    "Mi vecino tiene un perro que ladra toda la noche.",
    "Estábamos hablando de política cuando llegó ella.",
    "El médico le recetó un jarabe para la tos.",
    "Los excursionistas subieron hasta la cima del monte.",
    "Aquel verano hizo un calor insoportable.",
    "La biblioteca cierra a las ocho de la tarde.",
    "El concierto fue magnífico, aunque duró demasiado.",
    "Mi hermana estudia arquitectura en Valencia.",
    "El pescado estaba fresco y sabía delicioso.",
    "Nos quedamos en casa porque llovía a cántaros.",
    "El profesor explicó la lección con mucha paciencia.",
    "Aquella decisión cambió el rumbo de la empresa.",
    "Los vecinos se quejaron del ruido hasta la madrugada.",
    "El aeropuerto estaba lleno de viajeros nerviosos.",
)


def fingerprint():
    lines = []
    for text in PROBES:
        phones = phonemize(text)
        lines.append(" ".join(phones.symbols) + "|" + "".join(str(s) for s in phones.stress))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = {"language": LANGUAGE, "accent": ACCENT, "symbols": len(SYMBOLS), "espeak": espeak_version(),
              "probe": fingerprint()}
    print(json.dumps(result) if args.json else
          f"{result['language']}/{result['accent']}  {result['espeak']}\nprobe {result['probe']}")


if __name__ == "__main__":
    main()
