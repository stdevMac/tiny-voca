"""Spanish number reading, used where digits must become words.

The phonemizer does not need this: eSpeak reads numbers, dates and money from
context, which is the whole point of using it. Two places do need it:

* word-error-rate scoring, because Whisper writes "4090" while the reference
  sentence may say "cuatro mil noventa" (or the other way round), and
* corpus building, to keep digit-bearing templates out of the training text
  unless the intended reading is unambiguous.

Only what a sentence realistically contains is implemented: integers up to
999,999,999, decimals with a comma, and percentages. Anything else is left
alone for eSpeak to interpret.
"""

import re

UNITS = ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
         "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
         "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
         "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve")
TENS = ("", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa")
HUNDREDS = ("", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
            "seiscientos", "setecientos", "ochocientos", "novecientos")


def below_hundred(n):
    if n < 30:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    return TENS[tens] if not unit else f"{TENS[tens]} y {UNITS[unit]}"


def below_thousand(n):
    if n == 0:
        return "cero"
    if n == 100:
        return "cien"
    hundreds, rest = divmod(n, 100)
    words = HUNDREDS[hundreds] if hundreds else ""
    if rest:
        words = f"{words} {below_hundred(rest)}".strip()
    return words


def integer_words(n):
    """Cardinal reading of ``n`` (0 <= n <= 999,999,999)."""
    if not 0 <= n <= 999_999_999:
        raise ValueError("only 0..999,999,999 is implemented")
    if n < 1000:
        return below_thousand(n)
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1000)
    words = []
    if millions:
        words.append("un millón" if millions == 1 else f"{integer_words(millions)} millones")
    if thousands:
        words.append("mil" if thousands == 1 else f"{below_thousand(thousands)} mil")
    if units:
        words.append(below_thousand(units))
    return " ".join(words)


def decimal_words(text):
    """``3,14`` -> ``tres coma catorce``.

    The fraction is read as a number, not digit by digit, because that is what
    the front end does: eSpeak says "cincuenta y seis" for ``,56`` and "noventa"
    for ``,90``. Reading it digit by digit here made the word error rate
    disagree with the audio on every price and measurement. A fraction with a
    leading zero is the exception, and eSpeak reads that one digit by digit
    (``1,05`` -> "uno coma cero cinco").
    """
    whole, fraction = text.split(",")
    if fraction.startswith("0"):
        digits = " ".join(integer_words(int(d)) for d in fraction)
    else:
        digits = integer_words(int(fraction))
    return f"{integer_words(int(whole))} coma {digits}"


def expand(text):
    """Replace percentages, decimals and plain integers with their reading."""
    text = re.sub(r"(\d+)\s*%", lambda m: f"{integer_words(int(m[1]))} por ciento", text)
    text = re.sub(r"\b(\d{1,3}(?:\.\d{3})+)\b", lambda m: m[1].replace(".", ""), text)   # 1.500 -> 1500
    text = re.sub(r"\b\d+,\d+\b", lambda m: decimal_words(m[0]), text)
    return re.sub(r"\b\d+\b", lambda m: integer_words(int(m[0])), text)


def normalize(text):
    """Lower-case, strip punctuation and expand numbers: the form both sides of
    a word-error-rate comparison are reduced to."""
    import unicodedata
    text = expand(unicodedata.normalize("NFC", text).lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
