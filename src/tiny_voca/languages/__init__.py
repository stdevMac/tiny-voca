"""Language profiles and the accent selected for this process.

The language is chosen once, at import time, because the phoneme inventory is
part of the model configuration: ``tiny_voca.text`` exposes ``SYMBOLS`` and
``SYMBOL_TO_ID`` as module constants and every other module reads them at
import. Choose with ``TINY_VOCA_LANGUAGE`` (``en`` or ``es``; this fork defaults
to ``es``) and, for Spanish, ``TINY_VOCA_ACCENT`` (``es-ES``, ``es-AR``,
``es-CU``; default ``es-ES``).

A profile module provides: ``CODE``, ``NAME``, ``FRONTENDS``,
``DECIMAL_COMMA``, ``SPECIALS``, ``PUNCTUATION``, ``PHONES``, ``VOWELS``,
``SYMBOLS``, ``VOICELESS``, ``FRICATIVES``, ``FALLBACK``, ``ABBREVIATIONS``,
``NOT_AN_END`` and either ``ESPEAK_VOICE`` or ``ACCENTS`` + ``DEFAULT_ACCENT``.
"""

from importlib import import_module
import os

LANGUAGES = {"en": "tiny_voca.languages.english", "es": "tiny_voca.languages.spanish"}
VARIABLE = "TINY_VOCA_LANGUAGE"
ACCENT_VARIABLE = "TINY_VOCA_ACCENT"
DEFAULT_LANGUAGE = "es"


def language_code():
    code = os.environ.get(VARIABLE, DEFAULT_LANGUAGE)
    if code not in LANGUAGES:
        raise ValueError(f"{VARIABLE} must be one of {sorted(LANGUAGES)}, not {code!r}")
    return code


_code = language_code()
profile = import_module(LANGUAGES[_code])


def accent():
    """The active accent object, or ``None`` for a language without accents."""
    options = getattr(profile, "ACCENTS", None)
    if not options:
        return None
    code = os.environ.get(ACCENT_VARIABLE, profile.DEFAULT_ACCENT)
    if code not in options:
        raise ValueError(f"{ACCENT_VARIABLE} must be one of {sorted(options)}, not {code!r}")
    return options[code]


def espeak_voice():
    chosen = accent()
    return chosen.espeak_voice if chosen else profile.ESPEAK_VOICE


def apply_accent(words):
    """Rewrite phones for the active accent. ``words`` is the parsed eSpeak
    output: a list of words, each a list of ``(phone, stress)``."""
    chosen = accent()
    if chosen is None or chosen.transform is None:
        return words
    return chosen.transform(words)
