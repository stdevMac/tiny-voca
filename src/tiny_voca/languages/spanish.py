"""Spanish profile: one phone inventory shared by every accent, plus accent rules.

The inventory is the union of what eSpeak emits for Spanish over a 25k-sentence
sample of the Common Voice Spanish Wikipedia corpus, minus foreign-language
artefacts. Every symbol is also a label of the forced aligner
(``facebook/wav2vec2-lv-60-espeak-cv-ft``), so one aligned label is one token.

Accents differ in the *front end*, not in the inventory:

* ``es-ES`` Castilian: distinción (``θ`` for <z>, <ce>, <ci>), lleísmo (``ʎ``).
* ``es-AR`` Rioplatense: seseo (already in ``es-419``) and sheísmo, where <ll>
  and <y> become ``ʃ`` instead of ``ʝ``.
* ``es-CU`` Cuban: seseo, aspiration of syllable-final /s/ to ``h``, and loss of
  intervocalic /d/ in ``-ado``/``-ada``.

Only ``es-ES`` has been validated against a native ear. The other two are
rule-based approximations and should be checked before a model is trained on
them. The inventory is shared so one checkpoint can serve all three later.

The symbol order is append-only: changing it invalidates every checkpoint
trained on it.
"""

from dataclasses import dataclass

CODE = "es"
NAME = "Spanish"
FRONTEND_VERSION = 1              # first Spanish front end; no released model yet
FRONTENDS = ("espeak",)          # no permissive Spanish G2P yet; see docs/spanish.md
DECIMAL_COMMA = True             # "3,14" is a number; a full stop groups thousands
ONLY_LANGUAGE = "tiny-voca solo lee español por ahora"
SENTENCE_START = "A-ZÁÉÍÓÚÑÜ0-9¿¡"
ASR_MODEL = "small"               # evaluation only; multilingual, so it can read Spanish
ASR_LANGUAGE = "es"
TTS_LANGUAGE = "Spanish"          # the name the teacher TTS expects

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
# Spanish adds the opening marks. «» are normalised to " by ``text.normalize_text``.
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…", "¿", "¡")

CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "θ", "s", "x", "h", "tʃ", "dʒ", "ts",
              "m", "n", "ɲ", "ŋ", "l", "ʎ", "ɾ", "r", "j", "w", "ð", "β", "ɣ", "ʝ", "ʃ", "ʒ", "pː")
VOWELS = ("a", "e", "i", "o", "u", "ɛ", "aɪ", "aʊ", "eɪ", "eʊ", "oɪ")
PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES

VOICELESS = frozenset(("p", "t", "k", "f", "θ", "s", "ʃ", "x", "tʃ", "h"))
FRICATIVES = frozenset(("f", "θ", "s", "ʃ", "x"))

# Foreign or rare eSpeak output mapped onto the Spanish inventory. An empty
# tuple drops the phone (modifier letters that carry no segment of their own).
FALLBACK = {
    "c": ("k",), "ɟ": ("d",), "ʈ": ("t",), "ɳ": ("n",), "ʋ": ("b",), "ʔ": ("h",),
    "ɪ": ("i",), "ɫ": ("l",), "ɑː": ("a",), "ə": ("e",), "aː": ("a",), "iː": ("i",),
    "uː": ("u",), "oː": ("o",), "eː": ("e",), "bʰ": ("b",), "ʰχ": ("x",), "χ": ("x",),
    "ɔ": ("o",), "z": ("s",), "v": ("b",), "ɹ": ("r",), "ɻ": ("r",), "ʁ": ("r",),
    "ɬ": ("l",), "d̪": ("d",), "t̪": ("t",), "s̪": ("s",), "n̪": ("n",), "l̪": ("l",),
    "β̞": ("β",), "ð̞": ("ð",), "ɣ̞": ("ɣ",), "ʝ̞": ("ʝ",), "ɟʝ": ("ʝ",),
    "ʰ": (), "ʲ": (), "ˠ": (), "ˀ": (), "ː": (),
}

ABBREVIATIONS = frozenset(("sr", "sra", "srta", "dr", "dra", "prof", "profa", "ud", "uds", "etc",
                           "ej", "pág", "núm", "av", "tel", "dpto", "gral", "cnel", "aprox", "izq", "dcha"))
NOT_AN_END = ("sr", "sra", "srta", "dr", "dra", "prof", "profa", "ud", "uds", "etc", "ej",
              "pág", "núm", "av", "tel", "dpto", "gral", "cnel", "aprox")

# Held-out sentences for ``tiny_voca.evaluate``: in no split, chosen for phoneme
# coverage (ñ, ll, j/x, θ, rr, every diphthong).
NOVEL_SENTENCES = (
    "El pequeño murciélago voló sobre el jardín durante toda la noche.",
    "¿Cuántas personas van a venir a la reunión del jueves por la tarde?",
    "Necesito que me envíes el informe antes del viernes, por favor.",
    "La lluvia caía con fuerza sobre los tejados del pueblo.",
    "Mi abuelo prepara un guiso de garbanzos que huele de maravilla.",
    "El tren de las ocho y media sale del andén número cuatro.",
    "Aquella mañana, el cielo estaba tan limpio que se veían las montañas.",
    "Los niños corrían por la playa mientras sus padres preparaban la cena.",
    # A yes/no question, because Spanish marks it with intonation alone and the
    # rest of this list would never test that. Deliberately formal and unlike
    # anything in the corpus: the corpus's questions are conversational, so a
    # chatty sentence here would only measure how well a template was copied.
    # Three more of the same register, because one sample cannot tell a
    # systematic accent drift from a single unlucky clip.
    "¿Cree que el comité aprobará la propuesta antes del verano?",
    "¿Considera que la propuesta merece una segunda lectura?",
    "¿Le parece que el informe está listo para presentarlo?",
    "¿Piensa que la situación mejorará durante el próximo trimestre?",
    "¡Menuda sorpresa nos dio tu hermano con aquella noticia!",
    "¡Vaya alegría llevarme esta noticia después de tantos meses de espera!",
)


@dataclass(frozen=True)
class Accent:
    code: str
    name: str
    espeak_voice: str
    transform: object = None     # callable(words) -> words, or None


def _seseo(phone):
    """Latin American seseo: <z>, <ce>, <ci> are [s], never [θ]."""
    return "s" if phone == "θ" else phone


def _sheismo(words):
    """Rioplatense: seseo, and <ll>/<y> are [ʃ], not [ʎ]/[ʝ]."""
    return [[("ʃ" if phone in ("ʝ", "ʎ") else _seseo(phone), stress) for phone, stress in word] for word in words]


def _cuban(words):
    """Cuban: seseo, yeísmo, syllable-final /s/ is [h], and the lenited /d/ drops in -ado/-ada."""
    out = []
    for word in words:
        phones = [("ʝ" if phone == "ʎ" else _seseo(phone), stress) for phone, stress in word]
        for index, (phone, stress) in enumerate(phones):
            if phone != "s":
                continue
            following = phones[index + 1][0] if index + 1 < len(phones) else None
            # A glide after /s/ is the onset of the next syllable (ciudad, suave),
            # not a coda consonant, so it does not trigger aspiration.
            if following is None or (following not in VOWELS and following not in ("j", "w")):
                phones[index] = ("h", stress)
        # -ado / -ada / -ido / -ida: eSpeak writes the lenited /d/ as ð.
        if len(phones) >= 3 and phones[-2][0] in ("d", "ð") and phones[-3][0] in VOWELS and phones[-1][0] in VOWELS:
            phones.pop(-2)
        out.append(phones)
    return out


# All three accents start from the Castilian voice and are rewritten from there.
# ``es-419`` was the obvious choice for the Latin American ones, but it renders
# <ll> as ``j j`` and <y> as ``ʝ`` inconsistently, which makes sheísmo a
# guessing game. One voice plus explicit rules is predictable.
ACCENTS = {
    "es-ES": Accent("es-ES", "Castilian (Spain)", "es"),
    "es-AR": Accent("es-AR", "Rioplatense (Argentina)", "es", _sheismo),
    "es-CU": Accent("es-CU", "Cuban", "es", _cuban),
}
DEFAULT_ACCENT = "es-ES"
