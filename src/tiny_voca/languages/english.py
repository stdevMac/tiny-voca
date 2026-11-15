"""English (US) profile: inventory, espeak voice and text rules.

Moved here unchanged from ``tiny_voca.text`` so the machinery can serve more
than one language. The symbol order is append-only: changing it invalidates
every checkpoint trained on it.
"""

CODE = "en"
NAME = "English (US)"
FRONTEND_VERSION = 3              # the released English checkpoints record this
ESPEAK_VOICE = "en-us"
ONLY_LANGUAGE = "tiny-voca only reads English for now"
SENTENCE_START = "A-Z0-9"         # a sentence begins with a capital or a digit
ASR_MODEL = "small.en"            # evaluation only
ASR_LANGUAGE = "en"
TTS_LANGUAGE = "English"          # the name the teacher TTS expects
FRONTENDS = ("espeak", "own")     # "own" is dizione, the permissive CMUdict front end
DECIMAL_COMMA = False             # "3.14" is a number; a comma is a separator

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")

CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
              "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ")
VOWELS = ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
          "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
          "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩")
PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES

VOICELESS = frozenset(("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ"))
# Voiceless fricatives whose boundary against a vowel must flip the pitch
# tracker's voicing: the label-free timing probe in ``tiny_voca.annotate``.
FRICATIVES = frozenset(("f", "θ", "s", "ʃ"))

# eSpeak phonemes outside the inventory (foreign words, rare allophones) map to
# the closest inventory phone(s). Anything else becomes <unk> and is reported.
FALLBACK = {
    "r": ("ɹ",), "ɻ": ("ɹ",), "ʁ": ("ɹ",), "x": ("k",), "ç": ("h",), "ɬ": ("l",), "ɫ": ("l",), "ɲ": ("n", "j"),
    "ts": ("t", "s"), "dz": ("d", "z"), "β": ("v",), "ɣ": ("ɡ",), "ʎ": ("l", "j"),
    "a": ("æ",), "aː": ("ɑː",), "ɑ": ("ɑː",), "ɒ": ("ɑː",), "ɑ̃": ("ɑː",), "ɔ̃": ("ɔː",), "ɛ̃": ("æ",),
    "e": ("eɪ",), "eː": ("eɪ",), "o": ("oʊ",), "u": ("uː",), "y": ("uː",), "yː": ("uː",),
    "ø": ("ɜː",), "øː": ("ɜː",), "œ": ("ɜː",), "ɜ": ("ɜː",), "ɛː": ("ɛ",), "ɪː": ("ɪ",),
    "əʊ": ("oʊ",), "eə": ("ɛɹ",), "ɪə": ("iə",), "ʊə": ("ʊɹ",), "aʊɚ": ("aʊ", "ɚ"), "l̩": ("əl",), "m̩": ("ə", "m"),
}

ABBREVIATIONS = frozenset(("mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"))
# Words after which a full stop is not the end of a sentence (``tiny_voca.infer``).
NOT_AN_END = ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "inc", "ltd", "co",
              "e.g", "i.e", "a.m", "p.m")

# Held-out sentences for ``tiny_voca.evaluate``: in no split, chosen for phoneme coverage.
NOVEL_SENTENCES = (
    "The quick brown fox jumps over the lazy dog near the quiet river.",
    "Please remember to call your grandmother before the weekend starts.",
    "How many people are coming to the conference in September?",
    "I can't believe it's already half past nine; we should leave now!",
    "Tiny voca is a small text to speech model trained from scratch.",
    "After the storm passed, the mountain air felt fresh, clear, and surprisingly warm.",
    "Could you send me the report by Friday afternoon, please?",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
)
