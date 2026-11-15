"""Front-end tests for the active language (``TINY_VOCA_LANGUAGE``).

This fork defaults to Spanish; run with ``TINY_VOCA_LANGUAGE=en`` to exercise
the original English front end, whose fingerprint is pinned so released
checkpoints keep loading.
"""

import shutil
import unittest
from unittest.mock import patch

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.text import (BOS, EOS, LANGUAGE, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, canonical, chunks,
                            frontend_fingerprint, normalize_text, parse_espeak, phonemize, tokenize)

ESPEAK = shutil.which("espeak-ng")
SPANISH = LANGUAGE == "es"


class FrontEndTests(unittest.TestCase):
    def test_inventory_is_consistent_with_the_model_vocabulary(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(len(PHONES), 42 if SPANISH else 59)
        with self.assertRaises(ValueError):
            Config(vocab_size=259)     # the byte vocabulary is gone

    @unittest.skipIf(SPANISH, "English text rules")
    def test_chunks_keep_numbers_contractions_and_abbreviations(self):
        pieces = list(chunks(normalize_text("Dr. Smith can't pay 3.14 at 3:45 for 1,000 U.S. days, right?")))
        self.assertEqual([p for punctuation, p in pieces if punctuation], [".", ",", "?"])
        self.assertIn("3.14 at 3:45 for 1,000 U.S", pieces[0][1])
        self.assertEqual(normalize_text("“Wait…”  he said -- ‘no’ [aside]..."), '"Wait…" he said — \'no\' (aside)…')

    @unittest.skipUnless(SPANISH, "Spanish text rules")
    def test_chunks_keep_spanish_numbers_abbreviations_and_opening_marks(self):
        pieces = list(chunks(normalize_text("El Sr. García pagó 3,14 euros a las 3:45 por 1.000 días, ¿verdad?")))
        self.assertEqual([p for punctuation, p in pieces if punctuation], [",", "¿", "?"])
        self.assertIn("El Sr. García pagó 3,14 euros a las 3:45 por 1.000 días", pieces[0][1])
        self.assertEqual(normalize_text("«Hola» —dijo-- ‘adiós’ [aparte]..."), '"Hola" —dijo— \'adiós\' (aparte)…')

    @unittest.skipIf(SPANISH, "English eSpeak output")
    def test_separated_espeak_output_becomes_phones_with_stress_features(self):
        words = parse_espeak("h_ə_l_ˈoʊ w_ˈɜː_l_d\nð_ˌɛ_m æ_z__\n")
        self.assertEqual(words[0], [("h", 0), ("ə", 0), ("l", 0), ("oʊ", 1)])
        self.assertEqual(words[2], [("ð", 0), ("ɛ", 2), ("m", 0)])
        self.assertEqual(words[3], [("æ", 0), ("z", 0)])               # empty units are dropped
        result = tokenize([(True, '"'), (False, words[:2]), (True, ","), (True, ","), (False, words[2:]), (True, "?")])
        self.assertEqual(result.symbols, (BOS, '"', "h", "ə", "l", "oʊ", SP, "w", "ɜː", "l", "d", ",", "ð", "ɛ", "m", SP, "æ", "z", "?", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertTrue(all(s in PHONE_SET or s in SYMBOLS for s in result.symbols))
        # Stress is never a token, so nothing but phones, pauses and punctuation needs frames.
        self.assertNotIn("ˈ", "".join(result.symbols))

    @unittest.skipUnless(SPANISH, "Spanish eSpeak output")
    def test_separated_espeak_output_becomes_phones_with_stress_features(self):
        words = parse_espeak("k_ˈo_m_o n_ˈi_ɲ_o\np_ˌe_ɾ_ˈo ʝ_a__\n")
        self.assertEqual(words[0], [("k", 0), ("o", 1), ("m", 0), ("o", 0)])
        self.assertEqual(words[2], [("p", 0), ("e", 2), ("ɾ", 0), ("o", 1)])
        self.assertEqual(words[3], [("ʝ", 0), ("a", 0)])               # empty units are dropped
        result = tokenize([(True, "¡"), (False, words[:2]), (True, ","), (True, ","), (False, words[2:]), (True, "?")])
        self.assertEqual(result.symbols, (BOS, "¡", "k", "o", "m", "o", SP, "n", "i", "ɲ", "o", ",", "p", "e", "ɾ", "o", SP, "ʝ", "a", "?", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertTrue(all(s in PHONE_SET or s in SYMBOLS for s in result.symbols))
        self.assertNotIn("ˈ", "".join(result.symbols))

    @unittest.skipIf(SPANISH, "English fallbacks")
    def test_out_of_inventory_phones_fall_back_or_are_reported(self):
        self.assertEqual(canonical("r"), ("ɹ",))
        self.assertEqual(canonical("ɛː"), ("ɛ",))
        self.assertEqual(canonical("ɑ̃"), ("ɑː",))
        self.assertIsNone(canonical("ʘ"))
        result = tokenize([(False, [[("ts", 0), ("ʘ", 0), ("aʊɚ", 1)]])])
        self.assertEqual(result.symbols, (BOS, "t", "s", UNK, "aʊ", "ɚ", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 0, 1, 0, 0))
        self.assertEqual(result.unknown, ("ʘ",))
        with self.assertRaises(ValueError):
            tokenize([(True, ".")])

    @unittest.skipUnless(SPANISH, "Spanish fallbacks")
    def test_out_of_inventory_phones_fall_back_or_are_reported(self):
        self.assertEqual(canonical("c"), ("k",))
        self.assertEqual(canonical("ɔ"), ("o",))
        self.assertEqual(canonical("aː"), ("a",))
        self.assertEqual(canonical("ʰ"), ())                            # a modifier letter carries no segment
        self.assertIsNone(canonical("ʘ"))
        result = tokenize([(False, [[("ts", 0), ("ʘ", 0), ("eʊ", 1)]])])
        self.assertEqual(result.symbols, (BOS, "ts", UNK, "eʊ", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 1, 0))
        self.assertEqual(result.unknown, ("ʘ",))
        with self.assertRaises(ValueError):
            tokenize([(True, ".")])

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = frontend_fingerprint()
        with patch.object(text, "FRONTEND_VERSION", 999):
            self.assertNotEqual(frontend_fingerprint(), baseline)
        with patch.object(text, "SYMBOLS", SYMBOLS + ("new",)):
            self.assertNotEqual(frontend_fingerprint(), baseline)

    @unittest.skipUnless(ESPEAK, "requires the espeak-ng executable")
    @unittest.skipIf(SPANISH, "English pronunciation")
    def test_real_espeak_keeps_the_last_phoneme_of_every_clause(self):
        # Regression: without a final newline eSpeak NG 1.51 dropped the last character
        # ("lips" -> l ɪ p, "away" -> ɐ w æ), corrupting 98% of the old training rows.
        self.assertEqual(phonemize("your Majesty's own lips").symbols[-4:], ("ɪ", "p", "s", EOS))
        self.assertEqual(phonemize("right away").symbols[-2:], ("eɪ", EOS))
        self.assertEqual(phonemize("Hello").symbols[1:-1], ("h", "ə", "l", "oʊ"))
        result = phonemize("My choice, go! Is it déjà vu?")
        self.assertEqual([s for s in result.symbols if s in ",!?"], [",", "!", "?"])
        self.assertEqual(result.unknown, ())
        self.assertNotEqual(phonemize("You did.").ids, phonemize("You did?").ids)
        self.assertEqual(phonemize("You did.").ids, phonemize(" You  did. ").ids)

    @unittest.skipUnless(ESPEAK, "requires the espeak-ng executable")
    @unittest.skipUnless(SPANISH, "Spanish pronunciation")
    def test_real_espeak_keeps_the_last_phoneme_of_every_clause(self):
        # Same regression as the English test: the input must end with a newline.
        self.assertEqual(phonemize("El niño comió piña y jamón").symbols[-4:], ("m", "o", "n", EOS))
        self.assertEqual(phonemize("Vamos ya").symbols[-3:], ("ʝ", "a", EOS))
        self.assertEqual(phonemize("Hola").symbols[1:-1], ("o", "l", "a"))
        result = phonemize("¿Vienes, o no? ¡Qué barbaridad!")
        self.assertEqual([s for s in result.symbols if s in ",?!¡¿"], ["¿", ",", "?", "¡", "!"])
        self.assertEqual(result.unknown, ())
        self.assertNotEqual(phonemize("Tú vienes.").ids, phonemize("¿Tú vienes?").ids)
        self.assertEqual(phonemize("Tú vienes.").ids, phonemize(" Tú  vienes. ").ids)

    @unittest.skipUnless(ESPEAK, "requires the espeak-ng executable")
    @unittest.skipIf(SPANISH, "English pronunciation")
    def test_real_espeak_units_are_inventory_phones(self):
        sample = ("The button on the bottle was a little better than the water. Dr. Smith paid 3.14 dollars for the car "
                  "near the fire; she read the book I read yesterday, and the poor tour guide heard pure joy!")
        result = phonemize(sample)
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 30)
        self.assertEqual(sum(result.stress[i] > 0 and s not in PHONE_SET for i, s in enumerate(result.symbols)), 0)

    @unittest.skipUnless(ESPEAK, "requires the espeak-ng executable")
    @unittest.skipUnless(SPANISH, "Spanish pronunciation")
    def test_real_espeak_units_are_inventory_phones(self):
        sample = ("El pequeño murciélago voló sobre el jardín durante toda la noche. ¿Cuántas personas van a venir "
                  "a la reunión del jueves por la tarde? La lluvia caía con fuerza sobre los tejados del pueblo, "
                  "y mi abuelo preparaba un guiso de garbanzos que olía de maravilla.")
        result = phonemize(sample)
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 18)
        self.assertEqual(sum(result.stress[i] > 0 and s not in PHONE_SET for i, s in enumerate(result.symbols)), 0)


if __name__ == "__main__":
    unittest.main()
