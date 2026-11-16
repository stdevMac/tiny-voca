"""Spanish specifics: the inventory, the accent rules and the number reader.

These run only for ``TINY_VOCA_LANGUAGE=es``; the English path has its own
tests in ``test_frontend.py``.
"""

import unittest

from tiny_voca import numbers_es
from tiny_voca.languages import spanish
from tiny_voca.text import LANGUAGE, PHONE_SET, SYMBOLS, SYMBOL_TO_ID, VOWELS

SPANISH = LANGUAGE == "es"


@unittest.skipUnless(SPANISH, "Spanish front end")
class InventoryTests(unittest.TestCase):
    def test_every_phone_is_a_distinct_symbol_and_a_vowel_is_a_vowel(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(len(spanish.PHONES), 42)
        self.assertTrue(set(spanish.VOWELS) <= PHONE_SET)
        self.assertTrue(set(spanish.CONSONANTS) <= PHONE_SET)
        self.assertFalse(set(spanish.VOWELS) & set(spanish.CONSONANTS))
        # Symbols are append-only: this is the order the checkpoints record.
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(SYMBOLS[:5], ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>"))
        self.assertEqual(SYMBOLS[5:7], (",", "."))

    def test_the_inventory_covers_what_espeak_emits(self):
        # Every phone of the 25k-sentence probe, plus ʒ for Rioplatense sheísmo.
        for phone in ("a e i o u ɛ aɪ aʊ eɪ eʊ oɪ p b t d k ɡ f θ s x h tʃ dʒ ts m n ɲ ŋ l ʎ ɾ r j w ð β ɣ ʝ ʃ ʒ pː").split():
            self.assertIn(phone, PHONE_SET, phone)
        self.assertEqual(set(spanish.VOWELS), {"a", "e", "i", "o", "u", "ɛ", "aɪ", "aʊ", "eɪ", "eʊ", "oɪ"})

    def test_voiceless_and_fricative_sets_are_consistent(self):
        self.assertTrue(spanish.FRICATIVES <= spanish.VOICELESS)
        self.assertIn("x", spanish.VOICELESS)          # /x/ is Spanish, not English
        self.assertNotIn("ð", spanish.VOICELESS)       # the lenited allophones are voiced


@unittest.skipUnless(SPANISH, "Spanish accent rules")
class AccentTests(unittest.TestCase):
    def test_castilian_is_the_default_and_changes_nothing(self):
        self.assertEqual(spanish.DEFAULT_ACCENT, "es-ES")
        self.assertEqual(spanish.ACCENTS["es-ES"].espeak_voice, "es")
        self.assertIsNone(spanish.ACCENTS["es-ES"].transform)
        # Every accent starts from the same voice; the rules do the rest.
        self.assertEqual({accent.espeak_voice for accent in spanish.ACCENTS.values()}, {"es"})

    def test_rioplatense_turns_ll_and_y_into_she_and_drops_distincion(self):
        # "calle" -> cashe, "ayer" -> asher, "Zaragoza" -> saragosa
        words = [[("k", 0), ("a", 1), ("ʎ", 0), ("e", 0)],
                 [("a", 0), ("ʝ", 1), ("e", 0), ("ɾ", 0)],
                 [("s", 0), ("a", 0), ("ɾ", 0), ("a", 0), ("ɣ", 0), ("o", 0), ("θ", 0), ("a", 0)]]
        self.assertEqual(spanish._sheismo(words),
                         [[("k", 0), ("a", 1), ("ʃ", 0), ("e", 0)],
                          [("a", 0), ("ʃ", 1), ("e", 0), ("ɾ", 0)],
                          [("s", 0), ("a", 0), ("ɾ", 0), ("a", 0), ("ɣ", 0), ("o", 0), ("s", 0), ("a", 0)]])

    def test_cuban_aspirates_syllable_final_s_and_drops_lenited_d(self):
        # "los" -> loh, "esta" -> ehta, "casa" -> casa, "pasado" -> pasao, "ciudad" -> siuda
        words = [[("l", 0), ("o", 0), ("s", 0)],
                 [("e", 0), ("s", 0), ("t", 0), ("a", 0)],
                 [("k", 0), ("a", 0), ("s", 0), ("a", 0)],
                 [("p", 0), ("a", 0), ("s", 0), ("a", 0), ("ð", 0), ("o", 0)],
                 [("θ", 0), ("j", 0), ("u", 0), ("ð", 0), ("a", 0), ("d", 0)],
                 [("ʎ", 0), ("u", 0), ("β", 0), ("j", 0), ("a", 0)]]
        self.assertEqual(spanish._cuban(words),
                         [[("l", 0), ("o", 0), ("h", 0)],
                          [("e", 0), ("h", 0), ("t", 0), ("a", 0)],
                          [("k", 0), ("a", 0), ("s", 0), ("a", 0)],
                          [("p", 0), ("a", 0), ("s", 0), ("a", 0), ("o", 0)],
                          [("s", 0), ("j", 0), ("u", 0), ("ð", 0), ("a", 0), ("d", 0)],
                          [("ʝ", 0), ("u", 0), ("β", 0), ("j", 0), ("a", 0)]])

    def test_accent_symbols_stay_inside_the_shared_inventory(self):
        for accent in spanish.ACCENTS.values():
            if accent.transform is None:
                continue
            words = [[("ʝ", 0), ("a", 0)], [("l", 0), ("o", 0), ("s", 0)], [("p", 0), ("a", 0), ("d", 0), ("o", 0)]]
            for word in accent.transform(words):
                for phone, _ in word:
                    self.assertIn(phone, PHONE_SET, f"{accent.code}: {phone}")


@unittest.skipUnless(SPANISH, "Spanish number reader")
class NumberTests(unittest.TestCase):
    def test_cardinals(self):
        cases = {0: "cero", 1: "uno", 16: "dieciséis", 21: "veintiuno", 22: "veintidós", 30: "treinta",
                 31: "treinta y uno", 100: "cien", 101: "ciento uno", 500: "quinientos", 1000: "mil",
                 2000: "dos mil", 4090: "cuatro mil noventa", 1000000: "un millón",
                 2000000: "dos millones", 999999999: "novecientos noventa y nueve millones novecientos noventa y nueve mil novecientos noventa y nueve"}
        for number, words in cases.items():
            self.assertEqual(numbers_es.integer_words(number), words, number)
        with self.assertRaises(ValueError):
            numbers_es.integer_words(1_000_000_000)

    def test_expand_handles_decimals_thousands_and_percentages(self):
        # The fraction is read as a number because that is what the front end
        # does; a leading zero is read digit by digit, also as the front end does.
        self.assertEqual(numbers_es.expand("3,14"), "tres coma catorce")
        self.assertEqual(numbers_es.expand("1,05"), "uno coma cero cinco")
        self.assertEqual(numbers_es.expand("27,90"), "veintisiete coma noventa")
        self.assertEqual(numbers_es.expand("1.234,56"), "mil doscientos treinta y cuatro coma cincuenta y seis")
        self.assertEqual(numbers_es.expand("1.500"), "mil quinientos")
        self.assertEqual(numbers_es.expand("25%"), "veinticinco por ciento")
        self.assertEqual(numbers_es.expand("Son 4090 euros."), "Son cuatro mil noventa euros.")

    def test_normalize_reduces_both_sides_of_a_wer_comparison(self):
        self.assertEqual(numbers_es.normalize("¡Hola, 4090 euros!"), "hola cuatro mil noventa euros")
        # Digits and words land on the same string, which is the point.
        self.assertEqual(numbers_es.normalize("4090"), numbers_es.normalize("cuatro mil noventa"))
        self.assertEqual(numbers_es.normalize("¿Qué tal?"), "qué tal")


if __name__ == "__main__":
    unittest.main()
