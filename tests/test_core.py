"""Unit tests for the parts every result depends on.

Run: python -m unittest discover tests
"""
import random
import unittest

import numpy as np

from src.metrics import recall_threshold
from src.perturb import CYRILLIC, ZWSP, attack, codeswitch, structural
from src.preprocess import mask, normalise, preprocess
from src.split import template_ids, template_split


class MaskingTest(unittest.TestCase):
    def test_phone_numbers(self):
        self.assertEqual(mask("tuma kwa 0712345678 jina JUMA"), "tuma kwa <PHONE> jina JUMA")
        self.assertEqual(mask("piga +255 712 345 678 sasa"), "piga <PHONE> sasa")

    def test_amounts_and_links(self):
        self.assertEqual(mask("umepata Sh1,500,000 tembelea wa.me/255712345678"),
                         "umepata <AMOUNT> tembelea <URL>")
        self.assertEqual(mask("salary 4,000,000TZS"), "salary <AMOUNT>")
        self.assertEqual(mask("MK75000.00 zokwana"), "<AMOUNT> zokwana")

    def test_short_numbers_kept(self):
        # "666" is a Freemason cue and *150# a USSD code: neither is personal data.
        self.assertEqual(mask("(666) piga *150#"), "(666) piga *150#")


class DefenceTest(unittest.TestCase):
    def test_lookalikes_and_zero_width(self):
        disguised = "n" + CYRILLIC["a"] + ZWSP + "mb" + CYRILLIC["a"]
        self.assertEqual(preprocess(disguised, defend=True), "namba")

    def test_leet_and_spacing(self):
        self.assertEqual(normalise("p3s4 t u m a"), "pesa tuma")
        self.assertEqual(normalise("M - P E S A <PHONE>"), "MPESA <PHONE>")

    def test_clean_text_unchanged(self):
        text = "Nitapika wali na mboga leo jioni, unanunua nyama?"
        self.assertEqual(preprocess(text, defend=True), preprocess(text))


class TemplateTest(unittest.TestCase):
    def test_same_script_grouped_transitively(self):
        base = "Mimi mwenye nyumba wako hii namba yangu ya {}. Mbona kimya na siku zinazidi kwenda...?"
        texts = ["Habari za asubuhi. " + base.format("Vodacom"),
                 "Habari za asubuhi. " + base.format("Tigo"),   # differs from 1 by the network
                 "Habari za mchana. " + base.format("Tigo"),    # differs from 2 by the greeting
                 "Nitapika wali na mboga leo jioni."]
        ids = template_ids(texts)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])
        self.assertNotEqual(ids[0], ids[3])

    def test_template_split_is_disjoint(self):
        groups = np.repeat(np.arange(60), 5)
        labels = (groups % 2).astype(int)
        parts = template_split(labels, groups, seed=0)
        for g in np.unique(groups):
            self.assertEqual(len(set(parts[groups == g])), 1)
        self.assertEqual(set(parts), {"train", "val", "test"})


class AttackTest(unittest.TestCase):
    @staticmethod
    def scorer(texts):
        # Toy model: scam probability rises with the words "tuma" and "pesa".
        return np.array([0.2 + 0.4 * ("tuma" in t) + 0.3 * ("pesa" in t) for t in texts])

    def test_deterministic(self):
        text = "tuma pesa kwa <PHONE> haraka"
        self.assertEqual(attack(text, "lookalike", "all", self.scorer, seed=1),
                         attack(text, "lookalike", "all", self.scorer, seed=1))

    def test_intensity_targets_strongest_word(self):
        out = attack("tuma pesa kwa <PHONE>", "lookalike", "1", self.scorer)
        self.assertNotIn("tuma", out)
        self.assertIn("pesa", out)
        self.assertIn("<PHONE>", out)

    def test_structural_keeps_letters(self):
        for seed in range(5):
            out = structural("tuma pesa haraka", [0, 1], random.Random(seed))
            self.assertNotEqual(out, "tuma pesa haraka")
            self.assertEqual(out.replace(" ", "").replace("-", ""), "tumapesaharaka")

    def test_codeswitch(self):
        self.assertEqual(codeswitch("Tuma PESA kwenye namba hii", [], [], "all"), "Send MONEY to this number")


class ThresholdTest(unittest.TestCase):
    def test_never_above_half(self):
        self.assertEqual(recall_threshold(np.array([1, 1, 0]), np.array([0.99, 0.98, 0.01])), 0.5)

    def test_lowered_to_reach_recall(self):
        y = np.ones(20, dtype=int)
        p = np.linspace(0.05, 0.95, 20)
        thr = recall_threshold(y, p)
        self.assertGreaterEqual((p >= thr).mean(), 0.95)


if __name__ == "__main__":
    unittest.main()
