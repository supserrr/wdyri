"""Rule-based obfuscation attacks for RQ2 (experiments E6 and E7).

Attacks are plain code, not an LLM, so anyone can rerun them and get the same
text. Each attack edits the "trigger words" of a scam message: the words that
push rung 2 (character logistic regression) hardest towards "scam", measured
by leave-one-word-out occlusion. Every model is attacked on the same words.

Intensity: 1, 3 or all trigger words (all = every word with a positive effect).
Only scam messages are attacked; genuine messages stay as they are, since an
honest sender has no reason to disguise a text.
"""
from __future__ import annotations

import bisect
import random
import re
from typing import Callable

import numpy as np

from .preprocess import PLACEHOLDERS

ATTACKS = ("lookalike", "structural", "codeswitch")
INTENSITIES = ("1", "3", "all")

# Latin letter -> Cyrillic twin that renders the same.
CYRILLIC = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "X": "Х", "I": "І", "S": "Ѕ", "J": "Ј",
}
DIGITS = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "A": "4", "E": "3", "I": "1", "O": "0", "S": "5"}
ZWSP = "​"

# Held-out lookalikes for the E7 generalisation check: letters from other scripts
# (Latin IPA, Armenian, Cherokee, Coptic, other Cyrillic and Greek forms) that also
# pass for Latin ones, and a different invisible character. The normalisation
# defence was written knowing CYRILLIC, DIGITS and ZWSP; it knows none of these,
# so this attack measures whether the defence generalises. It is not one of the
# ATTACKS, so the adversarial training copies never contain it.
UNSEEN = {
    "a": "ɑ", "c": "ϲ", "e": "ҽ", "g": "ɡ", "h": "հ", "i": "ı", "l": "ǀ",
    "n": "ո", "o": "օ", "p": "ⲣ", "s": "ꜱ", "u": "ս", "y": "ү",
    "A": "Ꭺ", "B": "Ᏼ", "C": "Ꮯ", "E": "Ꭼ", "H": "Ꮋ", "K": "Ꮶ", "M": "Ꮇ",
    "O": "Օ", "P": "Ꮲ", "S": "Ꮪ", "T": "Ꭲ",
}
INVISIBLE_UNSEEN = "⁣"  # INVISIBLE SEPARATOR

# 30 Swahili scam phrases and an English rendering, taken from the words that
# separate scams from genuine texts in the BongoScam training data. Longer
# phrases come first so "kwenye namba hii" wins over "namba".
LEXICON: dict[str, str] = {
    "kwenye namba hii": "to this number",
    "namba hii": "this number",
    "mwenye nyumba": "landlord",
    "nitumie": "send me",
    "unitumie": "send me",
    "utanitumia": "you will send me",
    "itume": "send it",
    "tuma": "send",
    "namba": "number",
    "jina": "name",
    "pesa": "money",
    "hela": "money",
    "ela": "money",
    "iyo": "that",
    "hiyo": "that",
    "piga": "call",
    "simu": "phone",
    "zawadi": "prize",
    "hongera": "congratulations",
    "umepata": "you have won",
    "jiunge": "join",
    "kujiunga": "to join",
    "utajiri": "wealth",
    "mtaji": "capital",
    "biashara": "business",
    "wakala": "agent",
    "laini": "line",
    "nakuomba": "I beg you",
    "naomba": "please",
    "kodi": "rent",
}
_LEX_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in LEXICON) + r")(?!\w)", re.IGNORECASE)

Scorer = Callable[[list[str]], np.ndarray]


def _is_word(token: str) -> bool:
    return token not in PLACEHOLDERS and sum(ch.isalpha() for ch in token) >= 2


def word_effects(text: str, scorer: Scorer) -> list[tuple[int, float]]:
    """(token index, drop in scam probability when the token is removed), largest first."""
    tokens = text.split(" ")
    idx = [i for i, t in enumerate(tokens) if _is_word(t)]
    if not idx:
        return []
    variants = [text] + [" ".join(tokens[:i] + tokens[i + 1:]) for i in idx]
    probs = scorer(variants)
    effects = [(i, float(probs[0] - p)) for i, p in zip(idx, probs[1:])]
    return sorted(effects, key=lambda e: e[1], reverse=True)


def trigger_indices(effects: list[tuple[int, float]], intensity: str) -> list[int]:
    positive = [i for i, eff in effects if eff > 0]
    return positive if intensity == "all" else positive[: int(intensity)]


# --- the three attacks --------------------------------------------------------

def _lookalike_word(word: str, rng: random.Random) -> str:
    chars = list(word)
    swappable = [i for i, ch in enumerate(chars) if ch in CYRILLIC or ch in DIGITS]
    for i in swappable:
        if rng.random() < 0.6:
            ch = chars[i]
            use_digit = ch in DIGITS and (ch not in CYRILLIC or rng.random() < 0.3)
            chars[i] = DIGITS[ch] if use_digit else CYRILLIC[ch]
    if swappable and all(c == o for c, o in zip(chars, word)):
        i = swappable[0]
        chars[i] = CYRILLIC.get(word[i], DIGITS.get(word[i]))
    # A zero-width space in the middle splits the word for tokenisers but not for readers.
    mid = max(1, len(chars) // 2)
    return "".join(chars[:mid]) + ZWSP + "".join(chars[mid:])


def lookalike(text: str, targets: list[int], rng: random.Random) -> str:
    tokens = text.split(" ")
    for i in targets:
        tokens[i] = _lookalike_word(tokens[i], rng)
    return " ".join(tokens)


def _unseen_word(word: str, rng: random.Random) -> str:
    chars = list(word)
    swappable = [i for i, ch in enumerate(chars) if ch in UNSEEN]
    for i in swappable:
        if rng.random() < 0.6:
            chars[i] = UNSEEN[chars[i]]
    if swappable and chars == list(word):
        chars[swappable[0]] = UNSEEN[word[swappable[0]]]
    mid = max(1, len(chars) // 2)
    return "".join(chars[:mid]) + INVISIBLE_UNSEEN + "".join(chars[mid:])


def unseen_lookalike(text: str, targets: list[int], rng: random.Random) -> str:
    """The lookalike attack with held-out characters (E7 generalisation check)."""
    tokens = text.split(" ")
    for i in targets:
        tokens[i] = _unseen_word(tokens[i], rng)
    return " ".join(tokens)


def structural(text: str, targets: list[int], rng: random.Random) -> str:
    """Split a word into letters ("t u m a", "M-P-E-S-A") or glue it to its neighbour."""
    tokens = text.split(" ")
    glue_after = set()
    for i in targets:
        style = rng.choice(("spaced", "spaced", "dashed", "joined"))
        if style == "joined" and i + 1 < len(tokens):
            glue_after.add(i)
        elif style == "dashed":
            tokens[i] = "-".join(tokens[i])
        else:
            tokens[i] = " ".join(tokens[i])
    out = ""
    for i, tok in enumerate(tokens):
        out += tok
        if i < len(tokens) - 1 and i not in glue_after:
            out += " "
    return out


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def codeswitch(text: str, targets: list[int], effects: list[tuple[int, float]], intensity: str) -> str:
    """Swap Swahili trigger phrases for English ones, strongest phrases first.

    A phrase's strength is the occlusion effect of its strongest word; `targets`
    is unused here because the lexicon, not the model, decides what can be swapped.
    """
    del targets
    effect_by_token = dict(effects)
    # Map character offsets to token indices to rank lexicon matches.
    starts, pos = [], 0
    for tok in text.split(" "):
        starts.append(pos)
        pos += len(tok) + 1
    matches = []
    for m in _LEX_RE.finditer(text):
        first = bisect.bisect_right(starts, m.start()) - 1
        last = bisect.bisect_right(starts, m.end() - 1) - 1
        strength = max((effect_by_token.get(i, 0.0) for i in range(first, last + 1)), default=0.0)
        matches.append((strength, m))
    matches.sort(key=lambda x: x[0], reverse=True)
    chosen = matches if intensity == "all" else matches[: int(intensity)]
    for _, m in sorted(chosen, key=lambda x: x[1].start(), reverse=True):
        repl = _match_case(m.group(0), LEXICON[m.group(0).lower()])
        text = text[: m.start()] + repl + text[m.end():]
    return text


def adversarial_copies(texts: list[str], scorer: Scorer, share: float = 0.25, seed: int = 0,
                       return_sources: bool = False):
    """Perturbed copies of a random `share` of training scams (defence 2 in E7).

    Each copy gets a random attack and intensity. The copies are machine-made,
    so they stay a small part of the training data and are reported as such.
    """
    rng = random.Random(seed)
    picked = rng.sample(range(len(texts)), k=round(share * len(texts)))
    copies = []
    for i in picked:
        kind, k = rng.choice(ATTACKS), rng.choice(INTENSITIES)
        copies.append(attack(texts[i], kind, k, scorer, seed=seed))
    return (copies, picked) if return_sources else copies


def attack(text: str, kind: str, intensity: str, scorer: Scorer, seed: int = 0,
           effects: list[tuple[int, float]] | None = None) -> str:
    """Apply one attack at one intensity. Deterministic for a given seed."""
    rng = random.Random(f"{seed}-{kind}-{intensity}-{text}")
    effects = word_effects(text, scorer) if effects is None else effects
    targets = trigger_indices(effects, intensity)
    if kind == "lookalike":
        return lookalike(text, targets, rng)
    if kind == "structural":
        return structural(text, targets, rng)
    if kind == "codeswitch":
        return codeswitch(text, targets, effects, intensity)
    if kind == "unseen":
        return unseen_lookalike(text, targets, rng)
    raise ValueError(kind)
