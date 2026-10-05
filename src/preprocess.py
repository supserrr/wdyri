"""Text cleaning shared by training, evaluation and the web app.

Every model and the app call `preprocess()`, so the text a model is trained on
and the text it sees in production go through exactly the same steps.

Steps:
1. Unicode NFC and whitespace clean-up.
2. Mask links, phone numbers and money amounts with placeholder tokens, so a
   model cannot memorise one scammer's number and no personal number is kept.
3. Optionally (defence for RQ2) undo common obfuscation tricks: lookalike
   letters, zero-width characters, digits used as letters and spaced-out words.

Punctuation, casing and stop words are kept on purpose (see DECISIONS.md).
"""
from __future__ import annotations

import re
import unicodedata

PHONE, AMOUNT, URL = "<PHONE>", "<AMOUNT>", "<URL>"
PLACEHOLDERS = (PHONE, AMOUNT, URL)

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b(?:wa\.me|bit\.ly|t\.me|tinyurl\.com|chat\.whatsapp\.com)/\S*",
    re.IGNORECASE,
)

# Nine or more digits, optionally with a leading + and single spaces, dots or
# hyphens between digit groups: Tanzanian 07xx/06xx, Malawian 08xx/09xx and
# international 255/265 numbers all match.
_PHONE_RE = re.compile(r"(?<!\d)\+?\d(?:[ .\-]?\d){8,13}(?!\d)")

_CURRENCY = r"(?:tshs?|tzs|shs?|ksh|mwk|mk|usd|k)"
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_AMOUNT_RE = re.compile(
    # currency first: "Tsh 50,000", "MK75000.00", "Sh1,500,000", "MK-2199.20"
    rf"\b{_CURRENCY}\.?\s?-?\s?(?:{_NUMBER})(?:\s?/=)?"
    # currency last: "1,500,000TZS", "50000/=", "20,000 kwacha"
    rf"|\b(?:{_NUMBER})\s?(?:{_CURRENCY}|/=|shilingi|shillings?|kwacha)\b"
    rf"|(?<![\d.])\d+\s?/="
    # thousands separators or five or more digits on their own: "4,000,000", "50000"
    rf"|\b\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?\b|\b\d{{5,8}}\b",
    re.IGNORECASE,
)

_SPACES_RE = re.compile(r"\s+")


def mask(text: str) -> str:
    """Replace links, phone numbers and money amounts with placeholders."""
    text = _URL_RE.sub(f" {URL} ", text)
    text = _PHONE_RE.sub(f" {PHONE} ", text)
    text = _AMOUNT_RE.sub(f" {AMOUNT} ", text)
    return _SPACES_RE.sub(" ", text).strip()


# --- Obfuscation defence (experiment E7) -------------------------------------

# Cyrillic and Greek letters that look like Latin ones.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ӏ": "l", "һ": "h", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ү": "Y",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
})
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­]")
_LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"}
# A digit or symbol from _LEET touching a letter, e.g. "p3s4" or "t0ma".
_LEET_RE = re.compile(r"(?<=[A-Za-z])[431057@$]|[431057@$](?=[A-Za-z])")
# Three or more single letters split by spaces, dots or hyphens: "t u m a", "M - P E S A".
_SPACED_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-]+){2,}[A-Za-z](?![A-Za-z])")
_PLACEHOLDER_RE = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS))


def _undo_leet(token: str) -> str:
    if token in PLACEHOLDERS or not re.search(r"[A-Za-z]", token):
        return token
    return _LEET_RE.sub(lambda m: _LEET[m.group(0)], token)


def normalise(text: str) -> str:
    """Undo lookalike letters, zero-width characters, leetspeak and spaced words.

    Expects text that has already been masked: placeholders are left alone.
    """
    parts = _PLACEHOLDER_RE.split(text)
    holders = _PLACEHOLDER_RE.findall(text)
    cleaned = []
    for part in parts:
        part = " ".join(_undo_leet(tok) for tok in part.split(" "))
        part = _SPACED_RE.sub(lambda m: re.sub(r"[ .\-]+", "", m.group(0)), part)
        cleaned.append(part)
    out = cleaned[0]
    for holder, part in zip(holders, cleaned[1:]):
        out += holder + part
    return _SPACES_RE.sub(" ", out).strip()


def preprocess(text: str, *, do_mask: bool = True, defend: bool = False) -> str:
    """The one cleaning function used in training, evaluation and the app."""
    text = unicodedata.normalize("NFC", str(text))
    if defend:
        # Character-level fixes go first so a phone number hidden with
        # lookalike digits or zero-width spaces is still masked.
        text = _ZERO_WIDTH_RE.sub("", text).translate(_CONFUSABLES)
    text = _SPACES_RE.sub(" ", text).strip()
    if do_mask:
        text = mask(text)
    if defend:
        text = normalise(text)
    return text


_PLACEHOLDER_TOKEN_RE = re.compile(r"\s*<(?:PHONE|AMOUNT|URL)>\s*")


def strip_placeholders(text: str) -> str:
    """Delete <PHONE>, <AMOUNT> and <URL>: the text without any trace of a number or link."""
    return _SPACES_RE.sub(" ", _PLACEHOLDER_TOKEN_RE.sub(" ", text)).strip()
