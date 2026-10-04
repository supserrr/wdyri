"""Tokeniser for the BiLSTM and a cache of fastText Swahili vectors.

fastText builds a word's vector from its character n-grams, so it returns a
vector for any string, including misspellings and attacked words. Loading the
full model needs about 4 GB of RAM, so we look up every token the BiLSTM will
ever see once and keep only those vectors (.cache/fasttext_tokens.npz).

Usage:
    python -m src.embeddings     # needs .cache/cc.sw.300.bin (python -m src.download --fasttext)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .data import load_bongo, load_chichewa

TOKEN_RE = re.compile(r"<\w+>|\w+|[^\w\s]")
CACHE_FILE = config.CACHE / "fasttext_tokens.npz"
DIM = 300


def tokenize(text: str) -> list[str]:
    """Lower-cased words, punctuation marks and placeholders as separate tokens."""
    return TOKEN_RE.findall(text.lower())


def all_texts() -> list[str]:
    texts = load_bongo()["text"].tolist() + load_chichewa()["text"].tolist()
    for split in ("template", "random"):
        path = config.DATA_PROCESSED / f"eval_{split}.csv"
        if path.exists():
            texts += pd.read_csv(path, keep_default_na=False)["text"].tolist()
    adv = config.DATA_PROCESSED / "adversarial_copies_template.csv"
    if adv.exists():
        texts += pd.read_csv(adv)["text"].tolist()
    return texts


def build_cache() -> None:
    from gensim.models.fasttext import load_facebook_vectors

    tokens = sorted({tok for text in all_texts() for tok in tokenize(text)})
    print(f"{len(tokens)} distinct tokens; loading fastText (takes a few minutes)...")
    ft = load_facebook_vectors(str(config.CACHE / "cc.sw.300.bin"))
    in_vocab = np.array([tok in ft.key_to_index for tok in tokens])
    vectors = np.stack([ft[tok] for tok in tokens]).astype(np.float32)
    np.savez_compressed(CACHE_FILE, tokens=np.array(tokens), vectors=vectors, in_vocab=in_vocab)
    print(f"saved {CACHE_FILE.name}: {in_vocab.mean():.1%} of tokens are whole words in fastText's vocabulary")


def load_cache() -> dict[str, np.ndarray]:
    data = np.load(CACHE_FILE)
    return dict(zip(data["tokens"].tolist(), data["vectors"]))


if __name__ == "__main__":
    build_cache()
