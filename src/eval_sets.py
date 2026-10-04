"""Build every evaluation set once, so all models are scored on identical text.

data/processed/eval_<split>.csv holds one row per (set, message):
    val, test                         clean validation and test messages
    test_<attack>_<intensity>         test set with its scams obfuscated (RQ2)
    chichewa                          all 733 masked Chichewa messages (RQ3, template split only)
and a "+norm" copy of each set passed through the normalisation defence (E7).
"""
from __future__ import annotations

import os

import pandas as pd

from . import config
from .data import load_bongo, load_chichewa, split_frames
from .perturb import ATTACKS, INTENSITIES, Scorer, attack, word_effects
from .preprocess import preprocess


def path(split: str):
    return config.DATA_PROCESSED / f"eval_{split}.csv"


def build(split: str, scorer: Scorer | None = None) -> pd.DataFrame:
    _, val, test = split_frames(load_bongo(), split)
    frames = [val.assign(set="val"), test.assign(set="test")]

    if split == "template":
        assert scorer is not None, "attacks need rung 2 to rank trigger words"
        scams = test[test["label"] == 1]
        effects = {row.id: word_effects(row.text, scorer) for row in scams.itertuples()}
        for kind in ATTACKS:
            for k in INTENSITIES:
                attacked = test.copy()
                is_scam = attacked["label"] == 1
                attacked.loc[is_scam, "text"] = [
                    attack(row.text, kind, k, scorer, seed=config.SPLIT_SEED, effects=effects[row.id])
                    for row in attacked[is_scam].itertuples()]
                frames.append(attacked.assign(set=f"test_{kind}_{k}"))
        frames.append(load_chichewa().assign(set="chichewa"))

    clean = pd.concat(frames, ignore_index=True)[["set", "id", "label", "text"]]
    defended = clean.assign(set=clean["set"] + "+norm",
                            text=clean["text"].map(lambda t: preprocess(t, defend=True)))
    out = pd.concat([clean, defended], ignore_index=True)
    # Write then rename, so a training run reading the file never sees half of it.
    tmp = path(split).with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path(split))
    return out


def load(split: str) -> pd.DataFrame:
    return pd.read_csv(path(split), keep_default_na=False)
