"""Build every evaluation set once, so all models are scored on identical text.

data/processed/eval_<split>.csv holds one row per (set, message):
    val, test                         clean validation and test messages
    test_<attack>_<intensity>         test set with its scams obfuscated (RQ2)
    chichewa                          all 733 masked Chichewa messages (RQ3, template split only)
and a "+norm" copy of each set passed through the normalisation defence (E7).

Template split only, the shortcut stress tests (E10): minimal pairs that change
one placeholder and nothing else, so any change in a prediction is caused by
the number alone.
    stress_genuine+phone              genuine test texts with " <PHONE>" appended (should stay genuine)
    stress_genuine+amount             genuine test texts with " <AMOUNT>" appended (should stay genuine)
    stress_scam-number                test scams that had a placeholder, with it deleted (should stay scam)
    valstress_*                       the same three sets built from validation messages, used only
                                      to choose which system the app deploys (never the test sets)

Controls for E6 (template split only): the attacked test sets disguise scams only, so an
odd-looking text is always a scam there. The control sets disguise the *genuine* test
texts the same way, so a false alarm on them shows a model flags disguise as such.
    ctrl_<attack>                     genuine texts, their 3 most influential words disguised
    ctrlall_<attack>                  genuine texts, every word disguised (matches the "all"
                                      intensity of the scam attacks)
A second attacker checks that results do not hinge on the char LR that picks trigger
words (white-box for a char LR, transfer for everything else):
    test_xatk_<attack>_all            test scams attacked on the trigger words of a word-count
                                      Naive Bayes attacker instead
"""
from __future__ import annotations

import os

import pandas as pd

from . import config
from .data import load_bongo, load_chichewa, split_frames
from .perturb import ATTACKS, INTENSITIES, Scorer, attack, word_effects
from .preprocess import AMOUNT, PHONE, PLACEHOLDERS, preprocess, strip_placeholders


def path(split: str):
    return config.DATA_PROCESSED / f"eval_{split}.csv"


def build(split: str, scorer: Scorer | None = None, transfer_scorer: Scorer | None = None) -> pd.DataFrame:
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
        frames += control_sets(test, scorer)

    clean = pd.concat(frames, ignore_index=True)[["set", "id", "label", "text"]]
    defended = clean.assign(set=clean["set"] + "+norm",
                            text=clean["text"].map(lambda t: preprocess(t, defend=True)))
    out = pd.concat([clean, defended], ignore_index=True)
    if split == "template":
        out = pd.concat([out, stress_sets(test), stress_sets(val, prefix="valstress"),
                         *control_sets_all(test, scorer)], ignore_index=True)
        if transfer_scorer is not None:
            out = pd.concat([out, *transfer_attack_sets(test, transfer_scorer)], ignore_index=True)
    # Write then rename, so a training run reading the file never sees half of it.
    tmp = path(split).with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path(split))
    return out


def control_sets(test: pd.DataFrame, scorer: Scorer) -> list[pd.DataFrame]:
    """Genuine test texts disguised with each attack (their 3 most influential words)."""
    genuine = test[test["label"] == 0]
    # Rank words by how much they move rung 2 either way: genuine texts have few scam-ward words.
    effects = {row.id: sorted(word_effects(row.text, scorer), key=lambda e: abs(e[1]), reverse=True)
               for row in genuine.itertuples()}
    frames = []
    for kind in ATTACKS:
        k = "all" if kind == "codeswitch" else "3"
        texts = [attack(row.text, kind, k, scorer, seed=config.SPLIT_SEED,
                        effects=[(i, abs(e) + 1e-9) for i, e in effects[row.id]])
                 for row in genuine.itertuples()]
        frames.append(genuine.assign(set=f"ctrl_{kind}", text=texts))
    return frames


def control_sets_all(test: pd.DataFrame, scorer: Scorer) -> list[pd.DataFrame]:
    """Genuine test texts with every word disguised (every lexicon phrase for code-switching)."""
    genuine = test[test["label"] == 0]
    frames = []
    for kind in ATTACKS:
        texts = []
        for row in genuine.itertuples():
            tokens = row.text.split(" ")
            # every word of two or more letters is a "trigger" here
            effects = [(i, 1.0) for i, t in enumerate(tokens) if sum(c.isalpha() for c in t) >= 2
                       and t not in ("<PHONE>", "<AMOUNT>", "<URL>")]
            texts.append(attack(row.text, kind, "all", scorer, seed=config.SPLIT_SEED, effects=effects))
        frames.append(genuine.assign(set=f"ctrlall_{kind}", text=texts)[["set", "id", "label", "text"]])
    return frames


def transfer_attack_sets(test: pd.DataFrame, scorer: Scorer) -> list[pd.DataFrame]:
    """Full-intensity attacks whose trigger words come from a different attacker model."""
    frames = []
    scams = test[test["label"] == 1]
    effects = {row.id: word_effects(row.text, scorer) for row in scams.itertuples()}
    for kind in ATTACKS:
        attacked = test.copy()
        is_scam = attacked["label"] == 1
        attacked.loc[is_scam, "text"] = [
            attack(row.text, kind, "all", scorer, seed=config.SPLIT_SEED, effects=effects[row.id])
            for row in attacked[is_scam].itertuples()]
        frames.append(attacked.assign(set=f"test_xatk_{kind}_all")[["set", "id", "label", "text"]])
    return frames


def stress_sets(base: pd.DataFrame, prefix: str = "stress") -> pd.DataFrame:
    genuine = base[base["label"] == 0]
    scams = base[(base["label"] == 1) & base["text"].map(lambda t: any(p in t for p in PLACEHOLDERS))]
    frames = [genuine.assign(set=f"{prefix}_genuine+phone", text=genuine["text"] + f" {PHONE}"),
              genuine.assign(set=f"{prefix}_genuine+amount", text=genuine["text"] + f" {AMOUNT}"),
              scams.assign(set=f"{prefix}_scam-number", text=scams["text"].map(strip_placeholders))]
    return pd.concat(frames, ignore_index=True)[["set", "id", "label", "text"]]


def load(split: str) -> pd.DataFrame:
    return pd.read_csv(path(split), keep_default_na=False)
