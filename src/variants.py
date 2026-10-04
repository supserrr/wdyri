"""Training data and evaluation rows for each experiment variant, shared by all rungs.

    clean      template- or random-split training data          (E2-E5, E6, E8)
    advtrain   + the saved perturbed copies of 25% of scams     (E7)
    fewshot20  + 20 Chichewa messages, their templates removed  (E9)
    fewshot50  + 50 Chichewa messages, their templates removed  (E9)
    strip      placeholders deleted everywhere, so a model cannot use
               "this text contains a phone number" as a shortcut      (E3b)
"""
from __future__ import annotations

import re

import pandas as pd

from . import config, eval_sets
from .data import fewshot_ids, load_bongo, load_chichewa, split_frames

ADV_COPIES = config.DATA_PROCESSED / "adversarial_copies_template.csv"
_PLACEHOLDER_RE = re.compile(r"\s*<(?:PHONE|AMOUNT|URL)>\s*")


def strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text).strip()


def frames(split: str, variant: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, rows to predict) for one run."""
    train, val, _ = split_frames(load_bongo(), split)
    train = train[["text", "label"]]
    ev = eval_sets.load(split)

    if variant == "advtrain":
        train = pd.concat([train, pd.read_csv(ADV_COPIES)], ignore_index=True)
    elif variant.startswith("fewshot"):
        chi = load_chichewa()
        shot = chi[chi["id"].isin(fewshot_ids(chi, int(variant.removeprefix("fewshot")), seed))]
        held = set(chi[chi["template_id"].isin(shot["template_id"])]["id"])
        train = pd.concat([train, shot[["text", "label"]]], ignore_index=True)
        keep = ev["set"].isin(["val", "chichewa", "chichewa+norm"]) & ~ev["id"].isin(held)
        ev = ev[keep]
    elif variant == "strip":
        train, val, ev = (f.assign(text=f["text"].map(strip_placeholders)) for f in (train, val, ev))
    elif variant != "clean":
        raise ValueError(variant)
    return train.reset_index(drop=True), val.reset_index(drop=True), ev.reset_index(drop=True)
