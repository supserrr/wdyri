"""Training data and evaluation rows for each experiment variant, shared by all rungs.

    clean      template- or random-split training data          (E2-E5, E6, E8)
    advtrain   + the saved perturbed copies of 25% of scams     (E7)
    fewshot20  + 20 Chichewa messages, their templates removed  (E9)
    fewshot50  + 50 Chichewa messages, their templates removed  (E9)
    strip      placeholders deleted everywhere, so a model cannot use
               "this text contains a phone number" as a shortcut      (E3b)
    counterfactual  training texts edited so a number is equally common in
               both classes (number-balanced; see number_balanced)   (E11)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, eval_sets
from .data import fewshot_ids, load_bongo, load_chichewa, split_frames
from .preprocess import PHONE, PLACEHOLDERS, strip_placeholders

ADV_COPIES = config.DATA_PROCESSED / "adversarial_copies_template.csv"


def number_balanced(train: pd.DataFrame, seed: int = config.SPLIT_SEED) -> tuple[pd.DataFrame, dict]:
    """Counterfactual edits that make "contains a number" useless as a label cue (E11).

    In BongoScam 85% of scams and 0% of genuine texts contain a placeholder. We
    delete the placeholders from half of the scams that have one, and append
    " <PHONE>" to 40% of genuine texts, so a number appears in 44% of
    training scams and 38% of genuine training texts. Words are never touched, labels never
    change, and the training set keeps its size: the model must learn the
    language of a scam instead of the presence of a number. Validation and
    test stay unedited.
    """
    rng = np.random.default_rng(seed)
    out = train.copy()
    has = out["text"].map(lambda t: any(p in t for p in PLACEHOLDERS))
    drop = (out["label"] == 1) & has & (rng.random(len(out)) < 0.5)
    add = (out["label"] == 0) & (rng.random(len(out)) < 0.4)
    out.loc[drop, "text"] = out.loc[drop, "text"].map(strip_placeholders)
    out.loc[add, "text"] = out.loc[add, "text"] + f" {PHONE}"
    after = out["text"].map(lambda t: any(p in t for p in PLACEHOLDERS))
    info = {"edited_rows": int(drop.sum() + add.sum()), "edited_share": float((drop | add).mean()),
            "number_share_scam": float(after[out.label == 1].mean()),
            "number_share_genuine": float(after[out.label == 0].mean())}
    return out, info


def frames(split: str, variant: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, rows to predict) for one run."""
    train, val, _ = split_frames(load_bongo(), split)
    # template_id travels with every training row so classical tuning can use
    # template-disjoint cross-validation (neural models ignore it).
    train = train[["text", "label", "template_id"]].astype({"template_id": str})
    ev = eval_sets.load(split)

    if variant == "advtrain":
        # Each perturbed copy keeps its source message's template, so a copy and
        # its source never sit on opposite sides of a cross-validation fold.
        train = pd.concat([train, pd.read_csv(ADV_COPIES).astype({"template_id": str})], ignore_index=True)
    elif variant.startswith("fewshot"):
        chi = load_chichewa()
        shot = chi[chi["id"].isin(fewshot_ids(chi, int(variant.removeprefix("fewshot")), seed))]
        held = set(chi[chi["template_id"].isin(shot["template_id"])]["id"])
        shot = shot.assign(template_id="chi_" + shot["template_id"].astype(str))
        train = pd.concat([train, shot[["text", "label", "template_id"]]], ignore_index=True)
        keep = ev["set"].isin(["val", "chichewa", "chichewa+norm"]) & ~ev["id"].isin(held)
        ev = ev[keep]
    elif variant == "counterfactual":
        train, _ = number_balanced(train)
    elif variant == "strip":
        train, val, ev = (f.assign(text=f["text"].map(strip_placeholders)) for f in (train, val, ev))
    elif variant != "clean":
        raise ValueError(variant)
    return train.reset_index(drop=True), val.reset_index(drop=True), ev.reset_index(drop=True)
