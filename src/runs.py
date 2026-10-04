"""Saving and loading per-message predictions.

Every training script writes one CSV per run to results/predictions/ with the
scam probability of every message in every evaluation set. `src.evaluate`
turns these files into results/experiments.csv, so metrics are computed in one
place for all models.
"""
from __future__ import annotations

import pandas as pd

from . import config

COLUMNS = ["model", "variant", "split", "seed", "set", "id", "label", "prob"]


def run_name(model: str, variant: str, split: str, seed: int) -> str:
    return f"{model}__{variant}__{split}__s{seed}"


def save(model: str, variant: str, split: str, seed: int, frame: pd.DataFrame, prob) -> None:
    out = frame[["set", "id", "label"]].copy()
    out["prob"] = prob
    out = out.assign(model=model, variant=variant, split=split, seed=seed)[COLUMNS]
    out.to_csv(config.PREDICTIONS / f"{run_name(model, variant, split, seed)}.csv",
               index=False, float_format="%.6f")


def load_all() -> pd.DataFrame:
    files = sorted(config.PREDICTIONS.glob("*.csv"))
    return pd.concat((pd.read_csv(f, keep_default_na=False) for f in files), ignore_index=True)
