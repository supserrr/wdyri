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


def save(model: str, variant: str, split: str, seed: int, frame: pd.DataFrame, prob,
         merge: bool = False) -> None:
    """Write one run's predictions to its file.

    `merge` adds sets to the run's existing file (replacing any message scored
    again) instead of overwriting it, so a run always has exactly one file.
    """
    out = frame[["set", "id", "label"]].copy()
    out["prob"] = prob
    out = out.assign(model=model, variant=variant, split=split, seed=seed)[COLUMNS]
    path = config.PREDICTIONS / f"{run_name(model, variant, split, seed)}.csv"
    if merge and path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        old = old[~old.set_index(["set", "id"]).index.isin(out.set_index(["set", "id"]).index)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(path, index=False, float_format="%.6f")


def load_all() -> pd.DataFrame:
    """Every prediction file; a message scored twice for the same run keeps its latest score."""
    files = sorted(config.PREDICTIONS.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    preds = pd.concat((pd.read_csv(f, keep_default_na=False) for f in files), ignore_index=True)
    return preds.drop_duplicates(["model", "variant", "split", "seed", "set", "id"], keep="last")
