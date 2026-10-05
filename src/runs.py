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
         tag: str | None = None) -> None:
    """Write one run's predictions. A `tag` adds extra sets to a run in a separate file."""
    out = frame[["set", "id", "label"]].copy()
    out["prob"] = prob
    out = out.assign(model=model, variant=variant, split=split, seed=seed)[COLUMNS]
    suffix = f"__{tag}" if tag else ""
    out.to_csv(config.PREDICTIONS / f"{run_name(model, variant, split, seed)}{suffix}.csv",
               index=False, float_format="%.6f")


def load_all() -> pd.DataFrame:
    """Every prediction file; a message scored twice for the same run keeps its latest score."""
    files = sorted(config.PREDICTIONS.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    preds = pd.concat((pd.read_csv(f, keep_default_na=False) for f in files), ignore_index=True)
    return preds.drop_duplicates(["model", "variant", "split", "seed", "set", "id"], keep="last")
