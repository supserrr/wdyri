"""E12: a two-model ensemble built from complementary errors.

The character n-gram LR and AfroXLMR fail on different messages: the LR
catches the landlord script that has no number, AfroXLMR catches disguised
scams that break the LR's n-grams. The ensemble flags a message when either
model flags it at its own validation-chosen threshold. The "OR" rule follows
the recall-first cost asymmetry fixed at the start (a missed scam costs money,
a false alarm costs trust); averaging probabilities is reported too and loses,
because the transformer's probabilities sit at 0 or 1.

The idea is post hoc: the complementary errors were first seen on the test
set. It is therefore re-checked on two fresh template-disjoint splits
(template_r1, template_r2) whose test sets played no part in designing it.

No training: the ensemble is computed from the saved per-message predictions.

Usage:
    python -m src.ensemble
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, runs
from .metrics import recall_threshold

# (name, [(model, variant), ...]); the LR is deterministic (seed 0) and is
# paired with every seed of the transformer.
ENSEMBLES = [
    ("ensemble", [("lr_char", "clean"), ("afroxlmr", "clean")]),
    ("ensemble_cf", [("lr_char", "clean"), ("afroxlmr", "counterfactual")]),
    ("ensemble_cf2", [("lr_char", "counterfactual"), ("afroxlmr", "counterfactual")]),
]


SPLITS = ("template", "template_r1", "template_r2")


def load_run(model: str, variant: str, seed: int, split: str = "template") -> pd.DataFrame | None:
    path = config.PREDICTIONS / f"{runs.run_name(model, variant, split, seed)}.csv"
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


def combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """OR rule and mean rule; scores keep 0.5 as the decision point."""
    ta = recall_threshold(*a[a.set == "val"][["label", "prob"]].to_numpy().T)
    tb = recall_threshold(*b[b.set == "val"][["label", "prob"]].to_numpy().T)
    m = a.merge(b[["set", "id", "prob"]], on=["set", "id"], suffixes=("_a", "_b"))
    # Shift each score so its own threshold sits at 0.5; then max() flags if either flags.
    m["or"] = np.clip(0.5 + np.maximum(m.prob_a - ta, m.prob_b - tb), 0, 1)
    m["mean"] = (m.prob_a + m.prob_b) / 2
    return m


def main() -> None:
    for name, parts in ENSEMBLES:
        (ma, va), (mb, vb) = parts
        built = 0
        for split in SPLITS:
            for seed in config.SEEDS:
                a = load_run(ma, va, 0, split)
                b = load_run(mb, vb, seed, split)
                if a is None or b is None:
                    continue
                m = combine(a, b)
                runs.save(name, "clean", split, seed, m, m["or"].to_numpy())
                if name == "ensemble":
                    runs.save("ensemble_mean", "clean", split, seed, m, m["mean"].to_numpy())
                built += 1
        print("built", name, "runs:", built)


if __name__ == "__main__":
    main()
