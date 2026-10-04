"""Scam-class metrics, the validation-chosen threshold and bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)

from . import config


def scores(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision, recall and F1 for the scam class, plus PR-AUC and accuracy."""
    y = np.asarray(y)
    pred = (np.asarray(prob) >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "n_scam": int(y.sum()),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    # PR-AUC (average precision) is undefined when a set holds a single class.
    out["pr_auc"] = average_precision_score(y, prob) if 0 < y.sum() < len(y) else np.nan
    return out


def recall_threshold(y_val: np.ndarray, prob_val: np.ndarray, target: float = config.TARGET_RECALL) -> float:
    """Decision threshold that flags at least `target` of validation scams.

    A missed scam costs the user money; a false alarm costs trust. We fix the
    recall we need and then report what precision that buys. The threshold
    starts at 0.5 and is only lowered when 0.5 misses more than 5% of
    validation scams; it is never raised, because a perfectly separated
    validation set would otherwise push it to ~1.0 and overfit.
    """
    scam_probs = np.sort(np.asarray(prob_val)[np.asarray(y_val) == 1])
    if len(scam_probs) == 0:
        return 0.5
    # Allow at most floor((1 - target) * n) scams below the threshold.
    k = int(np.floor((1 - target) * len(scam_probs)))
    return float(min(0.5, scam_probs[k]))


def bootstrap_f1(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5,
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for scam F1 (resampling test messages)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred = (np.asarray(prob) >= threshold).astype(int)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        stats.append(f1_score(y[idx], pred[idx], zero_division=0))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)
