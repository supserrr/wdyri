"""Template detection and the two 70/15/15 splits (random and template-disjoint).

A scam "template" is one script sent many times with a different name, number
or amount. If copies of one template sit in both train and test, the test set
rewards memorisation. The template-disjoint split keeps every copy on one side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config


def template_ids(texts: list[str], threshold: float = config.TEMPLATE_JACCARD,
                 n: int = config.TEMPLATE_NGRAM) -> np.ndarray:
    """Group messages whose character n-gram sets overlap by >= threshold (Jaccard).

    Grouping is transitive (union-find): if A~B and B~C, all three share an id.
    """
    vec = CountVectorizer(analyzer="char", ngram_range=(n, n), binary=True, lowercase=True)
    X = vec.fit_transform(texts).astype(np.int32)
    sizes = np.asarray(X.sum(axis=1)).ravel()
    inter = (X @ X.T).tocoo()

    parent = np.arange(len(texts))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, shared in zip(inter.row, inter.col, inter.data):
        if i >= j:
            continue
        union = sizes[i] + sizes[j] - shared
        if union and shared / union >= threshold:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(texts))])
    # Renumber so ids are 0..k-1 in order of first appearance.
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def random_split(labels: np.ndarray, seed: int = config.SPLIT_SEED) -> np.ndarray:
    """Plain stratified 70/15/15 split, the setting prior work reports."""
    idx = np.arange(len(labels))
    rest, test = train_test_split(idx, test_size=0.15, stratify=labels, random_state=seed)
    train, val = train_test_split(rest, test_size=0.15 / 0.85, stratify=labels[rest], random_state=seed)
    out = np.empty(len(labels), dtype=object)
    out[train], out[val], out[test] = "train", "val", "test"
    return out


def _best_group_fold(labels, groups, n_splits, target_share, seed):
    """Pick the StratifiedGroupKFold fold closest to the target size and class balance."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best, best_cost = None, np.inf
    overall = labels.mean()
    for _, fold in cv.split(np.zeros(len(labels)), labels, groups):
        cost = abs(len(fold) / len(labels) - target_share) + abs(labels[fold].mean() - overall)
        if cost < best_cost:
            best, best_cost = fold, cost
    return best


def template_split(labels: np.ndarray, groups: np.ndarray, seed: int = config.SPLIT_SEED) -> np.ndarray:
    """70/15/15 split where no template appears in more than one part."""
    idx = np.arange(len(labels))
    test = _best_group_fold(labels, groups, n_splits=7, target_share=0.15, seed=seed)
    rest = np.setdiff1d(idx, test)
    val_local = _best_group_fold(labels[rest], groups[rest], n_splits=6, target_share=0.15 / 0.85, seed=seed)
    val = rest[val_local]
    out = np.full(len(labels), "train", dtype=object)
    out[val], out[test] = "val", "test"
    return out


def check_disjoint(df: pd.DataFrame, split_col: str, group_col: str = "template_id") -> int:
    """Number of templates that appear in more than one part of a split (0 = no leakage)."""
    spread = df.groupby(group_col)[split_col].nunique()
    return int((spread > 1).sum())
