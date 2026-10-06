"""Are the key differences larger than chance? Paired bootstrap and rank tests.

Paired bootstrap (two systems on the same messages): resample messages with
replacement 1,000 times, recompute the metric for both systems on the same
resample, and look at the distribution of the difference. Neural runs are
paired seed by seed (a deterministic model is paired with every seed) and the
resamples are pooled. Two-sided p = 2 x the smaller tail share around zero,
with exact ties split evenly between the tails (mid-p).

Repeated splits (RQ1) are not paired, so random vs template F1 over the 10
redraws uses a Mann-Whitney U test.

Usage:
    python -m src.significance      # writes results/significance.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, f1_score

from . import config, runs
from .metrics import recall_threshold

METRICS = {
    "f1": lambda y, p, t: f1_score(y, p >= t, zero_division=0),
    "pr_auc": lambda y, p, t: average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,
    "precision": lambda y, p, t: ((p >= t) & (y == 1)).sum() / max((p >= t).sum(), 1),
    "recall": lambda y, p, t: ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1),
}

# (label, system A, system B, set, metric); a system is (model, variant)
COMPARISONS = [
    ("AfroXLMR vs XLM-R, Chichewa ranking", ("afroxlmr", "clean"), ("xlmr", "clean"), "chichewa", "pr_auc"),
    ("AfroXLMR vs XLM-R, Chichewa F1", ("afroxlmr", "clean"), ("xlmr", "clean"), "chichewa", "f1"),
    ("AfroXLMR vs XLM-R, Swahili test F1", ("afroxlmr", "clean"), ("xlmr", "clean"), "test", "f1"),
    ("Plain ensemble (char LR OR AfroXLMR) vs AfroXLMR, Swahili test F1", ("ensemble", "clean"), ("afroxlmr", "clean"), "test", "f1"),
    ("Plain ensemble vs char LR, full lookalike attack F1", ("ensemble", "clean"), ("lr_char", "clean"), "test_lookalike_all", "f1"),
    ("Plain ensemble vs char LR, full structural attack F1", ("ensemble", "clean"), ("lr_char", "clean"), "test_structural_all", "f1"),
    ("Normalisation vs none, word LR under full lookalike", ("lr_word", "clean"), ("lr_word", "clean"), ("test_lookalike_all+norm", "test_lookalike_all"), "f1"),
    ("Normalisation vs none, word LR under held-out lookalikes", ("lr_word", "clean"), ("lr_word", "clean"), ("test_unseen_all+norm", "test_unseen_all"), "f1"),
    ("Number-balanced vs clean AfroXLMR, Chichewa F1", ("afroxlmr", "counterfactual"), ("afroxlmr", "clean"), "chichewa", "f1"),
    ("Number-balanced vs clean AfroXLMR, Chichewa precision", ("afroxlmr", "counterfactual"), ("afroxlmr", "clean"), "chichewa", "precision"),
    ("Number-balanced vs clean AfroXLMR, Swahili test F1", ("afroxlmr", "counterfactual"), ("afroxlmr", "clean"), "test", "f1"),
    ("Number-balanced vs clean char LR, Chichewa F1", ("lr_char", "counterfactual"), ("lr_char", "clean"), "chichewa", "f1"),
    ("AfroXLMR +50 Chichewa vs XLM-R +50, F1", ("afroxlmr", "fewshot50"), ("xlmr", "fewshot50"), "chichewa", "f1"),
    ("Char LR +50 Chichewa vs AfroXLMR +50, F1", ("lr_char", "fewshot50"), ("afroxlmr", "fewshot50"), "chichewa", "f1"),
    # the deployed ensemble: number-balanced char LR OR number-balanced AfroXLMR (chosen on validation)
    ("Deployed ensemble vs number-balanced AfroXLMR, Swahili test F1", ("ensemble_cf2", "clean"), ("afroxlmr", "counterfactual"), "test", "f1"),
    ("Deployed ensemble vs char LR, Swahili test F1", ("ensemble_cf2", "clean"), ("lr_char", "clean"), "test", "f1"),
    ("Deployed ensemble vs number-balanced char LR, Swahili test F1", ("ensemble_cf2", "clean"), ("lr_char", "counterfactual"), "test", "f1"),
    ("Deployed ensemble vs number-balanced char LR, full lookalike attack F1", ("ensemble_cf2", "clean"), ("lr_char", "counterfactual"), "test_lookalike_all", "f1"),
    ("Deployed ensemble vs char LR, full lookalike attack F1", ("ensemble_cf2", "clean"), ("lr_char", "clean"), "test_lookalike_all", "f1"),
    ("Deployed ensemble vs char LR, lookalike with NB attacker F1", ("ensemble_cf2", "clean"), ("lr_char", "clean"), "test_xatk_lookalike_all", "f1"),
    ("Deployed ensemble vs char LR, structural with NB attacker F1", ("ensemble_cf2", "clean"), ("lr_char", "clean"), "test_xatk_structural_all", "f1"),
    ("Deployed ensemble vs char LR, fresh split 1 test F1", ("ensemble_cf2", "clean", "template_r1"), ("lr_char", "clean", "template_r1"), "test", "f1"),
    ("Deployed ensemble vs char LR, fresh split 2 test F1", ("ensemble_cf2", "clean", "template_r2"), ("lr_char", "clean", "template_r2"), "test", "f1"),
    ("Deployed ensemble vs AfroXLMR, fresh split 1 test F1", ("ensemble_cf2", "clean", "template_r1"), ("afroxlmr", "clean", "template_r1"), "test", "f1"),
    ("Deployed ensemble vs AfroXLMR, fresh split 2 test F1", ("ensemble_cf2", "clean", "template_r2"), ("afroxlmr", "clean", "template_r2"), "test", "f1"),
]


def runs_of(preds: pd.DataFrame, model: str, variant: str, split: str = "template") -> dict[int, pd.DataFrame]:
    sub = preds[(preds.model == model) & (preds.variant == variant) & (preds.split == split)]
    return {seed: g for seed, g in sub.groupby("seed")}


def paired_bootstrap(a: dict, b: dict, set_a: str, set_b: str, metric: str, n_boot: int = 1000):
    fn = METRICS[metric]
    seeds = sorted(set(a) & set(b)) or [(sa, sb) for sa in a for sb in b]
    if seeds and not isinstance(seeds[0], tuple):
        seeds = [(s, s) for s in seeds]
    diffs, points = [], []
    rng = np.random.default_rng(0)
    for sa, sb in seeds:
        ra, rb = a[sa], b[sb]
        ta = recall_threshold(*ra[ra.set == "val"][["label", "prob"]].to_numpy().T)
        tb = recall_threshold(*rb[rb.set == "val"][["label", "prob"]].to_numpy().T)
        m = ra[ra.set == set_a][["id", "label", "prob"]].merge(
            rb[rb.set == set_b][["id", "prob"]], on="id", suffixes=("_a", "_b"))
        y, pa, pb = m.label.to_numpy(), m.prob_a.to_numpy(), m.prob_b.to_numpy()
        points.append(fn(y, pa, ta) - fn(y, pb, tb))
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            diffs.append(fn(y[i], pa[i], ta) - fn(y[i], pb[i], tb))
    diffs = np.array(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    # Mid-p: resamples with a difference of exactly 0 count half to each tail,
    # so many ties (identical decisions) do not inflate the p-value.
    below, above, ties = (diffs < 0).mean(), (diffs > 0).mean(), (diffs == 0).mean()
    p = 2 * min(below + ties / 2, above + ties / 2)
    return float(np.mean(points)), *np.percentile(diffs, [2.5, 97.5]), min(1.0, p), len(seeds)


def main() -> None:
    preds = runs.load_all()
    rows = []
    for label, sys_a, sys_b, sets, metric in COMPARISONS:
        set_a, set_b = sets if isinstance(sets, tuple) else (sets, sets)
        a, b = runs_of(preds, *sys_a), runs_of(preds, *sys_b)  # a system may name its split
        if not a or not b:
            continue
        diff, low, high, p, n = paired_bootstrap(a, b, set_a, set_b, metric)
        rows.append({"comparison": label, "metric": metric, "difference": diff, "ci_low": low,
                     "ci_high": high, "p_value": p, "seed_pairs": n, "test": "paired bootstrap"})

    by_seed = pd.read_csv(config.RESULTS / "experiments_by_seed.csv")
    rep = by_seed[(by_seed.variant == "repeat") & (by_seed.set == "test")]
    for model in ["nb_word_counts", "nb_char", "lr_word", "lr_char"]:
        r = rep[(rep.model == model) & (rep.split == "random")]["f1"]
        t = rep[(rep.model == model) & (rep.split == "template")]["f1"]
        rows.append({"comparison": f"Random vs template split, {model}", "metric": "f1",
                     "difference": r.mean() - t.mean(), "ci_low": np.nan, "ci_high": np.nan,
                     "p_value": mannwhitneyu(r, t, alternative="two-sided").pvalue,
                     "seed_pairs": len(r), "test": "Mann-Whitney U (10 vs 10 splits)"})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "significance.csv", index=False, float_format="%.4f")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
