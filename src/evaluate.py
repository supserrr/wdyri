"""Turn every saved prediction file into one metrics table.

Usage:
    python -m src.evaluate

Writes results/experiments_by_seed.csv with one row per (model, variant, split,
seed, evaluation set) and results/experiments.csv with the mean and standard
deviation over seeds. Thresholds:
  * f1, precision, recall at the default 0.5 threshold (comparable to prior work)
  * *_op columns at the operating point: the threshold chosen on validation to
    catch at least 95% of scams, then applied unchanged to every other set
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, runs
from .data import load_bongo, load_chichewa
from .metrics import bootstrap_f1, recall_threshold, scores


def chichewa_subsets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All 733 messages, the balanced D-CHI part, and one message per template."""
    chi = load_chichewa().set_index("id")
    base = frame.assign(source=frame["id"].map(chi["source"]), template=frame["id"].map(chi["template_id"]))
    return {
        "all": base,
        "dchi": base[base["source"] == "D_CHI"],
        "dedup": base.drop_duplicates("template"),
    }


def test_subsets(frame: pd.DataFrame, templates: pd.Series) -> dict[str, pd.DataFrame]:
    """The test set as is, and with one message per template.

    One landlord-impersonation template holds 31 of the 81 test scams, so the
    per-template view shows whether a score rests on a single script.
    """
    return {"": frame, "/tpl": frame.assign(t=frame["id"].map(templates)).drop_duplicates("t")}


def evaluate(preds: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    templates = load_bongo().set_index("id")["template_id"]
    rows = []
    for (model, variant, split, seed), run in preds.groupby(["model", "variant", "split", "seed"]):
        val = run[run["set"] == "val"]
        thr = recall_threshold(val["label"], val["prob"]) if len(val) else 0.5
        for set_name, part in run.groupby("set"):
            parts = {"": part}
            if set_name.startswith("chichewa"):
                parts = {f"/{k}": v for k, v in chichewa_subsets(part).items()}
            elif set_name.startswith("test") and split == "template":
                parts = test_subsets(part, templates)
            for suffix, p in parts.items():
                y, prob = p["label"].to_numpy(), p["prob"].to_numpy()
                row = {"model": model, "variant": variant, "split": split, "seed": seed,
                       "set": set_name + suffix, **scores(y, prob)}
                op = scores(y, prob, threshold=thr)
                pred = prob >= thr
                row.update({"threshold_op": thr, "precision_op": op["precision"],
                            "recall_op": op["recall"], "f1_op": op["f1"],
                            # confusion matrix at the operating point (scam = positive)
                            "tp_op": int((pred & (y == 1)).sum()), "fp_op": int((pred & (y == 0)).sum()),
                            "fn_op": int((~pred & (y == 1)).sum()), "tn_op": int((~pred & (y == 0)).sum())})
                if set_name in ("test", "chichewa") and suffix in ("", "/all", "/tpl"):
                    row["f1_ci_low"], row["f1_ci_high"] = bootstrap_f1(y, prob, n_boot=n_boot)
                rows.append(row)
    out = pd.DataFrame(rows)
    # Relative F1 drop against the same run's clean test set (Chiuseni et al.'s measure).
    # Attacked sets are compared with the clean test set in the same view
    # ("", "/tpl") and with the same defence ("+norm" or not).
    def view(sets: pd.Series) -> pd.Series:
        return sets.str.contains(r"\+norm").map({True: "+norm", False: ""}) + \
            sets.str.endswith("/tpl").map({True: "/tpl", False: ""})

    clean = out[out["set"].isin(["test", "test+norm", "test/tpl", "test+norm/tpl"])].copy()
    clean["key"] = view(clean["set"])
    clean = clean.set_index(["model", "variant", "split", "seed", "key"])["f1"]
    keys = view(out["set"])
    ref = [clean.get((m, v, s, sd, k), np.nan)
           for m, v, s, sd, k in zip(out.model, out.variant, out.split, out.seed, keys)]
    out["f1_clean"] = ref
    out["rel_f1_drop"] = (out["f1_clean"] - out["f1"]) / out["f1_clean"]
    # The drop only means something for the (attacked) Swahili test sets.
    out.loc[~out["set"].str.startswith("test"), ["f1_clean", "rel_f1_drop"]] = np.nan
    return out


def pooled_ci(preds: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """95% bootstrap interval for the seed-averaged F1.

    Each resample draws the same messages for every seed, computes F1 per seed
    and averages over seeds, so the interval covers the number we report.
    """
    templates = load_bongo().set_index("id")["template_id"]
    rows = []
    for (model, variant, split), run in preds.groupby(["model", "variant", "split"]):
        for set_name in ("test", "chichewa"):
            part = run[run["set"] == set_name]
            if part.empty:
                continue
            if set_name == "test" and split == "template":
                # One message per template, chosen once (same rule as test_subsets) and
                # then taken from every seed, so all seeds are scored on the same messages.
                first = part[part["seed"] == part["seed"].iloc[0]]
                keep = set(test_subsets(first, templates)["/tpl"]["id"])
                views = {"": part, "/tpl": part[part["id"].isin(keep)]}
            elif set_name == "chichewa":
                views = {"/all": part}
            else:
                views = {"": part}
            for suffix, view in views.items():
                wide = view.pivot_table(index="id", columns="seed", values="prob")
                if wide.isna().any().any():
                    # Seeds scored different messages (few-shot runs, repeated splits):
                    # a pooled interval would mix them, so none is reported.
                    continue
                y = view.drop_duplicates("id").set_index("id").loc[wide.index, "label"].to_numpy() == 1
                p = wide.to_numpy() >= 0.5
                idx = np.random.default_rng(0).integers(0, len(y), (n_boot, len(y)))
                yy, pp = y[idx][:, :, None], p[idx]                      # (boot, messages, seeds)
                tp = (pp & yy).sum(1)
                fp = (pp & ~yy).sum(1)
                fn = (~pp & yy).sum(1)
                f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
                low, high = np.percentile(f1.mean(1), [2.5, 97.5])
                rows.append({"model": model, "variant": variant, "split": split, "set": set_name + suffix,
                             "f1_ci_low": low, "f1_ci_high": high})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, plus the seed-pooled F1 interval."""
    metrics = ["n", "n_scam", "accuracy", "precision", "recall", "f1", "pr_auc", "precision_op", "recall_op",
               "f1_op", "tp_op", "fp_op", "fn_op", "tn_op", "rel_f1_drop"]
    keys = ["model", "variant", "split", "set"]
    g = table.groupby(keys)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.join(std, rsuffix="_sd")
    out["n_seeds"] = table.groupby(keys).size()
    return out.reset_index().merge(ci, on=keys, how="left")


def main() -> None:
    preds = runs.load_all()
    table = evaluate(preds)
    table.to_csv(config.RESULTS / "experiments_by_seed.csv", index=False, float_format="%.4f")
    summary = summarise(table, pooled_ci(preds))
    summary.to_csv(config.RESULTS / "experiments.csv", index=False, float_format="%.4f")
    view = summary[summary["set"] == "test"][["model", "variant", "split", "accuracy", "f1", "f1_sd", "pr_auc", "n_seeds"]]
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
