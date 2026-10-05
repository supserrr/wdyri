"""E10: shortcut stress tests from minimal pairs.

Each stress set changes exactly one placeholder in a test message (see
src/eval_sets.py). Comparing a model's decision on the pair isolates the
effect of the number alone:

    false alarms on genuine texts   before vs after appending <PHONE> / <AMOUNT>
    misses on number-bearing scams  before vs after deleting their placeholders
    flip rate                       share of pairs whose decision changes

Decisions use each run's validation-chosen threshold (the operating point).

Usage:
    python -m src.stress      # writes results/stress_tests.csv (test pairs) and
                              # results/stress_tests_validation.csv (validation pairs)
"""
from __future__ import annotations

import pandas as pd

from . import config, runs
from .metrics import recall_threshold

MODELS = ["phone_rule", "nb_word_counts", "lr_word", "lr_char", "bilstm_finetuned",
          "xlmr", "afroxlmr", "ensemble"]


def analyse(run: pd.DataFrame, base: str = "test", prefix: str = "stress") -> dict | None:
    """Stress metrics on the test pairs (prefix "stress") or the validation pairs ("valstress")."""
    if f"{prefix}_genuine+phone" not in set(run["set"]):
        return None
    val = run[run.set == "val"]
    thr = recall_threshold(val.label, val.prob) if len(val) else 0.5
    flag = run.assign(flag=run.prob >= thr).set_index(["set", "id"])["flag"]
    ref = run[run.set == base]
    genuine_ids = ref[ref.label == 0].id
    scam_ids = run[run.set == f"{prefix}_scam-number"].id

    def rate(set_name, ids):
        return flag.loc[[(set_name, i) for i in ids]].to_numpy()

    g0 = rate(base, genuine_ids)
    out = {"false_alarm_clean": g0.mean()}
    for name in ("phone", "amount"):
        g1 = rate(f"{prefix}_genuine+{name}", genuine_ids)
        out[f"false_alarm_+{name}"] = g1.mean()
        out[f"flip_+{name}"] = (g1 != g0).mean()
    s0, s1 = rate(base, scam_ids), rate(f"{prefix}_scam-number", scam_ids)
    out.update({"miss_clean": 1 - s0.mean(), "miss_-number": 1 - s1.mean(), "flip_-number": (s1 != s0).mean()})
    return out


def main() -> None:
    preds = runs.load_all()
    preds = preds[preds.split == "template"]
    rows, val_rows = [], []
    for (model, variant, seed), run in preds.groupby(["model", "variant", "seed"]):
        res = analyse(run)
        if res is not None:
            rows.append({"model": model, "variant": variant, "seed": seed, **res})
        res = analyse(run, base="val", prefix="valstress")
        if res is not None:
            val_rows.append({"model": model, "variant": variant, "seed": seed, **res})
    if val_rows:
        # Validation pairs: the only evidence used to pick the deployed system.
        v = pd.DataFrame(val_rows).groupby(["model", "variant"]).mean(numeric_only=True).drop(columns="seed")
        v.reset_index().to_csv(config.RESULTS / "stress_tests_validation.csv", index=False, float_format="%.4f")
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "stress_tests_by_seed.csv", index=False, float_format="%.4f")
    g = table.groupby(["model", "variant"])
    summary = g.mean(numeric_only=True).drop(columns="seed").join(
        g.std(numeric_only=True).drop(columns="seed"), rsuffix="_sd")
    summary["n_seeds"] = g.size()
    summary.reset_index().to_csv(config.RESULTS / "stress_tests.csv", index=False, float_format="%.4f")
    view = summary.reset_index()
    view = view[view.variant.isin(["clean", "counterfactual", "strip"])]
    print(view[["model", "variant", "false_alarm_clean", "false_alarm_+phone", "false_alarm_+amount",
                "miss_clean", "miss_-number", "n_seeds"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
