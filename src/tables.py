"""Markdown tables for the README and report, built from results/experiments*.csv.

Usage:
    python -m src.tables      # writes results/tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

NAMES = {
    "majority": "Majority class", "phone_rule": "Phone rule", "nb_word_counts": "Naive Bayes, word counts",
    "nb_char": "Naive Bayes, char 2-5", "lr_word": "Log. regression, word 1-2", "lr_char": "Log. regression, char 2-5",
    "bilstm_random": "BiLSTM, random init", "bilstm_frozen": "BiLSTM, fastText frozen",
    "bilstm_finetuned": "BiLSTM, fastText fine-tuned", "xlmr": "XLM-R base", "afroxlmr": "AfroXLMR base",
}
LADDER = ["majority", "phone_rule", "nb_word_counts", "nb_char", "lr_word", "lr_char",
          "bilstm_random", "bilstm_frozen", "bilstm_finetuned", "xlmr", "afroxlmr"]


def fmt(mean: float, sd: float = np.nan, n: int = 1) -> str:
    if mean is None or np.isnan(mean):
        return "-"
    return f"{mean:.3f} ± {sd:.3f}" if n > 1 and not np.isnan(sd) else f"{mean:.3f}"


def cell(s: pd.DataFrame, model: str, variant: str, split: str, set_name: str, metric: str = "f1") -> str:
    r = s[(s.model == model) & (s.variant == variant) & (s.split == split) & (s.set == set_name)]
    if r.empty:
        return "-"
    r = r.iloc[0]
    return fmt(r[metric], r.get(f"{metric}_sd", np.nan), int(r["n_seeds"]))


def md(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    s = pd.read_csv(config.RESULTS / "experiments.csv")
    by_seed = pd.read_csv(config.RESULTS / "experiments_by_seed.csv")
    tuning = json.loads((config.RESULTS / "classical_tuning.json").read_text())
    parts = []

    # Main results: every rung on the honest (template-disjoint) test set
    rows = []
    for m in LADDER:
        if s[(s.model == m) & (s.variant == "clean") & (s.split == "template")].empty:
            continue
        rows.append([NAMES[m], cell(s, m, "clean", "template", "test"), cell(s, m, "clean", "template", "test/tpl"),
                     cell(s, m, "clean", "template", "test", "pr_auc"),
                     cell(s, m, "clean", "template", "test", "precision_op"),
                     cell(s, m, "clean", "template", "test", "recall_op"),
                     cell(s, m, "clean", "template", "chichewa/all"),
                     cell(s, m, "clean", "template", "chichewa/all", "pr_auc")])
    parts.append("## Main results (template-disjoint split)\n\nScam F1 at threshold 0.5 unless noted; "
                 "neural models are mean ± sd over 3 seeds. *Per template* keeps one test message per template; "
                 "*op.* is the operating point chosen on validation (>= 95% scam recall).\n\n" + md(rows, [
                     "Model", "Test F1", "Test F1 (per template)", "Test PR-AUC", "Precision (op.)",
                     "Recall (op.)", "Chichewa F1 (zero-shot)", "Chichewa PR-AUC"]))

    # E1 + E2
    acc = tuning["published_accuracy"]
    rows = [["E1 published setup (1,508 raw rows, 80/20)", "Naive Bayes, word counts",
             f"accuracy {acc:.4f}, F1 {cell(s, 'nb_word_counts', 'published', 'published', 'test')}", "-"]]
    for m in ["nb_word_counts", "nb_char", "lr_word", "lr_char"]:
        rows.append(["E2 cleaned data, one split", NAMES[m], cell(s, m, "clean", "random", "test"),
                     cell(s, m, "clean", "template", "test")])
    rep = by_seed[(by_seed.variant == "repeat") & (by_seed.set == "test")]
    for m in ["nb_word_counts", "nb_char", "lr_word", "lr_char"]:
        g = rep[rep.model == m].groupby("split")["f1"]
        rows.append(["E2 cleaned data, 10 re-drawn splits", NAMES[m],
                     fmt(g.mean()["random"], g.std()["random"], 10), fmt(g.mean()["template"], g.std()["template"], 10)])
    parts.append("## E1-E2: does leakage inflate the scores? (RQ1)\n\n" + md(rows, ["Setting", "Model", "Random split F1", "Template split F1"]))

    # E3 + E3b
    rows = []
    for m in ["nb_word_counts", "nb_char", "lr_word", "lr_char"]:
        rows.append([NAMES[m], cell(s, m, "clean", "template", "test"), cell(s, m, "nomask", "template", "test"),
                     cell(s, m, "strip", "template", "test"), cell(s, m, "clean", "template", "chichewa/all"),
                     cell(s, m, "strip", "template", "chichewa/all")])
    rows.append([NAMES["phone_rule"], cell(s, "phone_rule", "clean", "template", "test"), "-", "-",
                 cell(s, "phone_rule", "clean", "template", "chichewa/all"), "-"])
    for m in ["bilstm_finetuned", "afroxlmr"]:
        rows.append([NAMES[m], cell(s, m, "clean", "template", "test"), "-", cell(s, m, "strip", "template", "test"),
                     cell(s, m, "clean", "template", "chichewa/all"), cell(s, m, "strip", "template", "chichewa/all")])
    parts.append("## E3-E3b: features, masking and the phone-number shortcut\n\nTemplate split, scam F1. "
                 "*Masked*: `<PHONE>` etc. as tokens (default). *Raw*: numbers left in. "
                 "*No placeholders*: masked tokens deleted, so a model cannot use the presence of a number.\n\n" + md(
                     rows, ["Model", "Masked", "Raw", "No placeholders", "Chichewa, masked", "Chichewa, no placeholders"]))

    # E4 + E5
    rows = [[NAMES[m], cell(s, m, "clean", "template", "val"), cell(s, m, "clean", "template", "test"),
             cell(s, m, "clean", "template", "test/tpl"), cell(s, m, "clean", "template", "test", "pr_auc")]
            for m in ["bilstm_random", "bilstm_frozen", "bilstm_finetuned", "xlmr", "afroxlmr"]]
    parts.append("## E4-E5: embeddings and transformers\n\n" + md(rows, ["Model", "Validation F1", "Test F1", "Test F1 (per template)", "Test PR-AUC"]))

    # E6 + E7
    rows = []
    for m, variant in [("nb_word_counts", "clean"), ("lr_word", "clean"), ("lr_char", "clean"),
                       ("bilstm_finetuned", "clean"), ("xlmr", "clean"), ("afroxlmr", "clean"),
                       ("nb_word_counts", "advtrain"), ("lr_char", "advtrain"), ("bilstm_finetuned", "advtrain"),
                       ("afroxlmr", "advtrain")]:
        if s[(s.model == m) & (s.variant == variant)].empty:
            continue
        label = NAMES[m] + (" + adversarial training" if variant == "advtrain" else "")
        row = [label]
        for a in ("lookalike", "structural", "codeswitch"):
            row.append(cell(s, m, variant, "template", f"test_{a}_all", "rel_f1_drop"))
            row.append(cell(s, m, variant, "template", f"test_{a}_all+norm", "rel_f1_drop"))
        rows.append(row)
    parts.append("## E6-E7: relative F1 drop under the strongest attacks (all trigger words)\n\n"
                 "(F1 clean - F1 attacked) / F1 clean; negative = the attack made the scam easier to catch. "
                 "*+norm*: with the normalisation defence.\n\n" + md(rows, [
                     "Model", "Lookalike", "Lookalike +norm", "Structural", "Structural +norm", "Code-switch", "Code-switch +norm"]))

    # E8 + E9
    rows = []
    for m in ["majority", "phone_rule", "nb_word_counts", "lr_char", "bilstm_finetuned", "xlmr", "afroxlmr"]:
        rows.append([NAMES[m], cell(s, m, "clean", "template", "chichewa/all"), cell(s, m, "clean", "template", "chichewa/dchi"),
                     cell(s, m, "clean", "template", "chichewa/dedup"), cell(s, m, "clean", "template", "chichewa/all", "pr_auc"),
                     cell(s, m, "fewshot20", "template", "chichewa/all"), cell(s, m, "fewshot50", "template", "chichewa/all")])
    parts.append("## E8-E9: Swahili to Chichewa transfer (RQ3)\n\nFraud F1. *D-CHI*: the balanced fraud/normal part; "
                 "*per template*: one message per template; few-shot columns add 20 or 50 Chichewa messages to training "
                 "(their templates removed from the test).\n\n" + md(rows, [
                     "Model", "All 733", "D-CHI only", "Per template", "PR-AUC (all)", "+20 Chichewa", "+50 Chichewa"]))

    errors = config.RESULTS / "error_buckets.csv"
    if errors.exists():
        e = pd.read_csv(errors)
        parts.append("## Error buckets (seed 42 / single run, operating-point threshold)\n\n" + md(
            e.values.tolist(), list(e.columns)))

    (config.RESULTS / "tables.md").write_text("# Results tables\n\nGenerated by `python -m src.tables`.\n\n" + "\n\n".join(parts) + "\n")
    print((config.RESULTS / "tables.md").read_text()[:3000])


if __name__ == "__main__":
    main()
