"""Markdown tables for docs/results.md and the report, built from results/experiments*.csv.

Usage:
    python -m src.tables      # writes results/tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, runs
from .evaluate import fewshot_pairs

NAMES = {
    "majority": "Majority class", "phone_rule": "Phone rule", "length_rule": "Length rule", "nb_word_counts": "Naive Bayes, word counts",
    "nb_char": "Naive Bayes, char 2-5", "lr_word": "Log. regression, word 1-2", "lr_char": "Log. regression, char 2-5",
    "bilstm_random": "BiLSTM, random init", "bilstm_frozen": "BiLSTM, fastText frozen",
    "bilstm_finetuned": "BiLSTM, fastText fine-tuned", "xlmr": "XLM-R base", "afroxlmr": "AfroXLMR base",
    "ensemble": "Ensemble: char LR OR AfroXLMR", "ensemble_cf": "Ensemble: char LR OR number-balanced AfroXLMR",
    "ensemble_mean": "Ensemble: mean of char LR and AfroXLMR",
    "ensemble_cf2": "Ensemble: number-balanced char LR OR number-balanced AfroXLMR",
}
VARIANT_NAMES = {"clean": "", "counterfactual": ", number-balanced (E11)", "strip": ", no placeholders",
                 "advtrain": " + adversarial training"}
LADDER = [(m, "clean") for m in ["majority", "phone_rule", "length_rule", "nb_word_counts", "nb_char", "lr_word", "lr_char",
                                 "bilstm_random", "bilstm_frozen", "bilstm_finetuned", "xlmr", "afroxlmr"]]
LADDER += [("afroxlmr", "counterfactual"), ("ensemble", "clean"), ("ensemble_cf", "clean"), ("ensemble_cf2", "clean")]


def label(model: str, variant: str) -> str:
    return NAMES[model] + VARIANT_NAMES.get(variant, f", {variant}")


def ci(s: pd.DataFrame, model: str, variant: str, set_name: str) -> str:
    r = s[(s.model == model) & (s.variant == variant) & (s.split == "template") & (s.set == set_name)]
    if r.empty or np.isnan(r.iloc[0].get("f1_ci_low", np.nan)):
        return "-"
    return f"[{r.iloc[0]['f1_ci_low']:.2f}, {r.iloc[0]['f1_ci_high']:.2f}]"


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
    for m, v in LADDER:
        if s[(s.model == m) & (s.variant == v) & (s.split == "template")].empty:
            continue
        rows.append([label(m, v), cell(s, m, v, "template", "test"), ci(s, m, v, "test"),
                     cell(s, m, v, "template", "test/tpl"),
                     cell(s, m, v, "template", "test", "pr_auc"),
                     cell(s, m, v, "template", "test", "precision_op"),
                     cell(s, m, v, "template", "test", "recall_op"),
                     cell(s, m, v, "template", "chichewa/all"),
                     cell(s, m, v, "template", "chichewa/all", "pr_auc")])
    thr = by_seed.loc[by_seed["split"].str.startswith("template"), "threshold_op"]
    op_note = ("every run already catches that share at 0.5, so the threshold is 0.5 throughout and these are the "
               "0.5 values" if (thr == 0.5).all() else f"thresholds range from {thr.min():.2f} to {thr.max():.2f}")
    parts.append("## Main results (template-disjoint split)\n\nScam F1 at threshold 0.5 unless noted; "
                 "neural models are mean ± sd over 3 seeds; *95% CI* is a bootstrap interval over test messages "
                 "(averaged over seeds). *Per template* keeps one test message per template. *Precision* and "
                 f"*recall* use the operating point chosen on validation (>= 95% scam recall, never above 0.5); {op_note}."
                 "\n\n" + md(rows, [
                     "Model", "Test F1", "95% CI", "Test F1 (per template)", "Test PR-AUC", "Precision",
                     "Recall", "Chichewa F1 (zero-shot)", "Chichewa PR-AUC"]))

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
    for rule in ("phone_rule", "length_rule"):
        rows.append([NAMES[rule], cell(s, rule, "clean", "template", "test"), "-", "-",
                     cell(s, rule, "clean", "template", "chichewa/all"), "-"])
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
        row = [label(m, variant)]
        for a in ("lookalike", "structural", "codeswitch", "unseen"):
            row.append(cell(s, m, variant, "template", f"test_{a}_all", "rel_f1_drop"))
            row.append(cell(s, m, variant, "template", f"test_{a}_all+norm", "rel_f1_drop"))
        rows.append(row)
    parts.append("## E6-E7: relative F1 drop under the strongest attacks (all trigger words)\n\n"
                 "(F1 clean - F1 attacked) / F1 clean; negative = the attack made the scam easier to catch. "
                 "*+norm*: with the normalisation defence. The defence's lookup table was written knowing the "
                 "lookalike attack's characters, so *Lookalike +norm* is a best case; *Held-out lookalike* disguises "
                 "the same words with characters the defence does not know (`perturb.UNSEEN`). The adversarially "
                 "trained AfroXLMR was not saved, so it has no held-out score.\n\n" + md(rows, [
                     "Model", "Lookalike", "Lookalike +norm", "Structural", "Structural +norm", "Code-switch",
                     "Code-switch +norm", "Held-out lookalike", "Held-out lookalike +norm"]))

    # E6 with a different attacker (word-count NB picks the trigger words)
    rows = []
    for m, v in [("nb_word_counts", "clean"), ("lr_word", "clean"), ("lr_char", "clean"), ("bilstm_finetuned", "clean"),
                 ("xlmr", "clean"), ("afroxlmr", "clean"), ("afroxlmr", "counterfactual"),
                 ("ensemble", "clean"), ("ensemble_cf", "clean"), ("ensemble_cf2", "clean")]:
        if s[(s.model == m) & (s.variant == v) & (s.set == "test_xatk_lookalike_all")].empty:
            continue
        row = [label(m, v), cell(s, m, v, "template", "test")]
        for a_ in ("lookalike", "structural", "codeswitch"):
            row += [cell(s, m, v, "template", f"test_{a_}_all"), cell(s, m, v, "template", f"test_xatk_{a_}_all")]
        rows.append(row)
    parts.append("## E6 with a second attacker\n\nThe main attacks take their trigger words from a char LR, which "
                 "makes them white-box for the char LR and transfer attacks for every other model. Here the same "
                 "attacks use a word-count Naive Bayes attacker instead (it changes 84% of test scams; its "
                 "probabilities saturate on the rest). Scam F1 at full intensity, char-LR attacker vs NB attacker."
                 "\n\n" + md(rows, ["Model", "Clean", "Lookalike (LR atk)", "Lookalike (NB atk)", "Structural (LR atk)",
                                     "Structural (NB atk)", "Code-switch (LR atk)", "Code-switch (NB atk)"]))

    # E8 + E9
    rows = []
    for m in ["majority", "phone_rule", "nb_word_counts", "lr_char", "bilstm_finetuned", "xlmr", "afroxlmr"]:
        rows.append([NAMES[m], cell(s, m, "clean", "template", "chichewa/all"), cell(s, m, "clean", "template", "chichewa/dchi"),
                     cell(s, m, "clean", "template", "chichewa/dedup"), cell(s, m, "clean", "template", "chichewa/all", "pr_auc")])
    parts.append("## E8: Swahili to Chichewa transfer, zero-shot (RQ3)\n\nFraud F1. *D-CHI*: the balanced fraud/normal part; "
                 "*per template*: one message per template. Few-shot results (E9) are in *E9 on the same messages* "
                 "below.\n\n" + md(rows, ["Model", "All 733", "D-CHI only", "Per template", "PR-AUC (all)"]))

    # E10 stress tests
    st_path = config.RESULTS / "stress_tests.csv"
    if st_path.exists():
        st = pd.read_csv(st_path)
        rows = []
        for m, v in [("phone_rule", "clean"), ("nb_word_counts", "clean"), ("lr_word", "clean"), ("lr_char", "clean"),
                     ("bilstm_finetuned", "clean"), ("xlmr", "clean"), ("afroxlmr", "clean"),
                     ("lr_char", "counterfactual"), ("bilstm_finetuned", "counterfactual"),
                     ("afroxlmr", "counterfactual"), ("ensemble", "clean"), ("ensemble_cf", "clean"),
                     ("ensemble_cf2", "clean")]:
            r = st[(st.model == m) & (st.variant == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            n = int(r["n_seeds"])
            rows.append([label(m, v)] + [fmt(r[c], r.get(f"{c}_sd", np.nan), n) for c in (
                "false_alarm_clean", "false_alarm_+phone", "false_alarm_+amount", "miss_clean", "miss_-number")])
        parts.append("## E10: shortcut stress tests (minimal pairs)\n\nEach pair differs by one placeholder only. "
                     "*False alarms*: share of the 71 genuine test texts flagged, as written and with ` <PHONE>` or "
                     "` <AMOUNT>` appended. *Misses*: share of the 48 test scams that contain a placeholder that are "
                     "missed, as written and with the placeholder deleted. A model that reads the language should "
                     "not move. Operating-point thresholds.\n\n" + md(rows, [
                         "Model", "False alarms", "+ phone", "+ amount", "Misses", "- number"]))

    # E11-E12: number-balanced training and the ensemble across every evaluation set
    sets = [("test", "Test"), ("test/tpl", "Per template"), ("test_lookalike_all", "Lookalike"),
            ("test_structural_all", "Structural"), ("test_codeswitch_all", "Code-switch"),
            ("chichewa/all", "Chichewa F1")]
    rows = []
    for m, v in [("lr_char", "clean"), ("lr_char", "counterfactual"), ("bilstm_finetuned", "clean"),
                 ("bilstm_finetuned", "counterfactual"), ("afroxlmr", "clean"), ("afroxlmr", "counterfactual"),
                 ("ensemble_mean", "clean"), ("ensemble", "clean"), ("ensemble_cf", "clean"), ("ensemble_cf2", "clean")]:
        if s[(s.model == m) & (s.variant == v)].empty:
            continue
        rows.append([label(m, v)] + [cell(s, m, v, "template", k) for k, _ in sets]
                    + [cell(s, m, v, "template", "chichewa/all", "precision")])
    parts.append("## E11-E12: number-balanced training and the complementary ensemble\n\nScam F1 on every "
                 "evaluation set (attacks at full intensity, no defence). The ensemble flags a message if either "
                 "member flags it at its own validation threshold.\n\n" + md(
                     rows, ["System"] + [n for _, n in sets] + ["Chichewa precision"]))

    sig_path = config.RESULTS / "significance.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        rows = [[r.comparison, r.metric, f"{r.difference:+.3f}",
                 "-" if np.isnan(r.ci_low) else f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                 "< 0.001" if r.p_value < 0.001 else f"{r.p_value:.3f}", r.test] for r in sig.itertuples()]
        parts.append("## Significance of the key differences\n\nDifference = first system minus second. "
                     "Paired bootstrap: 1,000 resamples of messages per seed pair, pooled.\n\n" + md(
                         rows, ["Comparison", "Metric", "Difference", "95% CI", "p", "Test"]))

    # E6 control: false alarms when *genuine* texts are disguised the same way
    def false_alarms(m: str, set_name: str, v: str = "clean") -> str:
        r = s[(s.model == m) & (s.variant == v) & (s.split == "template") & (s.set == set_name)]
        if r.empty:
            return "-"
        r = r.iloc[0]  # share of genuine texts flagged, at the operating point
        n_genuine = r["fp_op"] + r["tn_op"]
        return fmt(r["fp_op"] / n_genuine, r.get("fp_op_sd", np.nan) / n_genuine, int(r["n_seeds"]))

    control_sets = ["test", "ctrl_lookalike", "ctrl_structural", "ctrl_codeswitch",
                    "ctrlall_lookalike", "ctrlall_structural", "ctrlall_codeswitch"]
    rows = [[label(m, v)] + [false_alarms(m, st, v) for st in control_sets]
            for m, v in [("nb_word_counts", "clean"), ("lr_word", "clean"), ("lr_char", "clean"),
                         ("lr_char", "counterfactual"), ("bilstm_finetuned", "clean"), ("xlmr", "clean"),
                         ("afroxlmr", "clean"), ("afroxlmr", "counterfactual"), ("ensemble", "clean"),
                         ("ensemble_cf2", "clean")]
            if not s[(s.model == m) & (s.variant == v) & (s.set == "ctrl_lookalike")].empty]
    if rows:
        parts.append("## E6 control: disguising genuine texts\n\nThe attacked test sets disguise scams only, so an "
                     "odd-looking text is always a scam there. These controls disguise the 71 genuine test texts the "
                     "same way: first their 3 most influential words, then every word (matching the scam attacks' "
                     "full intensity; every lexicon phrase for code-switching). Cells are the share of genuine texts "
                     "flagged as scam at the operating point.\n\n" + md(rows, [
                         "Model", "As written", "Lookalike (3)", "Structural (3)", "Code-switch",
                         "Lookalike (all)", "Structural (all)", "Code-switch (all)"]))

    # E9 on equal footing: zero-shot vs few-shot on exactly the same Chichewa messages
    pairs = fewshot_pairs(runs.load_all())
    rows = []
    for m in ["lr_char", "xlmr", "afroxlmr"]:
        row = [NAMES[m]]
        for n in (20, 50):
            p = pairs[(pairs.model == m) & (pairs.n_shots == n)]
            row += [fmt(p.zero_f1.mean(), p.zero_f1.std(), len(p)) if len(p) else "-",
                    fmt(p.few_f1.mean(), p.few_f1.std(), len(p)) if len(p) else "-"]
        rows.append(row)
    sizes = {n: round(pairs[pairs.n_shots == n].n_messages.mean()) for n in (20, 50)}
    parts.append("## E9 on the same messages\n\nAdding Chichewa examples removes their templates from the test, so the "
                 f"few-shot test sets are smaller (about {sizes[20]} and {sizes[50]} messages). Here the zero-shot model is "
                 "scored on exactly the same messages as each few-shot run.\n\n" + md(rows, [
                     "Model", f"Zero-shot (same {sizes[20]})", "+20 Chichewa", f"Zero-shot (same {sizes[50]})",
                     "+50 Chichewa"]))

    # Confusion matrices at the operating point
    rows = []
    for m, v in [("nb_word_counts", "clean"), ("lr_char", "clean"), ("bilstm_finetuned", "clean"), ("xlmr", "clean"),
                 ("afroxlmr", "clean"), ("afroxlmr", "counterfactual"), ("ensemble", "clean"), ("ensemble_cf", "clean")]:
        for st, nm in (("test", "Swahili test"), ("chichewa/all", "Chichewa")):
            r = s[(s.model == m) & (s.variant == v) & (s.split == "template") & (s.set == st)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([label(m, v), nm] + [f"{r[c]:.1f}" if r["n_seeds"] > 1 else f"{r[c]:.0f}"
                                             for c in ("tp_op", "fn_op", "fp_op", "tn_op")])
    parts.append("## Confusion matrices (operating point, mean over seeds)\n\nTP = scams caught, FN = scams missed, "
                 "FP = genuine texts flagged, TN = genuine texts passed.\n\n" + md(rows, ["Model", "Set", "TP", "FN", "FP", "TN"]))

    # E12 out of sample: the ensemble on two fresh template-disjoint splits
    rows = []
    systems = [("lr_char", "clean"), ("afroxlmr", "clean"), ("ensemble_mean", "clean"), ("ensemble", "clean"),
               ("lr_char", "counterfactual"), ("afroxlmr", "counterfactual"), ("ensemble_cf", "clean"),
               ("ensemble_cf2", "clean")]
    for split in ("template", "template_r1", "template_r2"):
        row = [split.replace("template_r", "fresh split ").replace("template", "main split (designed on)")]
        for m, v in systems:
            row.append(cell(s, m, v, split, "test"))
        rows.append(row)
    parts.append("## E12 out of sample: the ensemble on fresh splits\n\nThe ensemble was designed after seeing the main "
                 "split's test errors. Two new template-disjoint splits (seeds 1 and 2), never used for design, check "
                 "whether the gain holds. Test scam F1.\n\n" + md(rows, ["Split"] + [label(m, v) for m, v in systems]))

    errors = config.RESULTS / "error_buckets.csv"
    if errors.exists():
        e = pd.read_csv(errors)
        parts.append("## Error buckets (seed 42 / single run, operating-point threshold)\n\n" + md(
            e.values.tolist(), list(e.columns)))

    (config.RESULTS / "tables.md").write_text("# Results tables\n\nGenerated by `python -m src.tables`.\n\n" + "\n\n".join(parts) + "\n")

    # Refresh the main table in docs/results.md between its markers.
    page = config.ROOT / "docs" / "results.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if page.exists() and start in page.read_text():
        text = page.read_text()
        main_table = parts[0].split("\n\n", 2)[2]
        text = text[: text.index(start) + len(start)] + "\n" + main_table + "\n" + text[text.index(end):]
        page.write_text(text)
    print((config.RESULTS / "tables.md").read_text()[:3000])


if __name__ == "__main__":
    main()
