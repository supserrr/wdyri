"""Report figures, drawn from results/experiments*.csv.

Usage:
    python -m src.figures

Colours follow the model, never its rank: every figure gives a model the same
hue (validated colour-blind-safe categorical palette, fixed order).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from . import config, runs
from .evaluate import fewshot_pairs

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODELS = {  # key: (label, colour)
    "nb_word_counts": ("Naive Bayes (word counts)", "#2a78d6"),
    "lr_char": ("Log. regression (char 2-5)", "#eb6834"),
    "bilstm_finetuned": ("BiLSTM + fastText", "#1baf7a"),
    "xlmr": ("XLM-R base", "#eda100"),
    "afroxlmr": ("AfroXLMR base", "#e87ba4"),
    "lr_word": ("Log. regression (words)", "#008300"),
    "nb_char": ("Naive Bayes (char)", "#4a3aa7"),
    "majority": ("Majority class", MUTED),
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def save(fig, name: str) -> None:
    fig.savefig(config.FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rq1_repeated_splits(by_seed: pd.DataFrame) -> None:
    """Test F1 over 10 redraws of each split type, per classical model."""
    rep = by_seed[(by_seed["variant"] == "repeat") & (by_seed["set"] == "test")]
    order = ["nb_word_counts", "nb_char", "lr_word", "lr_char"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rng = np.random.default_rng(0)
    styles = {"random": ("Random split", "#2a78d6", -0.17), "template": ("Template-disjoint split", "#eb6834", 0.17)}
    for split, (label, colour, dx) in styles.items():
        for i, model in enumerate(order):
            vals = rep[(rep["model"] == model) & (rep["split"] == split)]["f1"].to_numpy()
            x = i + dx + rng.uniform(-0.06, 0.06, len(vals))
            ax.scatter(x, vals, s=22, color=colour, edgecolor=SURFACE, linewidth=1,
                       label=label if i == 0 else None, zorder=3)
            ax.hlines(vals.mean(), i + dx - 0.12, i + dx + 0.12, color=INK, linewidth=2, zorder=4)
    ax.set_xticks(range(len(order)), [MODELS[m][0].replace(" (", "\n(") for m in order])
    ax.set_ylabel("Scam F1 on test")
    ax.set_title("RQ1: random splits flatter character n-gram models most", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)   # below the axis, clear of every point
    ax.grid(axis="x", visible=False)
    save(fig, "rq1_repeated_splits")


def rq2_attacks(summary: pd.DataFrame) -> None:
    """Relative F1 drop per attack and intensity (0 = no harm; below 0 = attack helped detection)."""
    models = ["nb_word_counts", "lr_word", "lr_char", "bilstm_finetuned", "xlmr", "afroxlmr"]
    s = summary[(summary["split"] == "template") & (summary["variant"] == "clean")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    xs, xl = [0, 1, 2, 3], ["clean", "1 word", "3 words", "all"]
    titles = {"lookalike": "Lookalike letters", "structural": "Split / joined words", "codeswitch": "Code-switching"}
    for ax, (attack, title) in zip(axes, titles.items()):
        for m in models:
            ys = [0.0] + [s[(s["model"] == m) & (s["set"] == f"test_{attack}_{k}")]["rel_f1_drop"].mean() * 100
                          for k in ("1", "3", "all")]
            if np.all(np.isnan(ys[1:])):
                continue
            label, colour = MODELS[m]
            ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(xs, xl)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Relative F1 drop (%)")
    fig.suptitle("RQ2: disguise breaks word-based models; transformers are not hurt (see the E6 control for why)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()   # one legend under the panels, clear of the lines
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), fontsize=7.5)
    save(fig, "rq2_attacks")


def rq2_defences(summary: pd.DataFrame) -> None:
    """Scam recall on fully attacked test sets: no defence, normalisation, adversarial training."""
    models = ["nb_word_counts", "lr_char", "bilstm_finetuned", "afroxlmr"]
    s = summary[(summary["split"] == "template")]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), sharey=True)
    titles = {"lookalike": "Lookalike letters", "structural": "Split / joined words", "codeswitch": "Code-switching",
              "unseen": "Held-out lookalikes"}
    defences = [("No defence", "clean", "", "#898781"), ("Normalisation", "clean", "+norm", "#2a78d6"),
                ("Adversarial training", "advtrain", "", "#eb6834")]
    width = 0.26
    for ax, (attack, title) in zip(axes, titles.items()):
        for j, (label, variant, suffix, colour) in enumerate(defences):
            vals = [s[(s["model"] == m) & (s["variant"] == variant) & (s["set"] == f"test_{attack}_all{suffix}")]["recall"].mean()
                    for m in models]
            ax.bar(np.arange(len(models)) + (j - 1) * (width + 0.02), vals, width=width, color=colour,
                   label=label if attack == "lookalike" else None, zorder=3)
        clean = [s[(s["model"] == m) & (s["variant"] == "clean") & (s["set"] == "test")]["recall"].mean() for m in models]
        ax.hlines(clean, np.arange(len(models)) - 0.45, np.arange(len(models)) + 0.45, color=INK, linewidth=1.2,
                  zorder=4, label="Clean test (no attack)" if attack == "lookalike" else None)
        ax.set_xticks(range(len(models)), [MODELS[m][0].split(" (")[0].replace("Log. regression", "Log. reg.") for m in models],
                      rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of disguised scams caught")
    fig.legend(loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.04), fontsize=8)
    fig.suptitle("E7: defences on the strongest attacks", x=0.01, y=1.0, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "rq2_defences")


def rq3_transfer(summary: pd.DataFrame) -> None:
    """Zero-shot Chichewa F1 per model, and how few Chichewa examples close the gap."""
    s = summary[summary["split"] == "template"]
    models = ["majority", "nb_word_counts", "lr_char", "bilstm_finetuned", "xlmr", "afroxlmr"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    vals = [s[(s["model"] == m) & (s["variant"] == "clean") & (s["set"] == "chichewa/all")] for m in models]
    f1 = [v["f1"].mean() if len(v) else np.nan for v in vals]
    sd = [v["f1_sd"].mean() if len(v) else np.nan for v in vals]
    auc = [v["pr_auc"].mean() if len(v) else np.nan for v in vals]
    ax1.barh(range(len(models)), f1, xerr=np.nan_to_num(sd), color=[MODELS[m][1] for m in models],
             height=0.6, zorder=3, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2}, label="F1 at 0.5")
    ax1.scatter(auc, range(len(models)), marker="D", s=26, facecolor=SURFACE, edgecolor=INK, linewidth=1.2,
                zorder=4, label="PR-AUC")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=7.5)
    for i, v in enumerate(f1):
        if not np.isnan(v):
            ax1.text(np.nanmax([v, auc[i]]) + 0.03, i, f"F1 {v:.2f}", va="center", color=INK_2, fontsize=8)
    ax1.set_yticks(range(len(models)), [MODELS[m][0] for m in models])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    n_chi = int(s[(s["model"] == "majority") & (s["set"] == "chichewa/all")]["n"].iloc[0]) if "n" in s else 0
    ax1.set_xlabel(f"Fraud detection on {n_chi} Chichewa SMS (no Chichewa training)")
    ax1.set_title("Zero-shot transfer", loc="left")
    ax1.grid(axis="y", visible=False)
    # Each few-shot run is compared with its zero-shot model on the same messages (its
    # added examples' templates removed); the zero-shot point averages those matched scores.
    pairs = fewshot_pairs(runs.load_all())
    ends = []
    for m in ["lr_char", "afroxlmr", "xlmr"]:
        p = pairs[pairs["model"] == m]
        if p.empty:
            continue
        ys = [p["zero_f1"].mean()] + [p[p["n_shots"] == n]["few_f1"].mean() for n in (20, 50)]
        ax2.plot([0, 20, 50], ys, color=MODELS[m][1], linewidth=2, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, label=MODELS[m][0])
        ends.append([ys[-1], ys[-1]])
    # End labels, nudged apart so close values stay readable.
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.025)
    for value, y in ends:
        ax2.text(51.5, y, f"{value:.2f}", va="center", color=INK_2, fontsize=8)
    ax2.set_xticks([0, 20, 50])
    ax2.set_xlim(-3, 58)
    ax2.set_ylim(0.5, 1)
    ax2.set_xlabel("Chichewa messages added to training\n(each point on the same messages as its zero-shot match)")
    ax2.set_ylabel("Fraud F1")
    ax2.set_title("Few-shot (E9)", loc="left")
    ax2.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("RQ3: Swahili knowledge transfers only partly to Chichewa", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "rq3_transfer")


def e10_stress(stress: pd.DataFrame) -> None:
    """Minimal pairs: how far one placeholder moves each model (before = grey, after = orange)."""
    rows = [("phone_rule", "clean", "Phone rule"), ("nb_word_counts", "clean", "Naive Bayes (word counts)"),
            ("lr_word", "clean", "Log. regression (words)"), ("lr_char", "clean", "Log. regression (char 2-5)"),
            ("bilstm_finetuned", "clean", "BiLSTM + fastText"), ("xlmr", "clean", "XLM-R base"),
            ("afroxlmr", "clean", "AfroXLMR base"), ("afroxlmr", "counterfactual", "AfroXLMR, number-balanced"),
            ("ensemble", "clean", "Ensemble (char LR OR AfroXLMR)"),
            ("ensemble_cf", "clean", "Ensemble with number-balanced AfroXLMR"),
            ("ensemble_cf2", "clean", "Ensemble, both members number-balanced")]
    rows = [r for r in rows if not stress[(stress.model == r[0]) & (stress.variant == r[1])].empty]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(rows) + 1.2), sharey=True)
    panels = [("false_alarm_clean", "false_alarm_+phone", "Genuine text + a phone number\n(share wrongly flagged)"),
              ("miss_clean", "miss_-number", "Scam with its number deleted\n(share missed)")]
    after_colour = "#eb6834"
    for ax, (before, after, title) in zip(axes, panels):
        for i, (m, v, _) in enumerate(rows):
            r = stress[(stress.model == m) & (stress.variant == v)].iloc[0]
            ax.plot([r[before], r[after]], [i, i], color=AXIS, linewidth=2, zorder=2)
            ax.scatter(r[before], i, s=40, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label="As written" if i == 0 else None)
            ax.scatter(r[after], i, s=40, color=after_colour, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                       label="After the one-token edit" if i == 0 else None)
            if r[after] > 0.85:  # label inside the line, left of the dot, so it never collides
                ax.text(r[after] - 0.03, i - 0.32, f"{r[after]:.0%}", ha="right", va="center", color=INK_2, fontsize=7.5)
            else:
                ax.text(r[after] + 0.03, i, f"{r[after]:.0%}", va="center", color=INK_2, fontsize=7.5)
        ax.set_xlim(-0.02, 1.08)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r[2] for r in rows])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("E10: one placeholder flips AfroXLMR; number-balanced training (E11) removes the shortcut",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "e10_stress_tests")


def training_curves() -> None:
    """Validation F1 and loss per epoch for every logged transformer run on the template split."""
    log = pd.read_json(config.RESULTS / "transformer_log.jsonl", lines=True)
    styles = {("xlmr", "clean"): ("XLM-R", MODELS["xlmr"][1], "-"),
              ("afroxlmr", "clean"): ("AfroXLMR", MODELS["afroxlmr"][1], "-"),
              ("afroxlmr", "counterfactual"): ("AfroXLMR, number-balanced", "#9c3d6e", "--")}
    log = log[(log["split"] == "template") & log.apply(lambda r: (r["model"], r["variant"]) in styles, axis=1)]
    log = log.assign(order=log.apply(lambda r: list(styles).index((r["model"], r["variant"])), axis=1)).sort_values(["order", "seed"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})
    seen = set()
    for row, (_, run) in enumerate(log.iterrows()):
        label, colour, dash = styles[(run["model"], run["variant"])]
        hist = pd.DataFrame(run["history"])
        best = hist.sort_values(["val_f1", "val_loss"], ascending=[False, True]).iloc[0]   # the epoch kept
        # (a) one row per run: dark = F1 1.000, pale = 0.994 (one validation error in 152), ring = epoch kept
        for _, h in hist.iterrows():
            ax1.scatter(h["epoch"], row, s=70, color=colour, alpha=1.0 if h["val_f1"] >= 0.9999 else 0.3, zorder=3)
        ax1.scatter(best["epoch"], row, s=170, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        # (b) validation loss per epoch
        name = None if label in seen else label
        seen.add(label)
        ax2.plot(hist["epoch"], hist["val_loss"], color=colour, linestyle=dash, linewidth=1.6, marker="o", markersize=3.5,
                 label=name)
        ax2.scatter([best["epoch"]], [best["val_loss"]], s=60, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    ax1.set_yticks(range(len(log)), [f"{styles[(r.model, r.variant)][0]}, seed {r.seed}" for r in log.itertuples()])
    ax1.invert_yaxis()
    ax1.set_xticks(range(1, 6))
    ax1.set_xlim(0.5, 5.5)
    ax1.set_xlabel("Epoch (dark: F1 1.000; pale: 0.994, one error in 152)")
    ax1.set_title("(a) Validation F1 per run", loc="left")
    ax1.grid(axis="y", visible=False)
    ax2.set_yscale("log")
    ax2.set_xticks(range(1, 6))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss (log scale)")
    ax2.set_title("(b) Validation loss", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5)   # below, clear of the lines
    fig.suptitle("Validation F1 is 0.99 to 1.00 from the first epoch in every run; rings mark the epoch kept",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "training_curves")


def main() -> None:
    summary = pd.read_csv(config.RESULTS / "experiments.csv")
    by_seed = pd.read_csv(config.RESULTS / "experiments_by_seed.csv")
    rq1_repeated_splits(by_seed)
    rq2_attacks(summary)
    rq2_defences(summary)
    rq3_transfer(summary)
    training_curves()
    stress = config.RESULTS / "stress_tests.csv"
    if stress.exists():
        e10_stress(pd.read_csv(stress))
    print("figures:", sorted(p.name for p in config.FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
