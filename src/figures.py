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
import numpy as np
import pandas as pd

from . import config

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
    ax.legend(loc="lower right", ncol=2)
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
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("RQ2: word-based models break under disguise; transformers read it as suspicious",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "rq2_attacks")


def rq2_defences(summary: pd.DataFrame) -> None:
    """Scam recall on fully attacked test sets: no defence, normalisation, adversarial training."""
    models = ["nb_word_counts", "lr_char", "bilstm_finetuned", "afroxlmr"]
    s = summary[(summary["split"] == "template")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    titles = {"lookalike": "Lookalike letters", "structural": "Split / joined words", "codeswitch": "Code-switching"}
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
    ends = []
    for m in ["lr_char", "afroxlmr", "xlmr"]:
        ys = []
        for v in ("clean", "fewshot20", "fewshot50"):
            r = s[(s["model"] == m) & (s["variant"] == v) & (s["set"] == "chichewa/all")]["f1"]
            ys.append(r.mean() if len(r) else np.nan)
        if np.isnan(ys[1:]).all():
            continue
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
    ax2.set_xlabel("Chichewa messages added to training")
    ax2.set_ylabel("Fraud F1")
    ax2.set_title("Few-shot (E9)", loc="left")
    ax2.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("RQ3: Swahili knowledge transfers only partly to Chichewa", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "rq3_transfer")


def main() -> None:
    summary = pd.read_csv(config.RESULTS / "experiments.csv")
    by_seed = pd.read_csv(config.RESULTS / "experiments_by_seed.csv")
    rq1_repeated_splits(by_seed)
    rq2_attacks(summary)
    rq2_defences(summary)
    rq3_transfer(summary)
    print("figures:", sorted(p.name for p in config.FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
