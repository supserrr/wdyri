"""Prepare app/ for deployment: thresholds and model choice from the experiments.

Usage:
    python scripts/export_app.py [--model afroxlmr] [--variant clean|counterfactual]
                                 [--baseline-variant clean|counterfactual] [--seed 42]

Writes app/settings.json. The decision thresholds are the validation-chosen
operating points (catch >= 95% of validation scams), the same ones the
report's *_op metrics use, so the app behaves like the evaluated models.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, runs  # noqa: E402
from src.metrics import recall_threshold  # noqa: E402

NAMES = {"afroxlmr": "AfroXLMR base, fine-tuned", "xlmr": "XLM-R base, fine-tuned"}


def threshold(model: str, seed: int, variant: str = "clean") -> float:
    pred = pd.read_csv(config.PREDICTIONS / f"{runs.run_name(model, variant, 'template', seed)}.csv",
                       keep_default_na=False)
    val = pred[pred["set"] == "val"]
    return recall_threshold(val["label"], val["prob"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="afroxlmr", choices=list(NAMES))
    parser.add_argument("--variant", default="clean", choices=["clean", "counterfactual"],
                        help="transformer variant")
    parser.add_argument("--baseline-variant", default="clean", choices=["clean", "counterfactual"],
                        help="char n-gram model variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = json.loads((config.RESULTS / "data_stats.json").read_text())
    train = stats["split_template"]["train"]
    settings = {
        "transformer_name": NAMES[args.model] + (", number-balanced" if args.variant == "counterfactual" else ""),
        "transformer_run": runs.run_name(args.model, args.variant, "template", args.seed),
        "threshold_transformer": round(threshold(args.model, args.seed, args.variant), 4),
        "baseline_run": runs.run_name("lr_char", args.baseline_variant, "template", 0),
        "threshold_baseline": round(threshold("lr_char", 0, args.baseline_variant), 4),
        "train_note": (f"{train['scam'] + train['not scam']} Tanzanian Swahili SMS from the BongoScam "
                       f"dataset ({train['scam']} scam, {train['not scam']} genuine)."),
    }
    source = config.MODELS / ("lr_char__template.joblib" if args.baseline_variant == "clean"
                              else "lr_char__counterfactual__template.joblib")
    shutil.copy(source, ROOT / "app" / "baseline_lr_char.joblib")
    (ROOT / "app" / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
