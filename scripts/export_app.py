"""Prepare app/ for deployment: thresholds and model choice from the experiments.

Usage:
    python scripts/export_app.py [--model afroxlmr] [--seed 42]

Writes app/settings.json. The decision thresholds are the validation-chosen
operating points (catch >= 95% of validation scams), the same ones the
report's *_op metrics use, so the app behaves like the evaluated models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, runs  # noqa: E402
from src.metrics import recall_threshold  # noqa: E402

NAMES = {"afroxlmr": "AfroXLMR base, fine-tuned", "xlmr": "XLM-R base, fine-tuned"}


def threshold(model: str, seed: int) -> float:
    pred = pd.read_csv(config.PREDICTIONS / f"{runs.run_name(model, 'clean', 'template', seed)}.csv",
                       keep_default_na=False)
    val = pred[pred["set"] == "val"]
    return recall_threshold(val["label"], val["prob"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="afroxlmr", choices=list(NAMES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = json.loads((config.RESULTS / "data_stats.json").read_text())
    train = stats["split_template"]["train"]
    settings = {
        "transformer_name": NAMES[args.model],
        "transformer_run": runs.run_name(args.model, "clean", "template", args.seed),
        "threshold_transformer": round(threshold(args.model, args.seed), 4),
        "threshold_baseline": round(threshold("lr_char", 0), 4),
        "train_note": (f"{train['scam'] + train['not scam']} Tanzanian Swahili SMS from the BongoScam "
                       f"dataset ({train['scam']} scam, {train['not scam']} genuine)."),
    }
    (ROOT / "app" / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
