"""Publish the fine-tuned model and the Gradio app to the Hugging Face Hub.

Needs a Hugging Face account and a write token:
    hf auth login
    python scripts/export_app.py
    python scripts/publish_hf.py --user supserrr

Creates (or updates):
    https://huggingface.co/<user>/wdyri-afroxlmr          the model the app loads by name
    https://huggingface.co/spaces/<user>/wdyri            the public Gradio app
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

MODEL_CARD = """---
language: [sw]
license: mit
base_model: {base}
pipeline_tag: text-classification
tags: [scam-detection, sms, swahili, mobile-money]
datasets: [henrydioniz/swahili-sms-detection-dataset]
---

# WDYRI: Swahili mobile-money scam SMS classifier

{base} fine-tuned to flag Tanzanian Swahili SMS that try to steal money
("ni tumie kwa namba hii" scams, fake prizes, fake agents, landlord impersonation).
Trained with number-balanced counterfactual edits so that the presence of a phone
number is not, by itself, evidence of a scam (experiment E11 in the repository).

* Labels: `0` = not scam, `1` = scam. Use the decision threshold {threshold} (chosen on validation to
  catch at least 95% of scams), not 0.5, to match the reported results.
* Input must go through the project's `preprocess()` (phone numbers, amounts and links are masked).
* Training data: the BongoScam dataset (MIT licence), template-disjoint split, {n_train} messages.
* Known blind spots: impersonation scams with no money words (landlord "this is my new number"),
  which the companion character n-gram model catches; Chichewa and other languages.
  See the repository for the full evaluation.

Code and results: https://github.com/supserrr/wdyri
"""

SPACE_CARD = """---
title: WDYRI Swahili Scam SMS Checker
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

Paste a Swahili SMS to check it for mobile-money scam patterns. Code: https://github.com/supserrr/wdyri
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="supserrr")
    parser.add_argument("--model-repo", default="wdyri-afroxlmr")
    parser.add_argument("--space", default="wdyri")
    args = parser.parse_args()
    api = HfApi()
    settings = json.loads((ROOT / "app" / "settings.json").read_text())
    stats = json.loads((config.RESULTS / "data_stats.json").read_text())
    train = stats["split_template"]["train"]
    model_dir = config.MODELS / settings["transformer_run"]
    base = config.TRANSFORMERS[settings["transformer_run"].split("__")[0]]

    model_id = f"{args.user}/{args.model_repo}"
    api.create_repo(model_id, exist_ok=True)
    # Marker the app checks, so it never serves an older upload under the wrong name.
    (model_dir / "wdyri_run.json").write_text(json.dumps({"run": settings["transformer_run"]}) + "\n")
    (model_dir / "README.md").write_text(MODEL_CARD.format(
        base=base, threshold=settings["threshold_transformer"], n_train=train["scam"] + train["not scam"]))
    api.upload_folder(repo_id=model_id, folder_path=model_dir, commit_message="Upload fine-tuned model")
    print("model:", f"https://huggingface.co/{model_id}")

    space_id = f"{args.user}/{args.space}"
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("app.py", "settings.json", "baseline_lr_char.joblib", "requirements.txt"):
            shutil.copy(ROOT / "app" / name, tmp / name)
        (tmp / "src").mkdir()
        for name in ("__init__.py", "preprocess.py", "perturb.py"):
            shutil.copy(ROOT / "src" / name, tmp / "src" / name)
        (tmp / "README.md").write_text(SPACE_CARD)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp, commit_message="Deploy app")
    api.add_space_variable(space_id, "WDYRI_MODEL", model_id)
    print("space:", f"https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
