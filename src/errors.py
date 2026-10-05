"""Sort every error of the main models into one of the report's error buckets.

Usage:
    python -m src.errors

Writes results/error_examples.csv (every error with its bucket) and
results/error_buckets.csv (counts per model and bucket).

Buckets, checked in this order. Names describe the message, not a proven cause:
    Ambiguous                  five words or fewer: needs sender or conversation context
    Telco service text flagged a genuine Chichewa telco message (balance, bundle) marked as scam
    Service-like text flagged  another genuine text with money or telco words marked as scam
    Genuine text flagged       any other false alarm
    Chichewa fraud missed      a Chichewa fraud the Swahili-trained model let through
                               (local_scheme marks Malawi-specific scripts: DODMA relief,
                               "Foundation" grants, Miracle Money)
    Obfuscation missed         an attacked scam whose clean version the model caught
    Impersonation missed       a scam that poses as a landlord, relative or boss
    Other scam missed          any other missed scam
"""
from __future__ import annotations

import re

import pandas as pd

from . import config, runs
from .metrics import recall_threshold

SERVICE = re.compile(
    r"salio|balance|balansi|bundle|kifurushi|\bMB\b|\bGB\b|muamala|umepokea|umetuma|akasitomala|customer"
    r"|\*\d+#|<AMOUNT>|m-?pesa|tigo ?pesa|airtel ?money|halopesa|mpamba|ndalama", re.IGNORECASE)
IMPERSONATION = re.compile(
    r"mwenye nyumba|landlord|\bmama\b|\bbaba\b|\bkaka\b|\bdada\b|mjomba|shangazi|bibi|babu|ndugu|rafiki|mjukuu"
    r"|namba yangu|ni mimi|mimi ni|\bboss\b|mkurugenzi|\bHR\b", re.IGNORECASE)

# The runs analysed: one seed per neural model (seed 42), the classical models as trained.
MAIN_RUNS = [
    ("nb_word_counts", "clean", 0),
    ("lr_char", "clean", 0),
    ("bilstm_finetuned", "clean", 42),
    ("xlmr", "clean", 42),
    ("afroxlmr", "clean", 42),
]
ATTACK_SETS = [f"test_{a}_all" for a in ("lookalike", "structural", "codeswitch")]


LOCAL_SCHEME = re.compile(r"dodma|foundation|miracle money|mtukula|grant|thandizo", re.IGNORECASE)


def bucket(text: str, label: int, set_name: str, clean_correct: bool, source: str = "") -> str:
    if len(text.split()) <= 5:
        return "Ambiguous"
    if label == 0:
        if source == "telcoSMS_CHI":
            return "Telco service text flagged"
        return "Service-like text flagged" if SERVICE.search(text) else "Genuine text flagged"
    if set_name == "chichewa":
        return "Chichewa fraud missed"
    if set_name.startswith("test_") and clean_correct:
        return "Obfuscation missed"
    return "Impersonation missed" if IMPERSONATION.search(text) else "Other scam missed"


def main() -> None:
    ev = pd.read_csv(config.DATA_PROCESSED / "eval_template.csv", keep_default_na=False)
    text = ev.set_index(["set", "id"])["text"]
    source = pd.read_csv(config.DATA_PROCESSED / "chichewa.csv").set_index("id")["source"]
    rows = []
    for model, variant, seed in MAIN_RUNS:
        path = config.PREDICTIONS / f"{runs.run_name(model, variant, 'template', seed)}.csv"
        if not path.exists():
            print("missing", path.name)
            continue
        pred = pd.read_csv(path, keep_default_na=False)
        val = pred[pred["set"] == "val"]
        thr = recall_threshold(val["label"], val["prob"])
        pred["pred"] = (pred["prob"] >= thr).astype(int)
        clean_ok = pred[pred["set"] == "test"].set_index("id").eval("pred == label")
        for set_name in ["test", "chichewa", *ATTACK_SETS]:
            part = pred[(pred["set"] == set_name) & (pred["pred"] != pred["label"])]
            if set_name in ATTACK_SETS:
                part = part[part["label"] == 1]  # genuine texts are not attacked; their errors repeat "test"
            for r in part.itertuples():
                t = text[(set_name, r.id)]
                ok = bool(clean_ok.get(r.id, False))
                src = source.get(r.id, "") if set_name == "chichewa" else ""
                rows.append({"model": model, "set": set_name, "id": r.id, "label": r.label,
                             "prob": round(r.prob, 3), "threshold": round(thr, 3),
                             "bucket": bucket(t, r.label, set_name, ok, src),
                             "local_scheme": bool(set_name == "chichewa" and r.label == 1 and LOCAL_SCHEME.search(t)),
                             "text": t})
    errors = pd.DataFrame(rows)
    errors["where"] = errors["set"].map(lambda s: "Swahili test" if s == "test"
                                        else "Chichewa" if s == "chichewa" else "Attacked test (all words)")
    errors.to_csv(config.RESULTS / "error_examples.csv", index=False)
    counts = errors.pivot_table(index=["where", "bucket"], columns="model", values="id",
                                aggfunc="count", fill_value=0)
    counts = counts[[m for m, _, _ in MAIN_RUNS if m in counts.columns]]
    counts.to_csv(config.RESULTS / "error_buckets.csv")
    print(counts.to_string())


if __name__ == "__main__":
    main()
