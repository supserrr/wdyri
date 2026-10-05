"""Load the raw datasets, clean them and write the processed files with splits.

Usage:
    python -m src.data

Writes:
    data/processed/bongo.csv     masked Swahili SMS + template ids + both splits
    data/processed/chichewa.csv  masked Chichewa SMS (sealed test set for RQ3)
    results/data_stats.json      counts used in the report
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import f1_score

from . import config
from .preprocess import preprocess
from .split import check_disjoint, random_split, template_ids, template_split


def load_bongo_raw() -> pd.DataFrame:
    """BongoScam with labels as 0/1 and exact duplicates removed (raw text kept)."""
    raw = pd.read_csv(config.BONGO_CSV)
    df = raw.rename(columns={"Category": "label_name", "Sms": "raw_text"})
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["label"] = (df["label_name"].str.strip().str.lower() == "scam").astype(int)
    df = df.drop_duplicates("raw_text").reset_index(drop=True)
    df.insert(0, "id", [f"bongo_{i:04d}" for i in range(len(df))])
    return df[["id", "raw_text", "label"]]


def load_chichewa_raw() -> pd.DataFrame:
    """Chichewa D-CHI (balanced fraud/normal) plus the telco service texts (all normal)."""
    frames = []
    for sheet in ("D_CHI", "telcoSMS_CHI"):
        part = pd.read_excel(config.CHICHEWA_XLSX, sheet_name=sheet)
        part = part.dropna(subset=["Text", "Label"])
        part = part.assign(source=sheet, orig_id=part["ID"].astype(str))
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["raw_text"] = df["Text"].astype(str).str.strip()
    df["label"] = (df["Label"].str.strip().str.lower() == "fraud").astype(int)
    return df[["orig_id", "source", "raw_text", "label"]]


def build() -> dict:
    stats: dict = {}

    # --- BongoScam -----------------------------------------------------------
    raw = pd.read_csv(config.BONGO_CSV)
    stats["bongo_raw_rows"] = len(raw)
    stats["bongo_raw_labels"] = raw["Category"].value_counts().to_dict()
    bongo = load_bongo_raw()
    stats["bongo_exact_duplicates_dropped"] = len(raw) - len(bongo)
    bongo["text"] = bongo["raw_text"].map(preprocess)
    stats["bongo_labels_after_exact_dedup"] = bongo["label"].map(config.LABEL_NAMES).value_counts().to_dict()
    # Messages that differ only in number, amount or link are the same message
    # to every model. Keeping the copies would let one campaign fill the test
    # set (35 copies of one job scam landed in a single split), so keep one.
    dup = bongo["text"].duplicated()
    stats["bongo_identical_after_masking_dropped"] = int(dup.sum())
    stats["bongo_identical_after_masking_by_label"] = (
        bongo[dup]["label"].map(config.LABEL_NAMES).value_counts().to_dict())
    bongo = bongo[~dup].reset_index(drop=True)
    stats["bongo_rows"] = len(bongo)
    stats["bongo_labels"] = bongo["label"].map(config.LABEL_NAMES).value_counts().to_dict()
    stats["bongo_words_median"] = float(bongo["text"].str.split().str.len().median())
    # Descriptive only (a fixed rule needs no training): how far the number shortcut goes.
    has_number = bongo["text"].str.contains("<PHONE>|<URL>", regex=True)
    stats["share_with_phone_or_link"] = {
        "scam": float(has_number[bongo.label == 1].mean()), "not scam": float(has_number[bongo.label == 0].mean())}
    stats["phone_rule_f1_all_messages"] = float(f1_score(bongo["label"], has_number))
    words = bongo["text"].str.split().str.len()
    stats["words_min_scam"] = int(words[bongo.label == 1].min())
    stats["words_median"] = {"scam": float(words[bongo.label == 1].median()),
                             "not scam": float(words[bongo.label == 0].median())}

    bongo["template_id"] = template_ids(bongo["text"].tolist())
    sizes = bongo.groupby("template_id").size()
    stats["bongo_templates"] = int(len(sizes))
    stats["bongo_templates_with_2plus"] = int((sizes > 1).sum())
    stats["bongo_messages_in_shared_templates"] = int(sizes[sizes > 1].sum())
    stats["bongo_largest_template"] = int(sizes.max())
    stats["bongo_scam_messages_in_shared_templates"] = int(
        bongo[bongo["template_id"].isin(sizes[sizes > 1].index)]["label"].sum())

    labels = bongo["label"].to_numpy()
    bongo["split_random"] = random_split(labels)
    bongo["split_template"] = template_split(labels, bongo["template_id"].to_numpy())
    for col in ("split_random", "split_template"):
        counts = bongo.groupby([col, "label"]).size().unstack(fill_value=0)
        stats[col] = {part: {"not scam": int(r[0]), "scam": int(r[1])} for part, r in counts.iterrows()}
        stats[f"{col}_templates_crossing_parts"] = check_disjoint(bongo, col)

    # Masked text only: phone numbers and amounts never reach the repo.
    bongo[["id", "label", "text", "template_id", "split_random", "split_template"]].to_csv(
        config.DATA_PROCESSED / "bongo.csv", index=False)

    # --- Chichewa (sealed until RQ3) ----------------------------------------
    chi = load_chichewa_raw()
    stats["chichewa_rows_raw"] = len(chi)
    stats["chichewa_by_source"] = {
        src: part["label"].map({0: "normal", 1: "fraud"}).value_counts().to_dict()
        for src, part in chi.groupby("source")}
    stats["chichewa_exact_duplicates"] = int(chi["raw_text"].duplicated().sum())
    chi["text"] = chi["raw_text"].map(preprocess)
    dup = chi["text"].duplicated()
    stats["chichewa_identical_after_masking_dropped"] = int(dup.sum())
    chi = chi[~dup].reset_index(drop=True)
    stats["chichewa_rows"] = len(chi)
    stats["chichewa_by_source_after_dedup"] = {
        src: part["label"].map({0: "normal", 1: "fraud"}).value_counts().to_dict()
        for src, part in chi.groupby("source")}
    chi["template_id"] = template_ids(chi["text"].tolist())
    csizes = chi.groupby("template_id").size()
    stats["chichewa_templates"] = int(len(csizes))
    stats["chichewa_fraud_templates"] = int(chi[chi["label"] == 1]["template_id"].nunique())
    chi.insert(0, "id", [f"chi_{i:04d}" for i in range(len(chi))])
    chi[["id", "orig_id", "source", "label", "text", "template_id"]].to_csv(
        config.DATA_PROCESSED / "chichewa.csv", index=False)

    (config.RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def load_bongo() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PROCESSED / "bongo.csv")


def load_chichewa() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PROCESSED / "chichewa.csv")


def fewshot_ids(chichewa: pd.DataFrame, n: int, seed: int) -> set[str]:
    """n Chichewa messages (half fraud, half normal) to add to training (E9).

    Callers must drop every message that shares a template with these from the
    test set, so no rewrite of a training message is scored.
    """
    picked: list[str] = []
    for label in (1, 0):
        pool = chichewa[chichewa["label"] == label]
        templates = pool["template_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
        need = n // 2
        for tid in templates:
            if need <= 0:
                break
            members = pool[pool["template_id"] == tid]["id"].tolist()[:need]
            picked += members
            need -= len(members)
    return set(picked)


def split_frames(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and test frames for split 'random', 'template' or 'template_r<k>'.

    'template_r<k>' is a fresh template-disjoint split drawn with seed k (the same
    draw as repeat k of E2); it checks the E12 ensemble on test sets that played
    no part in designing it.
    """
    if split.startswith("template_r"):
        assignment = template_split(df["label"].to_numpy(), df["template_id"].to_numpy(),
                                    seed=int(split.removeprefix("template_r")))
        return tuple(df[assignment == part].reset_index(drop=True) for part in ("train", "val", "test"))
    col = f"split_{split}"
    return tuple(df[df[col] == part].reset_index(drop=True) for part in ("train", "val", "test"))


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
