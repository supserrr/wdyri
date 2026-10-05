"""Train rungs 0-2 and run every classical experiment (E1, E2, E3, E6, E7, E8, E9).

Usage:
    python -m src.train_classical

Steps:
0. Reproduce BongoScam's published Naive Bayes setup.     (E1)
1. Both splits x {majority, NB, LR} x {word, char} x {masked, unmasked}: tune on
   validation, predict validation and test.            (E1, E2, E3)
2. Use the masked character LR on the template split to build the attacked and
   Chichewa evaluation sets (src.eval_sets).
3. Score every masked template-split model on all evaluation sets. (E6, E8, E7 normalisation)
4. Retrain NB and LR with perturbed training copies.  (E7 adversarial training)
5. Retrain LR with 20 or 50 Chichewa messages added.  (E9)
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from . import config, eval_sets, runs, variants
from .classical import GRIDS, build, fit_tuned, scam_proba, top_features
from .data import load_bongo, load_bongo_raw, split_frames
from .perturb import adversarial_copies
from .preprocess import preprocess
from .split import random_split, template_split

CONFIGS = [  # (model, features)
    ("majority", "word_counts"),
    ("nb", "word_counts"),
    ("nb", "char"),
    ("lr", "word"),
    ("lr", "char"),
]
SEED = 0  # classical models are deterministic; one run each


def name(model: str, features: str) -> str:
    return model if model == "majority" else f"{model}_{features}"


def main() -> None:
    bongo = load_bongo()
    unmasked = load_bongo_raw().set_index("id")["raw_text"].map(lambda t: preprocess(t, do_mask=False))
    tuning: dict = {}
    fitted: dict = {}

    # 0. E1 reproduction exactly as published: all 1,508 rows, no cleaning,
    #    80/20 split with random_state=42, default CountVectorizer + MultinomialNB.
    raw = pd.read_csv(config.BONGO_CSV)
    raw = raw.assign(id=[f"raw_{i:04d}" for i in range(len(raw))], text=raw["Sms"].astype(str),
                     label=(raw["Category"] == "scam").astype(int))
    tr_raw, te_raw = train_test_split(raw, test_size=0.2, random_state=42)
    pipe = make_pipeline(CountVectorizer(), MultinomialNB()).fit(tr_raw.text, tr_raw.label)
    runs.save("nb_word_counts", "published", "published", SEED, te_raw.assign(set="test"),
              scam_proba(pipe, te_raw.text))
    tuning["published_accuracy"] = float(pipe.score(te_raw.text, te_raw.label))
    print("published setup, accuracy:", round(tuning["published_accuracy"], 4))

    # 1. E1-E3 on clean validation and test sets
    for split in ("random", "template"):
        train, val, test = split_frames(bongo, split)
        for masked in (True, False):
            variant = "clean" if masked else "nomask"
            if not masked:
                train, val, test = (f.assign(text=f["id"].map(unmasked)) for f in (train, val, test))
            for model, features in CONFIGS:
                if model == "majority" and not masked:
                    continue
                # Template split: template-disjoint CV folds. Random split: plain folds, as in prior work.
                cv_groups = train.template_id if split == "template" else None
                pipe, info = fit_tuned(model, features, train.text, train.label, val.text, val.label, cv_groups)
                key = name(model, features)
                tuning[f"{key}/{variant}/{split}"] = info
                if masked and split == "template" and model != "majority":
                    # Sensitivity, for reporting only (selection never sees the test set):
                    # test F1 of every grid value, refit on the full training set.
                    (param, values), = GRIDS[model].items()
                    info["test_by_value"] = {}
                    for v in values:
                        m = build(model, features).set_params(**{param: v}).fit(train.text, train.label)
                        prob = scam_proba(m, test.text)
                        info["test_by_value"][str(v)] = {
                            "f1": round(f1_score(test.label, prob >= 0.5), 4),
                            "pr_auc": round(average_precision_score(test.label, prob), 4)}
                frame = pd.concat([val.assign(set="val"), test.assign(set="test")], ignore_index=True)
                runs.save(key, variant, split, SEED, frame, scam_proba(pipe, frame.text))
                if masked:
                    fitted[(key, split)] = pipe
                    joblib.dump(pipe, config.MODELS / f"{key}__{split}.joblib")
        print(f"{split}: trained {len(CONFIGS)} configs x 2 masking settings")

    # 1b. E2 repeated: one split is ~150 test messages, so redraw both kinds of
    #     split with 10 seeds and compare the distributions, not single numbers.
    labels, groups = bongo["label"].to_numpy(), bongo["template_id"].to_numpy()
    for split_seed in range(10):
        parts = {"random": random_split(labels, seed=split_seed),
                 "template": template_split(labels, groups, seed=split_seed)}
        for split, assignment in parts.items():
            train, val, test = (bongo[assignment == p] for p in ("train", "val", "test"))
            for model, features in CONFIGS:
                cv_groups = train.template_id if split == "template" else None
                pipe, _ = fit_tuned(model, features, train.text, train.label, val.text, val.label, cv_groups)
                runs.save(name(model, features), "repeat", split, split_seed, test.assign(set="test"),
                          scam_proba(pipe, test.text))
    print("repeated splits: 10 seeds x 2 split types")

    # 2. Attacked and Chichewa evaluation sets. Trigger words come from a fixed
    #    "attacker": char LR with C=100, trained on the template split's training
    #    data. It is kept separate from the tuned models so the attack text never
    #    changes when a grid changes (it is white-box only for a char LR with C=100).
    train_t, _, _ = split_frames(bongo, "template")
    attacker = build("lr", "char").set_params(clf__C=100.0).fit(train_t.text, train_t.label)
    joblib.dump(attacker, config.MODELS / "attacker_lr_char.joblib")
    scorer = lambda texts: scam_proba(attacker, texts)  # noqa: E731
    # A second, different attacker (word-count Naive Bayes, alpha 1) for the transfer-attack check.
    attacker_nb = build("nb", "word_counts").set_params(clf__alpha=1.0).fit(train_t.text, train_t.label)
    eval_sets.build("random")
    ev = eval_sets.build("template", scorer=scorer,
                         transfer_scorer=lambda texts: scam_proba(attacker_nb, texts))
    print("evaluation sets:", ev["set"].nunique(), "sets,", len(ev), "rows")

    # 3. Every masked template-split model on every evaluation set
    for model, features in CONFIGS:
        key = name(model, features)
        runs.save(key, "clean", "template", SEED, ev, scam_proba(fitted[(key, "template")], ev.text))

    # 4. E7 adversarial training: 25% of training scams get one perturbed copy.
    #    Saved so the BiLSTM and transformers train on exactly the same copies.
    train, _, _ = split_frames(bongo, "template")
    scams = train[train.label == 1].text.tolist()
    copies, sources = adversarial_copies(scams, scorer, share=0.25, seed=config.SPLIT_SEED, return_sources=True)
    source_templates = train[train.label == 1].template_id.to_numpy()[sources]
    tmp = variants.ADV_COPIES.with_suffix(".tmp")
    pd.DataFrame({"text": copies, "label": 1, "template_id": source_templates}).to_csv(tmp, index=False)
    tmp.replace(variants.ADV_COPIES)
    tuning["adversarial_copies"] = {"n_copies": len(copies), "n_train": len(train),
                                    "share_of_augmented_train": len(copies) / (len(train) + len(copies))}

    # E3b shortcut checks: does a one-line rule already solve the task?
    #   phone rule:  "contains a phone number or link"
    #   length rule: "has at least k words", k chosen on the training set
    rule = ev["text"].str.contains("<PHONE>|<URL>", regex=True).astype(float)
    runs.save("phone_rule", "clean", "template", SEED, ev, rule.to_numpy())
    train_t, _, _ = split_frames(bongo, "template")
    words = train_t.text.str.split().str.len()
    k_best = max(range(1, 61), key=lambda k: (f1_score(train_t.label, words >= k), -k))
    tuning["length_rule_k"] = k_best
    runs.save("length_rule", "clean", "template", SEED, ev,
              (ev["text"].str.split().str.len() >= k_best).astype(float).to_numpy())

    # 4-5. E7 adversarial training (NB, LR), E9 few-shot Chichewa (LR), E3b placeholders removed,
    #      E11 number-balanced counterfactual training
    jobs = [("nb", "word_counts", "advtrain", SEED), ("lr", "char", "advtrain", SEED)]
    jobs += [(m, f, "strip", SEED) for m, f in CONFIGS if m != "majority"]
    jobs += [(m, f, "counterfactual", SEED) for m, f in CONFIGS if m != "majority"]  # E11
    jobs += [("lr", "char", f"fewshot{n}", seed) for n in (20, 50) for seed in config.SEEDS]
    for model, features, variant, seed in jobs:
        train, val, rows = variants.frames("template", variant, seed)
        pipe, info = fit_tuned(model, features, train.text, train.label, val.text, val.label, train.template_id)
        tuning[f"{name(model, features)}/{variant}/template/s{seed}"] = info
        runs.save(name(model, features), variant, "template", seed, rows, scam_proba(pipe, rows.text))
        if variant == "counterfactual":
            joblib.dump(pipe, config.MODELS / f"{name(model, features)}__counterfactual__template.joblib")

    # E12 check on fresh splits: char LR (plain and number-balanced) on two new template-disjoint splits
    for split in ("template_r1", "template_r2"):
        eval_sets.build(split)
        for variant in ("clean", "counterfactual"):
            train, val, ev_r = variants.frames(split, variant, SEED)
            pipe, info = fit_tuned("lr", "char", train.text, train.label, val.text, val.label, train.template_id)
            tuning[f"lr_char/{variant}/{split}"] = info
            runs.save("lr_char", variant, split, SEED, ev_r, scam_proba(pipe, ev_r.text))

    # Interpretability: rung 2's strongest features, for the report and DECISIONS.md
    feats = {key: top_features(fitted[(key, "template")], k=25) for key in ("lr_char", "lr_word")}
    (config.RESULTS / "lr_top_features.json").write_text(json.dumps(feats, indent=1, ensure_ascii=False))
    (config.RESULTS / "classical_tuning.json").write_text(json.dumps(tuning, indent=1, default=str))
    print("done; tuning:", json.dumps({k: v.get("params") for k, v in tuning.items() if isinstance(v, dict) and "params" in v}))

    # The app's n-gram model is copied into app/ by scripts/export_app.py (plain or number-balanced).


if __name__ == "__main__":
    main()
