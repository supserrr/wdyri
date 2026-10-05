"""Rungs 0-2 of the model ladder: majority class, Naive Bayes, logistic regression.

All three are scikit-learn pipelines that take preprocessed text and return a
scam probability, so the same code serves training, attacks and the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Keeps <PHONE>, <AMOUNT> and <URL> as single tokens; otherwise scikit-learn's default.
TOKEN_PATTERN = r"(?u)<\w+>|\b\w\w+\b"

# Ordered from most to least regularised: ties on validation go to the first,
# simplest model (validation F1 saturates on this data, see fit_tuned).
GRIDS = {
    "nb": {"clf__alpha": [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]},
    "lr": {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "majority": {},
}


def vectoriser(features: str):
    if features == "word_counts":
        # The BongoScam baseline: raw word counts (unigrams).
        return CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)
    if features == "word":
        return TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True,
                               ngram_range=(1, 2), sublinear_tf=True)
    if features == "char":
        # Character 2-5-grams inside word boundaries: robust to small spelling changes.
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                               sublinear_tf=True, min_df=2)
    raise ValueError(features)


def build(model: str, features: str) -> Pipeline:
    if model == "majority":
        return Pipeline([("vec", CountVectorizer(token_pattern=TOKEN_PATTERN)),
                         ("clf", DummyClassifier(strategy="most_frequent"))])
    if model == "nb":
        clf = MultinomialNB()
    elif model == "lr":
        # Balanced weights so the minority class (here "not scam") still counts.
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    else:
        raise ValueError(model)
    return Pipeline([("vec", vectoriser(features)), ("clf", clf)])


def fit_tuned(model: str, features: str, x_train, y_train, x_val, y_val, groups=None) -> tuple[Pipeline, dict]:
    """Choose hyperparameters by 5-fold cross-validation on the training data, then refit.

    The 152-message validation set is saturated (F1 0.99-1.00 for most grid
    values), so it cannot choose. Cross-validation inside the training set uses
    five times more held-out messages; with `groups` (template ids) the folds
    are template-disjoint, like the test. Best = highest mean CV scam F1; ties
    go to the most regularised value (grids are ordered that way). Validation
    F1 is recorded for every value but not used to choose.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    x_train, y_train = pd.Series(list(x_train)), pd.Series(list(y_train))
    if groups is not None:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x_train, y_train, list(groups)))
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(x_train, y_train))
    grid = GRIDS[model]
    candidates = [{}] if not grid else [{k: v} for k, vals in grid.items() for v in vals]
    best_params, best_cv, scores = None, -1.0, []
    for params in candidates:
        cv_f1 = []
        for tr, te in folds:
            pipe = build(model, features).set_params(**params).fit(x_train.iloc[tr], y_train.iloc[tr])
            cv_f1.append(f1_score(y_train.iloc[te], pipe.predict(x_train.iloc[te]), zero_division=0))
        full = build(model, features).set_params(**params).fit(x_train, y_train)
        scores.append({**{k.split("__")[-1]: v for k, v in params.items()},
                       "cv_f1": round(float(np.mean(cv_f1)), 4),
                       "val_f1": round(f1_score(y_val, full.predict(list(x_val)), zero_division=0), 4)})
        if np.mean(cv_f1) > best_cv + 1e-9:  # strict: the first (most regularised) value wins ties
            best_params, best_cv, best = params, float(np.mean(cv_f1)), full
    return best, {"params": best_params, "cv_f1": best_cv, "grid": scores}


def scam_proba(pipe: Pipeline, texts) -> np.ndarray:
    return pipe.predict_proba(list(texts))[:, list(pipe.classes_).index(1)] \
        if 1 in pipe.classes_ else np.zeros(len(texts))


def top_features(pipe: Pipeline, k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Most scam-ward and most not-scam-ward features of a logistic regression."""
    names = pipe.named_steps["vec"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    return {
        "scam": [(names[i], float(coef[i])) for i in order[::-1][:k]],
        "not_scam": [(names[i], float(coef[i])) for i in order[:k]],
    }
