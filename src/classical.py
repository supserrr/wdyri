"""Rungs 0-2 of the model ladder: majority class, Naive Bayes, logistic regression.

All three are scikit-learn pipelines that take preprocessed text and return a
scam probability, so the same code serves training, attacks and the app.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Keeps <PHONE>, <AMOUNT> and <URL> as single tokens; otherwise scikit-learn's default.
TOKEN_PATTERN = r"(?u)<\w+>|\b\w\w+\b"

GRIDS = {
    "nb": {"clf__alpha": [0.01, 0.1, 0.3, 1.0]},
    "lr": {"clf__C": [0.1, 1.0, 10.0, 100.0]},
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


def fit_tuned(model: str, features: str, x_train, y_train, x_val, y_val) -> tuple[Pipeline, dict]:
    """Fit every grid value on train, keep the best on validation.

    Best = highest scam F1; ties (common when validation F1 saturates) go to
    the lower validation log-loss, i.e. the better-calibrated probabilities.
    """
    grid = GRIDS[model]
    candidates = [{}] if not grid else [{k: v} for k, vals in grid.items() for v in vals]
    best, best_params, best_key = None, None, (-1.0, 0.0)
    for params in candidates:
        pipe = build(model, features).set_params(**params)
        pipe.fit(x_train, y_train)
        f1 = f1_score(y_val, pipe.predict(x_val), zero_division=0)
        loss = log_loss(y_val, pipe.predict_proba(x_val), labels=pipe.classes_)
        if (f1, -loss) > best_key:
            best, best_params, best_key = pipe, params, (f1, -loss)
    return best, {"params": best_params, "val_f1": best_key[0], "val_log_loss": -best_key[1]}


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
