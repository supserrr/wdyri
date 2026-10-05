# Data and models

## Data

| Dataset | Language | Size used | Role | Licence |
| --- | --- | --- | --- | --- |
| [BongoScam](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset) (Dioniz, 2024) | Swahili, Tanzania | 1,508 raw → 1,064 after cleaning (567 scam, 497 genuine) | train / validation / test | MIT |
| [Chichewa SMS fraud](https://doi.org/10.5281/zenodo.14607454) (Taylor & Robert, 2025b) | Chichewa, Malawi | 824 raw → 733 (336 fraud) | zero-shot transfer test only | CC BY 4.0 |

Cleaning: drop exact duplicates, mask phone numbers / amounts / links as `<PHONE>`, `<AMOUNT>`, `<URL>`, drop messages that become identical, group near-duplicate *templates* (character 5-gram Jaccard ≥ 0.8), then make a random and a template-disjoint 70/15/15 split. Data quality, provenance, both shortcuts (numbers and length) and other limits: [data/README.md](../data/README.md). Every design choice and its reason: [DECISIONS.md](../DECISIONS.md).

## Models: a ladder from spam filter to Africa-centric transformer

| Rung | Model | Input | Why it is here |
| --- | --- | --- | --- |
| 0 | Majority class; phone rule; length rule | none; "contains `<PHONE>`/`<URL>`"; "≥ 9 words" | Floors: what you get for free, and with each shortcut |
| 1 | Multinomial Naive Bayes (**baseline**) | word counts | The model BongoScam and Mambina et al. used |
| 2 | Logistic regression | character 2-5-gram TF-IDF | Sub-words resist spelling tricks; weights explain decisions |
| 3 | BiLSTM | fastText Swahili vectors (300-d), random / frozen / fine-tuned | The course's embeddings + RNN core |
| 4 | XLM-R base, fine-tuned | SentencePiece sub-words | Multilingual transformer control (no Chichewa in pretraining) |
| 5 | AfroXLMR base, fine-tuned | SentencePiece sub-words | XLM-R adapted to 17 African languages incl. Swahili and Chichewa |
| + | Ensemble | rungs 2 and 5, both number-balanced | Flags when either flags; the deployed system |

BiLSTM: embedding (300) → BiLSTM (2 × 128) → max-pool over time → dropout 0.3 → linear → sigmoid. Transformers: SMS → sub-word tokens (max 128) → 12-layer encoder → `<s>` vector → dense + tanh → linear → softmax. AdamW, learning rate 3e-5, batch 16, up to 5 epochs with early stopping on validation F1, class-weighted loss, 3 seeds (mean ± sd). Classical hyperparameters are chosen by template-disjoint 5-fold cross-validation on the training set, because the 152-message validation set is saturated (F1 0.99-1.00 for most settings). Transparency note: this protocol replaced two earlier ones after review, and one of those changes was prompted by seeing a large test-F1 drop; the final protocol is chosen on principle (more held-out evidence), and the test F1 and PR-AUC of every grid value are kept as a sensitivity analysis (`results/classical_tuning.json`).
