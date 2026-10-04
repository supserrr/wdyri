# Design decisions

One line per choice and why. These are the viva notes: every number in the report traces back to a choice here.

## Problem and data

| Choice | Why |
| --- | --- |
| Text classification of Swahili SMS into scam / not scam | Mobile-money fraud by SMS is common in Tanzania and there is public, real-world labelled data (BongoScam). |
| Swahili, not an MT language | The brief's "no established MT" rule belongs to the MT option; the classification option sets no such limit. |
| BongoScam (Kaggle, MIT licence) for training and in-language test | 1,508 real Tanzanian SMS, public, the dataset behind the 98.7% accuracy we set out to test. |
| Chichewa SMS fraud set (Zenodo, CC BY 4.0) for transfer only | Real Malawian fraud SMS in a related Bantu language; kept sealed (never used for tuning) so RQ3 is a true zero-shot test. |
| Only the `D_CHI` (Chichewa) and `telcoSMS_CHI` sheets | The other sheets are English translations (human and Google). Using them would no longer test Chichewa. |
| Raw data is not committed; `src/download.py` fetches it | Both licences allow redistribution, but raw texts contain real phone numbers. Only masked text is committed. |
| Drop exact duplicates (131 rows) | Copies inflate the test set and can sit on both sides of a split. |
| Mask phone numbers, money amounts and links as `<PHONE>`, `<AMOUNT>`, `<URL>` | Stops a model memorising one scammer's number and keeps personal numbers out of the repo. |
| Drop messages that become identical after masking (313 rows, all scams) | They are one message to every model. Kept, one campaign filled a whole split (35 copies of one job scam). |
| Keep punctuation, casing and stop words | Removing them hurt every model in the Chichewa study (Taylor & Robert, 2025); casing is a real scam signal (ALL CAPS adverts). |
| Template = masked texts sharing >= 80% of lower-cased character 5-grams (Jaccard), grouped transitively | Catches the same script with a different name or network ("Habari za asubuhi/mchana. Mimi mwenye nyumba wako ... Vodacom/Tigo"). |
| Two 70/15/15 splits, both stratified by label | Random reproduces prior work; template-disjoint (`StratifiedGroupKFold`) puts every copy of a script on one side, so the test measures new scripts. |
| Split fixed with seed 42 before any modelling, never changed after | Changing the split after seeing test scores would be test-set tuning. |
| One template ended up holding 31 of the 81 test scams (landlord impersonation) | Reported, not fixed: it is exactly the kind of unseen script RQ1 is about. Every test score is also given "one per template" (`/tpl`). |
| Classical models also get 10 re-drawn splits of each type (E2 repeat) | A single test set of ~150 messages is too noisy to compare split types on one number. |

## Models

| Choice | Why |
| --- | --- |
| A ladder: majority, Naive Bayes, logistic regression, BiLSTM, XLM-R, AfroXLMR | Each rung adds one idea, so each gain or loss is measured, not assumed. |
| Majority class (rung 0) predicts "scam" | Scams are the majority in BongoScam (567 vs 497 after cleaning); F1 0.70 is the floor any model must beat. |
| Phone rule ("contains `<PHONE>` or `<URL>`") as an extra baseline (E3b) | 85% of scams and 0% of genuine texts contain a number, so this one line scores F1 0.92 on the clean data. Any model must beat a shortcut. |
| Naive Bayes on raw word counts as the baseline | It is what BongoScam and Mambina et al. used, so our numbers compare directly with theirs. |
| E1 reproduces BongoScam exactly (all 1,508 rows, 80/20, `random_state=42`, default `CountVectorizer` + `MultinomialNB`) | Confirms our pipeline reaches their 98.7% before we change anything (we get 98.68%). |
| Logistic regression on character 2-5-grams (`char_wb`, TF-IDF) | Sub-word features should survive misspellings; the weights are directly interpretable. |
| `class_weight="balanced"` for LR, class-weighted losses for neural models | Keeps the smaller class from being ignored on every split. |
| Grids: NB alpha {0.01, 0.1, 0.3, 1}, LR C {0.1, 1, 10, 100}, chosen on validation scam F1, ties broken by validation log-loss | Validation F1 often saturates at 1.0; log-loss then prefers the better-calibrated model instead of the first grid value. |
| BiLSTM: 300-d fastText Swahili vectors, BiLSTM 2 x 128, max-pool, dropout 0.3, sigmoid | The course's embeddings + RNN core. Max-pooling keeps the strongest signal anywhere in the SMS. |
| fastText `cc.sw.300.bin`, not `.vec` | The `.bin` model builds vectors from character n-grams, so misspelled and attacked words still get sensible vectors. |
| Three embedding variants: random, frozen, fine-tuned (embedding LR 1e-4, rest 1e-3) | E4 asks whether pretrained vectors help and whether adapting them to scam vocabulary helps more; the small embedding LR stops 760 messages from washing out the pretraining. |
| BiLSTM: Adam, batch 32, up to 30 epochs, early stopping (patience 4) on validation F1 | Small data overfits fast; early stopping on the metric we report. |
| XLM-R base vs AfroXLMR base | Same architecture and size; AfroXLMR is XLM-R further trained on 17 African languages. Any difference isolates African pretraining. |
| XLM-R's 100 languages include Swahili (`sw`) but not Chichewa (`ny`); AfroXLMR adds `ny` (checked on both model cards) | So on Chichewa (RQ3) the comparison also isolates exposure to the target language. |
| Transformer head: `<s>` (first token) vector -> dense + tanh -> linear -> 2 logits (the standard `XLMRobertaForSequenceClassification` head) | Standard, well-tested fine-tuning setup. |
| Max length 128 tokens | The longest SMS is 81 sub-word tokens; nothing is truncated. |
| AdamW, learning rate 3e-5, weight decay 0.01, 10% linear warm-up, batch 16, up to 5 epochs, early stopping (patience 2) on validation F1 | Middle of the usual 2e-5 to 5e-5 range for base models; validation F1 hit 1.0 within 1-3 epochs, so more epochs only add cost. |
| 3 seeds (13, 42, 2026) for every neural model, reported as mean +- sd | Fine-tuning on ~760 messages varies with the seed. |
| Trained on CPU (Apple M1 Pro, about 7-13 minutes per run) | The local sandbox has no GPU access; small data and short SMS make CPU feasible. |
| Predictions for every evaluation set are saved per run; one script computes all metrics | Metrics are computed one way for every model, and nothing needs retraining to add a metric. |

## Evaluation

| Choice | Why |
| --- | --- |
| Scam-class precision, recall, F1 and PR-AUC; accuracy only as context | A model that flags nothing still looks accurate when one class dominates; PR-AUC is threshold-free. |
| F1 at the default 0.5 threshold for experiment comparisons | Comparable with prior work. |
| Operating point: threshold that catches >= 95% of validation scams, never raised above 0.5 | A missed scam costs money, a false alarm costs trust. Raising the threshold on a perfectly separated validation set would push it to ~1.0 and overfit. |
| Bootstrap 95% intervals (1,000 resamples) on test F1 | The test set has ~150 messages. |
| Relative F1 drop = (F1 clean - F1 attacked) / F1 clean | The measure Chiuseni et al. (2026) used, so our robustness numbers compare with theirs. |
| Error buckets assigned by simple keyword rules (`src/errors.py`), then read by hand | Repeatable counts; the hand reading picks the examples for the report. |

## Robustness (RQ2)

| Choice | Why |
| --- | --- |
| Attacks are code, not an LLM | Anyone can rerun them and get the same text (seeded). |
| Three attacks: lookalike characters (Cyrillic twins, digits, zero-width space), structural (s p a c e d, d-a-s-h-e-d, joinedwords), code-switching (30-phrase Swahili-to-English lexicon) | Tricks scammers actually use; each targets a different kind of feature. |
| Trigger words = the words whose removal most lowers rung 2's scam probability (occlusion) | Every model is attacked on the same words, so the comparison is fair. |
| Intensity 1, 3 or all positive trigger words | Shows where each architecture starts to break. |
| Only scam messages are attacked | Honest senders have no reason to disguise a text. |
| Attacks are applied to masked text | Placeholders stay intact, so attacks change language, not numbers. |
| Defence 1: normalisation at inference (`preprocess(..., defend=True)`): map lookalikes back, remove zero-width characters, undo digit-for-letter swaps, rejoin spaced letters | Cheap, needs no retraining; prior work proposed it but did not test it. |
| Defence 2: adversarial training with perturbed copies of 25% of training scams (101 copies, 12% of the augmented training set) | Machine-made data stays a small share, as the brief requires; the same saved copies are used by every model. |
| Adversarial copies use the same attack code as the test attacks | Stated as a limitation: this is robustness to known tricks, an optimistic bound. |

## Transfer (RQ3)

| Choice | Why |
| --- | --- |
| Zero-shot: Swahili-trained models scored on Chichewa with no Chichewa training | The direct test of cross-language transfer. |
| The Zenodo files do not mark which fraud messages are originals vs rewrites | Reported on all 733 masked, de-duplicated messages, plus the balanced D-CHI part and one message per template. |
| Few-shot (E9): add 20 or 50 Chichewa messages (half fraud), drop every message sharing a template with them from the test, 3 seeds | Measures how fast the gap closes without leaking rewrites of training messages into the test. |

## App

| Choice | Why |
| --- | --- |
| Gradio on a free Hugging Face Space | Free, public link; the model loads from the Hub by name. |
| Transformer (AfroXLMR, seed 42) and baseline (char LR) side by side | Shows where a simple model and a large one disagree, live. |
| Same `preprocess()` as training, imported from `src/` | The app cannot drift from the experiments. |
| Word highlights by leave-one-word-out occlusion | Model-agnostic, easy to explain, and it is the same method that picks attack trigger words. |
| No logging of submitted text | Privacy: SMS can contain personal data. |
