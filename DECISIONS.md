# Design decisions

One line per choice and why. These are the viva notes: every number in the report traces back to a choice here.

## Problem and data

| Choice | Why |
| --- | --- |
| Text classification of Swahili SMS into scam / not scam | Mobile-money fraud by SMS is common in Tanzania and there is public, real-world labelled data (BongoScam). |
| Swahili, with the rubric's "established system" test applied to classification | The brief's "no established MT" rule is written for the MT option. For classification we apply the matching test, whether this task in this language is already solved, and show ([docs/research.md](docs/research.md)) that no public, robustly evaluated Swahili scam classifier exists. Scam detection is not in the brief's example list, so it needs the instructor's approval. |
| BongoScam (Kaggle, MIT licence) for training and in-language test | 1,508 Tanzanian Swahili SMS, public, the dataset behind the 98.7% accuracy we set out to test. The author does not document how the messages were collected or labelled; the scams contain real-looking numbers and names, while the genuine texts read like composed personal chat. This provenance gap is stated as a limitation. |
| Chichewa SMS fraud set (Zenodo, CC BY 4.0) for transfer only | Malawian fraud SMS in a related Bantu language; no model, threshold or hyperparameter was tuned on it, so RQ3 is a zero-shot test. One honest caveat: the masking rules were written after looking at samples of both datasets, so they also recognise Malawian number and currency formats (08/09 numbers, MK/K amounts). |
| Only the `D_CHI` (Chichewa) and `telcoSMS_CHI` sheets | The other sheets are English translations (human and Google). Using them would no longer test Chichewa. |
| Raw data is not committed; `src/download.py` fetches it | Both licences allow redistribution, but raw texts contain real phone numbers. Only masked text is committed. |
| Drop exact duplicates (131 rows) | Copies inflate the test set and can sit on both sides of a split. |
| Mask phone numbers, money amounts and links as `<PHONE>`, `<AMOUNT>`, `<URL>` | Stops a model memorising one scammer's number and keeps personal numbers out of the repo. |
| Drop messages that become identical after masking (313 rows, all scams) | They are one message to every model. Kept, one campaign filled a whole split (35 copies of one job scam). |
| Keep punctuation, casing and stop words | Removing them hurt every model in the Chichewa study (Taylor & Robert, 2025a); casing is a real scam signal (ALL CAPS adverts). |
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
| Phone rule ("contains `<PHONE>` or `<URL>`") as an extra baseline (E3b) | 86% of scams and 0% of genuine texts contain a phone number or link, so this one line scores F1 0.92 across all 1,064 messages (descriptive; it needs no training) and 0.74 on the template-split test set. Any model must beat a shortcut. |
| Naive Bayes on raw word counts as the baseline | It is what BongoScam and Mambina et al. used, so our numbers compare directly with theirs. |
| E1 reproduces BongoScam exactly (all 1,508 rows, 80/20, `random_state=42`, default `CountVectorizer` + `MultinomialNB`) | Confirms our pipeline reaches their 98.7% before we change anything (we get 98.68%). |
| Logistic regression on character 2-5-grams (`char_wb`, TF-IDF) | Sub-word features should survive misspellings; the weights are directly interpretable. |
| `class_weight="balanced"` for LR, class-weighted losses for neural models | Keeps the smaller class from being ignored on every split. |
| Classical hyperparameters (NB alpha 0.001-10, LR C 0.01-1000) are chosen by template-disjoint 5-fold cross-validation on the training set; ties go to the most regularised value; every grid value's CV F1, validation F1 and (for reporting only) test F1 is saved in `results/classical_tuning.json` | Revised twice after review. The 152-message validation set is saturated (F1 0.99-1.00 for most values), so choosing on it is arbitrary: breaking ties by log-loss always picked the least regularised edge, breaking them by simplicity picked the most regularised edge and dropped LR test F1 to 0.75. Cross-validation inside the training set has five times more held-out messages and does discriminate (differences up to 0.06 F1); it picks interior values for all four models (NB-word alpha 0.3, NB-char 3, LR-word C 10, LR-char C 100). The test-F1 sensitivity (0.73-1.00 for LR across C, with PR-AUC near 1.00 throughout) shows the choice mainly moves calibration, not ranking. |
| BiLSTM: 300-d fastText Swahili vectors, BiLSTM 2 x 128, max-pool, dropout 0.3, sigmoid | The course's embeddings + RNN core. Max-pooling keeps the strongest signal anywhere in the SMS. |
| The fastText vector cache holds every token in every evaluation set, not only training tokens | Transductive but label-free: fastText computes a vector for any string from its character n-grams, so a deployed BiLSTM would do the same at inference. No label information is used. |
| fastText `cc.sw.300.bin`, not `.vec` | The `.bin` model builds vectors from character n-grams, so misspelled and attacked words still get sensible vectors. |
| Three embedding variants: random, frozen, fine-tuned (embedding LR 1e-4, rest 1e-3) | E4 asks whether pretrained vectors help and whether adapting them to scam vocabulary helps more; the small embedding LR stops 760 messages from washing out the pretraining. |
| BiLSTM: Adam, batch 32, up to 30 epochs, early stopping (patience 4) on validation F1 | Small data overfits fast; early stopping on the metric we report. |
| XLM-R base vs AfroXLMR base | Same architecture and size; AfroXLMR is XLM-R further trained on 17 African languages. Any difference isolates African pretraining. |
| XLM-R's 100 languages include Swahili (`sw`) but not Chichewa (`ny`); AfroXLMR adds `ny` (checked on both model cards) | So on Chichewa (RQ3) the comparison also isolates exposure to the target language. |
| Transformer head: `<s>` (first token) vector -> dense + tanh -> linear -> 2 logits (the standard `XLMRobertaForSequenceClassification` head) | Standard, well-tested fine-tuning setup. |
| Max length 128 tokens (BiLSTM: 64 word tokens) | No clean message is truncated (the longest is 81 sub-word tokens, 46 words). Some attacked texts are longer (spaced-out letters reach 131 tokens) and lose their tail, which is part of what such an attack does to a real system. |
| AdamW, learning rate 3e-5, weight decay 0.01, 10% linear warm-up, batch 16, up to 5 epochs, early stopping (patience 2) on validation F1 | Middle of the usual 2e-5 to 5e-5 range for base models. No learning-rate search: validation F1 reaches 0.99-1.00 after the first epoch in every one of the 29 training runs, so this validation set cannot tell learning rates apart; a search would only fit noise. |
| 3 seeds (13, 42, 2026) for every neural model, reported as mean +- sd | Fine-tuning on ~760 messages varies with the seed. |
| Trained on CPU (Apple M1 Pro, 5-14 minutes per run, see results/transformer_log.jsonl) | The local sandbox has no GPU access; small data and short SMS make CPU feasible. |
| Predictions for every evaluation set are saved per run; one script computes all metrics | Metrics are computed one way for every model, and nothing needs retraining to add a metric. |

## Evaluation

| Choice | Why |
| --- | --- |
| Scam-class precision, recall, F1 and PR-AUC; accuracy only as context | A model that flags nothing still looks accurate when one class dominates; PR-AUC is threshold-free. |
| F1 at the default 0.5 threshold for experiment comparisons | Comparable with prior work. |
| Operating point: threshold that catches >= 95% of validation scams, never raised above 0.5 | A missed scam costs money, a false alarm costs trust. Raising the threshold on a perfectly separated validation set would push it to ~1.0 and overfit. |
| Bootstrap 95% intervals (1,000 resamples) on test F1; for multi-seed models each resample draws the same messages for every seed and averages F1 over seeds; no pooled interval is reported when seeds scored different messages | The test set has ~150 messages; averaging per-seed interval bounds would not give a valid interval. |
| Relative F1 drop = (F1 clean - F1 attacked) / F1 clean | The measure Chiuseni et al. (2026) used, so our robustness numbers compare with theirs. |
| Error buckets assigned by simple keyword rules (`src/errors.py`), then read by hand | Repeatable counts; the hand reading picks the examples for the report. Bucket names describe the message (e.g. "Chichewa fraud missed", "Telco service text flagged"), not a proven cause. |

## Robustness (RQ2)

| Choice | Why |
| --- | --- |
| Attacks are code, not an LLM | Anyone can rerun them and get the same text (seeded). |
| Three attacks: lookalike characters (Cyrillic twins, digits, zero-width space), structural (s p a c e d, d-a-s-h-e-d, joinedwords), code-switching (30-phrase Swahili-to-English lexicon) | Tricks scammers actually use; each targets a different kind of feature. |
| Trigger words = the words whose removal most lowers the scam probability of a fixed "attacker" (char LR, C=100) | Every model is attacked on the same words. The attack is therefore white-box against a char LR and a transfer (black-box) attack against every other model, which favours the other models; stated wherever robustness is compared. |
| Intensity 1, 3 or all positive trigger words | Shows where each architecture starts to break. |
| Only scam messages are attacked | Honest senders have no reason to disguise a text. |
| Attacks are applied to masked text | Placeholders stay intact, so attacks change language, not numbers. |
| Defence 1: normalisation at inference (`preprocess(..., defend=True)`): map lookalikes back, remove zero-width characters, undo digit-for-letter swaps, rejoin spaced letters | Cheap, needs no retraining; prior work proposed it but did not test it. |
| Defence 2: adversarial training with perturbed copies of 25% of training scams (101 copies, 12% of the augmented training set) | Machine-made data stays a small share, as the brief requires; the same saved copies are used by every model. |
| Adversarial copies use the same attack code as the test attacks | Stated as a limitation: this is robustness to known tricks, an optimistic bound. |

## The shortcut: diagnose, fix, combine (E10-E12)

| Choice | Why |
| --- | --- |
| E10 stress tests are minimal pairs: append ` <PHONE>` or ` <AMOUNT>` to each genuine test text; delete the placeholders from each test scam that has one | Only one token changes, so any change in the decision is caused by the number alone (behavioural testing in the spirit of CheckList, Ribeiro et al., 2020). |
| E6 controls disguise the genuine test texts too, at two intensities: their 3 most influential words, and every word (matching the scam attacks' full intensity) | The attacked sets disguise scams only, so "odd-looking" and "scam" coincide there; the controls separate "flags disguise" from "reads the scam". Added after review, when the 3-word controls proved too mild. |
| A second attacker (word-count NB, alpha 1) re-runs the full-intensity attacks | The main attacker is a char LR, so the attack is white-box for the char LR and a transfer attack for every other model; a second attacker checks that conclusions, including the ensemble's gain over the char LR, do not hinge on that. |
| E11 number-balanced training: delete placeholders from half the training scams that have one, append ` <PHONE>` to 40% of genuine training texts | Makes "contains a number" uninformative (44% of scams, 38% of genuine texts carry one) while words and labels stay untouched: a counterfactual augmentation (Kaushik et al., 2020) targeted at the measured shortcut. 42% of training rows change by one token; validation and test are never edited. |
| E12 ensemble = char LR OR AfroXLMR, each at its own validation threshold | The two fail on different messages. OR follows the recall-first cost asymmetry set at the start; the mean rule is reported and loses because the transformer's probabilities sit at 0 or 1. |
| The ensemble idea is post hoc, so it is re-checked on two fresh template-disjoint splits (seeds 1 and 2) never used for design | Guards against designing on the test set. |
| Which members the app deploys (plain or number-balanced char LR and AfroXLMR) is decided on the validation stress pairs (`valstress_*`), never on test: lowest combined false-alarm and miss rate wins | Validation F1 is saturated, but the validation stress pairs still discriminate: both number-balanced gives 0% / 0%, a plain char LR 10% false alarms, a plain AfroXLMR 96%. |
| Key differences get a paired bootstrap (1,000 resamples of messages, seed-paired) or a Mann-Whitney U test (repeated splits) | Many differences are a few messages on a 152-message test set; p-values and intervals show which ones are real. |

## Transfer (RQ3)

| Choice | Why |
| --- | --- |
| Zero-shot: Swahili-trained models scored on Chichewa with no Chichewa training | The direct test of cross-language transfer. |
| The Zenodo files do not mark which fraud messages are originals vs rewrites | Reported on all 733 masked, de-duplicated messages, plus the balanced D-CHI part and one message per template. |
| Few-shot (E9): add 20 or 50 Chichewa messages (half fraud), drop every message sharing a template with them from the test, 3 seeds | Measures how fast the gap closes without leaking rewrites of training messages into the test. |

## App

| Choice | Why |
| --- | --- |
| A static Hugging Face Space that runs both models in the browser (transformers.js + ONNX Runtime Web; a JavaScript port of the char LR) | Gradio and Docker Spaces on CPU now need a paid plan; static Spaces are free, never sleep, and keep every message on the user's device. The Gradio app stays for local and Colab use. |
| Browser model: 8-bit embedding table + fp16 weight storage, fp32 compute (364 MB) | Full 8-bit dynamic quantisation dropped agreement with PyTorch to 74% (outlier activations); this scheme agrees on 99.7% of evaluation messages. |
| JavaScript preprocessing and n-gram model checked against Python on every message (`scripts/web_parity.mjs`) | The app must see text exactly as the models did in training; Python's Unicode `\w`, `\d` and `\b` are spelled out explicitly in JavaScript. |
| The app's verdict is the E12 ensemble: number-balanced AfroXLMR (seed 42) OR number-balanced char LR, both shown | The two fail on different messages; the number-balanced members were chosen on the validation stress pairs, never on test. |
| The app falls back from the Hub model to a local copy, then to the n-gram model alone | A missing or private Hub model must not crash the demo. |
| Same `preprocess()` as training, imported from `src/` | The app cannot drift from the experiments. |
| Word highlights by leave-one-word-out occlusion | Model-agnostic, easy to explain, and it is the same method that picks attack trigger words. |
| No logging of submitted text | Privacy: SMS can contain personal data. |
