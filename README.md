# WDYRI: Why Did You Redeem It?

**Detecting mobile-money scam SMS in Swahili, and testing whether near-perfect scores survive harder tests.**

| | |
| --- | --- |
| GitHub repository | https://github.com/supserrr/wdyri |
| Live app | https://huggingface.co/spaces/supserrr/wdyri |
| Fine-tuned model | https://huggingface.co/supserrr/wdyri-afroxlmr |
| Demo video | *to be added* |
| Fallback demo | [notebooks/colab_app.ipynb](notebooks/colab_app.ipynb) (runs the same app on Colab with a public link) |

Mobile-money scams arrive as SMS that pose as relatives, agents, landlords or employers and ask the victim to send money; in Tanzania a common hook is *"ni tumie kwa namba hii"* (send it to me on this number). Published Swahili detectors report 98.7% (BongoScam) to 99.86% (Mambina et al., 2022) accuracy. WDYRI reproduces that result and then asks what prior work skipped:

1. **RQ1, leakage.** How much of the score survives when near-duplicate scam templates cannot appear in both train and test?
2. **RQ2, robustness.** How far does each model fall under rule-based obfuscation (lookalike letters, split words, English code-switching), and do simple defences recover it?
3. **RQ3, transfer.** Can a model trained only on Swahili catch real Chichewa fraud SMS from Malawi, and does Africa-centric pretraining (AfroXLMR) beat plain XLM-R there?

**Users:** mobile-money customers checking a suspicious SMS, and the telcos or fintechs that could run the check for them.

## Findings in one minute

* **The published 98.7% reproduces exactly** (98.68% accuracy, E1), but the dataset is easier than it looks: none of BongoScam's 497 genuine messages contains a phone number while 85% of scams do, so the one-line rule *"contains a phone number or link"* already scores F1 0.92 on the cleaned dataset (E3b).
* **RQ1:** template leakage inflates character n-gram models most (char LR 0.998 → 0.963, char NB 0.995 → 0.926 over 10 re-drawn splits) and word models barely (E2). The bigger effect is *which* scripts land in the test: one unseen landlord-impersonation script (31 of 81 test scams, no number, no link) is missed by both transformers and by most BiLSTM seeds, which pulls their test F1 to 0.76-0.86 while per-template F1 stays at 0.98-0.99.
* **RQ2:** lookalike letters cut word-level LR recall from 1.00 to 0.31; normalisation brings it back to 0.99 at no cost on clean text, but only partly repairs split words. Transformers are not hurt by lookalikes or split words, they flag *more* (F1 rises by 16-31%): odd characters look suspicious. Code-switching is the attack normalisation cannot undo; adversarial training cuts char LR's loss there from 19% to 4%.
* **RQ3:** zero-shot Chichewa transfer is weak: transformers reach fraud F1 0.72, barely above the phone rule (0.68). AfroXLMR ranks Chichewa fraud clearly better than XLM-R (PR-AUC 0.81 ± 0.04 vs 0.65 ± 0.09), consistent with its Chichewa pretraining. Adding just 20 Chichewa messages lifts char LR from 0.65 to 0.82; with 50, AfroXLMR reaches 0.87 while XLM-R reaches only 0.78.

## Main results

Template-disjoint split (no scam script in both train and test). Full tables for every experiment: [results/tables.md](results/tables.md).

<!-- RESULTS:START -->
| Model | Test F1 | Test F1 (per template) | Test PR-AUC | Precision (op.) | Recall (op.) | Chichewa F1 (zero-shot) | Chichewa PR-AUC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.695 | 0.585 | 0.533 | 0.533 | 1.000 | 0.629 | 0.458 |
| Phone rule | 0.744 | 0.969 | 0.810 | 0.533 | 1.000 | 0.681 | 0.692 |
| Naive Bayes, word counts | 0.988 | 0.980 | 1.000 | 0.976 | 1.000 | 0.607 | 0.639 |
| Naive Bayes, char 2-5 | 0.921 | 0.990 | 0.998 | 0.986 | 0.864 | 0.305 | 0.528 |
| Log. regression, word 1-2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.494 | 0.725 |
| Log. regression, char 2-5 | 0.968 | 1.000 | 1.000 | 1.000 | 0.938 | 0.651 | 0.730 |
| BiLSTM, random init | 0.791 ± 0.027 | 0.990 ± 0.000 | 0.966 ± 0.046 | 1.000 ± 0.000 | 0.654 ± 0.037 | 0.597 ± 0.063 | 0.803 ± 0.026 |
| BiLSTM, fastText frozen | 0.778 ± 0.018 | 0.983 ± 0.006 | 0.981 ± 0.005 | 0.987 ± 0.011 | 0.642 ± 0.021 | 0.733 ± 0.005 | 0.645 ± 0.031 |
| BiLSTM, fastText fine-tuned | 0.860 ± 0.107 | 0.980 ± 0.017 | 0.990 ± 0.007 | 0.980 ± 0.024 | 0.778 ± 0.174 | 0.738 ± 0.005 | 0.701 ± 0.100 |
| XLM-R base | 0.759 ± 0.007 | 0.983 ± 0.011 | 0.871 ± 0.035 | 0.987 ± 0.022 | 0.617 ± 0.000 | 0.717 ± 0.012 | 0.654 ± 0.090 |
| AfroXLMR base | 0.776 ± 0.021 | 0.990 ± 0.000 | 0.958 ± 0.038 | 1.000 ± 0.000 | 0.634 ± 0.029 | 0.719 ± 0.019 | 0.805 ± 0.042 |
<!-- RESULTS:END -->

![RQ1](results/figures/rq1_repeated_splits.png)
![RQ2 attacks](results/figures/rq2_attacks.png)
![RQ2 defences](results/figures/rq2_defences.png)
![RQ3](results/figures/rq3_transfer.png)

## Data

| Dataset | Language | Size used | Role | Licence |
| --- | --- | --- | --- | --- |
| [BongoScam](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset) (Dioniz, 2024) | Swahili, Tanzania | 1,508 raw → 1,064 after cleaning (567 scam, 497 genuine) | train / validation / test | MIT |
| [Chichewa SMS fraud](https://doi.org/10.5281/zenodo.14607454) (Taylor & Robert, 2025) | Chichewa, Malawi | 824 raw → 733 (336 fraud) | zero-shot transfer test only | CC BY 4.0 |

Cleaning: drop exact duplicates, mask phone numbers / amounts / links as `<PHONE>`, `<AMOUNT>`, `<URL>`, drop messages that become identical, group near-duplicate *templates* (character 5-gram Jaccard ≥ 0.8), then make a random and a template-disjoint 70/15/15 split. Details, counts and limits: [data/README.md](data/README.md). Every design choice and its reason: [DECISIONS.md](DECISIONS.md).

## Models: a ladder from spam filter to Africa-centric transformer

| Rung | Model | Input | Why it is here |
| --- | --- | --- | --- |
| 0 | Majority class; phone rule | none; "contains `<PHONE>`/`<URL>`" | Floors: what you get for free, and with one shortcut |
| 1 | Multinomial Naive Bayes (**baseline**) | word counts | The model BongoScam and Mambina et al. used |
| 2 | Logistic regression | character 2-5-gram TF-IDF | Sub-words resist spelling tricks; weights explain decisions |
| 3 | BiLSTM | fastText Swahili vectors (300-d), random / frozen / fine-tuned | The course's embeddings + RNN core |
| 4 | XLM-R base, fine-tuned | SentencePiece sub-words | Multilingual transformer control (no Chichewa in pretraining) |
| 5 | AfroXLMR base, fine-tuned | SentencePiece sub-words | XLM-R adapted to 17 African languages incl. Swahili and Chichewa |

BiLSTM: embedding (300) → BiLSTM (2 × 128) → max-pool over time → dropout 0.3 → linear → sigmoid. Transformers: SMS → sub-word tokens (max 128) → 12-layer encoder → `<s>` vector → dense + tanh → linear → softmax. AdamW, learning rate 3e-5, batch 16, up to 5 epochs with early stopping on validation F1, class-weighted loss, 3 seeds (mean ± sd).

## Experiments

| ID | Question | What changes | Main metric | Result |
| --- | --- | --- | --- | --- |
| E1 | Baseline | NB word counts, published setup | accuracy, scam F1 | 0.9868 accuracy (published: 98.7%), F1 0.991 |
| E2 | RQ1 leakage | random vs template split, 1 and 10 re-drawn splits | drop in scam F1 | char models lose 3.5-7 points, word models < 1 |
| E3 | Features | word vs char n-grams; masking on vs off | scam F1 | word features win on this data; masking changes little in-language |
| E3b | Shortcut | phone rule; placeholders deleted | scam F1 | rule F1 0.92 (cleaned data); without placeholders, Chichewa transfer of classical models collapses |
| E4 | Embeddings | BiLSTM random / frozen / fine-tuned fastText | scam F1, mean ± sd | fine-tuned best on test (0.86 ± 0.11, seed-dependent); pretrained vectors raise Chichewa F1 (0.74 vs 0.60) but not PR-AUC |
| E5 | Transformers | XLM-R vs AfroXLMR, 3 seeds | scam F1, mean ± sd | 0.76 ± 0.01 vs 0.78 ± 0.02 on test, 0.98-0.99 per template; both miss the landlord script |
| E6 | RQ2 attacks | 3 attacks × 3 intensities, every rung | relative F1 drop | word LR loses 53% of its F1 under full lookalike; transformers gain 25-31% |
| E7 | RQ2 defences | normalisation; adversarial training (12% machine-made copies) | relative F1 drop, clean F1 | normalisation erases the lookalike loss (word LR 53% → 1%); adversarial training cuts char LR's code-switch loss 19% → 4% |
| E8 | RQ3 transfer | Swahili-trained rungs on Chichewa | fraud F1, PR-AUC | F1 0.31-0.74 across rungs (phone rule 0.68); AfroXLMR PR-AUC 0.81 vs XLM-R 0.65 |
| E9 | RQ3 few-shot | + 20 / 50 Chichewa messages | fraud F1 | char LR 0.65 → 0.82 → 0.88; AfroXLMR 0.72 → 0.75 → 0.87; XLM-R 0.72 → 0.70 → 0.78 |

## Error analysis

Every error of the main runs is sorted into a bucket by `src/errors.py` ([results/error_buckets.csv](results/error_buckets.csv), examples in [notebooks/07_errors.ipynb](notebooks/07_errors.ipynb)):

| Bucket | Typical message | Likely cause |
| --- | --- | --- |
| Impersonation missed | *"Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya ...?"* (landlord, "this is my new number") | No number, amount or link, and the money request comes later in the conversation. Word NB/LR catch it through *namba yangu* and *mwenye nyumba* (only ever in training scams); the transformers lean on the number and miss it |
| Obfuscation missed | *"Iyo p3sа іtumе kwenyе n a m b a hii ..."* | Word tokens break; normalisation fixes lookalikes, only partly split words, not code-switching |
| Service text flagged | Chichewa telco balance and bundle messages | Money words and amounts shared with scams; BongoScam has no genuine service texts |
| Local tactic missed | Malawian "DODMA flood relief" and "JB Foundation" grant scams | Scam scripts are national, not only linguistic |
| Ambiguous | very short texts | Need sender or conversation context |

The transformer's verdict rests on the `<PHONE>` token: AfroXLMR scores the classic *"Iyo pesa itume kwenye namba hii `<PHONE>` ..."* at 1.000 but 0.085 once the number is removed, and the landlord message jumps from 0.000 to 0.985 when a number is appended. On Chichewa it flags 44% of genuine telco texts but 13% of genuine personal texts.

## Deployment

A Gradio app on a free Hugging Face Space ([app/app.py](app/app.py)). Paste an SMS and get *Scam* / *Not scam* with a probability from the fine-tuned AfroXLMR **and** from the char-LR baseline, side by side, plus highlighted words that drove each verdict (leave-one-word-out occlusion) and the masked text the models actually saw.

* **Wiring:** `app.py` loads the transformer from the Hub by name (`supserrr/wdyri-afroxlmr`) with `transformers` and the baseline pipeline with `joblib`. Both call the same `preprocess()` from `src/` as training. Thresholds come from validation (`app/settings.json`).
* **Built-in examples:** a genuine message, a classic scam, the same scam disguised (with and without the defence), the landlord scam that fools the transformer, and a Chichewa scam.
* **Privacy:** nothing is logged; numbers are masked before any model sees them.
* **Fallback:** free Spaces sleep when idle; [notebooks/colab_app.ipynb](notebooks/colab_app.ipynb) launches the same app with `share=True`.

## Reproduce

Python 3.12. Everything runs on a laptop CPU (transformer runs take about 10 minutes each; 1-2 minutes on a free Colab T4).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh          # every step below, in order
```

Or step by step:

```bash
python -m src.download --fasttext   # data/raw + fastText model in .cache/
python -m src.data                  # cleaning, masking, templates, splits
python -m src.train_classical       # E1-E3, E3b, attack sets, E7/E9 for classical models
python -m src.embeddings            # fastText vectors for the BiLSTM
python -m src.train_bilstm          # E4
python -m src.train_transformer --model afroxlmr xlmr --seeds 13 42 2026   # E5
python -m src.evaluate && python -m src.figures && python -m src.tables
python app/app.py                   # local app (set WDYRI_MODEL to a local model folder or Hub id)
```

Unit tests for masking, defences, templates, attacks and thresholds: `python -m unittest discover tests`.

Every run writes per-message predictions to `results/predictions/`; `src.evaluate` turns them into `results/experiments.csv` (mean ± sd over seeds) and `results/experiments_by_seed.csv`, so metrics can be recomputed without retraining.

## Repository

```
wdyri/
├── README.md, DECISIONS.md, LICENSE, requirements.txt
├── data/            README.md (sources, licences, cleaning), processed/ (masked text only)
├── src/             preprocess.py, split.py, data.py, classical.py, perturb.py, eval_sets.py,
│                    train_classical.py, embeddings.py, train_bilstm.py, train_transformer.py,
│                    variants.py, metrics.py, evaluate.py, errors.py, figures.py, tables.py
├── notebooks/       01_eda … 07_errors (analysis), colab_app (fallback demo)
├── app/             app.py (Gradio), settings.json, baseline_lr_char.joblib, requirements.txt
├── scripts/         run_all.sh, export_app.py, publish_hf.py, make_notebooks.py
├── tests/           test_core.py (unittest)
└── results/         experiments.csv, tables.md, figures/, predictions/, error_buckets.csv
```

## Limitations

* Small data: 1,064 Swahili messages; test sets of 152 messages, one of them dominated by a single script.
* BongoScam's genuine class is casual chat with no service texts, so in-language scores overstate real-world performance (the phone-number shortcut).
* The Chichewa fraud set is partly machine-augmented by label-preserving rewrites, and originals are not marked.
* Attacks and adversarial training share the same code, so E7 measures robustness to known tricks only.
* No sender metadata or conversation context, which the impersonation scams need.

## References

Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In *Proceedings of the 29th International Conference on Computational Linguistics* (pp. 4336-4349). https://aclanthology.org/2022.coling-1.382

Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135-146. https://doi.org/10.1162/tacl_a_00051

Chiuseni, D., Bahizire, A., Hama, S., & Ndibwile, J. D. (2026). *Adversarial robustness in smishing detection: A comparative analysis of adversarial fragility in classical vs. transformer-based detection systems* (arXiv:2608.12889). arXiv. https://arxiv.org/abs/2608.12889

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440-8451). https://doi.org/10.18653/v1/2020.acl-main.747

Dioniz, H. (2024). *Swahili SMS detection dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset

Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). Learning word vectors for 157 languages. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*. https://aclanthology.org/L18-1550

Mambina, I. S., Ndibwile, J. D., & Michael, K. F. (2022). Classifying Swahili smishing attacks for mobile money users: A machine-learning approach. *IEEE Access, 10*, 83061-83074. https://doi.org/10.1109/ACCESS.2022.3196464

Njame, R. A., Sanga, G., & Tende, I. (2026). A machine-learning model for phishing detection in Swahili messages: A case of Tanzania. *East African Journal of Information Technology, 9*(2), 97-114. https://doi.org/10.37284/eajit.9.2.5655

Taylor, A., & Robert, A. (2025). Using machine learning to detect fraudulent SMSs in Chichewa. In *Integrating AI in Science, Management, and Technology (AISMT 2025)*, Communications in Computer and Information Science, vol. 2699. Springer. https://doi.org/10.1007/978-3-032-08260-2_12

Taylor, A., & Robert, A. (2025). *SMS fraud classification dataset for Chichewa* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14607454

Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations* (pp. 38-45). https://doi.org/10.18653/v1/2020.emnlp-demos.6

## Acknowledgements

Datasets by Henry Dioniz (BongoScam) and Taylor & Robert (Chichewa). Pretrained models: XLM-R (Meta AI), AfroXLMR (Alabi et al.), fastText Swahili vectors (Grave et al.). Libraries: scikit-learn, PyTorch, Hugging Face Transformers, gensim, Gradio. Existing resources are cited above; the data pipeline, attacks, experiments, evaluation and app in this repository were implemented for this project.
