# Results

Template-disjoint split (no scam script in both train and test). Full tables for every experiment, with significance tests: [results/tables.md](../results/tables.md).

<!-- RESULTS:START -->
| Model | Test F1 | 95% CI | Test F1 (per template) | Test PR-AUC | Precision (op.) | Recall (op.) | Chichewa F1 (zero-shot) | Chichewa PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Majority class | 0.695 | [0.62, 0.76] | 0.585 | 0.533 | 0.533 | 1.000 | 0.629 | 0.458 |
| Phone rule | 0.744 | [0.66, 0.82] | 0.969 | 0.810 | 1.000 | 0.593 | 0.681 | 0.692 |
| Length rule | 0.850 | [0.79, 0.90] | 0.774 | 0.747 | 0.752 | 0.975 | 0.724 | 0.573 |
| Naive Bayes, word counts | 0.982 | [0.96, 1.00] | 0.971 | 1.000 | 0.964 | 1.000 | 0.608 | 0.648 |
| Naive Bayes, char 2-5 | 0.953 | [0.91, 0.98] | 0.926 | 0.991 | 0.910 | 1.000 | 0.683 | 0.738 |
| Log. regression, word 1-2 | 0.981 | [0.96, 1.00] | 0.990 | 1.000 | 1.000 | 0.963 | 0.456 | 0.727 |
| Log. regression, char 2-5 | 0.968 | [0.94, 0.99] | 1.000 | 1.000 | 1.000 | 0.938 | 0.651 | 0.730 |
| BiLSTM, random init | 0.791 ± 0.027 | [0.71, 0.86] | 0.990 ± 0.000 | 0.966 ± 0.046 | 1.000 ± 0.000 | 0.654 ± 0.037 | 0.597 ± 0.063 | 0.803 ± 0.026 |
| BiLSTM, fastText frozen | 0.778 ± 0.018 | [0.70, 0.85] | 0.983 ± 0.006 | 0.981 ± 0.005 | 0.987 ± 0.011 | 0.642 ± 0.021 | 0.733 ± 0.005 | 0.645 ± 0.031 |
| BiLSTM, fastText fine-tuned | 0.860 ± 0.107 | [0.81, 0.90] | 0.980 ± 0.017 | 0.990 ± 0.007 | 0.980 ± 0.024 | 0.778 ± 0.174 | 0.738 ± 0.005 | 0.701 ± 0.100 |
| XLM-R base | 0.759 ± 0.007 | [0.67, 0.83] | 0.983 ± 0.011 | 0.871 ± 0.035 | 0.987 ± 0.022 | 0.617 ± 0.000 | 0.717 ± 0.012 | 0.654 ± 0.090 |
| AfroXLMR base | 0.776 ± 0.021 | [0.70, 0.84] | 0.990 ± 0.000 | 0.958 ± 0.038 | 1.000 ± 0.000 | 0.634 ± 0.029 | 0.719 ± 0.019 | 0.805 ± 0.042 |
| AfroXLMR base, number-balanced (E11) | 0.820 ± 0.035 | [0.75, 0.88] | 0.990 ± 0.000 | 0.985 ± 0.015 | 1.000 ± 0.000 | 0.696 ± 0.050 | 0.716 ± 0.033 | 0.649 ± 0.005 |
| Ensemble: char LR OR AfroXLMR | 0.968 ± 0.000 | [0.94, 0.99] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.938 ± 0.000 | 0.720 ± 0.018 | 0.805 ± 0.043 |
| Ensemble: char LR OR number-balanced AfroXLMR | 0.968 ± 0.000 | [0.94, 0.99] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.938 ± 0.000 | 0.737 ± 0.017 | 0.648 ± 0.005 |
| Ensemble: number-balanced char LR OR number-balanced AfroXLMR | 0.994 ± 0.000 | [0.98, 1.00] | 0.990 ± 0.000 | 0.996 ± 0.001 | 0.988 ± 0.000 | 1.000 ± 0.000 | 0.715 ± 0.034 | 0.642 ± 0.005 |
<!-- RESULTS:END -->

## Figures

![RQ1](../results/figures/rq1_repeated_splits.png)
![E10 stress tests](../results/figures/e10_stress_tests.png)
![RQ2 attacks](../results/figures/rq2_attacks.png)
![RQ2 defences](../results/figures/rq2_defences.png)
![RQ3](../results/figures/rq3_transfer.png)

## Experiments

| ID | Question | What changes | Main metric | Result |
| --- | --- | --- | --- | --- |
| E1 | Baseline | NB word counts, published setup | accuracy, scam F1 | 0.9868 accuracy (published: 98.7%), F1 0.991 |
| E2 | RQ1 leakage | random vs template split, 1 and 10 re-drawn splits | drop in scam F1 | char models lose 2.6-5.7 points, word models about 1 or less; all significant |
| E3 | Features | word vs char n-grams; masking on vs off | scam F1 | word features score highest on this data; masking changes little in Swahili |
| E3b | Shortcuts | phone rule, length rule, placeholders deleted | scam F1 | the rules alone reach test F1 0.74 and 0.85; without placeholders, classical Chichewa transfer collapses (char LR 0.65 → 0.11) |
| E4 | Embeddings | BiLSTM random / frozen / fine-tuned fastText | scam F1, mean ± sd | fine-tuned 0.86 ± 0.11 (seed-dependent); pretrained vectors raise Chichewa F1 (0.74 vs 0.60) but not PR-AUC |
| E5 | Transformers | XLM-R vs AfroXLMR, 3 seeds | scam F1, mean ± sd | 0.76 ± 0.01 vs 0.78 ± 0.02 (difference not significant, p = 0.39); both miss the landlord script |
| E6 | RQ2 attacks | 3 attacks × 3 intensities, every rung; disguised-genuine controls; a second attacker | relative F1 drop, control false alarms | word LR loses 53% under full lookalike, char LR 21-24%; transformers lose nothing but flag 54-93% of fully disguised genuine texts; same picture with the second attacker |
| E7 | RQ2 defences | normalisation; adversarial training (12% machine-made copies) | relative F1 drop | normalisation: word LR 53% → 1% loss under lookalike; adversarial training: char LR code-switch loss 19% → 4% |
| E8 | RQ3 transfer | Swahili-trained rungs on Chichewa | fraud F1, PR-AUC | trained models 0.46-0.74 (phone rule 0.68, length rule 0.72); AfroXLMR PR-AUC 0.81 vs XLM-R 0.65 (p < 0.001) |
| E9 | RQ3 few-shot | + 20 / 50 Chichewa messages, scored on the same messages | fraud F1 | char LR 0.65 → 0.82 → 0.88; AfroXLMR 0.72 → 0.75 → 0.87; XLM-R 0.72 → 0.70 → 0.78 (AfroXLMR vs XLM-R p < 0.001; char LR vs AfroXLMR n.s.) |
| E10 | Shortcut diagnosis | minimal pairs: + number on genuine, − number on scams | false alarms, misses | AfroXLMR 96% / 20%; XLM-R 40% / 15%; char LR 15% / 2%; word LR 1% / 2% |
| E11 | Shortcut fix | number-balanced counterfactual training | stress tests, F1 everywhere | AfroXLMR 0% / 2%; test PR-AUC 0.958 → 0.985 (F1 0.78 → 0.82, n.s.); Chichewa precision -0.06; char LR and BiLSTM weaker under lookalike |
| E12 | Combination | char LR OR AfroXLMR (plain and number-balanced), chosen on validation; 2 fresh splits; 2 attackers | scam F1 | deployed (both number-balanced): test 0.994, lookalike 0.963, split words 0.948; fresh splits 0.994 and 0.988; +0.22-0.23 over char LR under attack (p < 0.001) |

## Error analysis

Every error of the main runs is sorted into a bucket by `src/errors.py` ([results/error_buckets.csv](../results/error_buckets.csv); two examples per bucket in [notebooks/07_errors.ipynb](../notebooks/07_errors.ipynb)). Bucket names describe the message, not a proven cause.

| Bucket | Typical message | What the evidence says |
| --- | --- | --- |
| Impersonation missed | *"Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya ...?"* (landlord, "this is my new number") | No number, amount or link; the money request comes later in the conversation. Its words occur only in training scams (*namba yangu* in 49, *mwenye nyumba* in 9), so word NB (31 of 31) and word LR (29 of 31) catch it; the plain transformers do not (0-13% caught). AfroXLMR scores it 0.000, and 0.985 once a number is appended. Number-balanced training helps (BiLSTM 68-100%, AfroXLMR 6-32%). |
| Obfuscation missed | *"Іyo p3sа іtumе kwenyе n a m b a hii ..."* | Disguised words become unseen tokens. Normalisation fixes lookalikes, only partly split words, not code-switching. |
| Telco service text flagged | Chichewa balance and bundle messages | BongoScam has no genuine service texts. AfroXLMR flags 67% ± 20% of Chichewa telco texts but 17% ± 4% of other genuine Chichewa texts (3 seeds). |
| Chichewa fraud missed | Malawian scams, e.g. DODMA flood relief and "Foundation" grants | Some scripts are national, but only 14 of AfroXLMR's 113 misses (seed 42) are such local schemes; most are ordinary money requests in an unseen language. |
| Ambiguous | very short texts | Need sender or conversation context. |

## Limitations

* **Small, easy data with undocumented provenance.** 1,064 Swahili messages; the genuine texts read like composed chat, and two shortcuts (numbers, length) separate the classes. In-language scores overstate real-world performance.
* **One split for most neural comparisons.** Validation F1 saturates, so neural comparisons rest on one template-disjoint test set (152 messages) dominated by one script. We add per-template views, bootstrap intervals, significance tests and two fresh splits for the transformer and the ensemble, but not ten re-drawn splits for every neural model.
* **Attacks.** The main trigger words come from a char-LR attacker (white-box for a char LR); a second, Naive Bayes attacker gives the same picture but changes only 84% of scams. Adversarial training uses the same attack code, so E7 measures robustness to known tricks. Transformers resist disguise largely by flagging odd text, including genuine text.
* **Chichewa.** The fraud set is partly machine-rewritten by its authors and originals are not marked; our masking rules were written after looking at both datasets.
* **Post-hoc ensemble and a test-informed tuning revision.** The ensemble idea came from test-set errors (fresh splits and validation-based selection reduce but do not remove that bias), and one change of the classical tuning protocol was prompted by a test result.
* **No sender metadata or conversation context,** which impersonation scams need.
