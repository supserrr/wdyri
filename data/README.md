# Data

Two public datasets, used for different jobs. They are never mixed.

| Dataset | Language | Rows | Labels | Used for | Licence | Source |
| --- | --- | --- | --- | --- | --- | --- |
| BongoScam / Swahili SMS Detection Dataset (Dioniz, 2024a) | Swahili (Tanzania) | 1,508 | scam 1,000 / trust 508 | training, validation, in-language test | MIT | [Kaggle](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset) |
| SMS Fraud Classification dataset for Chichewa (Taylor & Robert, 2025b) | Chichewa (Malawi) | 824 (sheets `D_CHI` + `telcoSMS_CHI`) | fraud 338 / normal 486 | zero-shot transfer test only (RQ3) | CC BY 4.0 | [Zenodo 10.5281/zenodo.14607454](https://doi.org/10.5281/zenodo.14607454) |

## Get the raw files

```bash
python -m src.download             # both datasets -> data/raw/
python -m src.download --fasttext  # also the 2.7 GB fastText Swahili model -> .cache/
```

`data/raw/` is git-ignored: the raw messages contain real phone numbers.

## Processed files (committed)

Built by `python -m src.data` and `python -m src.train_classical`. All text is masked: phone numbers, money amounts and links are replaced by `<PHONE>`, `<AMOUNT>` and `<URL>`.

| File | Contents |
| --- | --- |
| `processed/bongo.csv` | 1,064 masked Swahili SMS: `id, label, text, template_id, split_random, split_template` |
| `processed/chichewa.csv` | 733 masked Chichewa SMS (336 fraud): `id, orig_id, source, label, text, template_id` |
| `processed/eval_template.csv` | every evaluation set for the template split: validation, clean test, 9 attacked test sets, Chichewa and genuine-text controls (each with a `+norm` copy); a held-out lookalike attack the defence was not written for (`test_unseen_all`, with `+norm`); the stress-test minimal pairs for test and validation (`stress_*`, `valstress_*`); full-intensity controls (`ctrlall_*`); attacks from a second attacker (`test_xatk_*`) |
| `processed/eval_random.csv` | validation and test sets for the random split |
| `processed/eval_template_r1.csv`, `eval_template_r2.csv` | validation and test sets of the two fresh template-disjoint splits (E12 check) |
| `processed/adversarial_copies_template.csv` | the 101 machine-perturbed scam copies used for adversarial training (E7), with the template of the message each was copied from |

Labels: `1` = scam (BongoScam "scam", Chichewa "fraud"), `0` = not scam.

## Cleaning, in order

| Step | BongoScam | Chichewa |
| --- | --- | --- |
| Raw rows | 1,508 (1,000 scam, 508 trust) | 824 (338 fraud, 338 normal, 148 telco normal) |
| Drop exact duplicates | -131 | (counted together with the next step) |
| Mask numbers, amounts, links; drop texts that become identical | -313 (all scams) | -91 |
| Final | **1,064** (567 scam, 497 not scam) | **733** (336 fraud, 338 normal, 59 telco normal) |
| Near-duplicate templates (character 5-gram Jaccard >= 0.8) | 909 templates; 73 have 2+ members | 670 |

Template-disjoint split (seed 42): train 760 (405 scam), validation 152 (81 scam), test 152 (81 scam), no template in two parts. The random split leaks 40 templates across parts.

## Known limits

* **Provenance.** The BongoScam author describes the data as Tanzanian Swahili SMS "showcasing various scam patterns" but does not document how messages were collected or labelled. The scams carry real-looking numbers and names; the genuine texts read like composed personal chat. We treat the labels as given and test what that design does to models.
* **Shortcut 1: numbers.** None of the 497 genuine messages contains a phone number or link and only one contains an amount, while 86% of scams contain a phone number or link. The fixed rule "has a phone number or link" (no training) scores F1 0.92 across all 1,064 cleaned messages and 0.74 on the template-split test set, whose largest scam template has no number. Real inboxes also hold bank, telco and M-Pesa service texts full of numbers; the Chichewa set has such texts and shows what happens then (E8, E10).
* **Shortcut 2: length.** Every scam has at least 8 words after masking (median 12); genuine texts have a median of 7. A rule "at least 9 words" (k chosen on the training set) scores test F1 0.85. Both shortcuts are reported as baselines in E3b.
* The Chichewa fraud set grew from 126 original messages by label-preserving rewrites (Taylor & Robert, 2025a); the files do not mark which are originals, so we also report one message per template.
* Names in scam texts (payee names) are kept: masking them would need a Swahili name recogniser.
* One Chichewa fraud message gives a contact email address. It is the scammer's published contact, not a victim's, so it is kept as the dataset authors released it; masking it would change one evaluation message after every model was scored.
* The masking rules were written after looking at samples of both datasets, so they recognise Tanzanian and Malawian number and currency formats. No model, threshold or hyperparameter was tuned on the Chichewa set.
* The Chichewa rewrites were produced by the dataset authors (Taylor & Robert, 2025a) to enlarge 126 original fraud messages; their share among the 336 fraud messages we keep is not recorded in the files. The few-shot runs (E9) add 20 or 50 of these messages to training and drop every message sharing a template with them from the test.

## Licences

Both licences allow redistribution of the processed, masked text committed here. The MIT notice for BongoScam and the CC BY 4.0 attribution for the Chichewa set, with the changes made, are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Citations

* Dioniz, H. (2024). *Swahili SMS Detection Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset
* Taylor, A., & Robert, A. (2025a). Using machine learning to detect fraudulent SMSs in Chichewa. In *Integrating AI in Science, Management, and Technology (AISMT 2025)*, CCIS vol. 2699. Springer. https://doi.org/10.1007/978-3-032-08260-2_12
* Taylor, A., & Robert, A. (2025b). *SMS Fraud Classification dataset for Chichewa* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14607454
