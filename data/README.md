# Data

Two public datasets, used for different jobs. They are never mixed.

| Dataset | Language | Rows | Labels | Used for | Licence | Source |
| --- | --- | --- | --- | --- | --- | --- |
| BongoScam / Swahili SMS Detection Dataset (Dioniz, 2024) | Swahili (Tanzania) | 1,508 | scam 1,000 / trust 508 | training, validation, in-language test | MIT | [Kaggle](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset) |
| SMS Fraud Classification dataset for Chichewa (Taylor & Robert, 2025) | Chichewa (Malawi) | 824 (sheets `D_CHI` + `telcoSMS_CHI`) | fraud 338 / normal 486 | zero-shot transfer test only (RQ3) | CC BY 4.0 | [Zenodo 10.5281/zenodo.14607454](https://doi.org/10.5281/zenodo.14607454) |

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
| `processed/eval_template.csv` | every evaluation set for the template split: clean test, 9 attacked test sets, Chichewa, and a `+norm` copy of each |
| `processed/eval_random.csv` | validation and test sets for the random split |
| `processed/adversarial_copies_template.csv` | the 101 machine-perturbed scam copies used for adversarial training (E7) |

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

* BongoScam's genuine class is mostly casual personal chat: none of its 497 genuine messages contains a phone number and only one contains an amount, while 85% of scams contain a number. A rule "has a phone number or link" already reaches F1 0.92. Real inboxes also hold bank, telco and M-Pesa service texts full of numbers; the Chichewa set has such texts and shows what happens then.
* The Chichewa fraud set grew from 126 original messages by label-preserving rewrites (Taylor & Robert, 2025); the files do not mark which are originals, so we also report one message per template.
* Names in scam texts (payee names) are kept: masking them would need a Swahili name recogniser.

## Citations

* Dioniz, H. (2024). *Swahili SMS Detection Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset
* Taylor, A., & Robert, A. (2025). *SMS Fraud Classification dataset for Chichewa* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14607454
