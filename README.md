<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img src="docs/assets/logo.svg" alt="" height="72">
</picture>

# WDYRI: Why Did You Redeem It?

[![Live app](https://img.shields.io/badge/Live_app-Hugging_Face_Space-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/supserrr/wdyri)
[![Model](https://img.shields.io/badge/Model-wdyri--afroxlmr-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/supserrr/wdyri-afroxlmr)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Swahili mobile-money scam detectors look near-perfect. WDYRI shows they are often right for the wrong reason, measures how much, and reduces it.

[Overview](#overview) • [Key findings](#key-findings) • [Try it](#try-it) • [Getting started](#getting-started) • [Documentation](#documentation)

</div>

> [!TIP]
> Nothing to install: the [live app](https://huggingface.co/spaces/supserrr/wdyri) runs both models in your browser. Paste a Swahili SMS or pick one of the examples.

## Overview

Mobile-money scams arrive as SMS that pose as relatives, agents, landlords or employers and ask the victim to send money. In Tanzania a common hook is *"ni tumie kwa namba hii"* (send it to me on this number). Published Swahili detectors report 98.7% to 99.86% accuracy. WDYRI reproduces that result, then tests what earlier work skipped:

- **Leakage:** how much of the score survives when no scam template appears in both train and test?
- **Robustness:** how far does each model fall under lookalike letters, split words and English code-switching, and do simple defences recover it?
- **Transfer:** can a model trained only on Swahili catch real Chichewa fraud SMS from Malawi?

Answering these raised a fourth question that became the main contribution: what are these models actually reading? The work is aimed at mobile-money customers checking a suspicious SMS, and at the telcos and fintechs that could run the check for them.

| Dataset | Language | Messages used | Role |
| --- | --- | --- | --- |
| [BongoScam](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset) | Swahili, Tanzania | 1,064 (567 scam) | train, validation and test |
| [Chichewa SMS fraud](https://doi.org/10.5281/zenodo.14607454) | Chichewa, Malawi | 733 (336 fraud) | zero-shot transfer test |

The models form a ladder: rule baselines and Naive Bayes, character n-gram logistic regression, a BiLSTM over fastText vectors, and fine-tuned XLM-R and AfroXLMR. The deployed system flags a message if either the character n-gram model or AfroXLMR flags it.

## Key findings

- **The published score reproduces.** Naive Bayes reaches 98.68% accuracy, matching BongoScam's 98.7%.
- **The models lean on phone numbers.** 86% of scams and no genuine texts contain a phone number or link. Adding a phone number to a genuine message makes fine-tuned AfroXLMR call it a scam 96% of the time.
- **A targeted fix works.** Training with numbers in both classes cuts those false alarms to 0%, and misses on scams with their number removed from 20% to 2%.
- **Two models beat one.** The deployed ensemble scores test F1 0.994 and keeps 0.95–0.99 under lookalike-letter and split-word attacks.
- **Chichewa transfer is weak.** Zero-shot fraud F1 is 0.46–0.74; twenty Chichewa examples lift the character n-gram model from 0.65 to 0.82.

| System (template-disjoint test split) | Test F1 | Chichewa F1 (zero-shot) |
| --- | --- | --- |
| Phone rule: "contains a number" | 0.744 | 0.681 |
| Naive Bayes, word counts (published baseline) | 0.982 | 0.608 |
| Logistic regression, character 2–5-grams | 0.968 | 0.651 |
| AfroXLMR base | 0.776 ± 0.021 | 0.719 ± 0.019 |
| AfroXLMR base, number-balanced | 0.820 ± 0.035 | 0.716 ± 0.033 |
| **Ensemble, both number-balanced (deployed)** | **0.994** | 0.715 ± 0.034 |

> [!NOTE]
> The transformers score lower here because one unseen landlord-impersonation script, with no number or link, makes up 31 of the 81 test scams and they miss it. Every model, figure and significance test is in [docs/results.md](docs/results.md).

## Try it

The [live app](https://huggingface.co/spaces/supserrr/wdyri) is a free static Space. AfroXLMR runs through transformers.js and ONNX Runtime Web, and the character n-gram model through a JavaScript port checked against Python. For each message it shows both models' scam probabilities, the words that drove them and the masked text they read.

- **Fallback demo:** [notebooks/colab_app.ipynb](notebooks/colab_app.ipynb) runs the same ensemble on Colab with a public link.
- **Locally:** `python app/app.py` runs it with Gradio.
- **Demo video:** to be added.

> [!IMPORTANT]
> WDYRI is a research demo, not a guarantee. When in doubt about a message, call your provider on its official number.

## Getting started

You need Python 3.12. Everything runs on a laptop CPU, where each transformer run takes 5–14 minutes; a GPU is much faster.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh          # every experiment, table, figure and notebook, in order
python -m unittest discover tests
python app/app.py                # the Gradio app
```

[docs/reproduce.md](docs/reproduce.md) explains each pipeline step and where its outputs go.

## Project structure

```
wdyri/
├── src/          data pipeline, models, attacks, evaluation, figures and tables
├── notebooks/    01_eda … 08_shortcut_and_ensemble, plus colab_app (fallback demo)
├── web/          the deployed browser app
├── app/          the same app in Gradio
├── scripts/      run_all.sh, model exports and publishing
├── results/      metrics, per-message predictions, significance tests and figures
├── data/         dataset notes: sources, licences, cleaning and limits
└── docs/         the detailed write-up
```

## Documentation

| Page | Contents |
| --- | --- |
| [Research questions and contributions](docs/research.md) | the problem, prior work, what is new, findings and related work |
| [Data and models](docs/methods.md) | datasets, cleaning, splits, the model ladder and training settings |
| [Results](docs/results.md) | main results table, figures, experiments E1–E12, error analysis and limitations |
| [Deployment](docs/deployment.md) | how the browser app runs both models and how it was checked |
| [Reproduce](docs/reproduce.md) | the full pipeline and repository layout |
| [References](docs/references.md) | papers, datasets and acknowledgements |

Design choices and their reasons are logged in [DECISIONS.md](DECISIONS.md), dataset details are in [data/README.md](data/README.md), and every results table is in [results/tables.md](results/tables.md).
