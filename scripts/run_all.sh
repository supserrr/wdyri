#!/usr/bin/env bash
# Reproduce every result in the README, in order.
# Classical models take under a minute; each transformer run takes about
# 10 minutes on a laptop CPU or 1-2 minutes on a free Colab/Kaggle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.download --fasttext        # raw data + fastText Swahili model (2.7 GB download)
python -m src.data                        # clean, mask, find templates, make both splits
python -m src.train_classical             # E1-E3, E3b, attack sets, E7 and E9 for NB/LR
python -m src.embeddings                  # fastText vectors for every token the BiLSTM will see
python -m src.train_bilstm --emb random frozen finetuned --variant clean           # E4
python -m src.train_bilstm --emb finetuned frozen --variant advtrain              # E7
python -m src.train_bilstm --emb finetuned --variant strip                        # E3b
python -m src.train_transformer --model afroxlmr xlmr --seeds 42 --save            # E5 (models kept for the app)
python -m src.train_transformer --model afroxlmr xlmr --seeds 13 2026              # E5
python -m src.train_transformer --model afroxlmr --variant advtrain strip           # E7, E3b
python -m src.train_transformer --model afroxlmr xlmr --variant fewshot20 fewshot50  # E9
python -m src.evaluate                    # every prediction file -> results/experiments.csv
python -m src.errors                      # error buckets
python -m src.figures
python -m src.tables                      # results/tables.md
python scripts/export_app.py              # thresholds for the app
python scripts/make_notebooks.py
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
