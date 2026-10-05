#!/usr/bin/env bash
# Reproduce every result in the README, in order.
# Classical models take a few minutes; each transformer run took 5-14 minutes
# on a laptop CPU (a GPU is much faster).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.download --fasttext        # raw data + fastText Swahili model (2.7 GB download)
python -m src.data                        # clean, mask, find templates, make both splits
rm -f results/predictions/*.csv           # start clean: every prediction below is regenerated
python -m src.train_classical             # E1-E3, E3b, attack/stress/control sets, E7, E9, E11, fresh splits
python -m src.embeddings                  # fastText vectors for every token the BiLSTM will see

# BiLSTM (E4, E7, E3b, E11)
python -m src.train_bilstm --emb random frozen finetuned --variant clean
python -m src.train_bilstm --emb finetuned frozen --variant advtrain
python -m src.train_bilstm --emb finetuned --variant strip counterfactual

# Transformers. Weights are saved (--save) for the clean and number-balanced runs,
# which the app, the stress tests and notebook 07 reuse.
python -m src.train_transformer --model afroxlmr xlmr --variant clean --save                  # E5, E6, E8, E10
python -m src.train_transformer --model afroxlmr --variant counterfactual --save               # E11
python -m src.train_transformer --model afroxlmr --variant advtrain strip                      # E7, E3b
python -m src.train_transformer --model afroxlmr xlmr --variant fewshot20 fewshot50            # E9
python -m src.train_transformer --model afroxlmr --variant clean counterfactual \
    --split template_r1 template_r2 --seeds 42                                                   # E12 fresh splits

python -m src.ensemble                    # E12 from saved predictions
python -m src.evaluate                    # every prediction file -> results/experiments.csv
python -m src.stress                      # E10 minimal pairs (test and validation)
python -m src.significance                # paired bootstrap and rank tests
python -m src.errors                      # error buckets
python -m src.figures
python -m src.tables                      # results/tables.md and the README results table
python scripts/export_app.py --variant counterfactual --baseline-variant counterfactual   # chosen on validation stress pairs
python scripts/export_web.py              # browser model (ONNX) + JS assets, with a parity check
node scripts/web_parity.mjs               # JavaScript preprocessing and n-gram model vs Python
python scripts/make_notebooks.py
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
