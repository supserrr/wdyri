# Reproduce

Python 3.12. Everything runs on a laptop CPU; transformer runs took 5-14 minutes each here (a GPU is much faster).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh        # every experiment, table, figure and notebook, in order
python -m unittest discover tests
python app/app.py              # the app (WDYRI_MODEL = a Hub id or a local model folder)
```

[scripts/run_all.sh](../scripts/run_all.sh) is the canonical order: download, data, classical models, fastText cache, BiLSTM, transformers with saved weights, stress and control predictions, fresh splits, ensemble, evaluation, significance, figures, tables, app settings and notebooks. Every run writes per-message predictions to `results/predictions/`; `src.evaluate` turns them into `results/experiments.csv` (mean ± sd over seeds) and `results/experiments_by_seed.csv`, so every metric can be recomputed without retraining.

## Repository layout

```
wdyri/
├── README.md, DECISIONS.md, LICENSE, requirements.txt
├── data/            README.md (sources, licences, cleaning, limits), processed/ (masked text only)
├── src/             preprocess, split, data, classical, perturb, eval_sets, variants,
│                    train_classical, embeddings, train_bilstm, train_transformer,
│                    ensemble, metrics, evaluate, stress, significance, errors, figures, tables
├── notebooks/       01_eda … 08_shortcut_and_ensemble (analysis), colab_app (fallback demo)
├── web/             the deployed app: index.html, app.js, preprocess.js, ngram.js, plus ui.js and ribbon.js for the page (runs in the browser)
├── app/             app.py (Gradio version), settings.json, baseline_lr_char.joblib, requirements.txt
├── scripts/         run_all.sh, export_app.py, export_web.py, web_parity.mjs, publish_hf.py, make_notebooks.py
├── tests/           test_core.py (17 unit tests)
└── results/         experiments.csv, tables.md, significance.csv, stress_tests.csv, figures/, predictions/
```
