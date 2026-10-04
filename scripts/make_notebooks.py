"""Write the analysis notebooks in notebooks/ (run them with scripts/run_all.sh).

The notebooks only read results and call src/ functions; all training happens
in the src/ scripts, so a notebook re-runs in seconds.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys, os, json
sys.path.insert(0, os.path.abspath('..'))
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath('../.cache/mpl'))
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option('display.max_colwidth', 140)
pd.set_option('display.width', 200)
from src import config
S = pd.read_csv(config.RESULTS / 'experiments.csv')          # mean (and sd) over seeds
BY_SEED = pd.read_csv(config.RESULTS / 'experiments_by_seed.csv')

def table(models, sets, variant='clean', split='template', metric='f1'):
    t = S[S.model.isin(models) & S.set.isin(sets) & (S.variant == variant) & (S.split == split)]
    return t.pivot_table(index='model', columns='set', values=metric).reindex(models).round(3)"""

NOTEBOOKS = {
    "01_eda": [
        ("md", "# 01 Data exploration\n\nWhat is in BongoScam and the Chichewa set, how cleaning changes them, and the shortcut that makes BongoScam easy.\n\nRegenerate the processed files with `python -m src.data`."),
        ("code", SETUP),
        ("code", "stats = json.loads((config.RESULTS / 'data_stats.json').read_text())\n{k: v for k, v in stats.items() if not k.startswith('split')}"),
        ("md", "## Raw BongoScam: labels and duplicates\n\nThe raw file has 1,508 rows; 131 are exact copies. After masking numbers, amounts and links, another 313 scams become identical to an earlier message (same script, different number), so they are dropped too."),
        ("code", "from src.data import load_bongo_raw, load_bongo, load_chichewa\nraw = pd.read_csv(config.BONGO_CSV)\nprint(raw.Category.value_counts(), '\\nexact duplicates:', raw.Sms.duplicated().sum())\nbongo = load_bongo()\nbongo.label.map(config.LABEL_NAMES).value_counts()"),
        ("md", "## What masking does"),
        ("code", "from src.preprocess import preprocess\nfor t in load_bongo_raw().raw_text.sample(6, random_state=3):\n    print('RAW :', t[:120]); print('SEEN:', preprocess(t)[:120]); print()"),
        ("md", "## Message length"),
        ("code", "bongo['words'] = bongo.text.str.split().str.len()\nbongo.groupby(bongo.label.map(config.LABEL_NAMES)).words.describe().round(1)"),
        ("md", "## Templates: the same script sent many times\n\nTwo messages share a template when their masked, lower-cased texts share at least 80% of character 5-grams. The template-disjoint split keeps each template on one side."),
        ("code", "sizes = bongo.groupby('template_id').size().sort_values(ascending=False)\nprint('templates:', len(sizes), '| with 2+ members:', (sizes > 1).sum())\nbig = sizes.head(8).index\nbongo[bongo.template_id.isin(big)].groupby('template_id').agg(n=('id', 'size'), split=('split_template', 'first'), example=('text', 'first')).sort_values('n', ascending=False)"),
        ("code", "pd.DataFrame({k: stats[k] for k in ('split_random', 'split_template')}).T"),
        ("md", "Templates crossing parts of each split (0 means no leakage):"),
        ("code", "{k: stats[k] for k in ('split_random_templates_crossing_parts', 'split_template_templates_crossing_parts')}"),
        ("md", "## The shortcut: genuine BongoScam messages never contain a number\n\nThe genuine class is casual personal chat. So *whether a message contains a phone number* almost decides the label, in Swahili but not in Chichewa, where genuine texts include telco service messages."),
        ("code", "chi = load_chichewa()\nrows = []\nfor name, df in [('BongoScam', bongo), ('Chichewa', chi)]:\n    for p in ['<PHONE>', '<AMOUNT>', '<URL>']:\n        r = df.groupby('label').text.apply(lambda s: s.str.contains(p, regex=False).mean())\n        rows.append({'data': name, 'placeholder': p, 'share of genuine': r[0], 'share of scams': r[1]})\npd.DataFrame(rows).round(3)"),
        ("md", "## Chichewa set (sealed until RQ3)"),
        ("code", "print(chi.groupby(['source', 'label']).size())\nchi.sample(5, random_state=1)[['source', 'label', 'text']]"),
    ],
    "02_baselines": [
        ("md", "# 02 Baselines and leakage (E1, E2, E3, E3b)\n\nTrain with `python -m src.train_classical`, score with `python -m src.evaluate`."),
        ("code", SETUP),
        ("md", "## E1: reproduce the published BongoScam result\n\nAll 1,508 raw rows, `train_test_split(test_size=0.2, random_state=42)`, default `CountVectorizer` + `MultinomialNB`, as in the BongoScam notebook. They report 98.7% accuracy."),
        ("code", "S[(S.variant == 'published') & (S.set == 'test')][['model', 'n', 'accuracy', 'precision', 'recall', 'f1']]"),
        ("md", "## E2: random vs template-disjoint split (RQ1)\n\nOne split of each kind, then 10 re-drawn splits of each kind so one lucky test set cannot decide the answer."),
        ("code", "models = ['majority', 'nb_word_counts', 'nb_char', 'lr_word', 'lr_char']\nt = S[S.model.isin(models) & (S.variant == 'clean') & (S.set == 'test')]\nt.pivot_table(index='model', columns='split', values='f1').reindex(models).round(3)"),
        ("code", "rep = BY_SEED[(BY_SEED.variant == 'repeat') & (BY_SEED.set == 'test')]\nrep.groupby(['model', 'split']).f1.agg(['mean', 'std', 'min', 'max']).round(3)"),
        ("code", "display(Image(str(config.FIGURES / 'rq1_repeated_splits.png')))"),
        ("md", "**Reading.** Word-level models lose little; character n-gram models lose 3-7 points of F1 when templates cannot leak. Character n-grams memorise the surface form of a script (names, spellings), which a random split rewards."),
        ("md", "## E3: word vs character features, masking on vs off"),
        ("code", "t = S[S.model.isin(models[1:]) & S.variant.isin(['clean', 'nomask']) & (S.split == 'template') & (S.set == 'test')]\nt.pivot_table(index='model', columns='variant', values='f1').round(3)"),
        ("md", "## E3b: the phone-number shortcut\n\n`phone_rule` flags any message with a `<PHONE>` or `<URL>`. The `strip` variant deletes the placeholders everywhere, so models must use the words."),
        ("code", "t = S[S.model.isin(['phone_rule'] + models[1:]) & S.variant.isin(['clean', 'strip']) & (S.split == 'template') & S.set.isin(['test', 'test/tpl', 'chichewa/all'])]\nt.pivot_table(index=['model', 'variant'], columns='set', values='f1').round(3)"),
        ("md", "**Reading.** In Swahili the words alone separate the classes (stripping placeholders barely moves test F1). On Chichewa the classical models lose most of their transfer without the placeholder: what transferred was \"contains a number\", not language."),
        ("md", "## What logistic regression learned"),
        ("code", "feats = json.loads((config.RESULTS / 'lr_top_features.json').read_text())\npd.DataFrame({'scam (word LR)': [f'{w} ({c:.2f})' for w, c in feats['lr_word']['scam'][:15]],\n              'not scam (word LR)': [f'{w} ({c:.2f})' for w, c in feats['lr_word']['not_scam'][:15]],\n              'scam (char LR)': [repr(w) for w, c in feats['lr_char']['scam'][:15]]})"),
    ],
    "03_bilstm": [
        ("md", "# 03 BiLSTM with fastText embeddings (E4)\n\n`python -m src.embeddings` caches fastText vectors; `python -m src.train_bilstm` trains random / frozen / fine-tuned embeddings with 3 seeds each.\n\n**Architecture:** tokens -> 300-d embedding -> BiLSTM (2 x 128) -> max-pool over time -> dropout 0.3 -> linear -> sigmoid."),
        ("code", SETUP),
        ("code", "import torch\nfrom src.train_bilstm import BiLSTM\nm = BiLSTM(torch.zeros(5000, 300), trainable=True)\nprint(m)\nprint('trainable parameters without the embedding:', sum(p.numel() for n, p in m.named_parameters() if not n.startswith('emb')))"),
        ("md", "## Results over 3 seeds"),
        ("code", "models = ['bilstm_random', 'bilstm_frozen', 'bilstm_finetuned']\nt = S[S.model.isin(models) & (S.variant == 'clean') & S.set.isin(['val', 'test', 'test/tpl', 'chichewa/all'])]\nt.pivot_table(index='model', columns='set', values=['f1', 'f1_sd']).round(3)"),
        ("code", "BY_SEED[BY_SEED.model.isin(models) & (BY_SEED.variant == 'clean') & (BY_SEED.set == 'test')][['model', 'seed', 'precision', 'recall', 'f1', 'pr_auc']].round(3)"),
        ("md", "**Reading.** Validation F1 is 1.0 for every variant, yet test F1 is 0.78-0.86: the test set's largest template (31 landlord-impersonation scams with no phone number) is caught by some seeds and missed by others, hence the large spread for fine-tuned vectors. On one message per template all variants score 0.98-0.99. On Chichewa, pretrained vectors raise F1 (0.73-0.74 vs 0.60 from random init), but random init ranks better (PR-AUC 0.80 vs 0.65-0.70): the F1 gain is partly a calibration effect at the 0.5 threshold, not clearly better transfer."),
        ("code", "from src.embeddings import load_cache\nimport numpy as np\ndata = np.load(config.CACHE / 'fasttext_tokens.npz')\nprint(f\"{len(data['tokens'])} tokens cached, {data['in_vocab'].mean():.1%} are whole words in fastText's vocabulary; the rest get sub-word vectors\")"),
    ],
    "04_transformers": [
        ("md", "# 04 Transformers: XLM-R vs AfroXLMR (E5)\n\n`python -m src.train_transformer --model xlmr afroxlmr --seeds 13 42 2026`\n\n**Data flow:** SMS -> SentencePiece sub-words (max 128) -> 12-layer encoder -> `<s>` vector -> dense + tanh -> linear -> softmax -> P(scam).\n\n**Settings:** AdamW, lr 3e-5, weight decay 0.01, 10% warm-up, batch 16, up to 5 epochs, early stopping on validation F1, class-weighted cross-entropy."),
        ("code", SETUP),
        ("code", "from transformers import AutoTokenizer\nfor name in config.TRANSFORMERS.values():\n    tok = AutoTokenizer.from_pretrained(name)\n    print(name, tok.tokenize('Iyo pesa itume kwenye namba hii <PHONE> jina litakuja JUMA'))"),
        ("md", "## Training curves"),
        ("code", "log = pd.read_json(config.RESULTS / 'transformer_log.jsonl', lines=True)\nlog[['model', 'variant', 'seed', 'lr', 'best_val_f1', 'seconds', 'history']]"),
        ("md", "## Results"),
        ("code", "models = ['xlmr', 'afroxlmr']\nt = S[S.model.isin(models) & (S.variant == 'clean') & S.set.isin(['test', 'test/tpl', 'chichewa/all'])]\nt.pivot_table(index='model', columns='set', values=['f1', 'f1_sd', 'pr_auc']).round(3)"),
        ("code", "BY_SEED[BY_SEED.model.isin(models) & (BY_SEED.variant == 'clean') & (BY_SEED.set == 'test')][['model', 'seed', 'precision', 'recall', 'f1', 'pr_auc']].round(3)"),
        ("md", "**Reading.** Both transformers reach validation F1 1.0 and catch every test scam except the landlord template, so their test F1 is lower than the word-level baselines. Higher capacity did not help on this small, easy dataset: the transformers leaned on the strongest cue (a number in the text) and missed the script without one, although its words (*namba yangu*, *mwenye nyumba*) appear only in training scams. AfroXLMR edges XLM-R on test (0.78 vs 0.76) and ranks better (PR-AUC 0.96 vs 0.87)."),
    ],
    "05_robustness": [
        ("md", "# 05 Robustness to obfuscation (E6, E7)\n\nAttacks: `src/perturb.py`. Every model is attacked on the same trigger words (ranked by rung 2's occlusion). Defences: normalisation at inference (`+norm`) and adversarial training (`advtrain`)."),
        ("code", SETUP),
        ("md", "## What the attacks look like"),
        ("code", "ev = pd.read_csv(config.DATA_PROCESSED / 'eval_template.csv', keep_default_na=False)\nclean = ev[ev.set == 'test'].set_index('id').text\nsid = ev[(ev.set == 'test') & (ev.label == 1)].id.iloc[3]\nfor s in ['test', 'test_lookalike_3', 'test_structural_3', 'test_codeswitch_3', 'test_lookalike_all', 'test_lookalike_all+norm']:\n    print(f'{s:26s}', ev[(ev.set == s) & (ev.id == sid)].text.iloc[0])"),
        ("md", "## E6: relative F1 drop"),
        ("code", "display(Image(str(config.FIGURES / 'rq2_attacks.png')))"),
        ("code", "models = ['nb_word_counts', 'lr_word', 'lr_char', 'bilstm_finetuned', 'xlmr', 'afroxlmr']\nsets = [f'test_{a}_{k}' for a in ('lookalike', 'structural', 'codeswitch') for k in ('1', '3', 'all')]\ntable(models, sets, metric='rel_f1_drop')"),
        ("md", "Recall on attacked scams (the share of disguised scams still caught):"),
        ("code", "table(models, ['test'] + sets, metric='recall')"),
        ("md", "## E7: defences"),
        ("code", "display(Image(str(config.FIGURES / 'rq2_defences.png')))"),
        ("code", "rows = []\nfor m in ['nb_word_counts', 'lr_char', 'bilstm_finetuned', 'afroxlmr']:\n    for v in ['clean', 'advtrain']:\n        for suffix in ['', '+norm']:\n            r = S[(S.model == m) & (S.variant == v) & (S.split == 'template')]\n            if r.empty: continue\n            row = {'model': m, 'training': v, 'defence': suffix or 'none', 'clean F1': r[r.set == 'test' + suffix].f1.mean()}\n            for a in ('lookalike', 'structural', 'codeswitch'):\n                row[a] = r[r.set == f'test_{a}_all{suffix}'].f1.mean()\n            rows.append(row)\npd.DataFrame(rows).round(3)"),
        ("md", "**Reading.** Word-level models break under lookalike letters (the disguised words become unseen tokens). Normalisation undoes lookalikes almost completely for classical models at no cost on clean text, but only partly repairs structural attacks: neighbouring spaced-out words are rejoined into one token and glued words stay glued (for the BiLSTM it even removes the oddity that made it suspicious). Transformers *gain* recall under lookalike and structural attacks: unusual characters look suspicious to them, so disguise backfires. Code-switching is the attack normalisation cannot undo; adversarial training cuts the char LR's loss there from 19% to 4%."),
    ],
    "06_transfer": [
        ("md", "# 06 Swahili to Chichewa transfer (E8, E9)\n\nThe Chichewa set was never used for training or tuning (except the few-shot runs, which remove their templates from the test)."),
        ("code", SETUP),
        ("code", "display(Image(str(config.FIGURES / 'rq3_transfer.png')))"),
        ("code", "models = ['majority', 'phone_rule', 'nb_word_counts', 'lr_char', 'bilstm_finetuned', 'xlmr', 'afroxlmr']\nfor metric in ['f1', 'pr_auc', 'precision', 'recall']:\n    display(Markdown(f'**{metric}**'))\n    display(table(models, ['chichewa/all', 'chichewa/dchi', 'chichewa/dedup'], metric=metric))"),
        ("md", "## Few-shot (E9)"),
        ("code", "t = S[S.model.isin(['lr_char', 'afroxlmr', 'xlmr']) & S.variant.isin(['clean', 'fewshot20', 'fewshot50']) & (S.set == 'chichewa/all')]\nt.pivot_table(index='model', columns='variant', values=['f1', 'pr_auc']).round(3)"),
        ("md", "## Where transfer fails: telco service texts\n\nThe Chichewa genuine class includes telco balance and bundle messages, which carry amounts and short codes, unlike any genuine BongoScam text."),
        ("code", "chi = pd.read_csv(config.DATA_PROCESSED / 'chichewa.csv').set_index('id')\np = pd.read_csv(config.PREDICTIONS / 'afroxlmr__clean__template__s42.csv', keep_default_na=False)\np = p[p.set == 'chichewa'].assign(source=lambda d: d.id.map(chi.source), text=lambda d: d.id.map(chi.text))\np.assign(flagged=p.prob >= 0.5).groupby(['source', 'label']).flagged.mean().round(3)"),
    ],
    "07_errors": [
        ("md", "# 07 Error analysis\n\n`python -m src.errors` sorts every error of the main runs into buckets with keyword rules; this notebook reads them and explains the main failure."),
        ("code", SETUP),
        ("code", "counts = pd.read_csv(config.RESULTS / 'error_buckets.csv')\ncounts"),
        ("code", "err = pd.read_csv(config.RESULTS / 'error_examples.csv', keep_default_na=False)\nfor (where, b), g in err.groupby(['where', 'bucket']):\n    display(Markdown(f'**{where} / {b}** ({len(g)} errors over all models)'))\n    display(g.drop_duplicates('text').head(2)[['model', 'label', 'prob', 'text']])"),
        ("md", "## The landlord template: why the neural models miss it\n\n31 of the 81 test scams are one script: *\"Habari ... Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya ...\"* (\"Hello, I am your landlord, this is my new number. Why the silence?\"). No phone number, no amount, no link: the money request comes later in the conversation. Occlusion shows what each model relies on."),
        ("code", "import joblib\nfrom src.classical import scam_proba\nfrom src.perturb import word_effects\nlr = joblib.load(config.MODELS / 'lr_char__template.joblib')\nnb = joblib.load(config.MODELS / 'nb_word_counts__template.joblib')\nlandlord = 'Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya na siku zinazidi kwenda...?'\nclassic = 'Iyo pesa itume kwenye namba hii <PHONE> jina litakuja JUMA ALLY'\nfor name, m in [('char LR', lr), ('word NB', nb)]:\n    f = lambda t, m=m: scam_proba(m, t)\n    for text in (landlord, classic):\n        toks = text.split(' ')\n        eff = word_effects(text, f)[:4]\n        print(f'{name}: P(scam)={f([text])[0]:.2f}', [(toks[i], round(e, 2)) for i, e in eff])"),
        ("code", "path = config.MODELS / 'afroxlmr__clean__template__s42'\nif path.exists():\n    import torch\n    from transformers import AutoTokenizer, AutoModelForSequenceClassification\n    tok = AutoTokenizer.from_pretrained(path); model = AutoModelForSequenceClassification.from_pretrained(path).eval()\n    @torch.no_grad()\n    def tf(texts):\n        enc = tok(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')\n        return torch.softmax(model(**enc).logits, -1)[:, 1].numpy()\n    for label, text in [('landlord', landlord), ('landlord + number', landlord + ' <PHONE>'),\n                        ('classic scam', classic), ('classic scam - number', classic.replace(' <PHONE>', ''))]:\n        print(f'AfroXLMR  {label:22s} P(scam) = {tf([text])[0]:.3f}')\nelse:\n    print('Train and save the model first: python -m src.train_transformer --model afroxlmr --seeds 42 --save')"),
        ("md", "**Reading.** The landlord theme is in training: *namba yangu* (my number) occurs in 49 training messages and *mwenye nyumba* (landlord) in 9, all scams. Word-level NB and LR therefore catch every copy. The transformers do not: their verdict on the classic scam rests on the `<PHONE>` token, and a scam with no number, amount or link reads as genuine (0-13% caught; still only 13-23% for AfroXLMR trained without placeholders). The BiLSTM swings between 0% and 100% across seeds, which is where its large test-F1 spread comes from. The money request in this scam arrives later in the conversation, so a single-message classifier has to rely on the social cue alone."),
    ],
}


COLAB = [
    ("md", "# WDYRI app on Colab (fallback demo)\n\nIf the Hugging Face Space is asleep or down, this notebook runs the same `app/app.py` and prints a temporary public link (`share=True`). A CPU runtime is enough.\n\n[Open in Colab](https://colab.research.google.com/github/supserrr/wdyri/blob/main/notebooks/colab_app.ipynb)"),
    ("code", "!git clone -q https://github.com/supserrr/wdyri.git\n%cd wdyri\n!pip -q install -r app/requirements.txt gradio"),
    ("code", "import os\nos.environ['WDYRI_MODEL'] = 'supserrr/wdyri-afroxlmr'  # the fine-tuned model on the Hugging Face Hub\nos.environ['WDYRI_SHARE'] = '1'\n!python app/app.py"),
    ("md", "## Retrain on a free GPU instead\n\nEverything in the README can be reproduced here (`Runtime > Change runtime type > T4 GPU`); a transformer run takes 1-2 minutes on a T4."),
    ("code", "!pip -q install -r requirements.txt\n!python -m src.download\n!python -m src.data && python -m src.train_classical\n!python -m src.train_transformer --model afroxlmr --seeds 42 --save\n!python -m src.evaluate && python -m src.tables && head -30 results/tables.md"),
]


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    NOTEBOOKS["colab_app"] = COLAB
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, out / f"{name}.ipynb")
    print("wrote", len(NOTEBOOKS), "notebooks")


if __name__ == "__main__":
    main()
