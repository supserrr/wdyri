"""Paths, seeds and shared settings for every script in the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
MODELS = ROOT / "models"
CACHE = ROOT / ".cache"

BONGO_CSV = DATA_RAW / "bongo_scam.csv"
CHICHEWA_XLSX = DATA_RAW / "SMS_Fraud_Chichewa_Dataset_SM.xlsx"

BONGO_URL = "https://www.kaggle.com/api/v1/datasets/download/henrydioniz/swahili-sms-detection-dataset"
CHICHEWA_URL = "https://zenodo.org/api/records/14607454/files/SMS_Fraud_Chichewa_Dataset_SM.xlsx/content"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sw.300.bin.gz"
# SHA-256 of the raw files used for every reported result.
SHA256 = {
    "bongo_scam.csv": "7675f7646395367f068c16789ce23e5d6bee4f11e92ee3bdfff42b8c168b7f44",
    "SMS_Fraud_Chichewa_Dataset_SM.xlsx": "4f83cfaab196f8fab3bdbf9c89e15313ddaa889da066335fcc2f35cc6b3f487a",
}

# One fixed seed builds the data splits; neural models are trained with all three.
SPLIT_SEED = 42
SEEDS = (13, 42, 2026)

# The positive class everywhere is "scam" (BongoScam) or "fraud" (Chichewa) = 1.
LABEL_NAMES = {0: "not scam", 1: "scam"}

# Template detection: two messages belong to one template when their masked,
# lower-cased texts share at least this share of character 5-grams (Jaccard).
TEMPLATE_NGRAM = 5
TEMPLATE_JACCARD = 0.8

# Operating point: the threshold is the highest one that still catches this
# share of scams in the validation set.
TARGET_RECALL = 0.95

TRANSFORMERS = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "afroxlmr": "Davlan/afro-xlmr-base",
}

for _dir in (DATA_PROCESSED, FIGURES, PREDICTIONS, MODELS, CACHE):
    _dir.mkdir(parents=True, exist_ok=True)
