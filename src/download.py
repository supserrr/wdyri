"""Download the two raw datasets (and optionally the fastText vectors) into data/raw.

Usage:
    python -m src.download            # BongoScam + Chichewa
    python -m src.download --fasttext # also the 2.7 GB Swahili fastText model
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile

from . import config


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed, trusted URLs)
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext", action="store_true", help="also download cc.sw.300.bin.gz")
    args = parser.parse_args()
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    if not config.BONGO_CSV.exists():
        print("BongoScam (Kaggle, MIT licence)...")
        with zipfile.ZipFile(io.BytesIO(fetch(config.BONGO_URL))) as zf:
            zf.extract("bongo_scam.csv", config.DATA_RAW)
    if not config.CHICHEWA_XLSX.exists():
        print("Chichewa SMS fraud dataset (Zenodo, CC BY 4.0)...")
        config.CHICHEWA_XLSX.write_bytes(fetch(config.CHICHEWA_URL))

    if args.fasttext:
        target = config.CACHE / "cc.sw.300.bin.gz"
        if not target.exists():
            print("fastText Swahili vectors (2.7 GB)...")
            urllib.request.urlretrieve(config.FASTTEXT_URL, target)  # noqa: S310
    print("Done:", sorted(p.name for p in config.DATA_RAW.iterdir()))


if __name__ == "__main__":
    main()
