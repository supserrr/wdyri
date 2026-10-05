"""Download the two raw datasets (and optionally the fastText vectors) into data/raw.

Usage:
    python -m src.download            # BongoScam + Chichewa
    python -m src.download --fasttext # also the 2.7 GB Swahili fastText model (unpacked to 3.3 GB)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import io
import shutil
import time
import urllib.request
import zipfile

from . import config


def fetch(url: str, attempts: int = 4) -> bytes:
    """Download with retries: a dropped connection should not stop a reproduction."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed, trusted URLs)
                return resp.read()
        except (OSError, http.client.HTTPException) as err:
            if attempt == attempts:
                raise
            print(f"  retry {attempt} after: {err}")
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def verify(path) -> None:
    """Check the file is byte-identical to the one behind the reported results."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = config.SHA256.get(path.name)
    status = "ok" if digest == expected else f"MISMATCH (expected {expected}, got {digest})"
    print(f"  sha256 {path.name}: {status}")


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
        gz, binary = config.CACHE / "cc.sw.300.bin.gz", config.CACHE / "cc.sw.300.bin"
        if not gz.exists():
            print("fastText Swahili vectors (2.7 GB)...")
            urllib.request.urlretrieve(config.FASTTEXT_URL, gz)  # noqa: S310
        if not binary.exists():
            # gensim cannot read this model reliably from the .gz, so unpack it once.
            with gzip.open(gz, "rb") as src, binary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    for path in (config.BONGO_CSV, config.CHICHEWA_XLSX):
        verify(path)
    print("Done:", sorted(p.name for p in config.DATA_RAW.iterdir()))


if __name__ == "__main__":
    main()
