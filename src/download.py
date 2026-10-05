"""Download the two raw datasets (and optionally the fastText vectors) into data/raw.

Usage:
    python -m src.download            # BongoScam + Chichewa
    python -m src.download --fasttext # also the 2.7 GB Swahili fastText model (unpacked to 3.3 GB)
    python -m src.download --allow-mismatch   # continue if a dataset changed upstream
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


def verify(path, allow_mismatch: bool = False) -> None:
    """Check the file is byte-identical to the one behind the reported results.

    A changed upstream file would silently change every result, so a mismatch
    stops the pipeline unless --allow-mismatch is given.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = config.SHA256.get(path.name)
    if digest == expected:
        print(f"  sha256 {path.name}: ok")
        return
    message = f"sha256 {path.name}: MISMATCH (expected {expected}, got {digest})"
    if not allow_mismatch:
        raise SystemExit(message + "\nThe dataset differs from the one behind the reported results; "
                         "rerun with --allow-mismatch to continue anyway.")
    print(f"  {message} (continuing: --allow-mismatch)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext", action="store_true", help="also download cc.sw.300.bin.gz")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="continue when a dataset's checksum differs from the reported one")
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
        if not binary.exists():
            if not gz.exists():
                print("fastText Swahili vectors (2.7 GB)...")
                urllib.request.urlretrieve(config.FASTTEXT_URL, gz)  # noqa: S310
            # gensim cannot read this model reliably from the .gz, so unpack it once
            # and drop the archive (only the unpacked model is ever read).
            tmp = binary.with_suffix(".tmp")
            with gzip.open(gz, "rb") as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(binary)
            gz.unlink()
    for path in (config.BONGO_CSV, config.CHICHEWA_XLSX):
        verify(path, allow_mismatch=args.allow_mismatch)
    print("Done:", sorted(p.name for p in config.DATA_RAW.iterdir()))


if __name__ == "__main__":
    main()
