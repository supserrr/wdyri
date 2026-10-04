"""Rungs 4-5: fine-tune XLM-R base or AfroXLMR base (E5, plus E6-E9).

Data flow for one SMS:
    text -> SentencePiece sub-word ids (max 128) -> 12-layer Transformer encoder
         -> vector of the first token (<s>) -> dropout + linear head -> 2 logits
         -> softmax -> P(scam)

Training: AdamW, linear warm-up (10%) then decay, batch 16, up to 5 epochs,
class-weighted cross-entropy, early stopping on validation scam F1.

Usage:
    python -m src.train_transformer --model afroxlmr --variant clean --seeds 13 42 2026
    python -m src.train_transformer --model afroxlmr --seeds 42 --save   # keeps the weights in models/
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import config, runs, variants

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def encode(tokenizer, texts, max_len: int):
    return tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tokenizer, texts, max_len: int = 128, batch: int = 64) -> np.ndarray:
    model.eval()
    texts = list(texts)
    # Sorting by length keeps padding (and time) low; the order is restored at the end.
    order = np.argsort([len(t) for t in texts])
    probs = np.empty(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch):
        idx = order[start:start + batch]
        enc = encode(tokenizer, [texts[i] for i in idx], max_len).to(DEVICE)
        probs[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
    return probs


def train_one(model_key: str, variant: str, split: str, seed: int, lr: float, epochs: int = 5,
              batch: int = 16, max_len: int = 128, patience: int = 2, save: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = config.TRANSFORMERS[model_key]
    train, val, rows = variants.frames(split, variant, seed)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEVICE)

    steps = epochs * int(np.ceil(len(train) / batch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    counts = np.bincount(train.label, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32).to(DEVICE))

    best, best_key, bad, history = None, (-1.0, 0.0), 0, []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = encode(tokenizer, train.text.iloc[idx], max_len).to(DEVICE)
            y = torch.tensor(train.label.iloc[idx].values).to(DEVICE)
            loss = loss_fn(model(**enc).logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        p_val = predict(model, tokenizer, val.text, max_len)
        f1 = f1_score(val.label, p_val >= 0.5, zero_division=0)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.tensor(p_val).clamp(1e-6, 1 - 1e-6), torch.tensor(val.label.values, dtype=torch.float32)))
        history.append({"epoch": epoch + 1, "val_f1": round(f1, 4), "val_loss": round(val_loss, 4)})
        if (f1, -val_loss) > best_key:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_key, bad = (f1, -val_loss), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    runs.save(model_key, variant if lr == DEFAULT_LR else f"{variant}_lr{lr:g}", split, seed,
              rows, predict(model, tokenizer, rows.text, max_len))
    if save:
        out = config.MODELS / runs.run_name(model_key, variant, split, seed)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
    del model, best
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return {"history": history, "best_val_f1": best_key[0]}


DEFAULT_LR = 3e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["xlmr", "afroxlmr"], choices=list(config.TRANSFORMERS))
    parser.add_argument("--variant", nargs="+", default=["clean"])
    parser.add_argument("--split", nargs="+", default=["template"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    parser.add_argument("--lr", nargs="+", type=float, default=[DEFAULT_LR])
    parser.add_argument("--save", action="store_true", help="save each trained model under models/")
    args = parser.parse_args()
    log = config.RESULTS / "transformer_log.jsonl"
    print("device:", DEVICE, flush=True)
    for split in args.split:
        for variant in args.variant:
            for model_key in args.model:
                for lr in args.lr:
                    for seed in args.seeds:
                        start = time.time()
                        info = train_one(model_key, variant, split, seed, lr, save=args.save)
                        info.update(model=model_key, variant=variant, split=split, seed=seed, lr=lr,
                                    seconds=round(time.time() - start))
                        with log.open("a") as fh:
                            fh.write(json.dumps(info) + "\n")
                        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
