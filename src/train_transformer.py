"""Rungs 4-5: fine-tune XLM-R base or AfroXLMR base (E5, plus E6-E9).

Data flow for one SMS:
    text -> SentencePiece sub-word ids (max 128) -> 12-layer Transformer encoder
         -> vector of the first token (<s>) -> dense + tanh -> linear -> 2 logits
            (the standard XLMRobertaForSequenceClassification head, with dropout)
         -> softmax -> P(scam)

Training: AdamW, linear warm-up (10%) then decay, batch 16, up to 5 epochs,
class-weighted cross-entropy, early stopping on validation scam F1.

Usage:
    python -m src.train_transformer --model afroxlmr --variant clean --seeds 13 42 2026
    python -m src.train_transformer --model afroxlmr --seeds 42 --save   # keeps the weights in models/
    python -m src.train_transformer --model afroxlmr --seeds 42 --only-sets stress_ --from-saved
        # scores sets added later (here the stress tests) with a run's saved weights and
        # merges them into the run's prediction file; fails if the weights are missing

Device: CPU by default, because every reported run was trained on CPU and GPU
kernels give slightly different numbers. Set WDYRI_DEVICE=cuda (or mps) for speed.
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

DEVICE = torch.device(os.environ.get("WDYRI_DEVICE", "cpu"))
DEFAULT_LR = 3e-5


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
              batch: int = 16, max_len: int = 128, patience: int = 2, save: bool = False,
              only_sets: str | None = None, from_saved: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = config.TRANSFORMERS[model_key]
    train, val, rows = variants.frames(split, variant, seed)
    if only_sets:
        prefixes = tuple(p for p in only_sets.split(",") if p)
        rows = rows[rows["set"].str.startswith(prefixes)]
    run_variant = variant if lr == DEFAULT_LR else f"{variant}_lr{lr:g}"
    saved = config.MODELS / runs.run_name(model_key, run_variant, split, seed)
    if from_saved:
        # Never fall back to training: the new sets must come from the same weights as the rest.
        if not saved.exists():
            raise FileNotFoundError(f"--from-saved: no saved weights at {saved} (train with --save first)")
        tokenizer = AutoTokenizer.from_pretrained(saved)
        model = AutoModelForSequenceClassification.from_pretrained(saved).to(DEVICE)
        runs.save(model_key, run_variant, split, seed, rows, predict(model, tokenizer, rows.text, max_len),
                  merge=only_sets is not None)
        return {"history": [], "best_val_f1": None, "from_saved": True}
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
    runs.save(model_key, run_variant, split, seed, rows, predict(model, tokenizer, rows.text, max_len),
              merge=only_sets is not None)
    if save:
        model.save_pretrained(saved)
        tokenizer.save_pretrained(saved)
    del model, best
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return {"history": history, "best_val_f1": best_key[0]}


def log_run(log, info: dict) -> None:
    """One line per training run: a rerun replaces its earlier line instead of adding a duplicate."""
    key = lambda d: (d["model"], d["variant"], d["split"], d["seed"], d["lr"])  # noqa: E731
    lines = [json.loads(line) for line in log.read_text().splitlines() if line] if log.exists() else []
    lines = [d for d in lines if key(d) != key(info)] + [info]
    log.write_text("".join(json.dumps(d) + "\n" for d in lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["xlmr", "afroxlmr"], choices=list(config.TRANSFORMERS))
    parser.add_argument("--variant", nargs="+", default=["clean"])
    parser.add_argument("--split", nargs="+", default=["template"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    parser.add_argument("--lr", nargs="+", type=float, default=[DEFAULT_LR])
    parser.add_argument("--save", action="store_true", help="save each trained model under models/")
    parser.add_argument("--only-sets", default=None,
                        help="predict only sets with these comma-separated prefixes (e.g. stress_ or ctrl_,valstress_)")
    parser.add_argument("--from-saved", action="store_true", help="reuse saved weights instead of training")
    args = parser.parse_args()
    log = config.RESULTS / "transformer_log.jsonl"
    print("device:", DEVICE, flush=True)
    for split in args.split:
        for variant in args.variant:
            for model_key in args.model:
                for lr in args.lr:
                    for seed in args.seeds:
                        start = time.time()
                        info = train_one(model_key, variant, split, seed, lr, save=args.save,
                                         only_sets=args.only_sets, from_saved=args.from_saved)
                        info.update(model=model_key, variant=variant, split=split, seed=seed, lr=lr,
                                    seconds=round(time.time() - start))
                        if not info.get("from_saved"):
                            log_run(log, info)
                        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
