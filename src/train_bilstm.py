"""Rung 3: BiLSTM over fastText Swahili word vectors (experiment E4, plus E6-E8).

Architecture (per message):
    tokens -> embedding (300-d fastText) -> BiLSTM (2 x 128) -> max-pool over time
           -> dropout 0.3 -> linear -> sigmoid = P(scam)

Embedding variants (E4):
    random     trained from scratch; words unseen in training map to <unk>
    frozen     fastText vectors, never updated
    finetuned  fastText vectors, updated with a 10x smaller learning rate

Usage:
    python -m src.train_bilstm --emb frozen finetuned random --variant clean
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

from . import config, runs, variants
from .embeddings import DIM, load_cache, tokenize

PAD, UNK = 0, 1


class BiLSTM(nn.Module):
    def __init__(self, weights: torch.Tensor, trainable: bool, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(weights, freeze=not trainable, padding_idx=PAD)
        self.lstm = nn.LSTM(weights.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.emb(ids))                       # (batch, time, 2*hidden)
        h = h.masked_fill((ids == PAD).unsqueeze(-1), -1e4)   # padding never wins the max
        pooled = h.max(dim=1).values                           # strongest signal per feature
        return self.out(self.drop(pooled)).squeeze(-1)         # logit; sigmoid gives P(scam)


class Vocab:
    """Maps tokens to rows of the embedding matrix."""

    def __init__(self, tokens: list[str]):
        self.itos = ["<pad>", "<unk>"] + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len]
        return ids or [UNK]


def build_vocab(emb: str, train_texts: list[str], cache: dict[str, np.ndarray]):
    rng = np.random.default_rng(0)
    if emb == "random":
        tokens = sorted({t for text in train_texts for t in tokenize(text)})
        weights = rng.normal(0, 0.1, (len(tokens) + 2, DIM))
    else:
        # Every token the model will meet has a fastText vector (built from
        # sub-words if needed), so nothing falls back to <unk>.
        tokens = sorted(cache)
        weights = np.vstack([np.zeros((2, DIM)), np.stack([cache[t] for t in tokens])])
    weights[PAD] = 0
    return Vocab(tokens), torch.tensor(weights, dtype=torch.float32)


def batches(vocab: Vocab, texts, labels=None, size: int = 32, shuffle: bool = False, seed: int = 0):
    order = np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for start in range(0, len(texts), size):
        idx = order[start:start + size]
        seqs = [vocab.encode(texts[i]) for i in idx]
        ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
        for row, seq in enumerate(seqs):
            ids[row, :len(seq)] = torch.tensor(seq)
        y = None if labels is None else torch.tensor([labels[i] for i in idx], dtype=torch.float32)
        yield ids, y


@torch.no_grad()
def predict(model: BiLSTM, vocab: Vocab, texts) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(ids)).numpy() for ids, _ in batches(vocab, list(texts), size=256)])


def train_one(emb: str, variant: str, split: str, seed: int, cache, max_epochs: int = 30, patience: int = 4):
    torch.manual_seed(seed)
    train, val, rows = variants.frames(split, variant, seed)
    vocab, weights = build_vocab(emb, train.text.tolist(), cache)
    model = BiLSTM(weights, trainable=(emb != "frozen"))

    emb_params = [p for p in model.emb.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    groups = [{"params": other, "lr": 1e-3}]
    if emb_params:
        # Pretrained vectors move slowly so 760 messages do not wash them out.
        groups.append({"params": emb_params, "lr": 1e-3 if emb == "random" else 1e-4})
    opt = torch.optim.Adam(groups)
    n_pos, n_neg = int(train.label.sum()), int((train.label == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))  # class-weighted

    best, best_key, bad = None, (-1.0, 0.0), 0
    for epoch in range(max_epochs):
        model.train()
        for ids, y in batches(vocab, train.text.tolist(), train.label.tolist(), shuffle=True, seed=seed + epoch):
            opt.zero_grad()
            loss_fn(model(ids), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p_val = predict(model, vocab, val.text)
        f1 = f1_score(val.label, p_val >= 0.5, zero_division=0)
        val_loss = float(loss_fn(torch.logit(torch.tensor(p_val).clamp(1e-6, 1 - 1e-6)),
                                 torch.tensor(val.label.values, dtype=torch.float32)))
        if (f1, -val_loss) > best_key:  # early stopping on validation F1, ties to lower loss
            best, best_key, bad = copy.deepcopy(model.state_dict()), (f1, -val_loss), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    runs.save(f"bilstm_{emb}", variant, split, seed, rows, predict(model, vocab, rows.text))
    return {"epochs": epoch + 1, "val_f1": best_key[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", nargs="+", default=["random", "frozen", "finetuned"])
    parser.add_argument("--variant", nargs="+", default=["clean"])
    parser.add_argument("--split", nargs="+", default=["template"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cache = load_cache()
    for split in args.split:
        for variant in args.variant:
            for emb in args.emb:
                for seed in args.seeds:
                    start = time.time()
                    info = train_one(emb, variant, split, seed, cache)
                    print(f"bilstm_{emb} {variant} {split} seed {seed}: {info} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
