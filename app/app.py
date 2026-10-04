"""WDYRI (Why Did You Redeem It?): check a Swahili SMS for mobile-money scams.

The app runs two of the trained models side by side:
  * the fine-tuned transformer (AfroXLMR by default), loaded from the Hugging
    Face Hub by name, or from a local folder if WDYRI_MODEL points to one
  * the character n-gram logistic regression baseline (baseline_lr_char.joblib)

Both see the text through the same `preprocess()` used in training, so the app
cannot drift from the experiments. Word highlights come from leave-one-word-out
occlusion: how much the scam probability drops when each word is removed.

Run locally:  python app/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # repo layout: app/ next to src/
sys.path.insert(0, str(HERE))          # Space layout: src/ copied next to app.py

from src.perturb import CYRILLIC, DIGITS, ZWSP  # noqa: E402
from src.preprocess import preprocess  # noqa: E402

MODEL_ID = os.environ.get("WDYRI_MODEL", "supserrr/wdyri-afroxlmr")
SETTINGS = json.loads((HERE / "settings.json").read_text())

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
baseline = joblib.load(HERE / "baseline_lr_char.joblib")


@torch.no_grad()
def transformer_proba(texts: list[str]) -> np.ndarray:
    enc = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    return torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()


def baseline_proba(texts: list[str]) -> np.ndarray:
    return baseline.predict_proba(texts)[:, list(baseline.classes_).index(1)]


def highlights(text: str, scorer) -> list[tuple[str, float | None]]:
    """Each word with the drop in scam probability when it is removed (positive = scam-ward)."""
    words = text.split(" ")
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = scorer(variants)
    out = []
    for i, word in enumerate(words):
        effect = float(probs[0] - probs[i + 1])
        out.append((word + " ", round(effect, 3) if abs(effect) >= 0.01 else None))
    return out


def verdict(prob: float, threshold: float) -> dict[str, float]:
    return {"Scam": prob, "Not scam": 1 - prob} if prob >= threshold else {"Not scam": 1 - prob, "Scam": prob}


def check(sms: str, defend: bool):
    sms = (sms or "").strip()
    if not sms:
        raise gr.Error("Paste an SMS first.")
    text = preprocess(sms, defend=defend)
    p_tf = float(transformer_proba([text])[0])
    p_lr = float(baseline_proba([text])[0])
    t_tf, t_lr = SETTINGS["threshold_transformer"], SETTINGS["threshold_baseline"]
    headline = (f"### {'⚠️ Likely scam' if p_tf >= t_tf else '✅ Looks genuine'}\n"
                f"Transformer: **{p_tf:.0%}** scam probability (flags at ≥ {t_tf:.0%}). "
                f"Baseline: **{p_lr:.0%}** (flags at ≥ {t_lr:.0%}).")
    return (headline, verdict(p_tf, t_tf), verdict(p_lr, t_lr),
            highlights(text, transformer_proba), highlights(text, baseline_proba), text)


def disguise(text: str) -> str:
    """Example-only obfuscation: lookalike letters, digits and a zero-width space."""
    out = "".join(DIGITS[c] if c == "e" else CYRILLIC.get(c, c) for c in text)
    return out.replace(" ", ZWSP + " ", 1)


SCAM = "Iyo pesa itume kwenye namba hii 0712345678 jina litakuja JUMA ALLY"
EXAMPLES = [
    # A genuine personal message (BongoScam "trust" class).
    ["Nitapika wali na mboga leo jioni, unanunua nyama au mimi nibebe?", False],
    # The classic "send it to this number" scam (BongoScam template, fake number).
    [SCAM, False],
    # The same scam disguised with lookalike letters, digits and a zero-width space.
    [disguise(SCAM), False],
    # The same disguised scam with the normalisation defence switched on.
    [disguise(SCAM), True],
    # Landlord impersonation: no link, no amount. The template every neural model missed.
    ["Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya na siku zinazidi kwenda...?", False],
    # A Chichewa fraud SMS from Malawi (Taylor & Robert dataset); no Chichewa in training.
    ["MIRACLE MONEY: lowani mpingo wathu wa satanic kuti mulemele. imbani phone ku nambala iyi 0999000000 kuti mujoine", False],
]

with gr.Blocks(title="WDYRI: Swahili scam SMS checker") as demo:
    gr.Markdown(
        "# WDYRI: Why Did You Redeem It?\n"
        "Paste a Swahili SMS to check whether it looks like a mobile-money scam. "
        "A fine-tuned transformer and a simple baseline give their verdicts side by side, "
        "and the highlighted words show what drove each decision.\n\n"
        "🔒 **Privacy:** your message is processed in memory and never stored or logged. "
        "Phone numbers, amounts and links are masked before any model sees them.")
    with gr.Row():
        with gr.Column(scale=3):
            sms = gr.Textbox(label="SMS text", lines=4, placeholder="e.g. Iyo pesa itume kwenye namba hii ...")
            defend = gr.Checkbox(label="Undo disguise tricks first (lookalike letters, s p a c e d words)", value=False)
            button = gr.Button("Check message", variant="primary")
        with gr.Column(scale=2):
            headline = gr.Markdown()
            with gr.Row():
                out_tf = gr.Label(label=f"Transformer ({SETTINGS['transformer_name']})", num_top_classes=2)
                out_lr = gr.Label(label="Baseline (character n-gram logistic regression)", num_top_classes=2)
    gr.Markdown("**Why?** Red words pushed the verdict towards *scam*, blue words away from it.")
    hl_tf = gr.HighlightedText(label="Transformer: word influence", color_map=None, show_legend=False)
    hl_lr = gr.HighlightedText(label="Baseline: word influence", show_legend=False)
    seen = gr.Textbox(label="What the models see (after masking)", interactive=False)
    gr.Examples(EXAMPLES, inputs=[sms, defend], label="Try these")
    gr.Markdown(
        f"Trained on {SETTINGS['train_note']} Results, code and limits: "
        f"[github.com/supserrr/wdyri](https://github.com/supserrr/wdyri). "
        "This is a research demo, not a guarantee: when in doubt, call your provider on its official number.")
    outputs = [headline, out_tf, out_lr, hl_tf, hl_lr, seen]
    button.click(check, [sms, defend], outputs)
    sms.submit(check, [sms, defend], outputs)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("WDYRI_SHARE") == "1")
