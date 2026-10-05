"""WDYRI (Why Did You Redeem It?): check a Swahili SMS for mobile-money scams.

The verdict comes from the project's ensemble (E12): two trained models that
fail on different messages, combined so that a message is flagged when either
one flags it at its own validation-chosen threshold.
  * the fine-tuned transformer (AfroXLMR, number-balanced training, E11),
    loaded from the Hugging Face Hub by name, or from a local folder; it
    catches disguised scams (lookalike letters, split words) and no longer
    treats "contains a phone number" as proof of a scam
  * the character n-gram logistic regression (baseline_lr_char.joblib), also
    number-balanced; it catches scams that carry no phone number, such as
    landlord impersonation
Which variants to deploy was decided on validation stress tests (E10-E12).

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

from src.perturb import attack  # noqa: E402
from src.preprocess import preprocess  # noqa: E402

SETTINGS = json.loads((HERE / "settings.json").read_text())
# A local copy of the evaluated run comes first; the Hub copy is used on the Space.
CANDIDATES = [str(HERE.parent / "models" / SETTINGS["transformer_run"]),
              os.environ.get("WDYRI_MODEL", "supserrr/wdyri-afroxlmr")]


def run_of(source: str) -> str | None:
    """The training run a model folder or Hub repo was exported from (wdyri_run.json)."""
    try:
        path = Path(source) / "wdyri_run.json"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(source, "wdyri_run.json"))
        return json.loads(path.read_text())["run"]
    except Exception:  # noqa: BLE001 (no marker file: treat as unknown)
        return None


def load_transformer():
    """The evaluated run, locally or from the Hub; None if neither loads (n-gram model only)."""
    for source in CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(source)
            mdl = AutoModelForSequenceClassification.from_pretrained(source).eval()
        except Exception as err:  # noqa: BLE001 (any load failure means: try the next source)
            print(f"could not load {source}: {err}")
            continue
        run = run_of(source) or (SETTINGS["transformer_run"] if Path(source).is_dir() else None)
        if run != SETTINGS["transformer_run"]:
            # Never serve a model under the wrong name: an older upload would mislabel the verdict.
            print(f"{source} holds run {run}, expected {SETTINGS['transformer_run']}; skipped")
            continue
        return tok, mdl, source
    return None, None, None


tokenizer, model, MODEL_SOURCE = load_transformer()
baseline = joblib.load(HERE / "baseline_lr_char.joblib")


@torch.no_grad()
def transformer_proba(texts: list[str]) -> np.ndarray:
    if model is None:
        return np.full(len(texts), np.nan)
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
    p_lr = float(baseline_proba([text])[0])
    t_tf, t_lr = SETTINGS["threshold_transformer"], SETTINGS["threshold_baseline"]
    if model is None:
        flagged = p_lr >= t_lr
        headline = (f"### {'⚠️ Likely scam' if flagged else '✅ Looks genuine'}\n"
                    f"N-gram model only (the transformer could not be loaded): **{p_lr:.0%}** scam probability.")
        return (headline, {}, verdict(p_lr, t_lr), [], highlights(text, baseline_proba), text)
    p_tf = float(transformer_proba([text])[0])
    flagged_by = [name for name, p, t in (("the transformer", p_tf, t_tf), ("the n-gram model", p_lr, t_lr)) if p >= t]
    if flagged_by:
        headline = f"### ⚠️ Likely scam\nFlagged by {' and '.join(flagged_by)}."
    else:
        headline = "### ✅ Looks genuine\nNeither model flags it."
    headline += (f"\n\nTransformer: **{p_tf:.0%}** scam probability (flags at ≥ {t_tf:.0%}). "
                 f"N-gram model: **{p_lr:.0%}** (flags at ≥ {t_lr:.0%}).")
    return (headline, verdict(p_tf, t_tf), verdict(p_lr, t_lr),
            highlights(text, transformer_proba), highlights(text, baseline_proba), text)


HELP_SCAM = ("Xorry nipo xafari nina xhida kwel nakuomba unixaidie sh elfu 9 nitakurefund nikifika plx "
             "By HAMISI RAMADHANI,ni grp member")
# The disguised example is made by the project's own attack code (lookalike letters on every
# trigger word), so it is exactly what experiment E6 does to a test scam.
DISGUISED = attack(HELP_SCAM, "lookalike", "all", lambda texts: baseline_proba([preprocess(t) for t in texts]), seed=42)
SCAM = "Iyo pesa itume kwenye namba hii 0712345678 jina litakuja JUMA ALLY"
EXAMPLES = [
    # A genuine personal message (BongoScam "trust" class).
    ["Nitapika wali na mboga leo jioni, unanunua nyama au mimi nibebe?", False],
    # The same genuine message with a phone number: the shortcut test (E10). Should stay genuine.
    ["Nitapika wali na mboga leo jioni, unanunua nyama au mimi nibebe? Nipigie 0712345678", False],
    # The classic "send it to this number" scam (BongoScam template, fake number).
    [SCAM, False],
    # A "help me, I'm travelling" scam disguised with lookalike letters: the n-gram model is fooled, the transformer is not.
    [DISGUISED, False],
    # The same disguised scam with the normalisation defence on: the n-gram model recovers.
    [DISGUISED, True],
    # Landlord impersonation: no number, link or amount. The transformer misses it; the n-gram model catches it.
    ["Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. Mbona kimya na siku zinazidi kwenda...?", False],
    # A Chichewa fraud SMS from Malawi (Taylor & Robert dataset); no Chichewa in training. The transformer
    # catches it; the number-balanced n-gram model, which lost its "has a number" cue, does not.
    ["MIRACLE MONEY: lowani mpingo wathu wa satanic kuti mulemele. imbani phone ku nambala iyi 0999000000 kuti mujoine", False],
]

with gr.Blocks(title="WDYRI: Swahili scam SMS checker") as demo:
    gr.Markdown(
        "# WDYRI: Why Did You Redeem It?\n"
        "Paste a Swahili SMS to check whether it looks like a mobile-money scam. "
        "Two models that fail on different messages check it: a fine-tuned African-language transformer "
        "(good at disguised scams) and a character n-gram model (good at scams without a phone number). "
        "The message is flagged if either one flags it, and the highlighted words show what drove each decision.\n\n"
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
                out_lr = gr.Label(label="Character n-gram logistic regression", num_top_classes=2)
    gr.Markdown("**Why?** Red words pushed the verdict towards *scam*, blue words away from it.")
    hl_tf = gr.HighlightedText(label="Transformer: word influence", color_map=None, show_legend=False)
    hl_lr = gr.HighlightedText(label="N-gram model: word influence", show_legend=False)
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
