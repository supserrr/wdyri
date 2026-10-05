"""Export the deployed models for the browser app in web/.

    python scripts/export_web.py

1. The number-balanced AfroXLMR (app/settings.json) to ONNX, shrunk from
   1.1 GB to 364 MB (8-bit embedding table, fp16 weight storage), in the
   layout transformers.js expects:
   models/<run>/onnx/model_quantized.onnx next to config.json and tokenizer.json.
2. A parity check: the quantised ONNX model against PyTorch on the test,
   attacked and Chichewa sets (results/web_parity.json).
3. The char n-gram logistic regression to web/lr_char.json (vocabulary, idf,
   weights), which web/ngram.js re-implements exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

SETTINGS = json.loads((ROOT / "app" / "settings.json").read_text())
WEB = ROOT / "web"
PARITY_SETS = ["test", "test_lookalike_all", "test_structural_all", "test_codeswitch_all", "chichewa",
               "stress_genuine+phone", "stress_scam-number"]
# Characters on which Python's and JavaScript's regex classes differ (whitespace, case
# mapping), so the JavaScript parity test covers them even though no message has them.
EDGE_CASES = ["tuma\ufeffpesa", "tuma\x85pesa\x1c 0712345678", "Tsh\u3000 50,000 www.example.com/a\x1cb",
              "piga\u2028simu\x1f\x1e0999000000", "MK\u00a01,500 kwa \u0130DD\u0130 \u03a3\u0391\u03a3"]


def export_transformer() -> Path:
    src = config.MODELS / SETTINGS["transformer_run"]
    out = src / "onnx"
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    enc = tok(["Iyo pesa itume kwenye namba hii <PHONE>", "Habari"], padding=True, return_tensors="pt")
    fp32 = out / "model.onnx"
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), fp32,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                    "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # Full int8 dynamic quantisation breaks this model (about 74% decision agreement):
    # RoBERTa-family activations have outlier channels that per-tensor 8-bit
    # activation quantisation destroys. Instead: quantise only the 250k x 768
    # embedding table to 8 bits (Gather) and store the other large weight
    # matrices in fp16 behind a Cast, so every computation stays fp32.
    gather8 = out / "model_gather8.onnx"
    quantize_dynamic(fp32, gather8, weight_type=QuantType.QUInt8, op_types_to_quantize=["Gather"])
    m = onnx.load(gather8)
    consumers: dict[str, list] = {}
    for node in m.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node.op_type)
    for init in list(m.graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < 100_000 or not all(op in ("MatMul", "Gemm") for op in consumers.get(init.name, [])):
            continue
        m.graph.initializer.remove(init)
        m.graph.initializer.append(numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16"))
        m.graph.node.insert(0, helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=TensorProto.FLOAT))
    q8 = out / "model_quantized.onnx"   # the file name transformers.js loads for dtype "q8"
    onnx.save(m, q8)
    gather8.unlink()
    print(f"onnx: {fp32.stat().st_size / 1e6:.0f} MB -> deployed {q8.stat().st_size / 1e6:.0f} MB")
    return src


def parity(src: Path) -> dict:
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    session = ort.InferenceSession(str(src / "onnx" / "model_quantized.onnx"))
    ev = pd.read_csv(config.DATA_PROCESSED / "eval_template.csv", keep_default_na=False)
    ev = ev[ev["set"].isin(PARITY_SETS)]
    p_torch, p_onnx = [], []
    for start in range(0, len(ev), 64):
        texts = ev["text"].iloc[start:start + 64].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            p_torch.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
        logits = session.run(["logits"], {"input_ids": enc["input_ids"].numpy(),
                                          "attention_mask": enc["attention_mask"].numpy()})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        p_onnx.append((e / e.sum(1, keepdims=True))[:, 1])
    a, b = np.concatenate(p_torch), np.concatenate(p_onnx)
    agree = ((a >= 0.5) == (b >= 0.5))
    report = {"messages": int(len(a)), "decision_agreement": float(agree.mean()),
              "mean_abs_prob_diff": float(np.abs(a - b).mean()),
              "by_set": {s: float(agree[(ev["set"] == s).to_numpy()].mean()) for s in PARITY_SETS}}
    (config.RESULTS / "web_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_lr() -> None:
    pipe = joblib.load(ROOT / "app" / "baseline_lr_char.joblib")
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
    data = {"ngram_range": list(vec.ngram_range), "lowercase": vec.lowercase, "sublinear_tf": vec.sublinear_tf,
            "vocabulary": vocab, "idf": [round(float(x), 6) for x in vec.idf_],
            "coef": [round(float(x), 6) for x in clf.coef_.ravel()], "intercept": float(clf.intercept_[0]),
            "positive_class_index": int(list(clf.classes_).index(1)), "run": SETTINGS["baseline_run"]}
    WEB.mkdir(exist_ok=True)
    (WEB / "lr_char.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    # Reference outputs for the JavaScript parity test (scripts/web_parity.mjs).
    ev = pd.read_csv(config.DATA_PROCESSED / "eval_template.csv", keep_default_na=False)
    edge = pd.DataFrame({"set": "edge", "id": [f"edge_{i}" for i in range(len(EDGE_CASES))], "text": EDGE_CASES})
    ev = pd.concat([ev, edge], ignore_index=True)
    ref = ev.assign(prob=pipe.predict_proba(ev["text"].tolist())[:, data["positive_class_index"]])
    ref[["set", "id", "text", "prob"]].to_json(config.CACHE / "web_lr_reference.json", orient="records",
                                               force_ascii=False)
    print(f"lr_char.json: {len(vocab)} features, {(WEB / 'lr_char.json').stat().st_size / 1e6:.1f} MB")


def preprocess_reference() -> None:
    """Python's preprocess() on every raw and attacked message, for scripts/web_parity.mjs."""
    from src.data import load_bongo_raw, load_chichewa_raw
    from src.preprocess import preprocess
    texts = load_bongo_raw()["raw_text"].tolist() + load_chichewa_raw()["raw_text"].tolist()
    texts += pd.read_csv(config.DATA_PROCESSED / "eval_template.csv", keep_default_na=False)["text"].tolist()
    texts += EDGE_CASES
    ref = [{"raw": t, "masked": preprocess(t), "defended": preprocess(t, defend=True)} for t in dict.fromkeys(texts)]
    (config.CACHE / "web_preprocess_reference.json").write_text(json.dumps(ref, ensure_ascii=False))
    print(f"preprocess reference: {len(ref)} distinct texts")


def export_examples() -> None:
    """The app's examples and thresholds; the disguised example is made by the project's attack code."""
    from src.perturb import attack
    from src.preprocess import preprocess
    pipe = joblib.load(ROOT / "app" / "baseline_lr_char.joblib")
    help_scam = ("Xorry nipo xafari nina xhida kwel nakuomba unixaidie sh elfu 9 nitakurefund nikifika plx "
                 "By HAMISI RAMADHANI,ni grp member")
    disguised = attack(help_scam, "lookalike", "all",
                       lambda ts: pipe.predict_proba([preprocess(t) for t in ts])[:, 1], seed=42)
    examples = [
        ("Genuine message", "Nitapika wali na mboga leo jioni, unanunua nyama au mimi nibebe?", False,
         "Passes."),
        ("Genuine + phone number", "Nitapika wali na mboga leo jioni, unanunua nyama au mimi nibebe? Nipigie 0712345678",
         False, "Still passes: the number shortcut is gone (E10-E11)."),
        ("Classic scam", "Iyo pesa itume kwenye namba hii 0712345678 jina litakuja JUMA ALLY", False,
         "Both models flag it."),
        ("Disguised scam", disguised, False,
         "Lookalike letters fool the n-gram model; the transformer still flags it (E6)."),
        ("Disguised + defence", disguised, True,
         "With the normalisation defence on, the n-gram model recovers (E7)."),
        ("Landlord impersonation", "Habari za asubuhi. Mimi mwenye nyumba wako hii namba yangu ya Vodacom. "
         "Mbona kimya na siku zinazidi kwenda...?", False,
         "No number or link: the transformer misses it, the n-gram model catches it (E12)."),
        ("Chichewa scam", "MIRACLE MONEY: lowani mpingo wathu wa satanic kuti mulemele. imbani phone ku nambala "
         "iyi 0999000000 kuti mujoine", False, "No Chichewa in training: only the transformer flags it (E8)."),
    ]
    (WEB / "examples.json").write_text(json.dumps(
        [{"label": a, "text": t, "defend": d, "note": n} for a, t, d, n in examples], ensure_ascii=False, indent=1))
    keep = ("transformer_name", "transformer_run", "threshold_transformer", "baseline_run", "threshold_baseline",
            "train_note")
    (WEB / "settings.json").write_text(json.dumps({k: SETTINGS[k] for k in keep} | {
        "model_id": "supserrr/wdyri-afroxlmr"}, indent=1))


def main() -> None:
    export_lr()
    export_examples()
    preprocess_reference()
    src = export_transformer()
    print(json.dumps(parity(src), indent=2))


if __name__ == "__main__":
    main()
