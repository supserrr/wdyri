// WDYRI browser app: the E12 ensemble (number-balanced AfroXLMR OR number-balanced
// char n-gram LR), running entirely client-side. The text goes through the same
// preprocessing as training (web/preprocess.js, parity-tested against Python).
import { AutoTokenizer, AutoModelForSequenceClassification, env }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.web.min.js";
import { preprocess } from "./preprocess.js";
import { loadNgramModel } from "./ngram.js";
import { note } from "./ui.js";

env.allowLocalModels = false;

const $ = (sel) => document.querySelector(sel);
const status = $("#status");
const [settings, lrData, examples] = await Promise.all(
  ["settings.json", "lr_char.json", "examples.json"].map((f) => fetch(f).then((r) => r.json())));
const ngram = loadNgramModel(lrData);
$("#tf-name").textContent = settings.transformer_name;
$("#train-note").textContent = `Trained on ${settings.train_note}`;

let tokenizer = null, model = null, current = null, checks = 0;
$("#check").disabled = false;
note("ngram", "N-gram model · ready");
note("tf", "Transformer · 0%");
status.textContent = "N-gram model ready. Loading the transformer.";

const files = {};
(async () => {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(settings.model_id);
    model = await AutoModelForSequenceClassification.from_pretrained(settings.model_id, {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          files[p.file] = [p.loaded, p.total];
          const [l, t] = Object.values(files).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
          note("tf", `Transformer · ${Math.floor((100 * l) / t)}%`, { instant: true });
        }
      },
    });
    note("tf", "Transformer · ready");
    status.textContent = "Both models ready.";
    if (current) run({ reveal: false });   // re-run a message checked before the transformer finished loading
  } catch (err) {
    console.error(err);
    note("tf", "Transformer · unavailable");
    status.textContent = "The transformer could not be loaded; results use the n-gram model only.";
  }
})();

async function transformerProba(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const inputs = await tokenizer(texts.slice(i, i + 16), { padding: true, truncation: true, max_length: 128 });
    const { logits } = await model(inputs);
    for (const [a, b] of logits.tolist()) out.push(1 / (1 + Math.exp(a - b)));
  }
  return out;
}

// Leave-one-word-out occlusion: how much the scam probability drops without each word.
async function influence(text, scorer) {
  const words = text.split(" ");
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const probs = await scorer(variants);
  return { p: probs[0], words: words.map((w, i) => [w, probs[0] - probs[i + 1]]) };
}

function paint(el, res, threshold) {
  const flagged = res.p >= threshold;
  el.style.setProperty("--thr", threshold);
  el.querySelector(".prob").textContent = `${Math.round(100 * res.p)}%`;
  el.querySelector(".bar").classList.toggle("flag", flagged);
  el.querySelector(".bar span").style.width = `${100 * res.p}%`;
  el.querySelector(".flagtxt").textContent =
    `${flagged ? "Flags it as a scam" : "Does not flag it"} (flags at ≥ ${Math.round(100 * threshold)}%)`;
  const box = el.querySelector(".words");
  box.replaceChildren(...res.words.map(([w, e]) => {
    const s = document.createElement("span");
    s.textContent = w;
    s.title = `effect ${e >= 0 ? "+" : ""}${e.toFixed(3)}`;
    const a = Math.min(Math.abs(e), 1) * 0.75;
    if (Math.abs(e) >= 0.01) s.style.background = e > 0 ? `rgba(214,69,55,${a})` : `rgba(42,111,219,${a})`;
    return s;
  }).flatMap((s) => [s, document.createTextNode(" ")]));
}

async function run({ reveal = true } = {}) {
  const raw = $("#sms").value.trim();
  if (!raw) return;
  const defend = $("#defend").checked;
  const text = preprocess(raw, { defend });
  current = { raw, defend };
  $("#check").disabled = true;
  $("#seen").textContent = text;
  const lr = await influence(text, async (ts) => ts.map((t) => ngram.proba(t)));
  paint($("#m-lr"), lr, settings.threshold_baseline);
  let tf = null;
  if (model) {
    tf = await influence(text, transformerProba);
    paint($("#m-tf"), tf, settings.threshold_transformer);
  } else {
    $("#m-tf .prob").textContent = "…";
    $("#m-tf .flagtxt").textContent = "Still loading; the result will update when it is ready.";
    $("#m-tf .words").replaceChildren();
  }
  const by = [];
  if (tf && tf.p >= settings.threshold_transformer) by.push("the transformer");
  if (lr.p >= settings.threshold_baseline) by.push("the n-gram model");
  const v = $("#verdict");
  v.className = `verdict ${by.length ? "scam" : "ok"}`;
  v.querySelector("h2").textContent = by.length ? "Likely scam" : "Looks genuine";
  v.querySelector("p").textContent = by.length ? `Flagged by ${by.join(" and ")}.`
    : model ? "Neither model flags it." : "The n-gram model does not flag it (transformer still loading).";
  const ex = examples.find((e) => e.text === raw && e.defend === defend);
  v.querySelector(".note").textContent = ex ? `${ex.label}. ${ex.note}` : "";
  $("#result").hidden = false;
  if (reveal) note("count", `Checks this visit · ${++checks}`);
  note("l1", `Verdict · ${by.length ? "likely scam" : "looks genuine"}`);
  note("l2", `Transformer · ${tf ? `${Math.round(100 * tf.p)}%` : "loading"}`);
  note("l3", `N-gram model · ${Math.round(100 * lr.p)}%`);
  note("l4", `Disguise fix · ${defend ? "on" : "off"}`);
  $("#check").disabled = false;
  if (reveal) $("#result").scrollIntoView({ block: "start" });
}

$("#check").addEventListener("click", () => run());
$("#sms").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });

// Example chips: type the message into the box, then check it, as if pasted by hand.
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let typing = 0;
async function typeInto(box, text) {
  const token = ++typing;
  if (reduceMotion) { box.value = text; return true; }
  const step = Math.max(1, Math.ceil(text.length / 32));
  for (let i = step; i < text.length + step; i += step) {
    if (token !== typing) return false;   // another example was picked meanwhile
    box.value = text.slice(0, i);
    box.scrollTop = box.scrollHeight;
    await new Promise((r) => setTimeout(r, 16));
  }
  return token === typing;
}
const chips = examples.map((ex) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = ex.label;
  b.title = ex.note;
  b.setAttribute("aria-pressed", "false");
  b.addEventListener("click", async () => {
    chips.forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
    $("#defend").checked = ex.defend;
    if (await typeInto($("#sms"), ex.text)) run();
  });
  return b;
});
$("#examples").replaceChildren(...chips);
// Editing the box by hand means it no longer shows the chosen example.
$("#sms").addEventListener("input", () => chips.forEach((c) => c.setAttribute("aria-pressed", "false")));
