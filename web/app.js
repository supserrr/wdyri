// WDYRI browser app: the E12 ensemble (number-balanced AfroXLMR OR number-balanced
// char n-gram LR), running entirely client-side. The text goes through the same
// preprocessing as training (web/preprocess.js, parity-tested against Python).
import { preprocess } from "./preprocess.js";
import { loadNgramModel } from "./ngram.js";
import { note, appReady, settled } from "./ui.js";

// Loaded in the background once the page has appeared, so the n-gram model and the page
// are ready without waiting for it.
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.web.min.js";

const $ = (sel) => document.querySelector(sel);
const status = $("#status");
const [settings, lrData, examples] = await Promise.all(
  ["settings.json", "lr_char.json", "examples.json"].map((f) => fetch(f).then((r) => r.json())));
const ngram = loadNgramModel(lrData);
$("#train-note").textContent = `Trained on ${settings.train_note}`;

let tokenizer = null, model = null, current = null, checks = 0;
$("#check").disabled = false;
note("ngram", "N-gram model · ready");
note("tf", "Transformer · 0%");
status.textContent = "N-gram model ready. Loading the transformer.";

const files = {};
(async () => {
  try {
    await settled;
    const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS);
    env.allowLocalModels = false;
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

function paint(el, words, res, threshold) {
  const flagged = res.p >= threshold;
  el.classList.toggle("flagged", flagged);
  el.classList.toggle("passed", !flagged);
  el.style.setProperty("--thr", threshold);
  el.querySelector(".prob").textContent = `${Math.round(100 * res.p)}%`;
  el.querySelector(".bar").classList.toggle("flag", flagged);
  el.querySelector(".bar span").style.width = `${100 * res.p}%`;
  el.querySelector(".flagtxt").textContent = flagged ? "Flags it" : "Passes";
  el.querySelector(".flagtxt").title = `Flags at ${Math.round(100 * threshold)}% or more`;
  explain(words.closest(".why"), res);
  words.replaceChildren(...res.words.map(([w, e]) => {
    const s = document.createElement("span");
    s.textContent = w;
    s.title = Math.abs(points(e)) >= 1
      ? `${points(e) > 0 ? "Pushes towards scam" : "Pushes away from scam"}: without this word the score would be ${pct(res.p - e)}% (${signed(points(e))} points)`
      : "Little effect on the score";
    // Vivid red tint for words pointing to a scam, blue for genuine; stronger words get a deeper
    // tint and a solid underline. Text stays near-black so it is always easy to read.
    if (Math.abs(points(e)) >= 1) {
      const a = Math.min(0.14 + Math.abs(e) * 1.6, 0.5);
      s.style.background = e > 0 ? `rgba(255,45,45,${a})` : `rgba(20,110,255,${a * 0.9})`;
      if (Math.abs(e) >= 0.08) s.style.boxShadow = `inset 0 -2px 0 ${e > 0 ? "#e0141c" : "#1462e6"}`;
    }
    return s;
  }).flatMap((s) => [s, document.createTextNode(" ")]));
}

// "What drove it": the words that moved each model's score most, in plain language.
const PLACEHOLDER = { "<PHONE>": "the phone number", "<AMOUNT>": "the amount", "<URL>": "the link" };
const points = (e) => Math.round(100 * e);
const pct = (p) => Math.min(100, Math.max(0, Math.round(100 * p)));
const signed = (n) => (n > 0 ? `+${n}` : `\u2212${Math.abs(n)}`);
const wordName = (w) => PLACEHOLDER[w] ?? `\u201c${w}\u201d`;
const bold = (text) => { const b = document.createElement("b"); b.textContent = text; return b; };
function joinWords(items) {   // [a, b, c] -> a, b and c (as bold nodes)
  const out = [];
  items.forEach((x, i) => {
    if (i) out.push(i === items.length - 1 ? " and " : ", ");
    out.push(bold(wordName(x.w)));
  });
  return out;
}
function explain(box, res) {
  const ranked = res.words.map(([w, e]) => ({ w, e })).filter((x) => Math.abs(points(x.e)) >= 1);
  const up = ranked.filter((x) => x.e > 0).sort((a, b) => b.e - a.e).slice(0, 3);
  const down = ranked.filter((x) => x.e < 0).sort((a, b) => a.e - b.e).slice(0, 3);
  const rows = box.querySelector(".why-rows"), sum = box.querySelector(".why-sum");
  const chip = (x) => {
    const f = document.createElement("span");
    f.className = `factor ${x.e > 0 ? "up" : "down"}`;
    f.append(PLACEHOLDER[x.w] ? PLACEHOLDER[x.w].replace(/^the /, "") : x.w, bold(signed(points(x.e))));
    return f;
  };
  const row = (kind, label, items) => {
    const r = document.createElement("div");
    r.className = `why-row ${kind}`;
    const l = document.createElement("span");
    l.className = "why-row-label";
    l.textContent = label;
    const c = document.createElement("div");
    c.className = "why-chips";
    c.append(...items.map(chip));
    r.append(l, c);
    return r;
  };
  rows.replaceChildren(...[up.length && row("up", "Points to a scam", up), down.length && row("down", "Points to genuine", down)].filter(Boolean));
  const p = pct(res.p);
  if (!up.length && !down.length) {
    sum.replaceChildren("No single word stands out. The score comes from the message as a whole.");
    return;
  }
  const top = [...up, ...down].sort((a, b) => Math.abs(b.e) - Math.abs(a.e))[0];
  const without = pct(res.p - top.e);
  sum.replaceChildren("Biggest effect: without ", bold(wordName(top.w)), `, the scam score would ${without < p ? "drop" : "rise"} from ${p}% to ${without}%.`);
}

// The check button shows that the analysis is running, for at least a moment.
const wait = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms)));
function setChecking(on) {
  const b = $("#check");
  b.disabled = on;
  b.classList.toggle("loading", on);
  b.setAttribute("aria-busy", String(on));
  b.querySelector(".label").textContent = on ? "Checking…" : "Check message";
  if (on) status.textContent = "Checking the message.";
}

let busy = false, pending = null;
async function run(opts = {}) {
  if (busy) { pending = opts; return; }   // run again once the current check finishes
  const raw = $("#sms").value.trim();
  if (!raw) return;
  busy = true;
  setChecking(true);
  try {
    await check(raw, opts);
  } finally {
    busy = false;
    setChecking(false);
  }
  if (pending) { const next = pending; pending = null; run(next); }
}

async function check(raw, { reveal = true }) {
  const started = performance.now();
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));   // paint the loading state first
  const defend = $("#defend").checked;
  const text = preprocess(raw, { defend });
  current = { raw, defend };
  const lr = await influence(text, async (ts) => ts.map((t) => ngram.proba(t)));
  paint($("#m-lr"), $("#words-lr"), lr, settings.threshold_baseline);
  let tf = null;
  if (model) {
    tf = await influence(text, transformerProba);
    paint($("#m-tf"), $("#words-tf"), tf, settings.threshold_transformer);
  } else {
    $("#m-tf").classList.remove("flagged", "passed");
    $("#m-tf .prob").textContent = "…";
    $("#m-tf .flagtxt").textContent = "Loading";
    $("#words-tf").replaceChildren();
    $("#why-tf .why-sum").replaceChildren();
    $("#why-tf .why-rows").replaceChildren();
  }
  const by = [];
  if (tf && tf.p >= settings.threshold_transformer) by.push("the transformer");
  if (lr.p >= settings.threshold_baseline) by.push("the n-gram model");
  const v = $("#verdict");
  v.className = `verdict ${by.length ? "scam" : "ok"}`;
  v.querySelector("h2").textContent = by.length ? "Likely scam" : "Looks genuine";
  v.querySelector("p").textContent = by.length ? `Flagged by ${by.join(" and ")}.`
    : model ? "Neither model flags it." : "The n-gram model does not flag it (transformer still loading).";
  await wait(500 - (performance.now() - started));
  $("#result").hidden = false;
  status.textContent = by.length ? "Check complete: likely scam." : "Check complete: looks genuine.";
  if (reveal) note("count", `Checks this visit · ${++checks}`);
  note("l1", `Verdict · ${by.length ? "likely scam" : "looks genuine"}`);
  note("l2", `Transformer · ${tf ? `${Math.round(100 * tf.p)}%` : "loading"}`);
  note("l3", `N-gram model · ${Math.round(100 * lr.p)}%`);
  note("l4", `Disguise fix · ${defend ? "on" : "off"}`);
  if (reveal) $("#result").scrollIntoView({ block: "start" });
}

$("#check").addEventListener("click", () => run());
$("#sms").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });

// Example chips: type the message into the box, as if pasted by hand; the user then presses Check.
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
    await typeInto($("#sms"), ex.text);
  });
  return b;
});
$("#examples").replaceChildren(...chips);
// Editing the box by hand means it no longer shows the chosen example.
$("#sms").addEventListener("input", () => chips.forEach((c) => c.setAttribute("aria-pressed", "false")));

// "What drove it" shows one model's word influences at a time.
document.querySelectorAll("[data-words]").forEach((b, _, all) => b.addEventListener("click", () => {
  all.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  $("#why-tf").hidden = b.dataset.words !== "tf";
  $("#why-lr").hidden = b.dataset.words !== "lr";
}));
appReady();
