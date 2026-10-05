// Checks the browser port (web/preprocess.js, web/ngram.js) against Python.
// Run after scripts/export_web.py:  node scripts/web_parity.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { preprocess } from "../web/preprocess.js";
import { loadNgramModel } from "../web/ngram.js";

const root = new URL("..", import.meta.url).pathname;
const pre = JSON.parse(readFileSync(root + ".cache/web_preprocess_reference.json", "utf8"));
let maskOk = 0, defOk = 0;
const misses = [];
for (const r of pre) {
  const m = preprocess(r.raw), d = preprocess(r.raw, { defend: true });
  if (m === r.masked) maskOk++; else if (misses.length < 5) misses.push({ raw: r.raw, py: r.masked, js: m });
  if (d === r.defended) defOk++; else if (misses.length < 5) misses.push({ raw: r.raw, py: r.defended, js: d });
}

const model = loadNgramModel(JSON.parse(readFileSync(root + "web/lr_char.json", "utf8")));
const lr = JSON.parse(readFileSync(root + ".cache/web_lr_reference.json", "utf8"));
let maxDiff = 0, decisionOk = 0;
for (const r of lr) {
  const p = model.proba(r.text);
  maxDiff = Math.max(maxDiff, Math.abs(p - r.prob));
  if ((p >= 0.5) === (r.prob >= 0.5)) decisionOk++;
}
const report = {
  preprocess_texts: pre.length,
  masked_identical: maskOk / pre.length,
  defended_identical: defOk / pre.length,
  ngram_messages: lr.length,
  ngram_decision_agreement: decisionOk / lr.length,
  ngram_max_prob_diff: maxDiff,
};
writeFileSync(root + "results/web_parity_js.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report, null, 2));
if (misses.length) console.log("first mismatches:", JSON.stringify(misses, null, 1));
