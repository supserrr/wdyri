// The character n-gram logistic regression from src/classical.py, re-implemented
// from its exported weights (web/lr_char.json). It mirrors scikit-learn exactly:
// TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
// sublinear_tf=True) followed by LogisticRegression. scripts/web_parity.mjs
// checks it against the Python probabilities on every evaluation message.
import { WHITESPACE } from "./preprocess.js";

const RUNS_RE = new RegExp(`[${WHITESPACE}][${WHITESPACE}]+`, "gu");   // scikit-learn's \s\s+
const SPLIT_RE = new RegExp(`[${WHITESPACE}]+`, "u");                  // Python's str.split()

export function loadNgramModel(data) {
  const [minN, maxN] = data.ngram_range;
  const vocab = new Map(Object.entries(data.vocabulary));

  // scikit-learn's _char_wb_ngrams: pad each word with spaces, take n-grams within it.
  function ngrams(text) {
    if (data.lowercase) text = text.toLowerCase();
    text = text.replace(RUNS_RE, " ");
    const out = [];
    for (const word of text.split(SPLIT_RE).filter(Boolean)) {
      const w = Array.from(" " + word + " ");   // code points, like Python strings
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        out.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) {
          offset += 1;
          out.push(w.slice(offset, offset + n).join(""));
        }
        if (offset === 0) break;   // a word shorter than n is counted once
      }
    }
    return out;
  }

  function proba(text) {
    const counts = new Map();
    for (const g of ngrams(text)) {
      const j = vocab.get(g);
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
    let norm = 0;
    const weights = [];
    for (const [j, tf] of counts) {
      const v = (data.sublinear_tf ? 1 + Math.log(tf) : tf) * data.idf[j];
      weights.push([j, v]);
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    let z = data.intercept;
    for (const [j, v] of weights) z += (v / norm) * data.coef[j];
    return 1 / (1 + Math.exp(-z));
  }

  return { proba };
}
