// A line-by-line port of src/preprocess.py, so the browser app sees text exactly
// as the models did in training. scripts/web_parity.mjs checks it against the
// Python output on every message in the project.
//
// Python's re module treats \w, \d and \b as Unicode-aware for str patterns;
// JavaScript's are ASCII-only, so they are spelled out with Unicode properties.
// Python's \s (str.isspace) also differs from JavaScript's: it includes U+001C-U+001F
// and U+0085 but not U+FEFF, so whitespace is spelled out too.

const W = "[\\p{L}\\p{N}_]";                       // Python's \w for str
const D = "\\p{Nd}";                               // Python's \d for str
const B = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`; // Python's \b
export const WHITESPACE = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const S = `[${WHITESPACE}]`;                          // Python's \s for str
const NS = `[^${WHITESPACE}]`;                       // Python's \S for str

export const PHONE = "<PHONE>";
export const AMOUNT = "<AMOUNT>";
export const URL = "<URL>";
export const PLACEHOLDERS = [PHONE, AMOUNT, URL];

const URL_RE = new RegExp(
  `(?:https?://|www\\.)${NS}+` +
  `|${B}(?:wa\\.me|bit\\.ly|t\\.me|tinyurl\\.com|chat\\.whatsapp\\.com)/${NS}*`,
  "giu");

const PHONE_RE = new RegExp(`(?<!${D})\\+?${D}(?:[ .\\-]?${D}){8,13}(?!${D})`, "gu");

const CURRENCY = "(?:tshs?|tzs|shs?|ksh|mwk|mk|usd|k)";
const NUMBER = `${D}{1,3}(?:,${D}{3})+(?:\\.${D}+)?|${D}+(?:\\.${D}+)?`;
const AMOUNT_RE = new RegExp(
  `${B}${CURRENCY}\\.?${S}?-?${S}?(?:${NUMBER})(?:${S}?/=)?` +
  `|${B}(?:${NUMBER})${S}?(?:${CURRENCY}|/=|shilingi|shillings?|kwacha)${B}` +
  `|(?<![${"\\p{Nd}"}.])${D}+${S}?/=` +
  `|${B}${D}{1,3}(?:,${D}{3})+(?:\\.${D}+)?${B}|${B}${D}{5,8}${B}`,
  "giu");

const SPACES_RE = new RegExp(`${S}+`, "gu");

export function mask(text) {
  text = text.replace(URL_RE, ` ${URL} `);
  text = text.replace(PHONE_RE, ` ${PHONE} `);
  text = text.replace(AMOUNT_RE, ` ${AMOUNT} `);
  return text.replace(SPACES_RE, " ").trim();
}

// --- Obfuscation defence -----------------------------------------------------

const CONFUSABLES = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
  "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ӏ": "l", "һ": "h", "ԛ": "q", "ԝ": "w",
  "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
  "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ү": "Y",
  "α": "a", "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
  "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
  "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
};
const ZERO_WIDTH_RE = /[​‌‍⁠﻿­]/gu;
const LEET = { "4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s" };
const LEET_RE = /(?<=[A-Za-z])[431057@$]|[431057@$](?=[A-Za-z])/g;
const SPACED_RE = /(?<![A-Za-z])(?:[A-Za-z][ .\-]+){2,}[A-Za-z](?![A-Za-z])/g;
const PLACEHOLDER_RE = /<PHONE>|<AMOUNT>|<URL>/g;

function undoLeet(token) {
  if (PLACEHOLDERS.includes(token) || !/[A-Za-z]/.test(token)) return token;
  return token.replace(LEET_RE, (m) => LEET[m]);
}

export function normalise(text) {
  const parts = text.split(PLACEHOLDER_RE);
  const holders = text.match(PLACEHOLDER_RE) || [];
  const cleaned = parts.map((part) => {
    part = part.split(" ").map(undoLeet).join(" ");
    return part.replace(SPACED_RE, (m) => m.replace(/[ .\-]+/g, ""));
  });
  let out = cleaned[0];
  holders.forEach((h, i) => { out += h + cleaned[i + 1]; });
  return out.replace(SPACES_RE, " ").trim();
}

export function preprocess(text, { doMask = true, defend = false } = {}) {
  text = String(text).normalize("NFC");
  if (defend) {
    text = Array.from(text.replace(ZERO_WIDTH_RE, ""), (ch) => CONFUSABLES[ch] ?? ch).join("");
  }
  text = text.replace(SPACES_RE, " ").trim();
  if (doMask) text = mask(text);
  if (defend) text = normalise(text);
  return text;
}
