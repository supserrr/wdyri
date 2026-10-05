// Page behaviour that does not involve the models: the light ribbons, the hero that
// shrinks away as you scroll, the story card, the typed status notes,
// in-page links and the easter egg. app.js reports model status through note().
import { mountRibbon } from "./ribbon.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const smooth = reduceMotion ? "auto" : "smooth";

// --- Corner notes -------------------------------------------------------------------
// Two small live logs, one near the top and one near the bottom. Each types itself in line
// by line, holds for a while, fades out, then types in again in the opposite corner; the two
// run on different clocks so they never switch together. app.js updates their lines.

const CHAR_MS = 26, FADE_MS = 600, GAP_MS = 700;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const typing = new WeakMap();

// Types `text` into `el` one character at a time; resolves when done (or when replaced).
function typeText(el, text) {
  const token = {};
  typing.set(el, token);
  el.dataset.typed = "";
  if (reduceMotion) { el.textContent = text; el.dataset.typed = "1"; return Promise.resolve(); }
  el.textContent = "";
  return new Promise((resolve) => {
    let i = 0;
    const step = () => {
      if (typing.get(el) !== token) return resolve();
      el.textContent = text.slice(0, ++i);
      if (i < text.length) setTimeout(step, CHAR_MS);
      else { el.dataset.typed = "1"; resolve(); }
    };
    text ? step() : (el.dataset.typed = "1", resolve());
  });
}

// Updates every element showing the note `key`; `instant` skips the typing effect.
export function note(key, text, { instant = false } = {}) {
  for (const el of $$(`[data-note="${key}"]`)) {
    if (el.dataset.text === text) continue;
    el.dataset.text = text;
    const block = el.closest(".notes");
    if (!block) { el.textContent = text; continue; }            // the one-line status on phones
    if (!el.dataset.typed) continue;                              // the block types it when it gets there
    if (instant) { typing.set(el, {}); el.textContent = text; }
    else typeText(el, text);
  }
}

const noteBlocks = $$(".notes");
noteBlocks.forEach((b) => b.querySelectorAll("div").forEach((el) => {
  el.dataset.text ??= el.textContent;
  el.textContent = "";
}));

async function typeBlock(block) {
  const lines = [...block.querySelectorAll("div")];
  lines.forEach((el) => { typing.set(el, {}); el.textContent = ""; el.dataset.typed = ""; });
  block.classList.remove("fading");
  for (const el of lines) await typeText(el, el.dataset.text);
}

async function cycleNotes(block, { delay, hold, side }) {
  await sleep(delay);
  await typeBlock(block);
  if (reduceMotion) return;   // stays put
  for (;;) {
    await sleep(hold);
    block.classList.add("fading");
    await sleep(FADE_MS + GAP_MS);
    side = side === "left" ? "right" : "left";
    block.classList.toggle("right", side === "right");
    await typeBlock(block);
  }
}

function startNotes() {
  const [top, bottom] = noteBlocks;
  if (top) cycleNotes(top, { delay: 700, hold: 9000, side: "left" });
  if (bottom) cycleNotes(bottom, { delay: 1600, hold: 12500, side: "left" });
}

// --- Page loader -------------------------------------------------------------------
// The loader stays up until the fonts and the n-gram model (with the example chips) are
// ready, so the hero appears in one piece; then the ribbon fades in and the hero rises.

const loadStart = performance.now();
let fontsLoaded = false, appLoaded = false, revealed = false;
function reveal() {
  if (revealed) return;
  revealed = true;
  setTimeout(() => {
    const root = document.documentElement, loader = $(".loader");
    const fly = $(".loader-mark"), target = $("#hero .mark");
    const land = () => { root.classList.add("landed"); loader?.remove(); };
    root.classList.add("ready");   // loader background fades, ribbon blooms, hero rises
    if (reduceMotion || !fly || !target) land();
    else {
      // One logo throughout: the loader's mark flies to the hero's mark, then hands over.
      // Freeze the waiting pulse where it is and ease back to full opacity, instead of
      // cancelling it: cancelling made the logo jump whenever the pulse was mid-fade.
      const a = fly.getBoundingClientRect(), b = target.getBoundingClientRect();
      const opacity = getComputedStyle(fly).opacity;
      fly.style.animation = "none";
      fly.style.opacity = opacity;
      fly.getBoundingClientRect();   // commit the frozen state before animating from it
      fly.style.transition = "transform 0.85s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.35s ease";
      fly.style.opacity = "1";
      fly.style.transform = `translate(${b.left + b.width / 2 - (a.left + a.width / 2)}px, ` +
        `${b.top + b.height / 2 - (a.top + a.height / 2)}px) scale(${b.width / a.width})`;
      setTimeout(land, 900);
    }
    layout();
    startNotes();
    setTimeout(settle, 1600);   // the entrance has finished
  }, Math.max(0, 1100 - (performance.now() - loadStart)));   // let the logo finish drawing
}
// Resolves once the page has finished appearing; app.js waits for it before starting the
// multi-megabyte transformer download, so parsing it cannot stall the entrance animation.
let settle;
export const settled = new Promise((resolve) => { settle = resolve; });
const maybeReveal = () => { if (fontsLoaded && appLoaded) reveal(); };
export function appReady() { appLoaded = true; maybeReveal(); }
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { fontsLoaded = true; maybeReveal(); });
setTimeout(reveal, 6000);   // never keep anyone waiting longer than this

// --- Ribbons -------------------------------------------------------------------

let heroRibbonY = 0.5;
try {
  mountRibbon($("#ribbon-hero"), { y: () => heroRibbonY, start: 0.5 });
  mountRibbon($("#ribbon-story"), { y: 0.5, start: 9 });
  mountRibbon($("#ribbon-end"), { dark: true, y: 0.42, start: 7 });
} catch (err) {
  console.warn("Light ribbon unavailable:", err);   // the page still works on its plain background
}

// --- Hero shrinking away, story text scrolling through its card ------------------------

const heroWrap = $(".hero-wrap"), heroCard = $("#hero-card"), hero = $("#hero"), glass = $(".glass");
const notesLayer = $(".notes-layer");
const stage = $("#story"), storyCard = $(".story-card"), story = $("#story-text");
let vw = innerWidth, vh = innerHeight;

function layout() {
  vw = innerWidth;
  vh = innerHeight;
  stage.style.height = `${Math.round(vh + (storyCard.offsetHeight * 0.73 + story.offsetHeight) * 1.05)}px`;
  // The ribbon runs behind the text box. Offsets ignore the entrance transforms and scrolling.
  heroRibbonY = clamp(1 - (glass.offsetTop + glass.offsetHeight / 2) / heroCard.offsetHeight, 0.2, 0.8);
  update();
}

function update() {
  const k = easeOut(clamp(scrollY / (heroWrap.offsetHeight * 0.7), 0, 1));
  const small = vw < 720;
  const y = k * (small ? 10 : 24), x = k * (small ? 10 : Math.max(24, vw * 0.06)), r = k * (small ? 28 : 40);
  heroCard.style.clipPath = k > 0 ? `inset(${y}px ${x}px round ${r}px)` : "none";
  const fade = 1 - clamp(k * 1.5, 0, 1);
  hero.style.opacity = fade;
  hero.style.transform = fade < 1 ? `scale(${1 - k * 0.04})` : "";
  hero.inert = fade < 0.3;
  if (notesLayer) notesLayer.style.opacity = fade;

  const q = clamp(-stage.getBoundingClientRect().top / (stage.offsetHeight - vh), 0, 1);
  const h = storyCard.offsetHeight, top = h * 0.85, end = h * 0.12 - story.offsetHeight;
  story.style.transform = `translateY(${top + (end - top) * q}px)`;

}

let queued = false;
addEventListener("scroll", () => {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; update(); });
}, { passive: true });
addEventListener("resize", layout);
document.fonts?.ready.then(layout);
layout();

// --- Links and buttons ---------------------------------------------------------------

for (const el of $$("[data-focus]")) {
  el.addEventListener("click", (e) => {
    e.preventDefault();
    // The text box is inert while the hero is faded out, so focus it once back at the top.
    const focusBox = () => $("#sms").focus({ preventScroll: true });
    if (scrollY < 2) return focusBox();
    addEventListener("scrollend", focusBox, { once: true });
    setTimeout(focusBox, 1500);   // browsers without scrollend
    scrollTo({ top: 0, behavior: smooth });
  });
}

// --- Easter egg and links to a question ----------------------------------------------

const egg = $("#egg");
let eggRibbon = null;
function openEgg() {
  if (egg.open) return;
  egg.showModal();
  try { eggRibbon ??= mountRibbon($("#ribbon-egg"), { y: 0.36, start: 5 }); } catch { /* plain background */ }
}
function openFromHash() {
  if (location.hash === "#why") return openEgg();
  const el = location.hash && document.getElementById(location.hash.slice(1));
  if (el && el.tagName === "DETAILS") el.open = true;
}
addEventListener("hashchange", openFromHash);
openFromHash();
for (const el of $$("[data-egg]")) el.addEventListener("click", openEgg);
egg.querySelector("[data-close]").addEventListener("click", () => egg.close());
egg.addEventListener("click", (e) => { if (e.target === egg) egg.close(); });   // click on the backdrop
