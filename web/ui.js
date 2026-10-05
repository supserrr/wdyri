// Page behaviour that does not involve the models: the light ribbons, the hero that
// shrinks into the story card as you scroll, the floating nav, the typed status notes,
// in-page links and the easter egg. app.js reports model status through note().
import { mountRibbon } from "./ribbon.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const smooth = reduceMotion ? "auto" : "smooth";

// --- Typed notes ---------------------------------------------------------------

const timers = new WeakMap();
function type(el, text, delay = 0) {
  clearTimeout(timers.get(el));
  if (reduceMotion) { el.textContent = text; return; }
  let i = 0;
  const step = () => {
    el.textContent = text.slice(0, ++i);
    if (i < text.length) timers.set(el, setTimeout(step, 22));
  };
  el.textContent = "";
  timers.set(el, setTimeout(step, delay));
}

// Updates every element showing the note `key`; `instant` skips the typing effect.
export function note(key, text, { instant = false } = {}) {
  for (const el of $$(`[data-note="${key}"]`)) {
    if (el.dataset.text === text) continue;
    el.dataset.text = text;
    if (instant) { clearTimeout(timers.get(el)); el.textContent = text; } else type(el, text);
  }
}

$$(".notes div").forEach((el, i) => {
  el.dataset.text = el.textContent;
  type(el, el.textContent, 500 + i * 240);
});

// --- Ribbons -------------------------------------------------------------------

let heroRibbonY = 0.5;
try {
  mountRibbon($("#ribbon-hero"), { y: () => heroRibbonY, start: 3 });
  mountRibbon($("#ribbon-cta"), { y: 0.47, start: 11 });
  mountRibbon($("#ribbon-end"), { dark: true, y: 0.42, start: 7 });
} catch (err) {
  console.warn("Light ribbon unavailable:", err);   // the page still works on its plain background
}

// --- Stage: hero -> story card ------------------------------------------------------

const stage = $(".stage"), card = $("#stage-card"), hero = $("#hero");
const win = $("#story-window"), story = $("#story-text"), nav = $("#nav"), glass = $(".glass");
const notes = $$(".notes");
let vw = innerWidth, vh = innerHeight, shrink = 1, storyStart = 0, storyEnd = 1;
const insetY = (k) => k * (vw < 720 ? vh * 0.09 : vh * 0.15);
const insetX = (k) => k * (vw < 720 ? 12 : Math.max(24, vw * 0.12));

function layout() {
  vw = innerWidth;
  vh = innerHeight;
  shrink = vh * 0.5;
  const cardH = vh - 2 * insetY(1);
  stage.style.height = `${Math.round(vh + shrink + (cardH * 0.7 + story.offsetHeight) * 1.05)}px`;
  storyStart = shrink * 0.55;
  storyEnd = stage.offsetHeight - vh;
  $("#story").style.top = `${Math.round(storyStart + (storyEnd - storyStart) * 0.22)}px`;
  if (scrollY < 4) {   // the ribbon runs behind the text box
    const r = glass.getBoundingClientRect();
    heroRibbonY = clamp(1 - (r.top + r.height / 2) / vh, 0.2, 0.8);
  }
  update();
}

function update() {
  const scrolled = clamp(-stage.getBoundingClientRect().top, 0, stage.offsetHeight);
  const k = easeOut(clamp(scrolled / shrink, 0, 1));
  const y = insetY(k), x = insetX(k), r = k * (vw < 720 ? 28 : 40);
  card.style.clipPath = k > 0 ? `inset(${y}px ${x}px round ${r}px)` : "none";
  win.style.inset = `${y}px ${x}px`;

  const fade = 1 - clamp(k * 1.7, 0, 1);
  hero.style.opacity = fade;
  hero.style.transform = fade < 1 ? `translateY(${-k * 80}px) scale(${1 - k * 0.04})` : "";
  hero.inert = fade < 0.3;
  for (const n of notes) n.style.opacity = fade;

  const winH = vh - 2 * y;
  const q = clamp((scrolled - storyStart) / (storyEnd - storyStart), 0, 1);
  const top = winH * 0.85, end = winH * 0.12 - story.offsetHeight;
  story.style.transform = `translateY(${top + (end - top) * q}px)`;
  story.style.opacity = clamp((scrolled - shrink * 0.35) / (shrink * 0.4), 0, 1);

  nav.classList.toggle("show", scrolled > vh * 0.3);
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

for (const el of $$("[data-top]")) {
  el.addEventListener("click", (e) => { e.preventDefault(); scrollTo({ top: 0, behavior: smooth }); });
}
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
function openFromHash() {
  if (location.hash === "#why") return egg.showModal();
  const el = location.hash && document.getElementById(location.hash.slice(1));
  if (el && el.tagName === "DETAILS") el.open = true;
}
addEventListener("hashchange", openFromHash);
openFromHash();
for (const el of $$("[data-egg]")) el.addEventListener("click", () => egg.showModal());
egg.querySelector("[data-close]").addEventListener("click", () => egg.close());
egg.addEventListener("click", (e) => { if (e.target === egg) egg.close(); });   // click on the backdrop
