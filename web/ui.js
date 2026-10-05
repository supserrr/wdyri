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

$$(".notes div").forEach((el) => { el.dataset.text = el.textContent; el.textContent = ""; });
function typeNotes() {
  $$(".notes div").forEach((el, i) => { if (!el.textContent) type(el, el.dataset.text, 700 + i * 240); });
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
      const a = fly.getBoundingClientRect(), b = target.getBoundingClientRect();
      fly.style.animation = "none";
      fly.style.transition = "transform 0.85s cubic-bezier(0.22, 1, 0.36, 1)";
      fly.style.transform = `translate(${b.left + b.width / 2 - (a.left + a.width / 2)}px, ` +
        `${b.top + b.height / 2 - (a.top + a.height / 2)}px) scale(${b.width / a.width})`;
      setTimeout(land, 900);
    }
    layout();
    typeNotes();
  }, Math.max(0, 1100 - (performance.now() - loadStart)));   // let the logo finish drawing
}
const maybeReveal = () => { if (fontsLoaded && appLoaded) reveal(); };
export function appReady() { appLoaded = true; maybeReveal(); }
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { fontsLoaded = true; maybeReveal(); });
setTimeout(reveal, 6000);   // never keep anyone waiting longer than this

// --- Ribbons -------------------------------------------------------------------

let heroRibbonY = 0.5;
try {
  mountRibbon($("#ribbon-hero"), { y: () => heroRibbonY, start: 3 });
  mountRibbon($("#ribbon-story"), { y: 0.5, start: 9 });
  mountRibbon($("#ribbon-end"), { dark: true, y: 0.42, start: 7 });
} catch (err) {
  console.warn("Light ribbon unavailable:", err);   // the page still works on its plain background
}

// --- Hero shrinking away, story text scrolling through its card ------------------------

const heroWrap = $(".hero-wrap"), heroCard = $("#hero-card"), hero = $("#hero"), glass = $(".glass");
const notes = $$(".notes");
const stage = $("#story"), storyCard = $(".story-card"), story = $("#story-text");
let vw = innerWidth, vh = innerHeight;

function layout() {
  vw = innerWidth;
  vh = innerHeight;
  stage.style.height = `${Math.round(vh + (storyCard.offsetHeight * 0.73 + story.offsetHeight) * 1.05)}px`;
  if (scrollY < 4) {   // the ribbon runs behind the text box
    const r = glass.getBoundingClientRect(), h = heroCard.getBoundingClientRect();
    heroRibbonY = clamp(1 - (r.top + r.height / 2 - h.top) / h.height, 0.2, 0.8);
  }
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
  for (const n of notes) n.style.opacity = fade;

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
