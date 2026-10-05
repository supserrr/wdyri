// The animated light ribbon behind the hero, the call-to-action card and the footer:
// a dispersed band of light that fans out from a focal point, with soft coloured haze
// above and below it. One full-screen fragment shader, WebGL 1, no libraries.

const VERT = `
attribute vec2 p;
varying vec2 vUv;
void main() { vUv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform float uTime;
uniform float uDark;   // 0: light page, 1: dark footer
uniform float uY;      // height of the ribbon, 0 = bottom, 1 = top
varying vec2 vUv;

// Prism colours from one edge of the band to the other.
vec3 spectrum(float t) {
  t = clamp(t, 0.0, 1.0) * 8.0;
  vec3 c0 = vec3(0.20, 0.33, 1.00), c1 = vec3(0.30, 0.72, 1.00), c2 = vec3(0.70, 0.97, 1.00);
  vec3 c3 = vec3(1.00, 1.00, 0.97), c4 = vec3(1.00, 0.93, 0.45), c5 = vec3(1.00, 0.62, 0.20);
  vec3 c6 = vec3(0.98, 0.30, 0.25), c7 = vec3(0.90, 0.28, 0.70), c8 = vec3(0.45, 0.35, 1.00);
  if (t < 1.0) return mix(c0, c1, t);
  if (t < 2.0) return mix(c1, c2, t - 1.0);
  if (t < 3.0) return mix(c2, c3, t - 2.0);
  if (t < 4.0) return mix(c3, c4, t - 3.0);
  if (t < 5.0) return mix(c4, c5, t - 4.0);
  if (t < 6.0) return mix(c5, c6, t - 5.0);
  if (t < 7.0) return mix(c6, c7, t - 6.0);
  return mix(c7, c8, t - 7.0);
}

// Height of the ribbon's centre line at horizontal position s (0..1).
float centre(float s, float t) {
  if (uDark > 0.5)   // a horizon: high on the left, low in the middle, rising again
    return uY + 0.02 * sin(s * 5.0 + t * 0.25) + 0.012 * sin(s * 9.0 - t * 0.2 + 2.0)
              + 0.17 * smoothstep(0.5, -0.15, s) + 0.07 * smoothstep(0.5, 1.15, s);
  return uY + 0.03 * sin(s * 4.0 - 0.6 + t * 0.27) + 0.018 * sin(s * 8.0 + 1.0 - t * 0.21)
            + 0.26 * smoothstep(0.5, 1.2, s) - 0.02 * smoothstep(0.0, 0.45, s);
}

void main() {
  float s = vUv.x, t = uTime;
  float dy = vUv.y - centre(s, t);

  // The band is thinnest at a drifting focal point and fans out towards both edges;
  // the colour order flips across the focus, like light through a twisted prism.
  float fs = s - (0.5 + 0.07 * sin(t * 0.19));
  float dir = clamp(fs / 0.04, -1.0, 1.0);
  float w = 0.012 + (fs > 0.0 ? 0.17 : 0.12) * pow(abs(fs), 1.05);
  float d = dy / w;
  float drift = 0.13 * sin(t * 0.23 + s * 3.1);

  vec3 col = spectrum(0.5 + 0.5 * d * dir + drift);
  float core = exp(-pow((d + 0.25 * dir) / 0.22, 2.0)) * smoothstep(0.02, 0.12, abs(fs));
  col = mix(col, vec3(1.0), core * 0.55);
  float streak = 0.72 + 0.28 * pow(abs(sin(d * 4.0 + t * 0.5 + s * 2.0)), 3.0);
  float ba = clamp(exp(-pow(abs(d), 2.6)) * streak, 0.0, 1.0);

  // Soft light around the band, then two broad hazes above and below it.
  float soft = exp(-pow(abs(d) / 2.4, 2.0)) * 0.35;
  vec3 softCol = spectrum(0.5 + 0.25 * d * dir + drift);
  float ha = exp(-pow((dy - 0.08) / 0.13, 2.0));
  float hb = exp(-pow((dy + (uDark > 0.5 ? 0.06 : 0.09)) / (uDark > 0.5 ? 0.045 : 0.13), 2.0));
  float m1 = smoothstep(0.25, 0.85, s + 0.12 * sin(t * 0.21));
  float m2 = smoothstep(0.30, 0.90, s - 0.12 * sin(t * 0.17 + 1.0));
  vec3 above = mix(vec3(0.34, 0.50, 1.00), vec3(1.00, 0.60, 0.56), m1);
  vec3 below = mix(vec3(1.00, 0.64, 0.46), vec3(0.42, 0.50, 1.00), m2);
  float aA = ha * 0.6, aB = hb * 0.5;
  if (uDark > 0.5) {
    above = mix(vec3(0.25, 0.75, 0.95), vec3(0.95, 0.62, 0.25), m1);
    below = vec3(0.22, 0.36, 0.95);
    aB = hb * 0.22;
  }
  float hazeA = clamp(aA + aB * (1.0 - aA), 0.0, 1.0);
  vec3 haze = above * aA + below * aB * (1.0 - aA);
  haze = softCol * soft + haze * (1.0 - soft);
  hazeA = soft + hazeA * (1.0 - soft);

  // Premultiplied alpha, so the page's own background shows through.
  gl_FragColor = vec4(col * ba + haze * (1.0 - ba), ba + hazeA * (1.0 - ba));
}`;

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

// Draws a ribbon into `canvas` and keeps it moving while it is on screen.
// `y` is a number or a function returning the ribbon's height (0 = bottom, 1 = top).
export function mountRibbon(canvas, { dark = false, y = 0.5, speed = 1, start = 0 } = {}) {
  const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false });
  if (!gl) return null;
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const uTime = gl.getUniformLocation(prog, "uTime");
  const uY = gl.getUniformLocation(prog, "uY");
  gl.uniform1f(gl.getUniformLocation(prog, "uDark"), dark ? 1 : 0);
  gl.clearColor(0, 0, 0, 0);

  let time = start, last = 0, raf = 0, visible = true;
  const size = () => {
    const scale = Math.min(devicePixelRatio || 1, 1.5);
    const w = Math.round(canvas.clientWidth * scale), h = Math.round(canvas.clientHeight * scale);
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
  };
  const draw = () => {
    size();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform1f(uTime, time);
    gl.uniform1f(uY, typeof y === "function" ? y() : y);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
  const frame = (now) => {
    raf = 0;
    if (last) time += Math.min((now - last) / 1000, 0.1) * speed;
    last = now;
    draw();
    schedule();
  };
  const schedule = () => {
    if (!raf && visible && !document.hidden && !reduceMotion.matches) raf = requestAnimationFrame(frame);
  };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; last = 0; schedule(); }, { rootMargin: "120px" })
    .observe(canvas);
  document.addEventListener("visibilitychange", () => { last = 0; schedule(); });
  new ResizeObserver(() => { if (!raf) draw(); }).observe(canvas);
  draw();
  schedule();
  return { redraw: () => { if (!raf) draw(); } };
}
