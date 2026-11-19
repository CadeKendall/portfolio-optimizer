"use strict";

// Portfolio Optimizer front end. Plain JS + canvas, no chart library.
// Optimize: random portfolios flow out of the ticker list into a risk/return cloud, the exact
// efficient frontier (from the server) traces along its edge, the capital market line draws in,
// and the weights settle and lock. Backtest: equity and drawdown lines draw in from the left.

const STORAGE_KEY = "portfolio-optimizer:inputs:v2";
const FIELDS = ["risk_free", "min_weight", "max_weight", "opt_lookback", "bt_years", "bt_lookback", "bt_frequency", "bt_cost", "bt_benchmark"];
const DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "QQQ"];
const CLOUD_SIZE = 3000;
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => (v == null || !Number.isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));
const money = (v) => `$${v.toFixed(2)}`;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const FONT = () => css("--font"), MONO = () => css("--mono");

const state = {
  mode: "optimize",
  tickers: [...DEFAULT_TICKERS],
  source: "yahoo",
  uploadId: null,
  uploadMeta: null,
  busy: false,
  opt: null,      // { data, model, anim }
  bt: null,       // { data, model, anim }
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c instanceof Node ? c : String(c));
  return node;
}

// ---------------------------------------------------------------------------
// Persisted inputs (a per-browser convenience; never required)
// ---------------------------------------------------------------------------
function saveInputs() {
  const values = Object.fromEntries(FIELDS.map((id) => [id, $(id).value]));
  Object.assign(values, { tickers: state.tickers, source: state.source, uploadId: state.uploadId, uploadMeta: state.uploadMeta });
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(values)); } catch { /* storage unavailable */ }
}
function restoreInputs() {
  let v = null;
  try { v = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* ignore */ }
  if (!v) return;
  for (const id of FIELDS) if (v[id] != null) $(id).value = v[id];
  if (Array.isArray(v.tickers) && v.tickers.length) state.tickers = v.tickers;
  state.source = v.source === "upload" ? "upload" : "yahoo";
  state.uploadId = v.uploadId || null;
  state.uploadMeta = v.uploadMeta || null;
}

// ---------------------------------------------------------------------------
// Tickers rail
// ---------------------------------------------------------------------------
const X_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

function renderTickers() {
  const list = $("ticker-list");
  const model = state.opt && state.opt.model;
  const weights = model && model.best && sameTickers() ? model.best.w : null;
  list.replaceChildren(...state.tickers.map((t) => {
    const idx = model ? model.tickers.indexOf(t) : -1;
    const missing = model && sameTickers() && idx < 0;
    const li = el("li", { "data-ticker": t, class: missing ? "missing" : "" },
      el("span", { class: "sym" }, t),
      el("span", { class: "trk" }, el("span", { class: "fill" })),
      el("span", { class: "pct" }, weights && idx >= 0 ? pct(weights[idx]) : missing ? "no data" : "–"),
    );
    const rm = el("button", { type: "button", class: "remove", "aria-label": `Remove ${t}` });
    rm.innerHTML = X_ICON;
    rm.addEventListener("click", () => removeTicker(t));
    li.append(rm);
    if (weights && idx >= 0) li.querySelector(".fill").style.width = `${weights[idx] * 100}%`;
    return li;
  }));
  $("ticker-hint").textContent = state.tickers.length < 2
    ? "Add at least two tickers."
    : `${state.tickers.length} of 30 tickers. Paste a list to add several at once.`;
}
const tickerKey = (list) => [...list].sort().join(",");
function sameTickers() { return state.opt && state.opt.key === tickerKey(state.tickers); }

function addTickers(text) {
  const incoming = text.split(/[\s,;]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
  const bad = incoming.filter((t) => !/^[A-Z0-9.\-^=]{1,15}$/.test(t));
  const next = [...state.tickers];
  for (const t of incoming) if (!bad.includes(t) && !next.includes(t) && next.length < 30) next.push(t);
  state.tickers = next;
  renderTickers();
  saveInputs();
  return bad;
}
function removeTicker(t) {
  state.tickers = state.tickers.filter((x) => x !== t);
  renderTickers();
  saveInputs();
}
const rowFor = (t) => $("ticker-list").querySelector(`li[data-ticker="${CSS.escape(t)}"]`);

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------
function errorMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const field = (d.loc || []).filter((p) => p !== "body").join(".");
      const msg = String(d.msg || "").replace(/^Value error, /, "");
      return field ? `${field}: ${msg}` : msg;
    }).join("; ");
  }
  return "Something went wrong.";
}
async function request(url, options) {
  let res;
  try { res = await fetch(url, options); } catch { throw new Error("Couldn't reach the server. Is it still running?"); }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(body && body.detail) || `HTTP ${res.status}`);
  return body;
}
const postJson = (url, payload) => request(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

function commonPayload() {
  const p = {
    tickers: state.tickers,
    source: state.source,
    risk_free: Number($("risk_free").value) / 100,
    min_weight: Number($("min_weight").value) / 100,
    max_weight: Number($("max_weight").value) / 100,
  };
  if (state.source === "upload") p.upload_id = state.uploadId;
  return p;
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }
function clearError() { $("error").hidden = true; }
function showNotes(lines) {
  const box = $("notes");
  box.hidden = !lines.length;
  if (lines.length) box.replaceChildren(el("ul", {}, ...lines.map((l) => el("li", {}, l))));
}

async function uploadCsv(file) {
  const body = new FormData();
  body.append("file", file);
  $("upload-status").textContent = "Uploading…";
  try {
    const meta = await request("/api/prices/upload", { method: "POST", body });
    state.uploadId = meta.upload_id;
    state.uploadMeta = meta;
    clearError();
    if (state.tickers.length < 2) { state.tickers = meta.tickers.slice(0, 30); renderTickers(); }
    saveInputs();
  } catch (err) { showError(err.message); }
  renderUploadStatus();
}
function renderUploadStatus() {
  const m = state.uploadMeta;
  $("upload-status").textContent = m
    ? `Loaded ${m.tickers.length} tickers, ${m.start} to ${m.end}: ${m.tickers.join(", ")}`
    : "A date column plus one adjusted-price column per ticker.";
}
function syncSource() {
  document.querySelectorAll("[data-source]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.source === state.source)));
  $("upload-box").hidden = state.source !== "upload";
  renderUploadStatus();
}

// ---------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------
function fit(canvas) {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: r.width, H: r.height };
}
function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
// a label on a solid plate, so it stays readable over dense dots
function plate(ctx, text, x, y, font, align = "left") {
  ctx.font = font;
  const w = ctx.measureText(text).width, x0 = align === "right" ? x - w : x;
  ctx.fillStyle = css("--panel"); ctx.fillRect(x0 - 5, y - 13, w + 10, 19);
  ctx.strokeStyle = css("--line-strong"); ctx.lineWidth = 1; ctx.strokeRect(x0 - 5.5, y - 13.5, w + 11, 20);
  ctx.fillStyle = css("--text"); ctx.textAlign = "left"; ctx.fillText(text, x0, y);
}
// a solid tag sitting on an axis (crosshair readout)
function axisTag(ctx, text, x, y, align) {
  ctx.font = `11px ${MONO()}`;
  const w = ctx.measureText(text).width + 10, x0 = align === "center" ? x - w / 2 : x - w;
  ctx.fillStyle = css("--text"); ctx.fillRect(x0, y, w, 18);
  ctx.fillStyle = css("--panel"); ctx.textAlign = "left"; ctx.fillText(text, x0 + 5, y + 13);
}
function placeTip(tip, wrap, px, py) {
  tip.hidden = false;
  const W = wrap.clientWidth, H = wrap.clientHeight, tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = `${px + 16 + tw > W - 8 ? Math.max(8, px - 16 - tw) : px + 16}px`;
  tip.style.top = `${clamp(py - th / 2, 8, H - th - 8)}px`;
}

// Sharpe colour ramp for the cloud: low = warm sand, high = deep violet (reads in both themes)
const RAMP = [[226, 184, 104], [93, 186, 160], [64, 132, 210], [96, 84, 196]];
function ramp(t) {
  t = clamp(t, 0, 0.999) * (RAMP.length - 1);
  const i = Math.floor(t), f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return `rgb(${a.map((c, j) => Math.round(c + (b[j] - c) * f)).join(",")})`;
}

// ---------------------------------------------------------------------------
// Optimize: model
// ---------------------------------------------------------------------------
function buildOptModel(data, rf, minW, maxW) {
  const tickers = data.data.tickers;
  const n = tickers.length;
  const mu = tickers.map((t) => data.assets[t].expected_return);
  const cov = tickers.map((a) => tickers.map((b) => data.covariance[a][b]));
  const toVec = (p) => tickers.map((t) => p.weights[t] || 0);
  const port = (p) => ({ w: toVec(p), r: p.expected_return, v: p.volatility, s: p.sharpe != null ? p.sharpe : (p.expected_return - rf) / p.volatility });

  // random long-only portfolios that respect the weight bounds (rejection sampling; capped)
  let seed = 20261119;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const cloud = [];
  for (let tries = 0; cloud.length < CLOUD_SIZE && tries < CLOUD_SIZE * 30; tries++) {
    let w = mu.map(() => Math.pow(-Math.log(rnd()), 1.6));
    const sum = w.reduce((a, b) => a + b, 0);
    w = w.map((x) => x / sum);
    if (w.some((x) => x < minW - 1e-9 || x > maxW + 1e-9)) continue;
    let r = 0, v = 0;
    for (let a = 0; a < n; a++) { r += w[a] * mu[a]; for (let b = 0; b < n; b++) v += w[a] * w[b] * cov[a][b]; }
    v = Math.sqrt(Math.max(v, 0));
    cloud.push({ w, r, v, s: (r - rf) / v, k: w.indexOf(Math.max(...w)), d: Infinity, du: 0 });  // flight times set in setupFlight()
  }

  const minv = port(data.min_volatility);
  const best = data.max_sharpe ? port(data.max_sharpe) : null;
  const frontier = [minv, ...data.frontier.map(port)].sort((a, b) => a.r - b.r).filter((p, i, arr) => i === 0 || p.r - arr[i - 1].r > 1e-7);
  const assets = tickers.map((t, i) => ({ t, w: tickers.map((_, j) => (j === i ? 1 : 0)), r: mu[i], v: Math.sqrt(cov[i][i]), s: (mu[i] - rf) / Math.sqrt(cov[i][i]) }));

  const sAll = cloud.map((p) => p.s);
  const sLo = Math.min(...sAll), sHi = Math.max(best ? best.s : -Infinity, ...sAll);
  cloud.forEach((p) => { p.col = ramp((p.s - sLo) / (sHi - sLo || 1)); });

  const pts = [...cloud, ...frontier, ...assets];
  const vs = pts.map((p) => p.v), rs = pts.map((p) => p.r);
  let X0 = Math.min(...vs), X1 = Math.max(...vs), Y0 = Math.min(...rs), Y1 = Math.max(...rs);
  const px = (X1 - X0) * 0.07 || 0.01, py = (Y1 - Y0) * 0.08 || 0.01;
  X0 = Math.max(0, X0 - px); X1 += px; Y0 -= py; Y1 += py;
  return { tickers, mu, cov, cloud, frontier, assets, minv, best, rf, X0, X1, Y0, Y1 };
}

function optGeom(W, H) {
  const narrow = W < 560;
  return { x0: narrow ? 52 : 70, x1: W - (narrow ? 14 : 24), y0: 22, y1: H - (narrow ? 46 : 56) };
}

// ---------------------------------------------------------------------------
// Optimize: drawing. `t` is ms since results arrived (Infinity = final, static state).
// ---------------------------------------------------------------------------
const TIMING = { spread: 1300, flyMin: 650, flyRange: 650, frontier: 650, cml: 800, settle: 3500 };

function drawOpt(t) {
  const o = state.opt; if (!o) return;
  const m = o.model, a = o.anim;
  const { ctx, W, H } = fit($("opt-cv"));
  const g = optGeom(W, H);
  const sx = (v) => g.x0 + (v - m.X0) / (m.X1 - m.X0) * (g.x1 - g.x0);
  const sy = (r) => g.y1 - (r - m.Y0) / (m.Y1 - m.Y0) * (g.y1 - g.y0);
  o.sx = sx; o.sy = sy; o.g = g;
  const ink = css("--text"), muted = css("--muted"), faint = css("--faint"), lineS = css("--line-strong"), line = css("--line");
  ctx.clearRect(0, 0, W, H);

  // axes + grid
  ctx.globalAlpha = Math.min(1, t / 300);
  ctx.strokeStyle = line; ctx.lineWidth = 1;
  const xt = niceTicks(m.X0, m.X1, W < 560 ? 4 : 6), yt = niceTicks(m.Y0, m.Y1, 5);
  ctx.beginPath();
  for (const v of xt) { const x = Math.round(sx(v)) + 0.5; ctx.moveTo(x, g.y0); ctx.lineTo(x, g.y1); }
  for (const r of yt) { const y = Math.round(sy(r)) + 0.5; ctx.moveTo(g.x0, y); ctx.lineTo(g.x1, y); }
  ctx.stroke();
  ctx.strokeStyle = lineS; ctx.beginPath(); ctx.moveTo(g.x0 + 0.5, g.y0); ctx.lineTo(g.x0 + 0.5, g.y1 + 0.5); ctx.lineTo(g.x1, g.y1 + 0.5); ctx.stroke();
  ctx.fillStyle = faint; ctx.font = `11px ${MONO()}`; ctx.textAlign = "center";
  for (const v of xt) ctx.fillText(pct(v, 0), sx(v), g.y1 + 18);
  ctx.textAlign = "right";
  for (const r of yt) ctx.fillText(pct(r, 0), g.x0 - 10, sy(r) + 4);
  ctx.font = `12px ${FONT()}`; ctx.fillStyle = muted; ctx.textAlign = "center";
  ctx.fillText("Annualized volatility", (g.x0 + g.x1) / 2, g.y1 + 40);
  ctx.save(); ctx.translate(14, (g.y0 + g.y1) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Expected return", 0, 0); ctx.restore();
  ctx.globalAlpha = 1;

  // landed cloud points (in-flight ones are drawn on the flow layer)
  ctx.globalAlpha = 0.85;
  for (const p of m.cloud) {
    if (t < p.d + p.du) continue;
    ctx.fillStyle = p.col; ctx.fillRect(sx(p.v) - 1.25, sy(p.r) - 1.25, 2.5, 2.5);
  }
  ctx.globalAlpha = 1;

  // frontier + assets
  if (t > a.TF) {
    const k = easeInOut(Math.min(1, (t - a.TF) / TIMING.frontier));
    const n = Math.max(2, Math.round(m.frontier.length * k));
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.setLineDash([6, 3, 1.5, 3]);
    ctx.beginPath(); m.frontier.slice(0, n).forEach((p, i) => (i ? ctx.lineTo(sx(p.v), sy(p.r)) : ctx.moveTo(sx(p.v), sy(p.r)))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = k;
    for (const s of m.assets) {
      const x = sx(s.v), y = sy(s.r);
      ctx.fillStyle = css("--panel"); ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.strokeRect(x - 4, y - 4, 8, 8);
      ctx.font = `11px ${MONO()}`; ctx.textAlign = "left";
      const lw = ctx.measureText(s.t).width;
      ctx.fillStyle = css("--panel"); ctx.fillRect(x + 7, y - 7, lw + 6, 14);   // knock out the dots behind the label
      ctx.fillStyle = ink; ctx.fillText(s.t, x + 10, y + 4);
    }
    ctx.globalAlpha = 1;
  }

  // capital market line draws in from the left, then the markers and labels lock in
  if (t > a.TS) {
    const kl = Math.min(1, (t - a.TS) / TIMING.cml), vEnd = m.X0 + (m.X1 - m.X0) * easeOut(kl);
    const mx = sx(m.minv.v), my = sy(m.minv.r);
    if (m.best) {
      const cml = (v) => sy(m.rf + m.best.s * v);
      ctx.save(); ctx.beginPath(); ctx.rect(g.x0, g.y0, g.x1 - g.x0, g.y1 - g.y0); ctx.clip();
      ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sx(m.X0), cml(m.X0)); ctx.lineTo(sx(vEnd), cml(vEnd)); ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5); ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5); ctx.stroke();
    const bx = m.best ? sx(m.best.v) : 0, by = m.best ? sy(m.best.r) : 0;
    if (m.best && vEnd >= m.best.v) {
      ctx.fillStyle = ink; ctx.beginPath(); ctx.moveTo(bx, by - 8); ctx.lineTo(bx + 7, by + 5); ctx.lineTo(bx - 7, by + 5); ctx.closePath(); ctx.fill();
    }
    if (a.locked) {
      if (m.best) {
        ctx.setLineDash([2, 3]); ctx.strokeStyle = lineS; ctx.beginPath();
        ctx.moveTo(bx, by + 6); ctx.lineTo(bx, g.y1); ctx.moveTo(bx - 8, by); ctx.lineTo(g.x0, by); ctx.stroke(); ctx.setLineDash([]);
        plate(ctx, `Max Sharpe ${num(m.best.s)}`, bx + 12, by + 22, `500 13px ${FONT()}`);
      }
      plate(ctx, "Min volatility", mx + 12, my + 4, `500 12px ${FONT()}`);
    }
  }
}

// in-flight particles, drawn across the whole shell so they can leave the ticker rows
function drawFlow(t) {
  const { ctx, W, H } = fit($("flow"));
  ctx.clearRect(0, 0, W, H);
  const o = state.opt; if (!o || o.anim.locked || t > o.anim.TF) return;
  const shell = $("shell").getBoundingClientRect(), chart = $("opt-cv").getBoundingClientRect();
  const offX = chart.left - shell.left, offY = chart.top - shell.top;
  ctx.fillStyle = css("--faint");
  for (const p of o.model.cloud) {
    const u = (t - p.d) / p.du;
    if (u <= 0 || u >= 1) continue;
    const e = easeInOut(u), a = 1 - e;
    const tx = offX + o.sx(p.v), ty = offY + o.sy(p.r);
    const cx = (p.ox + tx) / 2 + p.jx, cy = p.oy + (ty - p.oy) * p.jr + p.jy;
    ctx.globalAlpha = 0.35 + 0.45 * u;
    ctx.fillRect(a * a * p.ox + 2 * a * e * cx + e * e * tx - 1.25, a * a * p.oy + 2 * a * e * cy + e * e * ty - 1.25, 2.5, 2.5);
  }
  ctx.globalAlpha = 1;
}

function setupFlight() {
  const o = state.opt, shell = $("shell").getBoundingClientRect();
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  let latest = 0;
  for (const p of o.model.cloud) {
    const row = rowFor(o.model.tickers[p.k]);
    const r = row ? row.querySelector(".trk").getBoundingClientRect() : shell;
    p.ox = r.left - shell.left + r.width * (0.2 + 0.8 * rnd());
    p.oy = r.top - shell.top + r.height / 2 + (rnd() - 0.5) * 10;
    p.jx = (rnd() - 0.5) * 120; p.jy = (rnd() - 0.5) * 140; p.jr = rnd();
    p.d = Math.pow(rnd(), 0.8) * TIMING.spread;
    p.du = TIMING.flyMin + rnd() * TIMING.flyRange;
    latest = Math.max(latest, p.d + p.du);
  }
  o.anim.TF = latest + 80;
  o.anim.TS = o.anim.TF + TIMING.frontier;
}

// ---------------------------------------------------------------------------
// Optimize: animation loop + weight springs
// ---------------------------------------------------------------------------
function spring(x, v, target, dt, zeta = 0.3, omega = 11) {
  const acc = -omega * omega * (x - target) - 2 * zeta * omega * v;
  v += acc * dt; x += v * dt;
  return [x, v];
}

function animateOptimize() {
  const o = state.opt, m = o.model;
  o.anim = { start: performance.now(), last: performance.now(), locked: false, TF: 0, TS: 0, springs: m.tickers.map(() => [0, 0]) };
  clearHover("opt");
  drawOpt(0);  // sets scales so particle targets are known
  setupFlight();
  const fills = m.tickers.map((t) => { const r = rowFor(t); return r && r.querySelector(".fill"); });
  const pcts = m.tickers.map((t) => { const r = rowFor(t); return r && r.querySelector(".pct"); });
  fills.forEach((f) => f && (f.style.width = "0"));
  pcts.forEach((p) => p && (p.textContent = "–"));
  setVerdict(`Sampling ${m.cloud.length.toLocaleString()} portfolios`);

  const step = (now) => {
    if (state.opt !== o) return;
    const a = o.anim, t = REDUCE ? Infinity : now - a.start, dt = Math.min(0.032, (now - a.last) / 1000);
    a.last = now;
    if (t > a.TF && t <= a.TS) setVerdict("Tracing the efficient frontier");
    if (t > a.TS && !a.locked) {
      if (m.best) {
        a.springs = a.springs.map(([x, v], i) => spring(x, v, m.best.w[i], dt));
        fills.forEach((f, i) => f && (f.style.width = `${clamp(a.springs[i][0], 0, 1.15) * 100}%`));
        setVerdict("Finding the best mix");
      }
      const lineDone = t - a.TS >= TIMING.cml;
      const still = !m.best || a.springs.every(([x, v], i) => Math.abs(x - m.best.w[i]) < 0.001 && Math.abs(v) < 0.01);
      if ((lineDone && still) || t - a.TS > TIMING.settle) lock();
    }
    drawOpt(t);
    drawFlow(t);
    if (!a.locked) a.raf = requestAnimationFrame(step);
  };
  const lock = () => {
    o.anim.locked = true;
    if (m.best) m.best.w.forEach((w, i) => { if (fills[i]) fills[i].style.width = `${w * 100}%`; if (pcts[i]) pcts[i].textContent = pct(w); });
    setVerdict(m.best
      ? `<strong>${pct(m.best.r)} return at ${pct(m.best.v)} volatility.</strong> Sharpe ${num(m.best.s)}.`
      : `No max-Sharpe portfolio for these inputs. Min volatility is <strong>${pct(m.minv.v)}</strong>.`, true);
    drawFlow(Infinity);
  };
  o.anim.raf = requestAnimationFrame(step);
}

function setVerdict(text, html = false) { if (html) $("verdict").innerHTML = text; else $("verdict").textContent = text; }

function renderFacts(data, m) {
  const fact = (title, p, extra) => {
    if (!p) return [el("h3", { class: "lbl" }, title), el("p", { class: "na" }, "Not available for these inputs.")];
    const dl = el("dl", {},
      el("dt", {}, "Volatility"), el("dd", {}, pct(p.v)),
      el("dt", {}, "Sharpe"), el("dd", {}, num(p.s)),
      ...extra);
    return [el("h3", { class: "lbl" }, title), el("p", { class: "big" }, pct(p.r)), dl];
  };
  const top = (p) => p ? m.tickers.map((t, i) => [t, p.w[i]]).filter(([, w]) => w > 0.0005).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
  const holdings = (p) => top(p).flatMap(([t, w]) => [el("dt", {}, t), el("dd", {}, pct(w))]);
  $("fact-sharpe").replaceChildren(...fact("Max Sharpe · expected return", m.best, holdings(m.best)));
  $("fact-minvol").replaceChildren(...fact("Min volatility · expected return", m.minv, holdings(m.minv)));
  $("assets-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Ticker"), el("th", {}, "Expected return"), el("th", {}, "Volatility"), el("th", {}, "Max Sharpe"), el("th", {}, "Min vol"))),
    el("tbody", {}, ...m.tickers.map((t, i) => el("tr", {},
      el("td", {}, t), el("td", {}, pct(m.mu[i])), el("td", {}, pct(Math.sqrt(m.cov[i][i]))),
      el("td", {}, m.best ? pct(m.best.w[i]) : "–"), el("td", {}, pct(m.minv.w[i]))))),
  );
  $("opt-facts").hidden = false;
}

async function runOptimize() {
  const payload = { ...commonPayload(), lookback_years: Number($("opt_lookback").value), frontier_points: 60 };
  const data = await postJson("/api/optimize", payload);
  const model = buildOptModel(data, payload.risk_free, payload.min_weight, payload.max_weight);
  if (state.opt && state.opt.anim) cancelAnimationFrame(state.opt.anim.raf);
  state.opt = { data, model, key: tickerKey(state.tickers), anim: null };
  $("opt-empty").hidden = true;
  renderTickers();
  renderMeta(data.data);
  showNotes([...data.data.warnings, ...data.warnings]);
  renderFacts(data, model);
  animateOptimize();
}

// ---------------------------------------------------------------------------
// Optimize: crosshair
// ---------------------------------------------------------------------------
function hoverOpt(e) {
  const o = state.opt, wrap = $("opt-chart");
  if (!o || !o.anim || !o.anim.locked) return;
  const r = wrap.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = o.g, m = o.model;
  if (x < g.x0 || x > g.x1 || y < g.y0 || y > g.y1) return clearHover("opt");
  wrap.style.cursor = "crosshair";
  let hit = null, kind = "", best = 22 * 22;
  const test = (p, k) => { const dx = o.sx(p.v) - x, dy = o.sy(p.r) - y, d = dx * dx + dy * dy; if (d < best) { best = d; hit = p; kind = k; } };
  m.cloud.forEach((p) => test(p, "Random portfolio"));
  m.frontier.forEach((p) => test(p, "Efficient frontier"));
  m.assets.forEach((p) => test(p, `${p.t} alone`));
  best = Math.min(best, 22 * 22) * 1.0001; // markers win ties
  test(m.minv, "Min volatility");
  if (m.best) test(m.best, "Max Sharpe");

  const { ctx, W, H } = fit($("opt-hv"));
  ctx.clearRect(0, 0, W, H);
  const px = hit ? o.sx(hit.v) : x, py = hit ? o.sy(hit.r) : y;
  const v = hit ? hit.v : m.X0 + (x - g.x0) / (g.x1 - g.x0) * (m.X1 - m.X0);
  const rr = hit ? hit.r : m.Y0 + (g.y1 - y) / (g.y1 - g.y0) * (m.Y1 - m.Y0);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, g.y0); ctx.lineTo(Math.round(px) + 0.5, g.y1); ctx.moveTo(g.x0, Math.round(py) + 0.5); ctx.lineTo(g.x1, Math.round(py) + 0.5); ctx.stroke();
  ctx.setLineDash([]);
  if (hit) { ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5; ctx.strokeRect(px - 5, py - 5, 10, 10); }
  axisTag(ctx, pct(v), px, g.y1 + 3, "center");
  axisTag(ctx, pct(rr), g.x0 - 3, py - 9, "right");

  const tip = $("opt-tip");
  if (!hit) { tip.hidden = true; return; }
  tip.innerHTML = `<div class="k">${kind}</div><div class="rows"><span>Return</span><span>${pct(hit.r)}</span><span>Volatility</span><span>${pct(hit.v)}</span><span>Sharpe</span><span>${num(hit.s)}</span></div>`
    + m.tickers.map((t, i) => `<div class="mix"><span>${t}</span><i><b style="width:${(hit.w[i] * 100).toFixed(1)}%"></b></i><span>${pct(hit.w[i])}</span></div>`).join("");
  placeTip(tip, wrap, px, py);
}
function clearHover(which) {
  const cv = $(`${which}-hv`), tip = $(`${which}-tip`);
  if (cv.width) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  tip.hidden = true;
  $(`${which}-chart`).style.cursor = "";
}

// ---------------------------------------------------------------------------
// Backtest
// ---------------------------------------------------------------------------
const STRATEGY_STYLE = {
  "Max Sharpe": { color: "--s-sharpe", dash: [] },
  "Min volatility": { color: "--s-minvol", dash: [] },
  "Equal weight": { color: "--s-equal", dash: [5, 3] },
};
const styleFor = (name) => STRATEGY_STYLE[name] || { color: "--s-bench", dash: [] };

function buildBtModel(data) {
  const r = data.result, dates = r.dates;
  const eq = r.strategies.flatMap((s) => s.equity), dd = r.strategies.flatMap((s) => s.drawdown);
  const E0 = Math.min(...eq), E1 = Math.max(...eq), D0 = Math.min(...dd, -0.01);
  // year boundaries for x ticks
  const years = [];
  dates.forEach((d, i) => { if (i === 0 || d.slice(0, 4) !== dates[i - 1].slice(0, 4)) years.push(i); });
  return { dates, strategies: r.strategies, E0: E0 - (E1 - E0) * 0.05, E1: E1 + (E1 - E0) * 0.05, D0: D0 * 1.08, years };
}
function btGeom(W, H) {
  const narrow = W < 560, x0 = narrow ? 52 : 70, x1 = W - (narrow ? 14 : 24), y0 = 18, y1 = H - 32;
  const split = y0 + (y1 - y0) * 0.68;
  return { x0, x1, y0, eqY1: split - 14, ddY0: split + 18, y1 };
}

function drawBt(t) {
  const b = state.bt; if (!b) return;
  const m = b.model, a = b.anim;
  const { ctx, W, H } = fit($("bt-cv"));
  const g = btGeom(W, H), n = m.dates.length;
  const sx = (i) => g.x0 + i / (n - 1) * (g.x1 - g.x0);
  const se = (v) => g.eqY1 - (v - m.E0) / (m.E1 - m.E0) * (g.eqY1 - g.y0);
  const sd = (v) => g.ddY0 + (v / m.D0) * (g.y1 - g.ddY0);
  Object.assign(b, { sx, se, sd, g });
  const faint = css("--faint"), line = css("--line"), lineS = css("--line-strong"), muted = css("--muted");
  ctx.clearRect(0, 0, W, H);

  // grids
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath();
  const eqTicks = niceTicks(m.E0, m.E1, 5), ddTicks = niceTicks(m.D0, 0, 3);
  for (const v of eqTicks) { const y = Math.round(se(v)) + 0.5; ctx.moveTo(g.x0, y); ctx.lineTo(g.x1, y); }
  for (const v of ddTicks) { const y = Math.round(sd(v)) + 0.5; ctx.moveTo(g.x0, y); ctx.lineTo(g.x1, y); }
  const step = Math.ceil(m.years.length / (W < 560 ? 4 : 8));
  const yearTicks = m.years.filter((_, i) => i % step === 0);
  for (const i of yearTicks) { const x = Math.round(sx(i)) + 0.5; ctx.moveTo(x, g.y0); ctx.lineTo(x, g.eqY1); ctx.moveTo(x, g.ddY0); ctx.lineTo(x, g.y1); }
  ctx.stroke();
  ctx.strokeStyle = lineS; ctx.beginPath();
  ctx.moveTo(g.x0 + 0.5, g.y0); ctx.lineTo(g.x0 + 0.5, g.eqY1); ctx.moveTo(g.x0 + 0.5, g.ddY0); ctx.lineTo(g.x0 + 0.5, g.y1 + 0.5); ctx.lineTo(g.x1, g.y1 + 0.5);
  ctx.stroke();
  ctx.fillStyle = faint; ctx.font = `11px ${MONO()}`; ctx.textAlign = "right";
  for (const v of eqTicks) ctx.fillText(money(v), g.x0 - 10, se(v) + 4);
  for (const v of ddTicks) ctx.fillText(pct(v, 0), g.x0 - 10, sd(v) + 4);
  ctx.textAlign = "center";
  for (const i of yearTicks) ctx.fillText(m.dates[i].slice(0, 4), sx(i), g.y1 + 18);
  ctx.textAlign = "left"; ctx.font = `12px ${FONT()}`; ctx.fillStyle = muted;
  ctx.fillText("Growth of $1", g.x0 + 8, g.y0 + 14);
  ctx.fillText("Drawdown", g.x0 + 8, g.ddY0 + 14);

  // series, drawn in from the left
  const k = REDUCE ? 1 : easeInOut(clamp(t / 1400, 0, 1));
  const last = Math.max(1, Math.floor(k * (n - 1)));
  const order = [...m.strategies].sort((p, q) => (p.name === "Max Sharpe") - (q.name === "Max Sharpe")); // Max Sharpe on top
  for (const s of order) {
    const st = styleFor(s.name);
    ctx.strokeStyle = css(st.color); ctx.setLineDash(st.dash); ctx.lineWidth = s.name === "Max Sharpe" ? 1.75 : 1.25;
    ctx.beginPath(); for (let i = 0; i <= last; i++) { const x = sx(i), y = se(s.equity[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
    ctx.beginPath(); for (let i = 0; i <= last; i++) { const x = sx(i), y = sd(s.drawdown[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
  }
  ctx.setLineDash([]);
  if (a.locked) {
    // lock: a square end-cap on each equity line
    for (const s of m.strategies) { ctx.fillStyle = css(styleFor(s.name).color); ctx.fillRect(g.x1 - 3, se(s.equity[n - 1]) - 3, 6, 6); }
  }
}

function renderBtTable(data, m) {
  const swatch = (name) => { const st = styleFor(name); return el("i", { class: st.dash.length ? "dash" : "", style: `background:var(${st.color});color:var(${st.color})` }); };
  $("bt-legend").replaceChildren(...m.strategies.map((s) => el("li", {}, swatch(s.name), s.name)));
  const head = ["Strategy", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Turnover / yr", "Final $1"];
  $("metrics-table").replaceChildren(
    el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))),
    el("tbody", {}, ...m.strategies.map((s) => {
      const x = s.metrics;
      return el("tr", {}, el("td", {}, s.name), el("td", {}, pct(x.cagr)), el("td", {}, pct(x.volatility)), el("td", {}, num(x.sharpe)),
        el("td", {}, pct(x.max_drawdown)), el("td", {}, pct(x.annual_turnover, 0)), el("td", {}, money(s.equity[s.equity.length - 1])));
    })),
  );
  $("metrics-table").hidden = false;
  const skips = $("skips"), list = data.result.skips;
  skips.hidden = list.length === 0;
  skips.querySelector("summary").textContent = `${list.length} rebalance${list.length === 1 ? "" : "s"} couldn't be optimized`;
  skips.querySelector("ul").replaceChildren(...list.map((s) => el("li", {}, `${s.date} · ${s.strategy}: ${s.reason}`)));
}

function animateBacktest() {
  const b = state.bt;
  b.anim = { start: performance.now(), locked: false };
  clearHover("bt");
  const best = b.model.strategies.find((s) => s.name === "Max Sharpe");
  setVerdict("Replaying the backtest");
  const step = (now) => {
    if (state.bt !== b) return;
    const t = REDUCE ? Infinity : now - b.anim.start;
    if (t >= 1400) b.anim.locked = true;
    drawBt(t);
    if (!b.anim.locked) b.anim.raf = requestAnimationFrame(step);
    else setVerdict(best
      ? `<strong>Max Sharpe grew $1 to ${money(best.equity[best.equity.length - 1])}</strong> (${pct(best.metrics.cagr)} a year, worst drop ${pct(best.metrics.max_drawdown)}).`
      : "Backtest complete.", true);
  };
  b.anim.raf = requestAnimationFrame(step);
}

async function runBacktest() {
  const payload = {
    ...commonPayload(),
    years: Number($("bt_years").value),
    lookback_years: Number($("bt_lookback").value),
    frequency: $("bt_frequency").value,
    cost_bps: Number($("bt_cost").value),
    benchmark: $("bt_benchmark").value.trim() || null,
  };
  const data = await postJson("/api/backtest", payload);
  if (state.bt && state.bt.anim) cancelAnimationFrame(state.bt.anim.raf);
  const model = buildBtModel(data);
  state.bt = { data, model, anim: null };
  $("bt-empty").hidden = true;
  renderMeta(data.data, data.result);
  showNotes(data.data.warnings);
  renderBtTable(data, model);
  animateBacktest();
}

function hoverBt(e) {
  const b = state.bt, wrap = $("bt-chart");
  if (!b || !b.anim || !b.anim.locked) return;
  const r = wrap.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = b.g, m = b.model;
  if (x < g.x0 || x > g.x1 || y < g.y0 || y > g.y1) return clearHover("bt");
  wrap.style.cursor = "crosshair";
  const n = m.dates.length, i = clamp(Math.round((x - g.x0) / (g.x1 - g.x0) * (n - 1)), 0, n - 1), px = b.sx(i);
  const { ctx, W, H } = fit($("bt-hv"));
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, g.y0); ctx.lineTo(Math.round(px) + 0.5, g.y1);
  const inEq = y <= g.eqY1, inDd = y >= g.ddY0;
  if (inEq || inDd) { ctx.moveTo(g.x0, Math.round(y) + 0.5); ctx.lineTo(g.x1, Math.round(y) + 0.5); }
  ctx.stroke(); ctx.setLineDash([]);
  for (const s of m.strategies) {
    const c = css(styleFor(s.name).color);
    for (const yy of [b.se(s.equity[i]), b.sd(s.drawdown[i])]) { ctx.fillStyle = css("--panel"); ctx.fillRect(px - 4, yy - 4, 8, 8); ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.strokeRect(px - 4, yy - 4, 8, 8); }
  }
  axisTag(ctx, m.dates[i], px, g.y1 + 3, "center");
  if (inEq) axisTag(ctx, money(m.E0 + (g.eqY1 - y) / (g.eqY1 - g.y0) * (m.E1 - m.E0)), g.x0 - 3, y - 9, "right");
  if (inDd) axisTag(ctx, pct((y - g.ddY0) / (g.y1 - g.ddY0) * m.D0), g.x0 - 3, y - 9, "right");

  const tip = $("bt-tip");
  tip.innerHTML = `<div class="k">${m.dates[i]}</div><div class="rows3"><span class="h"></span><span class="h">Value</span><span class="h">Drawdown</span>`
    + m.strategies.map((s) => `<span><i class="sw" style="background:var(${styleFor(s.name).color})"></i>${s.name}</span><span>${money(s.equity[i])}</span><span>${pct(s.drawdown[i])}</span>`).join("")
    + "</div>";
  placeTip(tip, wrap, px, y);
}

// ---------------------------------------------------------------------------
// Header meta, modes, run button
// ---------------------------------------------------------------------------
function renderMeta(info, result) {
  const span = result && result.dates.length ? `Backtest ${result.dates[0]} to ${result.dates[result.dates.length - 1]} · ${result.rebalances} rebalances` : `Prices ${info.start} to ${info.end}`;
  $("data-meta").textContent = `${span} · ${info.rows.toLocaleString()} trading days`;
}

function selectMode(mode) {
  state.mode = mode;
  for (const name of ["optimize", "backtest"]) {
    $(`tab-${name}`).setAttribute("aria-selected", String(name === mode));
    $(`view-${name}`).hidden = name !== mode;
  }
  document.querySelectorAll("[data-mode]").forEach((f) => { f.hidden = f.dataset.mode !== mode; });
  $("run").textContent = mode === "optimize" ? "Optimize" : "Run backtest";
  if (mode === "optimize" && state.opt && state.opt.anim && state.opt.anim.locked) { drawOpt(Infinity); }
  if (mode === "backtest" && state.bt && state.bt.anim && state.bt.anim.locked) { drawBt(Infinity); }
}

async function onRun() {
  if (state.busy) return;
  if (state.tickers.length < 2) { showError("Add at least two tickers first."); $("add-ticker").focus(); return; }
  clearError();
  state.busy = true;
  const btn = $("run"), label = btn.textContent;
  btn.setAttribute("aria-busy", "true"); btn.textContent = "Fetching prices…";
  $("progress").hidden = false;
  saveInputs();
  try {
    if (state.mode === "optimize") await runOptimize(); else await runBacktest();
  } catch (err) {
    showError(err.message);
    setVerdict("Fix the issue above, then run again.");
  } finally {
    state.busy = false;
    btn.removeAttribute("aria-busy"); btn.textContent = label;
    $("progress").hidden = true;
  }
}

function redrawStatic() {
  if (state.opt && state.opt.anim && state.opt.anim.locked && !$("view-optimize").hidden) { drawOpt(Infinity); clearHover("opt"); }
  if (state.bt && state.bt.anim && state.bt.anim.locked && !$("view-backtest").hidden) { drawBt(Infinity); clearHover("bt"); }
}

document.addEventListener("DOMContentLoaded", () => {
  restoreInputs();
  renderTickers();
  syncSource();

  $("add-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("add-ticker"), bad = addTickers(input.value);
    input.value = "";
    if (bad.length) showError(`${bad.join(", ")} ${bad.length === 1 ? "isn't a valid ticker" : "aren't valid tickers"}.`); else clearError();
  });
  $("add-ticker").addEventListener("paste", (e) => {
    const text = (e.clipboardData || window.clipboardData).getData("text");
    if (/[\s,;]/.test(text.trim())) { e.preventDefault(); addTickers(text); }
  });
  document.querySelectorAll("[data-source]").forEach((b) => b.addEventListener("click", () => { state.source = b.dataset.source; syncSource(); saveInputs(); }));
  $("csv-file").addEventListener("change", (e) => { if (e.target.files[0]) uploadCsv(e.target.files[0]); });
  for (const id of FIELDS) $(id).addEventListener("change", saveInputs);
  $("tab-optimize").addEventListener("click", () => selectMode("optimize"));
  $("tab-backtest").addEventListener("click", () => selectMode("backtest"));
  $("run").addEventListener("click", onRun);

  $("opt-chart").addEventListener("pointermove", hoverOpt);
  $("opt-chart").addEventListener("pointerleave", () => clearHover("opt"));
  $("bt-chart").addEventListener("pointermove", hoverBt);
  $("bt-chart").addEventListener("pointerleave", () => clearHover("bt"));

  new ResizeObserver(redrawStatic).observe($("shell"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawStatic);
});
