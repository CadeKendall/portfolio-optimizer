"use strict";

const STORAGE_KEY = "portfolio-optimizer:inputs";
const FIELDS = [
  "tickers", "risk_free", "min_weight", "max_weight", "opt_lookback",
  "bt_years", "bt_lookback", "bt_frequency", "bt_cost", "bt_benchmark",
];
const $ = (id) => document.getElementById(id);

const state = { uploadId: null, uploadMeta: null, lastOptimize: null, lastBacktest: null };

// ---------- Formatting ----------

const pct = (v, digits = 1) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);
const num = (v, digits = 2) => (v == null ? "–" : v.toFixed(digits));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child != null) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------- Persisting inputs (per-browser convenience only) ----------

function saveInputs() {
  const values = Object.fromEntries(FIELDS.map((id) => [id, $(id).value]));
  values.source = currentSource();
  values.uploadId = state.uploadId;
  values.uploadMeta = state.uploadMeta;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(values)); } catch { /* storage unavailable */ }
}

function restoreInputs() {
  let values = null;
  try { values = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* ignore */ }
  if (!values) return;
  for (const id of FIELDS) if (values[id] != null) $(id).value = values[id];
  if (values.source) document.querySelector(`input[name="source"][value="${values.source}"]`).checked = true;
  state.uploadId = values.uploadId || null;
  state.uploadMeta = values.uploadMeta || null;
}

// ---------- Inputs ----------

const currentSource = () => document.querySelector('input[name="source"]:checked').value;

function parseTickers() {
  return $("tickers").value.split(/[\s,;]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
}

function commonPayload() {
  const payload = {
    tickers: parseTickers(),
    source: currentSource(),
    risk_free: Number($("risk_free").value) / 100,
    min_weight: Number($("min_weight").value) / 100,
    max_weight: Number($("max_weight").value) / 100,
  };
  if (payload.source === "upload") payload.upload_id = state.uploadId;
  return payload;
}

function syncSourceUi() {
  $("upload-box").hidden = currentSource() !== "upload";
  renderUploadStatus();
}

function renderUploadStatus() {
  const meta = state.uploadMeta;
  $("upload-status").textContent = meta
    ? `Loaded ${meta.tickers.length} tickers, ${meta.start} to ${meta.end} (${meta.rows} rows): ${meta.tickers.join(", ")}`
    : "No file uploaded yet.";
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
    if (!$("tickers").value.trim()) $("tickers").value = meta.tickers.join(", ");
    saveInputs();
  } catch (err) {
    showError(err.message);
  }
  renderUploadStatus();
}

// ---------- Requests ----------

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
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Couldn't reach the server. Is it still running?");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(body && body.detail) || `HTTP ${response.status}`);
  return body;
}

const postJson = (url, payload) =>
  request(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function clearError() { $("error").hidden = true; }

async function run(button, output, empty, work) {
  clearError();
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "Running…";
  output.classList.add("busy"); // keep the previous result visible, dimmed
  try {
    await work();
    output.hidden = false;
    empty.hidden = true;
  } catch (err) {
    showError(err.message);
  } finally {
    output.classList.remove("busy");
    button.disabled = false;
    button.textContent = label;
  }
}

// ---------- Charts ----------

function baseLayout(extra = {}) {
  const ink = cssVar("--text-secondary");
  const axis = (more) => ({
    gridcolor: cssVar("--grid"),
    linecolor: cssVar("--axis"),
    zerolinecolor: cssVar("--axis"),
    tickfont: { color: cssVar("--text-muted"), size: 12 },
    title: { font: { color: ink, size: 12 } },
    automargin: true,
    ...more,
  });
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: ink, size: 13 },
    margin: { l: 56, r: 16, t: 8, b: 44 },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: ink } },
    hoverlabel: {
      bgcolor: cssVar("--surface"),
      bordercolor: cssVar("--axis"),
      font: { color: cssVar("--text-primary"), family: 'system-ui, sans-serif', size: 13 },
    },
    ...extra,
    xaxis: axis(extra.xaxis || {}),
    yaxis: axis(extra.yaxis || {}),
  };
}

const PLOT_CONFIG = { responsive: true, displayModeBar: false };

const PORTFOLIO_COLORS = { max_sharpe: "--series-2", min_volatility: "--series-3" };
const STRATEGY_COLORS = {
  "Max Sharpe": "--series-2",
  "Min volatility": "--series-3",
  "Equal weight": "--series-1",
};
const strategyColor = (name) => cssVar(STRATEGY_COLORS[name] || "--reference");

function portfolioHover(label, p) {
  return `<b>${label}</b><br>Return ${pct(p.expected_return)}<br>Volatility ${pct(p.volatility)}`
    + `<br>Sharpe ${num(p.sharpe)}<extra></extra>`;
}

function renderFrontier(data) {
  const traces = [];
  if (data.frontier.length) {
    traces.push({
      name: "Efficient frontier",
      x: data.frontier.map((p) => p.volatility),
      y: data.frontier.map((p) => p.expected_return),
      mode: "lines",
      line: { color: cssVar("--series-1"), width: 2, shape: "spline" },
      hovertemplate: data.frontier.map((p) => portfolioHover("Frontier", p)),
    });
  }
  const tickers = Object.keys(data.assets);
  traces.push({
    name: "Individual assets",
    x: tickers.map((t) => data.assets[t].volatility),
    y: tickers.map((t) => data.assets[t].expected_return),
    text: tickers,
    mode: "markers+text",
    textposition: "top center",
    cliponaxis: false, // keep the top asset's label from being cut off at the plot edge
    textfont: { color: cssVar("--text-secondary"), size: 12 },
    marker: { color: cssVar("--text-muted"), size: 9, line: { color: cssVar("--surface"), width: 2 } },
    hovertemplate: "<b>%{text}</b><br>Return %{y:.1%}<br>Volatility %{x:.1%}<extra></extra>",
  });
  for (const [key, label] of [["min_volatility", "Min volatility"], ["max_sharpe", "Max Sharpe"]]) {
    const p = data[key];
    if (!p) continue;
    traces.push({
      name: label,
      x: [p.volatility],
      y: [p.expected_return],
      mode: "markers",
      marker: { color: cssVar(PORTFOLIO_COLORS[key]), size: 14, line: { color: cssVar("--surface"), width: 2 } },
      hovertemplate: portfolioHover(label, p),
    });
  }
  const layout = baseLayout({
    hovermode: "closest",
    hoverdistance: 30,
    xaxis: { title: { text: "Volatility (annualized)" }, tickformat: ".0%", rangemode: "tozero" },
    yaxis: { title: { text: "Expected return (annualized)" }, tickformat: ".0%" },
  });
  Plotly.react($("frontier-chart"), traces, layout, PLOT_CONFIG);
}

function renderLines(target, data, valuesKey, yaxis) {
  const traces = data.result.strategies.map((s) => ({
    name: s.name,
    x: data.result.dates,
    y: s[valuesKey],
    mode: "lines",
    line: { color: strategyColor(s.name), width: 2 },
  }));
  const layout = baseLayout({
    hovermode: "x unified",
    xaxis: { type: "date", showspikes: true, spikemode: "across", spikethickness: 1, spikedash: "solid", spikecolor: cssVar("--axis") },
    yaxis,
  });
  Plotly.react($(target), traces, layout, PLOT_CONFIG);
}

// ---------- Optimize tab ----------

function renderDataInfo(target, info, extraWarnings = []) {
  const box = $(target);
  box.replaceChildren(
    `Prices ${info.start} to ${info.end} · ${info.rows.toLocaleString()} shared trading days · ${info.tickers.join(", ")}`,
  );
  const warnings = [...info.warnings, ...extraWarnings];
  if (warnings.length) box.append(el("ul", {}, ...warnings.map((w) => el("li", {}, w))));
}

function tile(label, colorVar, portfolio, fallback) {
  const title = el("div", { class: "tile-title" },
    el("span", { class: "swatch", style: `background: var(${colorVar})` }), label);
  if (!portfolio) return el("div", { class: "panel tile" }, title, el("p", { class: "unavailable" }, fallback));
  const stat = (value, name) => el("div", {}, el("div", { class: "stat-value" }, value), el("div", { class: "stat-label" }, name));
  return el("div", { class: "panel tile" }, title, el("div", { class: "stats" },
    stat(pct(portfolio.expected_return), "Expected return"),
    stat(pct(portfolio.volatility), "Volatility"),
    stat(num(portfolio.sharpe), "Sharpe"),
  ));
}

function weightCell(weight, colorVar) {
  if (weight == null) return el("td", {}, "–");
  const width = Math.max(0, Math.min(1, weight)) * 100;
  return el("td", {}, el("div", { class: "bar-cell" },
    el("span", {}, pct(weight)),
    el("span", { class: "bar-track" }, el("span", { class: "bar-fill", style: `display:block;width:${width}%;background:var(${colorVar})` })),
  ));
}

function renderOptimize(data) {
  renderDataInfo("opt-data", data.data, data.warnings);
  $("opt-tiles").replaceChildren(
    tile("Max Sharpe", PORTFOLIO_COLORS.max_sharpe, data.max_sharpe, "Not available. See the warning above."),
    tile("Min volatility", PORTFOLIO_COLORS.min_volatility, data.min_volatility),
  );

  const tickers = data.data.tickers;
  const header = el("tr", {}, el("th", {}, "Ticker"), el("th", {}, "Max Sharpe"), el("th", {}, "Min volatility"));
  const rows = tickers.map((t) => el("tr", {},
    el("td", {}, t),
    weightCell(data.max_sharpe ? data.max_sharpe.weights[t] : null, PORTFOLIO_COLORS.max_sharpe),
    weightCell(data.min_volatility.weights[t], PORTFOLIO_COLORS.min_volatility),
  ));
  $("weights-table").replaceChildren(el("thead", {}, header), el("tbody", {}, ...rows));

  const assetRows = tickers.map((t) => {
    const a = data.assets[t];
    return el("tr", {}, el("td", {}, t), el("td", {}, pct(a.expected_return)), el("td", {}, pct(a.volatility)));
  });
  $("assets-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Ticker"), el("th", {}, "Expected return"), el("th", {}, "Volatility"))),
    el("tbody", {}, ...assetRows),
  );
  renderFrontier(data);
}

async function runOptimize() {
  saveInputs();
  const payload = { ...commonPayload(), lookback_years: Number($("opt_lookback").value) };
  await run($("run-optimize"), $("optimize-output"), $("optimize-empty"), async () => {
    const data = await postJson("/api/optimize", payload);
    state.lastOptimize = data;
    $("optimize-output").hidden = false; // Plotly needs a visible container to size itself.
    renderOptimize(data);
  });
}

// ---------- Backtest tab ----------

function renderBacktest(data) {
  const result = data.result;
  renderDataInfo("bt-data", data.data);
  if (result.dates.length) {
    $("bt-data").prepend(
      `Backtest ${result.dates[0]} to ${result.dates[result.dates.length - 1]} · ${result.rebalances} rebalances. `,
    );
  }

  renderLines("equity-chart", data, "equity", { tickformat: "$.2f", title: { text: "Value of $1 invested" } });
  renderLines("drawdown-chart", data, "drawdown", { tickformat: ".0%", title: { text: "Below previous peak" } });

  const header = el("tr", {}, ...["Strategy", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Turnover / yr", "Final $1"]
    .map((h) => el("th", {}, h)));
  const rows = result.strategies.map((s) => {
    const m = s.metrics;
    const swatch = el("span", { class: "swatch line", style: `background:${strategyColor(s.name)}` });
    return el("tr", {},
      el("td", {}, el("span", { class: "name-cell" }, swatch, s.name)),
      el("td", {}, pct(m.cagr)),
      el("td", {}, pct(m.volatility)),
      el("td", {}, num(m.sharpe)),
      el("td", {}, pct(m.max_drawdown)),
      el("td", {}, pct(m.annual_turnover, 0)),
      el("td", {}, `$${s.equity[s.equity.length - 1].toFixed(2)}`),
    );
  });
  $("metrics-table").replaceChildren(el("thead", {}, header), el("tbody", {}, ...rows));

  const skips = $("skips");
  skips.hidden = result.skips.length === 0;
  skips.querySelector("summary").textContent =
    `${result.skips.length} rebalance${result.skips.length === 1 ? "" : "s"} couldn't be optimized`;
  skips.querySelector("ul").replaceChildren(
    ...result.skips.map((s) => el("li", {}, `${s.date} · ${s.strategy}: ${s.reason}`)),
  );
}

async function runBacktest() {
  saveInputs();
  const payload = {
    ...commonPayload(),
    years: Number($("bt_years").value),
    lookback_years: Number($("bt_lookback").value),
    frequency: $("bt_frequency").value,
    cost_bps: Number($("bt_cost").value),
    benchmark: $("bt_benchmark").value.trim() || null,
  };
  await run($("run-backtest"), $("backtest-output"), $("backtest-empty"), async () => {
    const data = await postJson("/api/backtest", payload);
    state.lastBacktest = data;
    $("backtest-output").hidden = false;
    renderBacktest(data);
  });
}

// ---------- Tabs, theme and wiring ----------

function selectTab(name) {
  for (const tab of ["optimize", "backtest"]) {
    const selected = tab === name;
    $(`tab-${tab}`).setAttribute("aria-selected", String(selected));
    $(`panel-${tab}`).hidden = !selected;
  }
  // Charts drawn while hidden have zero width, so resize after showing them.
  for (const id of ["frontier-chart", "equity-chart", "drawdown-chart"]) {
    const node = $(id);
    if (node.data && node.offsetParent) Plotly.Plots.resize(node);
  }
}

function rerender() {
  if (state.lastOptimize) renderOptimize(state.lastOptimize);
  if (state.lastBacktest) renderBacktest(state.lastBacktest);
}

document.addEventListener("DOMContentLoaded", () => {
  restoreInputs();
  syncSourceUi();

  $("tab-optimize").addEventListener("click", () => selectTab("optimize"));
  $("tab-backtest").addEventListener("click", () => selectTab("backtest"));
  $("run-optimize").addEventListener("click", runOptimize);
  $("run-backtest").addEventListener("click", runBacktest);
  $("inputs").addEventListener("change", saveInputs);
  $("inputs").addEventListener("submit", (e) => e.preventDefault());
  for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", syncSourceUi);
  }
  $("csv-file").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadCsv(e.target.files[0]);
  });
  for (const id of ["opt_lookback", "bt_years", "bt_lookback", "bt_frequency", "bt_cost", "bt_benchmark"]) {
    $(id).addEventListener("change", saveInputs);
  }
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
});
