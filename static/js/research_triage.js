/**
 * MediHaven Research Triage (DATA_MODE=mimic): the synthetic triage dashboard's layout over the
 * MIMIC-IV Demo research cohort. Every risk shown is an out-of-fold estimate, classified on the
 * six-level scale from common.js (Very low ... Extreme). Elements are built with DOM APIs and
 * textContent only (no innerHTML), and nothing is kept in browser storage.
 */

const VOTE_MODELS = ["logistic_regression", "decision_tree", "random_forest", "neural_network"];
const VOTE_LABELS = {
  logistic_regression: "Logistic regression",
  decision_tree: "Decision tree",
  random_forest: "Random forest",
  neural_network: "Neural network",
};
const VOTE_SHORT = { logistic_regression: "LR", decision_tree: "DT", random_forest: "RF", neural_network: "NN" };
const HIGH_LEVELS = ["High", "Very high", "Extreme"];

let roster = null;
let activeTierFilter = "";
let drawerRequestSeq = 0;

document.addEventListener("DOMContentLoaded", () => {
  setupControls();
  setupDrawer();
  loadRoster();
});

// ==============================================================================
// Small DOM helpers
// ==============================================================================
function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  if (options.title) node.title = options.title;
  if (options.style) node.style.cssText = options.style;
  if (options.dataset) Object.entries(options.dataset).forEach(([k, v]) => (node.dataset[k] = v));
  children.forEach((child) => child && node.appendChild(child));
  return node;
}

function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

const pct = (p) => (p === null || p === undefined ? "--" : `${(p * 100).toFixed(1)}%`);
const fmt = (v, digits = 0) => (v === null || v === undefined ? "--" : Number(v).toFixed(digits));

function levelLegend(container) {
  container.textContent = "";
  RISK_LEVELS.forEach((entry, i) => {
    container.appendChild(el("span", { style: "display: inline-flex; align-items: center; gap: 5px;" }, [
      riskBadge(entry.level, "small"),
      el("span", { text: riskRangeText(entry, RISK_LEVELS[i + 1]) }),
    ]));
  });
}

// ==============================================================================
// 1. Roster, KPIs and highest-risk stays
// ==============================================================================
async function loadRoster() {
  const tbody = document.getElementById("research-triage-tbody");
  try {
    const res = await fetch("/api/research/stays");
    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.success) {
      showTableMessage(tbody, `Stay view unavailable: ${json.error?.message || "server error"}`);
      return;
    }
    roster = json.data;
    renderKpis();
    renderHighestRisk();
    renderTable();
  } catch (err) {
    showTableMessage(tbody, "Connection error while loading the research cohort.");
  }
}

function showTableMessage(tbody, message) {
  tbody.textContent = "";
  const cell = el("td", { text: message, style: "text-align: center; padding: 40px; color: var(--text-secondary);" });
  cell.colSpan = 10;
  tbody.appendChild(el("tr", {}, [cell]));
}

function renderKpis() {
  const k = roster.kpis;
  setText("kpi-stays", k.stays);
  setText("kpi-top-levels", k.tiers["Extreme"] + k.tiers["Very high"]);
  setText("kpi-high", k.tiers["High"]);
  setText("kpi-medium", k.tiers["Medium"]);
  setText("kpi-deaths", k.deaths);
  setText("kpi-death-rate", `${pct(roster.death_rate)} of stays`);
  setText("kpi-captured", `${k.deaths_high_or_above} / ${k.deaths}`);
  setText("count-all", k.stays);
  document.querySelectorAll("[data-count]").forEach((span) => (span.textContent = k.tiers[span.dataset.count] ?? 0));
}

function renderHighestRisk() {
  const container = document.getElementById("research-alerts-container");
  container.textContent = "";
  const top = roster.stays
    .filter((s) => HIGH_LEVELS.includes(s.tier))
    .sort((a, b) => b.risk - a.risk)
    .slice(0, 3);
  if (top.length === 0) {
    container.style.display = "none";
    return;
  }
  container.style.display = "block";
  top.forEach((stay) => {
    const chips = el("div", { className: "alert-triggers-list" },
      VOTE_MODELS.map((m) => el("span", { className: "trigger-chip",
        text: `${VOTE_LABELS[m]}: ${pct(stay.models[m].probability)} (${stay.models[m].tier})` })));
    const headline = el("div", { className: "alert-patient-headline",
      text: `Stay ${stay.stay} · ${stay.age ?? "--"}y ${stay.sex ?? ""} · ${stay.variables_measured} of 14 variables measured` });
    const sub = el("div", { style: "font-size: 11px; color: var(--text-secondary); margin-top: 2px;" }, [
      document.createTextNode("Overall risk: "),
      el("strong", { text: pct(stay.risk), style: "color: #ffffff;" }),
      document.createTextNode(` · Outcome: ${stay.died ? "died in hospital" : "survived"}`),
    ]);
    const content = el("div", { className: "alert-banner-content" }, [
      riskBadge(stay.tier),
      el("div", {}, [headline, sub]),
      chips,
    ]);
    const button = el("button", { className: "btn btn-chart btn-sm", text: "📊 View Chart" });
    button.addEventListener("click", () => openStay(stay.stay));
    container.appendChild(el("div", { className: "alert-banner" }, [content, el("div", {}, [button])]));
  });
}

function visibleStays() {
  const outcome = document.getElementById("outcome-filter-select").value;
  const query = document.getElementById("stay-search-input").value.trim();
  const sort = document.getElementById("sort-select").value;
  const spread = (s) => {
    const values = VOTE_MODELS.map((m) => s.models[m].probability);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length);
  };
  const stays = roster.stays.filter((s) => {
    if (activeTierFilter && s.tier !== activeTierFilter) return false;
    if (outcome === "died" && !s.died) return false;
    if (outcome === "survived" && s.died) return false;
    if (query && !String(s.stay).startsWith(query.replace(/^#/, ""))) return false;
    return true;
  });
  const sorters = {
    risk: (a, b) => b.risk - a.risk,
    stay: (a, b) => a.stay - b.stay,
    age: (a, b) => (b.age ?? -1) - (a.age ?? -1),
    agreement: (a, b) => spread(b) - spread(a),
  };
  return stays.sort(sorters[sort] || sorters.risk);
}

function voteChip(stay, m) {
  const color = riskLevelColor(stay.models[m].tier);
  return el("span", {
    text: VOTE_SHORT[m],
    title: `${VOTE_LABELS[m]}: ${pct(stay.models[m].probability)} (${stay.models[m].tier})`,
    style: `font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 999px; color: ${color}; ` +
           `border: 1px solid ${color}; background: ${color}22;`,
  });
}

function renderTable() {
  const tbody = document.getElementById("research-triage-tbody");
  const stays = visibleStays();
  tbody.textContent = "";
  if (stays.length === 0) {
    showTableMessage(tbody, "No stays match these filters.");
    return;
  }
  stays.forEach((s) => {
    const l = s.latest;
    const votes = el("div", { style: "display: flex; gap: 4px; flex-wrap: wrap;" }, VOTE_MODELS.map((m) => voteChip(s, m)));
    const outcome = el("span", {
      text: s.died ? "Died" : "Survived",
      style: `font-weight: 600; font-size: 12px; color: ${s.died ? "var(--risk-critical)" : "var(--risk-low)"};`,
    });
    const chartBtn = el("button", { className: "btn btn-chart btn-sm", text: "📊 View Chart" });
    chartBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openStay(s.stay);
    });
    const lactate = l.lactate === null ? "--" : `${fmt(l.lactate, 1)} mmol/L`;
    const lactateWarn = l.lactate !== null && l.lactate >= 2.0;
    const row = el("tr", { dataset: { stay: s.stay } }, [
      el("td", {}, [riskBadge(s.tier)]),
      el("td", { className: "mono", text: pct(s.risk), style: "font-weight: 700; color: #ffffff;" }),
      el("td", {}, [el("div", { className: "patient-name-cell", text: `Stay ${s.stay}` }),
                    el("div", { className: "patient-mrn-cell", text: `${s.variables_measured}/14 variables measured` })]),
      el("td", { text: `${s.age ?? "--"}y / ${s.sex ?? "--"}` }),
      el("td", { className: "mono", text: `${fmt(l.heart_rate)} bpm · ${fmt(l.sbp)} · ${l.spo2 === null ? "--" : fmt(l.spo2) + "%"}` }),
      el("td", { className: "mono", text: l.resp_rate === null ? "--" : `${fmt(l.resp_rate)} /min` }),
      el("td", {}, [el("span", { className: `hemo-pill ${lactateWarn ? "hemo-warn" : "hemo-normal"}`, text: lactate })]),
      el("td", {}, [votes]),
      el("td", {}, [outcome]),
      el("td", { style: "text-align: right;" }, [chartBtn]),
    ]);
    row.addEventListener("click", () => openStay(s.stay));
    tbody.appendChild(row);
  });
}

function setupControls() {
  document.querySelectorAll(".filter-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTierFilter = btn.dataset.filter;
      if (roster) renderTable();
    });
  });
  ["outcome-filter-select", "sort-select"].forEach((id) =>
    document.getElementById(id)?.addEventListener("change", () => roster && renderTable()));
  document.getElementById("stay-search-input")?.addEventListener("input", () => roster && renderTable());
}

// ==============================================================================
// 2. Stay chart drawer
// ==============================================================================
function setupDrawer() {
  const drawer = document.getElementById("drawer-backdrop");
  const close = () => {
    drawer.classList.remove("active");
    drawer.setAttribute("aria-hidden", "true");
  };
  document.getElementById("drawer-close-btn").addEventListener("click", close);
  drawer.addEventListener("click", (e) => {
    if (e.target === drawer) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer.classList.contains("active")) close();
  });
}

async function openStay(number) {
  const requestSeq = ++drawerRequestSeq;
  const drawer = document.getElementById("drawer-backdrop");
  drawer.classList.add("active");
  drawer.setAttribute("aria-hidden", "false");
  setText("drawer-stay-title", `Stay ${number}`);
  setText("drawer-stay-meta", "Loading stay chart...");

  try {
    const res = await fetch(`/api/research/stays/${number}`);
    const json = await res.json().catch(() => ({}));
    if (requestSeq !== drawerRequestSeq) return; // another stay was opened meanwhile
    if (!res.ok || !json.success) {
      setText("drawer-stay-meta", json.error?.message || "This stay could not be loaded.");
      return;
    }
    renderStay(json.data);
  } catch (err) {
    if (requestSeq === drawerRequestSeq) setText("drawer-stay-meta", "Connection error while loading this stay.");
  }
}

function renderStay(d) {
  const meta = [`${d.age ?? "--"}y ${d.sex ?? ""}`, `cross-validation fold ${d.fold} of 5`];
  if (d.bp_source) meta.push(`blood pressure by ${d.bp_source}`);
  setText("drawer-stay-meta", meta.join(" · "));

  const badgeSlot = document.getElementById("drawer-risk-badge");
  badgeSlot.className = "";
  badgeSlot.textContent = "";
  badgeSlot.appendChild(riskBadge(d.tier, "large"));

  setText("drawer-risk-val", pct(d.risk));
  setText("drawer-risk-context", `cohort death rate ${pct(d.death_rate)}`);
  const outcome = document.getElementById("drawer-outcome-val");
  outcome.textContent = d.died ? "Died in hospital" : "Survived to discharge";
  outcome.style.color = d.died ? "var(--risk-critical)" : "var(--risk-low)";
  const note = document.getElementById("drawer-outcome-note");
  note.textContent = d.outcome_note || "";
  note.style.display = d.outcome_note ? "block" : "none";

  VOTE_MODELS.forEach((m) => {
    const slot = document.getElementById(`vote-${m}`);
    if (!slot) return;
    slot.textContent = "";
    slot.appendChild(riskBadge(d.models[m].tier, "small"));
    slot.appendChild(el("span", { className: "mono", text: ` ${pct(d.models[m].probability)}`, style: "margin-left: 4px;" }));
  });

  renderModelsTable(d);
  renderVariables(d);

  setText("drawer-measure-meta", `${d.variables_measured} of 14 variables measured in the first 24 h`);
  setText("drawer-scoring-note",
    `Each model that scored this stay was trained on the other four folds, which contain none of this patient's ICU ` +
    `stays, so the score is an honest estimate of how the model performs on a new patient. The overall risk is the ` +
    `mean of the four models. Risk levels are fixed probability bands (Very low < 2.5%, Low 2.5–7.5%, Medium 7.5–15%, ` +
    `High 15–30%, Very high 30–60%, Extreme ≥ 60%; cohort death rate ${pct(d.death_rate)}). They are display bands ` +
    `for research, not clinical thresholds.`);
}

function renderModelsTable(d) {
  const container = document.getElementById("drawer-models-table");
  container.textContent = "";
  const table = el("table", { className: "clinical-table", style: "font-size: 12px;" });
  table.appendChild(el("tr", {}, ["Model", "Predicted risk", "Risk level"].map((h) => el("th", { text: h }))));
  VOTE_MODELS.forEach((m) => {
    table.appendChild(el("tr", {}, [
      el("td", { text: VOTE_LABELS[m] }),
      el("td", { className: "mono", text: pct(d.models[m].probability), style: "font-weight: 700; color: #ffffff;" }),
      el("td", {}, [riskBadge(d.models[m].tier)]),
    ]));
  });
  container.appendChild(table);
  levelLegend(document.getElementById("drawer-level-legend"));
}

function rangeBar(v) {
  const [refLow, refHigh] = v.reference || [null, null];
  const values = [v.min, v.max, v.last, refLow, refHigh].filter((x) => x !== null && x !== undefined);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pos = (x) => `${(((x - lo) / span) * 100).toFixed(1)}%`;
  const track = el("div", { style: "position: relative; height: 10px; min-width: 120px; background: rgba(148, 163, 184, 0.12); border-radius: 5px;" });
  if (refLow !== null) {
    track.appendChild(el("div", { title: `Typical range ${refLow}–${refHigh}`,
      style: `position: absolute; top: 0; bottom: 0; left: ${pos(refLow)}; width: calc(${pos(refHigh)} - ${pos(refLow)}); background: rgba(16, 185, 129, 0.25); border-radius: 5px;` }));
  }
  track.appendChild(el("div", { title: `24 h range ${v.min}–${v.max}`,
    style: `position: absolute; top: 3px; height: 4px; left: ${pos(v.min)}; width: max(2px, calc(${pos(v.max)} - ${pos(v.min)})); background: #38bdf8; border-radius: 2px;` }));
  track.appendChild(el("div", { title: `Last ${v.last}`,
    style: `position: absolute; top: -2px; width: 4px; height: 14px; left: calc(${pos(v.last)} - 2px); background: #ffffff; border-radius: 2px;` }));
  return track;
}

function renderVariables(d) {
  const tbody = document.getElementById("drawer-variables-tbody");
  tbody.textContent = "";
  d.variables.forEach((v) => {
    const [refLow, refHigh] = v.reference || [null, null];
    const outside = (x) => x !== null && refLow !== null && (x < refLow || x > refHigh);
    const digits = Math.abs(v.last ?? 0) < 10 ? 1 : 0;
    if (!v.measured) {
      tbody.appendChild(el("tr", {}, [
        el("td", { text: v.label }),
        el("td", { text: "not measured", style: "color: var(--text-muted); font-style: italic;" }),
        el("td", {}), el("td", {}), el("td", { text: "0" }),
        el("td", { className: "mono", text: refLow === null ? "--" : `${refLow}–${refHigh} ${v.unit}`, style: "color: var(--text-muted);" }),
      ]));
      return;
    }
    const lastCell = el("td", { className: "mono", text: `${fmt(v.last, digits)} ${v.unit}`,
      style: `font-weight: 700; color: ${outside(v.last) ? "#f87171" : "#ffffff"};` });
    tbody.appendChild(el("tr", {}, [
      el("td", { text: v.label }),
      lastCell,
      el("td", {}, [
        el("div", { className: "mono", text: `${fmt(v.min, digits)} – ${fmt(v.max, digits)}`,
          style: `font-size: 11px; color: ${outside(v.min) || outside(v.max) ? "#fbbf24" : "var(--text-secondary)"};` }),
        rangeBar(v),
      ]),
      el("td", { className: "mono", text: fmt(v.mean, digits) }),
      el("td", { className: "mono", text: String(v.count) }),
      el("td", { className: "mono", text: refLow === null ? "--" : `${refLow}–${refHigh} ${v.unit}`, style: "color: var(--text-muted);" }),
    ]));
  });
}
