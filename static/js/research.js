/**
 * MediHaven Research Dashboard Controller (research mode only)
 * Renders cohort-level results of the verified Phase 2 models with Chart.js and drives the
 * what-if estimator. Only aggregate data is fetched; no individual patient is ever displayed.
 * All server text is inserted with textContent (no innerHTML) and nothing is kept in browser storage.
 */

const MODEL_ORDER = ["dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network"];
// The prior-only baseline predicts the training death rate: a reference, not one of the four models
const REAL_MODELS = MODEL_ORDER.filter((m) => m !== "dummy_prior");
const MODEL_LABELS = {
  dummy_prior: "Baseline (death rate)",
  logistic_regression: "Logistic regression",
  decision_tree: "Decision tree",
  random_forest: "Random forest",
  neural_network: "Neural network",
};
const MODEL_COLORS = {
  dummy_prior: "#64748b",
  logistic_regression: "#38bdf8",
  decision_tree: "#f59e0b",
  random_forest: "#10b981",
  neural_network: "#a78bfa",
};
const VARIABLE_LABELS = {
  heart_rate: "Heart rate", resp_rate: "Respiratory rate", spo2: "SpO2", temperature: "Temperature",
  sbp: "Systolic BP", dbp: "Diastolic BP", map: "Mean arterial pressure", creatinine: "Creatinine",
  bun: "BUN", wbc: "WBC", hemoglobin: "Hemoglobin", platelets: "Platelets", lactate: "Lactate", glucose: "Glucose",
};
const LEGEND = { labels: { color: "#94a3b8", font: { size: 11 } } };
const AXIS = { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } };

let dashboardData = null;
let weightsChart = null;

document.addEventListener("DOMContentLoaded", loadResearchDashboard);

async function loadResearchDashboard() {
  try {
    const res = await fetch("/api/research/dashboard", { credentials: "same-origin" });
    const json = await res.json();
    if (!json.success) {
      showDashboardError(json.error ? json.error.message : "Research data unavailable.");
      return;
    }
    dashboardData = json.data;
    renderKpis(dashboardData);
    renderMetricCiChart("researchAurocChart", "auroc", 0.5, "Chance (0.5)");
    renderMetricCiChart("researchApChart", "average_precision", dashboardData.kpis.prevalence, "Death rate");
    renderCohortFlow(dashboardData.cohort);
    renderCoverage(dashboardData.cohort);
    renderSensitivity(dashboardData.sensitivity);
    renderFolds(dashboardData);
    setupWeights(dashboardData.feature_weights);
    renderQuality(dashboardData.cohort);
    renderMetricsTable(dashboardData);
    renderConfusionMatrices(dashboardData.models);
    buildWhatIfForm(dashboardData.whatif_variables);
    setText("note-bootstrap", `Intervals: ${dashboardData.definitions.bootstrap}. ${dashboardData.definitions.average_precision}.`);
  } catch (err) {
    console.error("Research dashboard failed to load:", err);
    showDashboardError("Research data could not be loaded.");
  }
}

// ------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function fmt(value, digits = 2) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function ciText(ci) {
  return ci && ci.low !== null && ci.low !== undefined ? `${fmt(ci.low)}–${fmt(ci.high)}` : "n/a";
}

function prettyStep(step) {
  return step.charAt(0).toUpperCase() + step.slice(1);
}

function makeEl(tag, text, style) {
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = text;
  if (style) el.style.cssText = style;
  return el;
}

function showDashboardError(message) {
  const banner = makeEl("div", message,
    "border: 1px solid rgba(239, 68, 68, 0.5); background: rgba(239, 68, 68, 0.1); color: #f87171;" +
    "border-radius: var(--radius-md); padding: var(--space-md); margin-bottom: var(--space-lg);");
  const page = document.querySelector(".research-page .kpi-grid");
  if (page) page.parentNode.insertBefore(banner, page);
}

function emptyChart(canvasId, message) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const note = makeEl("div", message, "color: var(--text-muted); font-size: 12px; padding: 24px 0; text-align: center;");
  canvas.parentNode.replaceChild(note, canvas);
}

// ------------------------------------------------------------------------------
// KPIs
// ------------------------------------------------------------------------------
function renderKpis(data) {
  const k = data.kpis;
  setText("kpi-stays", k.stays);
  setText("kpi-subjects", `${k.subjects} patients`);
  setText("kpi-deaths", k.deaths);
  setText("kpi-prevalence", `${(k.prevalence * 100).toFixed(1)}% of stays`);
  setText("kpi-features", k.features);
  setText("kpi-evaluation", k.evaluation_status);
  if (k.top_model) {
    setText("kpi-top-auroc", fmt(k.top_auroc));
    setText("kpi-top-model", `${MODEL_LABELS[k.top_model]} · CI ${ciText(k.top_auroc_ci)} · intervals overlap`);
  }
  const anomalies = data.cohort ? data.cohort.outcome_anomalies : null;
  setText("kpi-anomalies", anomalies === null || anomalies === undefined ? "n/a" : anomalies);
}

// ------------------------------------------------------------------------------
// Charts
// ------------------------------------------------------------------------------
function renderMetricCiChart(canvasId, metric, reference, referenceLabel) {
  const models = dashboardData.models;
  const labels = MODEL_ORDER.map((m) => MODEL_LABELS[m]);
  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "95% CI",
          data: MODEL_ORDER.map((m) => {
            const ci = models[m].ci_95[metric];
            return ci && ci.low !== null ? [ci.low, ci.high] : null;
          }),
          backgroundColor: MODEL_ORDER.map((m) => MODEL_COLORS[m] + "40"),
          borderColor: MODEL_ORDER.map((m) => MODEL_COLORS[m]),
          borderWidth: 1,
          barPercentage: 0.45,
        },
        {
          type: "line",
          label: "Pooled estimate",
          data: MODEL_ORDER.map((m) => models[m][metric]),
          showLine: false,
          pointRadius: 6,
          pointHoverRadius: 7,
          pointBackgroundColor: MODEL_ORDER.map((m) => MODEL_COLORS[m]),
          pointBorderColor: "#0b0f17",
        },
        {
          type: "line",
          label: referenceLabel,
          data: MODEL_ORDER.map(() => reference),
          borderColor: "#94a3b8",
          borderDash: [5, 4],
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: LEGEND,
        tooltip: {
          callbacks: {
            label: (ctx) => ctx.dataset.type === "bar" && Array.isArray(ctx.raw)
              ? `95% CI: ${fmt(ctx.raw[0])}–${fmt(ctx.raw[1])}`
              : `${ctx.dataset.label}: ${fmt(ctx.raw, 3)}`,
          },
        },
      },
      scales: { x: AXIS, y: { ...AXIS, min: 0, max: 1 } },
    },
  });
}

function renderCohortFlow(cohort) {
  if (!cohort || !cohort.flow) return emptyChart("researchFlowChart", "Cohort flow not recorded in the dataset manifest.");
  const steps = cohort.flow.filter((s, i) => i === 0 || s.excluded > 0);
  setText("flow-source", `Stays remaining after each rule · ${cohort.source_release || "source unknown"}`);
  new Chart(document.getElementById("researchFlowChart"), {
    type: "bar",
    data: {
      labels: steps.map((s) => prettyStep(s.step)),
      datasets: [{
        label: "Stays remaining",
        data: steps.map((s) => s.remaining),
        backgroundColor: steps.map((s, i) => (i === steps.length - 1 ? "#10b981" : "rgba(56, 189, 248, 0.55)")),
        borderColor: "#38bdf8",
        borderWidth: 1,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { afterLabel: (ctx) => (ctx.dataIndex > 0 ? `Excluded at this step: ${steps[ctx.dataIndex].excluded}` : "") } },
      },
      scales: { x: { ...AXIS, beginAtZero: true }, y: AXIS },
    },
  });
}

function renderCoverage(cohort) {
  if (!cohort || !cohort.coverage || !Object.keys(cohort.coverage).length) {
    return emptyChart("researchCoverageChart", "Coverage not recorded in the dataset manifest.");
  }
  const names = Object.keys(cohort.coverage);
  new Chart(document.getElementById("researchCoverageChart"), {
    type: "bar",
    data: {
      labels: names.map((n) => VARIABLE_LABELS[n] || n),
      datasets: [{
        label: "Stays with ≥ 1 measurement (%)",
        data: names.map((n) => Math.round(cohort.coverage[n] * 1000) / 10),
        backgroundColor: names.map((n) => (cohort.coverage[n] < 0.8 ? "rgba(245, 158, 11, 0.7)" : "rgba(16, 185, 129, 0.6)")),
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: LEGEND },
      scales: { x: { ...AXIS, ticks: { ...AXIS.ticks, maxRotation: 60, minRotation: 45 } }, y: { ...AXIS, min: 0, max: 100 } },
    },
  });
}

function renderSensitivity(sensitivity) {
  const analyses = [
    ["primary", "Primary (all stays)", "#38bdf8"],
    ["sensitivity_evaluation_only", "A: anomalies removed from scoring", "#a78bfa"],
    ["sensitivity_retrained", "B: anomalies removed before training", "#f472b6"],
  ];
  new Chart(document.getElementById("researchSensitivityChart"), {
    type: "bar",
    data: {
      labels: MODEL_ORDER.map((m) => MODEL_LABELS[m]),
      datasets: analyses.map(([key, label, color]) => ({
        label: `${label} (${sensitivity[key].population.positives} deaths)`,
        data: MODEL_ORDER.map((m) => (sensitivity[key].auroc ? sensitivity[key].auroc[m] : null)),
        backgroundColor: color,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: LEGEND, tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.raw, 3)}` } } },
      scales: { x: AXIS, y: { ...AXIS, min: 0, max: 1 } },
    },
  });
}

function renderFolds(data) {
  const folds = data.folds || [];
  const positives = folds.map((f) => f.test_positives);
  setText("fold-note", `Test deaths per fold: ${positives.join(", ")} · large swings are expected`);
  new Chart(document.getElementById("researchFoldChart"), {
    type: "line",
    data: {
      labels: folds.map((f) => `Fold ${f.fold}`),
      datasets: REAL_MODELS.map((m) => ({
        label: MODEL_LABELS[m],
        data: data.models[m].per_fold_auroc,
        borderColor: MODEL_COLORS[m],
        backgroundColor: MODEL_COLORS[m],
        tension: 0.25,
        borderWidth: 2,
        pointRadius: 4,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: LEGEND },
      scales: { x: AXIS, y: { ...AXIS, min: 0, max: 1 } },
    },
  });
}

function setupWeights(weights) {
  if (!weights) return emptyChart("researchWeightsChart", "Feature weights unavailable for these models.");
  const rf = document.getElementById("weights-rf");
  const lr = document.getElementById("weights-lr");
  const show = (key) => {
    rf.setAttribute("aria-pressed", String(key === "random_forest_importance"));
    lr.setAttribute("aria-pressed", String(key === "logistic_regression_coefficient"));
    rf.classList.toggle("btn-primary", key === "random_forest_importance");
    lr.classList.toggle("btn-primary", key === "logistic_regression_coefficient");
    renderWeights(weights[key], key);
  };
  rf.addEventListener("click", () => show("random_forest_importance"));
  lr.addEventListener("click", () => show("logistic_regression_coefficient"));
  show("random_forest_importance");
}

function renderWeights(rows, key) {
  const isLr = key === "logistic_regression_coefficient";
  setText("weights-note", isLr
    ? "Standardized coefficients of the final logistic regression (red = higher risk, blue = lower). Not causal."
    : "Impurity-based importance of the final random forest. Unstable with few deaths; not causal.");
  if (weightsChart) weightsChart.destroy();
  weightsChart = new Chart(document.getElementById("researchWeightsChart"), {
    type: "bar",
    data: {
      labels: rows.map((r) => r.feature),
      datasets: [{
        label: isLr ? "Coefficient" : "Importance",
        data: rows.map((r) => r.weight),
        backgroundColor: rows.map((r) => (isLr ? (r.weight >= 0 ? "rgba(239, 68, 68, 0.7)" : "rgba(56, 189, 248, 0.7)") : "rgba(16, 185, 129, 0.65)")),
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: AXIS, y: { ...AXIS, ticks: { ...AXIS.ticks, font: { size: 9 } } } },
    },
  });
}

function renderQuality(cohort) {
  if (!cohort || !cohort.data_quality) return emptyChart("researchQualityChart", "Quality counts not recorded in the dataset manifest.");
  const dropped = cohort.data_quality.dropped;
  const labels = {
    outside_window: "Outside 24h window",
    stored_after_index: "Recorded after prediction time",
    non_numeric: "Non-numeric",
    unit_mismatch: "Unexpected unit",
    implausible: "Implausible value",
  };
  setText("quality-kept", `${cohort.data_quality.kept.toLocaleString()} measurements kept · removed rows by rule`);
  const keys = Object.keys(labels);
  new Chart(document.getElementById("researchQualityChart"), {
    type: "bar",
    data: {
      labels: keys.map((k) => labels[k]),
      datasets: [{ label: "Rows removed", data: keys.map((k) => dropped[k] || 0), backgroundColor: "rgba(249, 115, 22, 0.7)" }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: AXIS, y: { ...AXIS, beginAtZero: true } },
    },
  });
}

// ------------------------------------------------------------------------------
// Tables
// ------------------------------------------------------------------------------
function renderMetricsTable(data) {
  const body = document.getElementById("research-metrics-body");
  MODEL_ORDER.forEach((m) => {
    const r = data.models[m];
    const tr = document.createElement("tr");
    const nameCell = makeEl("td");
    const swatch = makeEl("span", "", `display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:${MODEL_COLORS[m]};`);
    nameCell.appendChild(swatch);
    nameCell.appendChild(document.createTextNode(MODEL_LABELS[m]));
    tr.appendChild(nameCell);
    [
      `${fmt(r.auroc)} (${ciText(r.ci_95.auroc)})`,
      `${fmt(r.average_precision)} (${ciText(r.ci_95.average_precision)})`,
      fmt(r.brier_score, 3), fmt(r.sensitivity), fmt(r.specificity), fmt(r.precision), fmt(r.f1),
    ].forEach((value) => {
      const td = makeEl("td", value);
      td.className = "mono";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  setText("threshold-note", `Thresholded metrics: ${data.definitions.threshold_rule}. Undefined metrics are shown as n/a.`);
}

function renderConfusionMatrices(models) {
  const container = document.getElementById("research-confusion");
  MODEL_ORDER.forEach((m) => {
    const cm = models[m].confusion_matrix;
    const card = makeEl("div", null,
      `background: var(--bg-surface-subtle); border: 1px solid var(--border-hairline); border-left: 3px solid ${MODEL_COLORS[m]};` +
      "border-radius: var(--radius-md); padding: var(--space-md);");
    card.appendChild(makeEl("div", MODEL_LABELS[m], "font-weight: 600; font-size: 13px; margin-bottom: 8px;"));
    const grid = makeEl("div", null, "display: grid; grid-template-columns: auto 1fr 1fr; gap: 4px; font-size: 12px; text-align: center;");
    const cell = (text, style) => grid.appendChild(makeEl("div", text, style));
    const head = "color: var(--text-muted); font-size: 11px;";
    const good = "background: rgba(16, 185, 129, 0.15); border-radius: 4px; padding: 6px 0;";
    const bad = "background: rgba(239, 68, 68, 0.15); border-radius: 4px; padding: 6px 0;";
    cell("", head); cell("Predicted death", head); cell("Predicted survival", head);
    cell("Died", head + "text-align: right;"); cell(String(cm.tp), good); cell(String(cm.fn), bad);
    cell("Survived", head + "text-align: right;"); cell(String(cm.fp), bad); cell(String(cm.tn), good);
    card.appendChild(grid);
    container.appendChild(card);
  });
}

// ------------------------------------------------------------------------------
// What-if estimator
// ------------------------------------------------------------------------------
function buildWhatIfForm(variables) {
  const fields = document.getElementById("whatif-fields");
  variables.forEach((v) => {
    const group = makeEl("div");
    group.className = "form-group";
    const label = makeEl("label", `${VARIABLE_LABELS[v.name] || v.name} (${v.unit})`);
    label.className = "form-label";
    label.htmlFor = `whatif-${v.name}`;
    const input = document.createElement("input");
    input.className = "form-control";
    input.type = "number";
    input.step = "any";
    input.min = v.min;
    input.max = v.max;
    input.id = `whatif-${v.name}`;
    input.dataset.variable = v.name;
    input.placeholder = `${v.min}–${v.max}`;
    group.appendChild(label);
    group.appendChild(input);
    fields.appendChild(group);
  });
  document.getElementById("research-whatif-form").addEventListener("submit", submitWhatIf);
  document.getElementById("research-whatif-form").addEventListener("reset", () => {
    document.getElementById("whatif-error").style.display = "none";
    document.getElementById("whatif-result").style.display = "none";
  });
}

async function submitWhatIf(event) {
  event.preventDefault();
  const errorBox = document.getElementById("whatif-error");
  errorBox.style.display = "none";
  const features = {};
  document.querySelectorAll("#whatif-fields input[data-variable]").forEach((input) => {
    if (input.value.trim() === "") return;
    features[`${input.dataset.variable}_count`] = 1;
    features[`${input.dataset.variable}_last`] = Number(input.value);
  });
  if ("sbp_count" in features || "dbp_count" in features) {
    features.bp_source_invasive = Number(document.getElementById("whatif-bp-source").value);
  }
  if ("map_count" in features) features.map_derived = Number(document.getElementById("whatif-map-source").value);
  const age = document.getElementById("whatif-age").value.trim();
  if (age !== "") features.age = Number(age);
  const sex = document.getElementById("whatif-sex").value;
  if (sex !== "") features.sex_male = Number(sex);

  if (!Object.keys(features).length) {
    errorBox.textContent = "Enter at least one value.";
    errorBox.style.display = "block";
    return;
  }
  try {
    const res = await fetch("/api/research/predict", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const json = await res.json();
    if (!json.success) {
      errorBox.textContent = json.error ? json.error.message : "The estimate could not be computed.";
      errorBox.style.display = "block";
      return;
    }
    renderWhatIf(json.data);
  } catch (err) {
    console.error("What-if request failed:", err);
    errorBox.textContent = "The estimate could not be computed.";
    errorBox.style.display = "block";
  }
}

function renderWhatIf(result) {
  document.getElementById("whatif-result").style.display = "block";
  const a = result.assessment;
  const box = document.getElementById("whatif-assessment");
  box.textContent = "";
  const pct = (p) => `${(p * 100).toFixed(1)}%`;

  // Overall level: mean of the four models
  const headline = makeEl("div", null,
    "display: flex; align-items: center; gap: 14px; flex-wrap: wrap; padding: 12px 14px; border-radius: 10px; " +
    `border: 1px solid ${riskLevelColor(a.level)}; background: ${riskLevelColor(a.level)}14;`);
  const label = makeEl("div", null);
  label.appendChild(makeEl("div", "Overall risk of in-hospital death", "font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;"));
  label.appendChild(makeEl("div", pct(a.consensus), "font-size: 26px; font-weight: 800; color: #ffffff; font-family: var(--font-mono, monospace);"));
  headline.appendChild(label);
  headline.appendChild(riskBadge(a.level, "large"));
  const agree = a.models_at_level === 4 ? "All 4 models agree on this level."
    : `${a.models_at_level} of 4 models give this level; ${a.models_high_or_above} of 4 rate the risk High or above.`;
  headline.appendChild(makeEl("div", agree, "font-size: 13px; color: #cbd5e1; flex: 1 1 220px;"));
  box.appendChild(headline);

  // Each model: its own percentage and level
  const table = makeEl("table", null, "width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px;");
  const head = makeEl("tr", null);
  ["Model", "Predicted risk", "Risk level"].forEach((h) =>
    head.appendChild(makeEl("th", h, "text-align: left; color: #94a3b8; font-weight: 600; padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.08);")));
  table.appendChild(head);
  REAL_MODELS.forEach((m) => {
    const row = makeEl("tr", null);
    row.appendChild(makeEl("td", MODEL_LABELS[m], "padding: 7px 8px; color: #e2e8f0;"));
    row.appendChild(makeEl("td", pct(a.models[m].probability), "padding: 7px 8px; color: #ffffff; font-weight: 700; font-family: var(--font-mono, monospace);"));
    const cell = makeEl("td", null, "padding: 7px 8px;");
    cell.appendChild(riskBadge(a.models[m].level));
    row.appendChild(cell);
    table.appendChild(row);
  });
  box.appendChild(table);

  // Scale legend
  const legend = makeEl("div", null, "display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 10px; font-size: 11px; color: #94a3b8;");
  RISK_LEVELS.forEach((entry, i) => {
    const item = makeEl("span", null, "display: inline-flex; align-items: center; gap: 5px;");
    item.appendChild(riskBadge(entry.level, "small"));
    item.appendChild(makeEl("span", riskRangeText(entry, RISK_LEVELS[i + 1])));
    legend.appendChild(item);
  });
  box.appendChild(legend);
  const s = result.input_summary;
  const measured = s.variables_measured.map((v) => VARIABLE_LABELS[v] || v).join(", ") || "none";
  setText("whatif-summary",
    `Measured: ${measured}. Not measured (encoded as in training): ${s.variables_not_measured.length} variables.` +
    (s.demographics_imputed.length ? ` Imputed: ${s.demographics_imputed.join(", ")}.` : ""));
}
