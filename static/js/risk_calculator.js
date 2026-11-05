/**
 * What-if Risk Calculator: sends manually entered vitals and labs to /api/risk-calculator and
 * shows the ensemble's tier and score, each model's vote, SHAP attributions and the K-Means
 * cohort map (charts from explain_charts.js). Nothing entered is stored.
 */

const CALC_PRESETS = {
  stable: { sex: "female", age: 45, heart_rate: 72, sbp: 122, dbp: 78, spo2: 98, respiratory_rate: 15,
            temperature: 36.7, glucose: 96, creatinine: 0.9, bun: 14, wbc: 6.5, hemoglobin: 13.8,
            platelets: 260, active_medications: 1 },
  watch: { sex: "male", age: 62, heart_rate: 98, sbp: 138, dbp: 88, spo2: 94, respiratory_rate: 20,
           temperature: 37.6, glucose: 240, creatinine: 1.3, bun: 24, wbc: 10.5, hemoglobin: 12.6,
           platelets: 230, active_medications: 3 },
  septic: { sex: "male", age: 71, heart_rate: 126, sbp: 84, dbp: 50, spo2: 88, respiratory_rate: 29,
            temperature: 39.3, glucose: 175, creatinine: 2.5, bun: 46, wbc: 19.5, hemoglobin: 10.4,
            platelets: 110, active_medications: 5 },
};

window.cohortCurrentLabel = "Your entry";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("calc-form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    calculateRisk();
  });
  form.addEventListener("reset", () => {
    hideError();
    setTimeout(clearInvalid, 0);
  });
  document.querySelectorAll(".calc-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      fillPreset(CALC_PRESETS[btn.dataset.preset]);
      calculateRisk();
    });
  });
});

function fillPreset(values) {
  clearInvalid();
  hideError();
  Object.entries(values).forEach(([key, value]) => {
    const el = document.getElementById(`calc-${key}`);
    if (el) el.value = value;
  });
}

function readForm() {
  const body = {};
  document.querySelectorAll("#calc-form [name]").forEach((el) => {
    const v = el.value.trim();
    if (v !== "") body[el.name] = el.name === "sex" ? v : Number(v);
  });
  return body;
}

function clearInvalid() {
  document.querySelectorAll("#calc-form .is-invalid").forEach((el) => {
    el.classList.remove("is-invalid");
    el.removeAttribute("aria-invalid");
  });
}

function showError(message, field) {
  const box = document.getElementById("calc-error");
  box.textContent = message;
  box.hidden = false;
  if (field) {
    const el = document.getElementById(`calc-${field}`);
    if (el) {
      el.classList.add("is-invalid");
      el.setAttribute("aria-invalid", "true");
      el.focus();
    }
  }
}

function hideError() {
  const box = document.getElementById("calc-error");
  box.hidden = true;
  box.textContent = "";
}

async function calculateRisk() {
  const button = document.getElementById("calc-submit");
  clearInvalid();
  hideError();
  button.disabled = true;
  button.textContent = "Calculating…";
  try {
    const res = await fetch("/api/risk-calculator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok || !payload.success) {
      const err = payload.error || {};
      showError(err.message || "The risk calculator is unavailable right now.", err.details && err.details.field);
      return;
    }
    renderResult(payload.data);
  } catch (err) {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    button.disabled = false;
    button.textContent = "Calculate risk";
  }
}

function renderResult(data) {
  const pred = data.prediction || {};
  document.getElementById("calc-empty").hidden = true;
  document.getElementById("calc-summary").hidden = false;
  document.getElementById("calc-explain").hidden = false;

  const tier = String(pred.risk_tier || "");
  document.getElementById("calc-score").textContent = Number(pred.risk_score).toFixed(2);
  const badge = document.getElementById("calc-tier");
  badge.className = `risk-badge risk-badge-${tier.toLowerCase()}`;
  badge.textContent = `${tier} risk tier`;
  document.getElementById("calc-confidence").textContent =
    `Consensus confidence ${Math.round((pred.confidence || 0) * 100)}% · 30-day readmission estimate ${Math.round((pred.readmission_30d_risk || 0) * 100)}%`;
  document.getElementById("calc-alert").hidden = !pred.early_warning_alert;

  const scores = (pred.model_consensus && pred.model_consensus.model_scores) || {};
  const names = { decision_tree: "Decision Tree", neural_network: "Neural Net", knn: "KNN", kmeans: "K-Means cluster" };
  document.getElementById("calc-votes").innerHTML = Object.entries(names).map(([key, label]) => {
    const v = scores[key];
    if (!v) return "";
    return `<div class="calc-vote"><span class="calc-vote-label">${escapeChartHtml(label)}</span>` +
      `<span class="risk-badge risk-badge-${escapeChartHtml(v.tier.toLowerCase())}">${escapeChartHtml(v.tier)}</span>` +
      `<span class="mono calc-vote-score">${Number(v.risk_score).toFixed(2)}</span></div>`;
  }).join("");

  const d = data.derived || {};
  document.getElementById("calc-derived").innerHTML =
    `<span>MAP <strong class="mono">${d.mean_arterial_pressure ?? "—"}</strong> mmHg</span>` +
    `<span>Shock index <strong class="mono">${d.shock_index ?? "—"}</strong></span>` +
    `<span>Pulse pressure <strong class="mono">${d.pulse_pressure ?? "—"}</strong> mmHg</span>`;
  document.getElementById("calc-assumptions").innerHTML =
    (data.assumptions || []).map((a) => `<li>${escapeChartHtml(a)}</li>`).join("");

  currentShapExplanations = pred.shap_explanations || null;
  renderShapChart();
  renderClusterMap(data.cohort_map && pred.kmeans_cluster ? { ...data.cohort_map, patient: pred.kmeans_cluster } : null);
  renderSimilar(pred.similar_patients || []);

  if (window.matchMedia("(max-width: 900px)").matches) {
    document.getElementById("calc-result").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderSimilar(similar) {
  const box = document.getElementById("calc-similar");
  if (!similar.length) {
    box.innerHTML = `<div class="calc-hint">No similar reference patients found.</div>`;
    return;
  }
  box.innerHTML = similar.map((p, i) =>
    `<div class="calc-similar-item">` +
      `<span class="calc-similar-rank">#${i + 1}</span>` +
      `<span class="mono">${escapeChartHtml(p.mrn)}</span>` +
      `<span class="risk-badge risk-badge-${escapeChartHtml(String(p.risk_tier).toLowerCase())}">${escapeChartHtml(p.risk_tier)}</span>` +
      `<span class="calc-similar-meta">similarity ${Number(p.similarity_score).toFixed(2)}` +
      `${p.readmitted_30d ? " · readmitted" : ""}${p.icu_transfer ? " · ICU transfer" : ""}</span>` +
    `</div>`).join("");
}
