/**
 * Shared explanation charts: the K-Means patient cohort map and the SHAP attribution chart.
 * Used by the physician dashboard (patient chart drawer) and the What-if Risk Calculator.
 * Pages provide the element ids (clusterChart, cluster-*, shapChart, shap-*), and optionally
 * window.onCohortPointClick(patientId) to make reference points clickable.
 */

let activeClusterChart = null;
let activeShapChart = null;
let currentShapExplanations = null;
let currentShapModel = "neural_network";

function escapeChartHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function isLightTheme() {
  return document.documentElement.dataset.theme === "light";
}

function gridLineColor() {
  return isLightTheme() ? "rgba(15, 23, 42, 0.07)" : "rgba(255, 255, 255, 0.05)";
}

function zeroLineColor() {
  return isLightTheme() ? "rgba(15, 23, 42, 0.35)" : "rgba(255, 255, 255, 0.35)";
}

// Redraw both charts when the theme switches so text and outlines stay readable
document.addEventListener("medihaven:themechange", () => {
  if (activeShapChart) renderShapChart();
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".shap-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      currentShapModel = tab.dataset.model;
      renderShapChart();
    });
  });
});

const TIER_COLORS = {
  Low: "#10b981",
  Medium: "#f59e0b",
  High: "#f97316",
  Critical: "#ef4444",
};
const CLUSTER_POINT_STYLES = ["circle", "triangle", "rect", "rectRot", "crossRot", "star"];

function tierColor(tier, alpha) {
  const hex = TIER_COLORS[tier] || "#94a3b8";
  if (alpha === undefined) return hex;
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function drawCanvasMessage(canvasId, message) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function fmt(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function fmtMeasure(value) {
  // Whole numbers stay whole; small ratios (e.g. shock index) keep two decimals
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) < 10 ? value.toFixed(2) : value.toFixed(1);
}

function shapFeatureLabel(f) {
  if (f.feature === "gender_encoded") return `Sex = ${f.value >= 0.5 ? "Male" : "Female"}`;
  return `${f.label} = ${fmtMeasure(f.value)}${f.unit ? " " + f.unit : ""}`;
}

let lastClusterMapData = null;

// Redraw the drawer charts when the theme switches, so outlines stay visible
document.addEventListener("medihaven:themechange", () => {
  if (activeClusterChart) renderClusterMap(lastClusterMapData);
});

function renderClusterMap(data) {
  lastClusterMapData = data;
  // K-Means cohort map: every reference patient in 2-D (first two principal components),
  // coloured by the tier its cluster was mapped to, with centroids and the current patient on top
  if (activeClusterChart) {
    activeClusterChart.destroy();
    activeClusterChart = null;
  }
  const assignment = document.getElementById("cluster-assignment");
  const stats = document.getElementById("cluster-stats");
  const legend = document.getElementById("cluster-legend");
  const profile = document.getElementById("cluster-profile");
  stats.innerHTML = "";
  legend.innerHTML = "";
  profile.innerHTML = "";

  if (!data || !Array.isArray(data.points) || !data.patient) {
    assignment.textContent = "Cluster: unavailable";
    assignment.style.color = "";
    drawCanvasMessage("clusterChart", "The K-Means cohort map is unavailable for this patient.");
    return;
  }

  const me = data.patient;
  const clusters = data.clusters || [];
  const byId = Object.fromEntries(clusters.map((c) => [c.cluster_id, c]));
  const myCluster = byId[me.cluster_id] || {};

  assignment.textContent = `Cluster C${me.cluster_id} → ${me.mapped_tier}`;
  assignment.style.color = tierColor(me.mapped_tier);
  assignment.style.borderColor = tierColor(me.mapped_tier, 0.45);

  const margin = me.distance_to_nearest_other - me.distance_to_centroid;
  const statItems = [
    ["Silhouette", fmt(data.silhouette_score, 2), "How well separated the clusters are (-1 to 1)"],
    ["Cluster size", `${myCluster.size ?? "—"} pts`, "Reference patients in this patient's cluster"],
    ["Tier purity", myCluster.purity !== undefined ? `${Math.round(myCluster.purity * 100)}%` : "—",
      "Share of the cluster's members whose tier matches the cluster's tier"],
    ["Margin", fmt(margin, 2), `Distance to C${me.nearest_other_cluster} (${me.nearest_other_tier}) minus distance to own centroid`],
  ];
  statItems.forEach(([label, value, title]) => {
    const div = document.createElement("div");
    div.className = "cluster-stat";
    div.title = title;
    div.innerHTML = `<div class="cluster-stat-label">${escapeChartHtml(label)}</div><div class="cluster-stat-value">${escapeChartHtml(value)}</div>`;
    stats.appendChild(div);
  });

  const datasets = clusters.map((c) => ({
    label: `C${c.cluster_id} · ${c.mapped_tier}`,
    data: data.points
      .filter((pt) => pt.cluster_id === c.cluster_id)
      .map((pt) => ({ x: pt.x, y: pt.y, patient_id: pt.patient_id, mrn: pt.mrn, risk_tier: pt.risk_tier })),
    pointStyle: CLUSTER_POINT_STYLES[c.cluster_id % CLUSTER_POINT_STYLES.length],
    backgroundColor: tierColor(c.mapped_tier, c.cluster_id === me.cluster_id ? 0.75 : 0.35),
    borderColor: tierColor(c.mapped_tier, 0.9),
    borderWidth: 1,
    pointRadius: c.cluster_id === me.cluster_id ? 5 : 4,
    pointHoverRadius: 7,
    order: 3,
  }));
  datasets.push({
    label: "Centroids",
    data: clusters.map((c) => ({ x: c.centroid_2d[0], y: c.centroid_2d[1], cluster_id: c.cluster_id, mapped_tier: c.mapped_tier, centroid: true })),
    pointStyle: "crossRot",
    pointRadius: 11,
    pointHoverRadius: 12,
    borderWidth: 3,
    borderColor: clusters.map((c) => tierColor(c.mapped_tier)),
    backgroundColor: "transparent",
    order: 2,
  });
  if (Array.isArray(me.coords_2d)) {
    datasets.push({
      label: "This patient",
      data: [{ x: me.coords_2d[0], y: me.coords_2d[1], current: true }],
      pointStyle: "star",
      pointRadius: 14,
      pointHoverRadius: 15,
      borderWidth: 2.5,
      borderColor: isLightTheme() ? "#0f172a" : "#ffffff",
      backgroundColor: tierColor(me.mapped_tier),
      order: 1,
    });
  }

  const variance = data.explained_variance || [];
  const axisTitle = (i) => `PC${i + 1}${variance[i] !== undefined ? ` (${Math.round(variance[i] * 100)}% of variance)` : ""}`;

  activeClusterChart = new Chart(document.getElementById("clusterChart").getContext("2d"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 450 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => {
              const d = item.raw;
              if (d.current) return `${window.cohortCurrentLabel || "This patient"} · cluster C${me.cluster_id} (${me.mapped_tier})`;
              if (d.centroid) return `Centroid of C${d.cluster_id} · mapped to ${d.mapped_tier}`;
              const hint = typeof window.onCohortPointClick === "function" ? " · click to open" : "";
              return `${d.mrn || "Patient " + d.patient_id} · labelled ${d.risk_tier}${hint}`;
            },
          },
        },
      },
      onHover: (evt, elements) => {
        const hit = elements.find((el) => {
          const raw = activeClusterChart.data.datasets[el.datasetIndex].data[el.index];
          return raw && raw.patient_id;
        });
        evt.native.target.style.cursor = hit && typeof window.onCohortPointClick === "function" ? "pointer" : "default";
      },
      onClick: (evt, elements) => {
        if (typeof window.onCohortPointClick !== "function") return;
        for (const el of elements) {
          const raw = activeClusterChart.data.datasets[el.datasetIndex].data[el.index];
          if (raw && raw.patient_id) {
            window.onCohortPointClick(raw.patient_id);
            return;
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: axisTitle(0), color: "#64748b", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 10 } },
          grid: { color: gridLineColor() },
        },
        y: {
          title: { display: true, text: axisTitle(1), color: "#64748b", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 10 } },
          grid: { color: gridLineColor() },
        },
      },
    },
  });

  clusters.forEach((c) => {
    const chip = document.createElement("span");
    chip.className = `cluster-chip${c.cluster_id === me.cluster_id ? " is-current" : ""}`;
    chip.style.color = c.cluster_id === me.cluster_id ? tierColor(c.mapped_tier) : "";
    const readmit = c.outcome_rates && typeof c.outcome_rates.readmitted_30d === "number"
      ? ` · ${Math.round(c.outcome_rates.readmitted_30d * 100)}% readmitted` : "";
    chip.title = Object.entries(c.tier_composition || {}).map(([t, n]) => `${t}: ${n}`).join(" · ");
    chip.innerHTML = `<span class="dot" style="background:${tierColor(c.mapped_tier)}"></span>` +
      `C${c.cluster_id} → ${escapeChartHtml(c.mapped_tier)} · ${c.size} pts · ${Math.round(c.purity * 100)}% pure${escapeChartHtml(readmit)}`;
    legend.appendChild(chip);
  });

  const rows = (me.profile_comparison || []).filter((r) => r.patient !== null || r.cluster_mean !== null);
  if (rows.length) {
    const body = rows.map((r) => {
      const diff = typeof r.patient === "number" && typeof r.cluster_mean === "number" ? r.patient - r.cluster_mean : null;
      const cls = diff === null || Math.abs(diff) < 1e-9 ? "" : diff > 0 ? "profile-diff-up" : "profile-diff-down";
      const sign = diff !== null && diff > 0 ? "+" : "";
      const unit = r.unit ? ` <span style="color: var(--text-muted);">${escapeChartHtml(r.unit)}</span>` : "";
      return `<tr><td>${escapeChartHtml(r.label)}${unit}</td><td class="num">${fmtMeasure(r.patient)}</td>` +
        `<td class="num">${fmtMeasure(r.cluster_mean)}</td><td class="num ${cls}">${diff === null ? "—" : sign + fmtMeasure(diff)}</td></tr>`;
    }).join("");
    profile.innerHTML = `<table class="profile-table"><thead><tr><th>Cluster C${me.cluster_id} profile</th>` +
      `<th class="num">Patient</th><th class="num">Cluster mean</th><th class="num">Diff.</th></tr></thead><tbody>${body}</tbody></table>`;
  }
}

function renderShapChart() {
  if (activeShapChart) {
    activeShapChart.destroy();
    activeShapChart = null;
  }
  document.querySelectorAll(".shap-tab").forEach((tab) => {
    const active = tab.dataset.model === currentShapModel;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });

  const subtitle = document.getElementById("shap-subtitle");
  const footer = document.getElementById("shap-footer");
  const exp = currentShapExplanations ? currentShapExplanations[currentShapModel] : null;
  if (!exp || !Array.isArray(exp.features) || exp.features.length === 0) {
    subtitle.textContent = "Which measurements moved this model's prediction, and by how much.";
    footer.textContent = "";
    drawCanvasMessage("shapChart", "SHAP values are unavailable for this prediction.");
    return;
  }

  subtitle.textContent = `${exp.model} · ${exp.method} · explaining the "${exp.target_tier}" tier`;
  const labels = exp.features.map(shapFeatureLabel);
  const values = exp.features.map((f) => f.shap_value);
  const toward = tierColor(exp.target_tier);

  activeShapChart = new Chart(document.getElementById("shapChart").getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "SHAP value",
        data: values,
        backgroundColor: values.map((v) => (v >= 0 ? tierColor(exp.target_tier, 0.75) : "rgba(56, 189, 248, 0.7)")),
        borderColor: values.map((v) => (v >= 0 ? toward : "#38bdf8")),
        borderWidth: 1,
        borderRadius: 3,
        barPercentage: 0.75,
        maxBarThickness: 26,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 450 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => `${item.raw >= 0 ? "+" : ""}${item.raw.toFixed(3)} to P(${exp.target_tier})`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: `Contribution to probability of "${exp.target_tier}"`, color: "#64748b", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 10 } },
          grid: { color: (c) => (c.tick && c.tick.value === 0 ? zeroLineColor() : gridLineColor()) },
        },
        y: {
          ticks: { color: isLightTheme() ? "#334155" : "#cbd5e1", font: { size: 10.5 } },
          grid: { display: false },
        },
      },
    },
  });

  const others = exp.n_nonzero_features - exp.features.length;
  const otherText = others > 0 ? ` The ${others} other features add <strong>${exp.other_features_total >= 0 ? "+" : ""}${exp.other_features_total.toFixed(3)}</strong>.` : "";
  const sparse = currentShapModel === "decision_tree"
    ? ` The tree only splits on ${exp.n_nonzero_features} feature${exp.n_nonzero_features === 1 ? "" : "s"} on this patient's path, so the rest contribute nothing.`
    : "";
  footer.innerHTML =
    `Average prediction <strong>${exp.base_value.toFixed(3)}</strong> + contributions = this patient's ` +
    `<strong>${exp.predicted_probability.toFixed(3)}</strong> probability of <strong>${escapeChartHtml(exp.target_tier)}</strong>. ` +
    `Coloured bars push toward ${escapeChartHtml(exp.target_tier)}, blue bars push away.${otherText}${sparse} ` +
    `These describe the model, not clinical causes.`;
}
