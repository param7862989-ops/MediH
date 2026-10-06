/**
 * MediHaven Physician Clinical Dashboard Controller (Phase 7 Presentation Layer)
 * Manages live telemetry polling, filterable triage rosters, Chart.js trends,
 * Decision Tree explainability visualizers, and early warning alert handling.
 */

let allPatients = [];
let activeVitalsChart = null;
let currentSelectedPatientId = null;

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

async function initDashboard() {
  await Promise.all([
    fetchDashboardSummary(),
    fetchDeteriorationAlerts(),
    fetchPatientRoster(),
  ]);

  setupFilterEventListeners();
  setupDrawerEventListeners();
  setupAlertModalEventListeners();

  // Polling every 10 seconds for live triage updates
  setInterval(async () => {
    await fetchDashboardSummary();
    await fetchDeteriorationAlerts();
  }, 10000);
}

// ==============================================================================
// 1. Dashboard Cohort Summary Telemetry
// ==============================================================================
async function fetchDashboardSummary() {
  try {
    const res = await fetch("/api/dashboard/summary");
    const json = await res.json();
    if (!json.success) return;

    const d = json.data;
    document.getElementById("kpi-total-admitted").textContent = d.total_admitted;
    document.getElementById("kpi-critical-count").textContent = d.risk_distribution.Critical || 0;
    document.getElementById("kpi-high-count").textContent = d.risk_distribution.High || 0;
    document.getElementById("kpi-active-alerts").textContent = d.active_deterioration_alerts || 0;
    document.getElementById("kpi-icu-occupancy").textContent = d.icu_occupancy || 0;
    document.getElementById("kpi-readmit-risk").textContent = Math.round(d.average_30d_readmission_risk * 100) + "%";
  } catch (err) {
    console.error("Failed to load dashboard summary:", err);
  }
}

// ==============================================================================
// 2. Early Warning Deterioration Alerts
// ==============================================================================
async function fetchDeteriorationAlerts() {
  try {
    const res = await fetch("/api/alerts");
    const json = await res.json();
    const container = document.getElementById("alerts-marquee-container");
    container.innerHTML = "";

    if (!json.success || !json.data || json.data.length === 0) {
      container.style.display = "none";
      return;
    }

    container.style.display = "block";
    json.data.slice(0, 3).forEach((alert) => {
      const banner = document.createElement("div");
      banner.className = "alert-banner";

      const triggerChips = alert.physiological_triggers
        .map((t) => `<span class="trigger-chip">${escapeHtml(t)}</span>`)
        .join("");

      banner.innerHTML = `
        <div class="alert-banner-content">
          <span class="alert-urgency-pill">⚠️ ${escapeHtml(alert.urgency_level)}</span>
          <div>
            <div class="alert-patient-headline">
              ${escapeHtml(alert.full_name)} (${escapeHtml(alert.mrn)}) · ${escapeHtml(alert.ward)} [Bed ${escapeHtml(alert.bed_number)}]
            </div>
            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">
              Predicted Window: <strong style="color: #ffffff;">${escapeHtml(alert.early_warning_window)}</strong> · Risk Score: <strong>${alert.risk_score}</strong>
            </div>
          </div>
          <div class="alert-triggers-list">${triggerChips}</div>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-outline btn-sm btn-open-chart" data-id="${alert.patient_id}">View Chart</button>
          <button class="btn btn-primary btn-sm btn-ack-alert" data-id="${alert.patient_id}" data-name="${escapeHtml(alert.full_name)}">Acknowledge</button>
        </div>
      `;

      container.appendChild(banner);
    });

    // Attach click events
    container.querySelectorAll(".btn-open-chart").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        openPatientChart(parseInt(e.target.dataset.id, 10));
      });
    });

    container.querySelectorAll(".btn-ack-alert").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        openAckModal(parseInt(e.target.dataset.id, 10), e.target.dataset.name);
      });
    });
  } catch (err) {
    console.error("Failed to load alerts feed:", err);
  }
}

// ==============================================================================
// 3. Patient Triage Roster & Filters
// ==============================================================================
async function fetchPatientRoster() {
  try {
    const res = await fetch("/api/patients");
    const json = await res.json();
    if (!json.success) return;

    allPatients = json.data;
    updateCohortCounts();
    renderPatientTable();
  } catch (err) {
    console.error("Failed to load patient roster:", err);
  }
}

function updateCohortCounts() {
  document.getElementById("count-all").textContent = allPatients.length;
  document.getElementById("count-critical").textContent = allPatients.filter((p) => p.risk_tier === "Critical").length;
  document.getElementById("count-high").textContent = allPatients.filter((p) => p.risk_tier === "High").length;
  document.getElementById("count-medium").textContent = allPatients.filter((p) => p.risk_tier === "Medium").length;
  document.getElementById("count-low").textContent = allPatients.filter((p) => p.risk_tier === "Low").length;
}

function renderPatientTable() {
  const tbody = document.getElementById("triage-tbody");
  tbody.innerHTML = "";

  const activeFilterBtn = document.querySelector(".filter-pill.active");
  const selectedTier = activeFilterBtn ? activeFilterBtn.dataset.filter : "";
  const selectedWard = document.getElementById("ward-filter-select").value.toLowerCase();
  const searchKeyword = document.getElementById("patient-search-input").value.trim().toLowerCase();

  const filtered = allPatients.filter((p) => {
    if (selectedTier && p.risk_tier !== selectedTier) return false;
    if (selectedWard && !p.ward.toLowerCase().includes(selectedWard)) return false;
    if (searchKeyword) {
      const matchName = p.full_name.toLowerCase().includes(searchKeyword);
      const matchMrn = p.mrn.toLowerCase().includes(searchKeyword);
      if (!matchName && !matchMrn) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" style="text-align: center; padding: 40px; color: var(--text-muted);">
          No admitted patients match current filter parameters.
        </td>
      </tr>
    `;
    return;
  }

  filtered.forEach((p) => {
    const tr = document.createElement("tr");
    tr.dataset.patientId = p.patient_id;

    const vs = p.vitals_snapshot || {};
    const hr = vs.heart_rate ? Math.round(vs.heart_rate) : "--";
    const bp = vs.sbp && vs.dbp ? `${Math.round(vs.sbp)}/${Math.round(vs.dbp)}` : "--/--";
    const spo2 = vs.spo2 ? `${vs.spo2.toFixed(1)}%` : "--%";

    const si = vs.shock_index || 0.6;
    const siClass = si >= 0.9 ? "hemo-warn" : "hemo-normal";

    const mapVal = vs.mean_arterial_pressure || 85;
    const mapClass = mapVal < 65 ? "hemo-warn" : "hemo-normal";

    tr.innerHTML = `
      <td><span class="risk-badge risk-badge-${p.risk_tier.toLowerCase()}">${escapeHtml(p.risk_tier)}</span></td>
      <td class="mono" style="font-weight: 700; color: #ffffff;">${p.risk_score.toFixed(2)}</td>
      <td>
        <div class="patient-name-cell">${escapeHtml(p.full_name)}</div>
        <div class="patient-mrn-cell">${escapeHtml(p.mrn)}</div>
      </td>
      <td>${p.age}y / ${escapeHtml(p.gender)}</td>
      <td>${escapeHtml(p.ward)} · <span class="mono">${escapeHtml(p.bed_number)}</span></td>
      <td class="mono">${hr} bpm · ${bp} · ${spo2}</td>
      <td><span class="hemo-pill ${siClass}">${si.toFixed(2)}</span></td>
      <td><span class="hemo-pill ${mapClass}">${Math.round(mapVal)} mmHg</span></td>
      <td style="font-size: 11px; color: var(--text-secondary); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        ${escapeHtml(p.recommended_protocol)}
      </td>
      <td style="text-align: right;">
        <button class="btn btn-outline btn-sm btn-chart-row" data-id="${p.patient_id}">Open Chart →</button>
      </td>
    `;

    tr.addEventListener("click", (e) => {
      openPatientChart(p.patient_id);
    });

    tbody.appendChild(tr);
  });
}

function setupFilterEventListeners() {
  document.querySelectorAll(".filter-pill").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".filter-pill").forEach((b) => b.classList.remove("active"));
      e.target.classList.add("active");
      renderPatientTable();
    });
  });

  document.getElementById("ward-filter-select").addEventListener("change", () => {
    renderPatientTable();
  });

  document.getElementById("patient-search-input").addEventListener("input", () => {
    renderPatientTable();
  });

  document.getElementById("btn-refresh-triage").addEventListener("click", async () => {
    await fetchPatientRoster();
    await fetchDashboardSummary();
    await fetchDeteriorationAlerts();
  });
}

// ==============================================================================
// 4. Slide-Over Clinical Chart Drawer & Live AI
// ==============================================================================
async function openPatientChart(patientId) {
  currentSelectedPatientId = patientId;
  const drawer = document.getElementById("drawer-backdrop");
  drawer.classList.add("active");
  drawer.setAttribute("aria-hidden", "false");

  // Reset Drawer fields to loading state
  document.getElementById("drawer-patient-name").textContent = "Loading Patient Chart...";
  document.getElementById("drawer-patient-meta").textContent = "Patient #" + patientId;
  document.getElementById("dt-rules-list").innerHTML = "<div style='color: var(--text-muted); font-size: 12px;'>Evaluating physiological rule paths...</div>";
  document.getElementById("knn-precedents-list").innerHTML = "<div style='color: var(--text-muted); font-size: 12px;'>Retrieving historical cases...</div>";

  try {
    const [chartRes, predRes, vitalsRes, simRes] = await Promise.all([
      fetch(`/api/patients/${patientId}`).then((r) => r.json()),
      fetch(`/api/patients/${patientId}/predict`).then((r) => r.json()),
      fetch(`/api/patients/${patientId}/vitals?hours=72`).then((r) => r.json()),
      fetch(`/api/patients/${patientId}/similar?k=5`).then((r) => r.json()),
    ]);

    if (!chartRes.success) return;
    const patient = chartRes.data;
    const pred = predRes.data || {};
    const vitalsSeries = vitalsRes.data || [];
    const simData = simRes.data || {};

    // 1. Populate Demographic Header
    document.getElementById("drawer-patient-name").textContent = patient.full_name;
    document.getElementById("drawer-patient-meta").textContent = `MRN: ${patient.mrn} · ${patient.age}y ${patient.gender} · Ward: ${patient.ward} [Bed ${patient.bed_number}]`;

    const badge = document.getElementById("drawer-risk-badge");
    badge.className = `risk-badge risk-badge-${(pred.risk_tier || "low").toLowerCase()}`;
    badge.textContent = `${pred.risk_tier || "Low"} Risk Tier`;

    // 2. Multi-Model Consensus Metrics
    document.getElementById("drawer-score-val").textContent = (pred.risk_score || 0.1).toFixed(2);
    document.getElementById("drawer-confidence-val").textContent = `Confidence: ${Math.round((pred.confidence || 0.85) * 100)}%`;
    document.getElementById("drawer-readmit-val").textContent = `${Math.round((pred.readmission_30d_risk || 0.15) * 100)}%`;

    const consensus = pred.model_consensus || {};
    document.getElementById("vote-kmeans").textContent = consensus.kmeans_tier || "Low";
    document.getElementById("vote-dt").textContent = consensus.decision_tree_tier || "Low";
    document.getElementById("vote-nn").textContent = consensus.neural_network_tier || "Low";

    // 3. Render Longitudinal Vitals Chart (Chart.js)
    renderVitalsChart(vitalsSeries);

    // 4. Render Decision Tree Rule Breadcrumb Path
    renderDecisionTreeRules(pred.decision_tree_explanation || []);

    // 5. Render KNN Case Precedents
    renderKnnPrecedents(simData);
  } catch (err) {
    console.error("Failed to load full clinical chart:", err);
  }
}

function renderVitalsChart(series) {
  const ctx = document.getElementById("vitalsChart").getContext("2d");

  if (activeVitalsChart) {
    activeVitalsChart.destroy();
  }

  if (!series || series.length === 0) {
    return;
  }

  const labels = series.map((s) => {
    const d = new Date(s.recorded_at);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  });

  const hrData = series.map((s) => s.heart_rate);
  const sbpData = series.map((s) => s.blood_pressure_sys);
  const dbpData = series.map((s) => s.blood_pressure_dia);
  const spo2Data = series.map((s) => s.oxygen_saturation);

  activeVitalsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Heart Rate (bpm)",
          data: hrData,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          yAxisID: "y",
        },
        {
          label: "Systolic BP (mmHg)",
          data: sbpData,
          borderColor: "#ef4444",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          yAxisID: "y",
        },
        {
          label: "Diastolic BP (mmHg)",
          data: dbpData,
          borderColor: "#f472b6",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 1,
          yAxisID: "y",
        },
        {
          label: "SpO2 (%)",
          data: spo2Data,
          borderColor: "#0284c7",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 10 }, maxRotation: 0 },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        y: {
          type: "linear",
          position: "left",
          min: 40,
          max: 200,
          ticks: { color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          title: { display: true, text: "BPM / mmHg", color: "#64748b", font: { size: 10 } },
        },
        y1: {
          type: "linear",
          position: "right",
          min: 80,
          max: 100,
          ticks: { color: "#38bdf8", font: { size: 10 } },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "SpO2 %", color: "#38bdf8", font: { size: 10 } },
        },
      },
    },
  });
}

function renderDecisionTreeRules(rules) {
  const container = document.getElementById("dt-rules-list");
  container.innerHTML = "";

  if (!rules || rules.length === 0) {
    container.innerHTML = "<div style='color: var(--text-muted); font-size: 12px;'>Standard baseline clinical classification path traversed.</div>";
    return;
  }

  rules.forEach((step, idx) => {
    const isLeaf = step.includes("Outcome:");
    const row = document.createElement("div");
    row.className = "dt-step";
    row.innerHTML = `
      <div class="dt-step-num" style="${isLeaf ? 'background-color: var(--color-brand); color: #ffffff;' : ''}">${idx + 1}</div>
      <div class="dt-step-rule" style="${isLeaf ? 'font-weight: 700; color: #38bdf8;' : ''}">${escapeHtml(step)}</div>
    `;
    container.appendChild(row);
  });
}

function renderKnnPrecedents(simData) {
  const container = document.getElementById("knn-precedents-list");
  container.innerHTML = "";

  const protoBadge = document.getElementById("drawer-rec-protocol");
  protoBadge.textContent = `${simData.recommended_protocol || "Protocol S"} (${Math.round((simData.protocol_success_rate || 0.85) * 100)}% Success)`;

  const matches = simData.similar_patients || [];
  if (matches.length === 0) {
    container.innerHTML = "<div style='color: var(--text-muted); font-size: 12px;'>No close historical precedents located.</div>";
    return;
  }

  matches.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "knn-precedent-card";
    const readmitText = p.readmitted_30d ? "Readmitted 30d" : "Discharged Stable";
    const readmitColor = p.readmitted_30d ? "var(--risk-critical)" : "var(--risk-low)";

    row.innerHTML = `
      <div>
        <strong style="color: #ffffff;">Match #${idx + 1}: ${escapeHtml(p.mrn)}</strong>
        <div style="font-size: 11px; color: var(--text-muted);">
          Euclidean Distance: <span class="mono">${p.distance.toFixed(3)}</span> · Tier: ${escapeHtml(p.risk_tier)}
        </div>
      </div>
      <div style="text-align: right;">
        <span class="knn-proto-badge">${escapeHtml(p.protocol)}</span>
        <div style="font-size: 11px; font-weight: 600; color: ${readmitColor}; margin-top: 2px;">
          ${readmitText}
        </div>
      </div>
    `;
    container.appendChild(row);
  });
}

function setupDrawerEventListeners() {
  const backdrop = document.getElementById("drawer-backdrop");
  const closeBtn = document.getElementById("drawer-close-btn");

  closeBtn.addEventListener("click", () => {
    backdrop.classList.remove("active");
    backdrop.setAttribute("aria-hidden", "true");
  });

  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) {
      backdrop.classList.remove("active");
      backdrop.setAttribute("aria-hidden", "true");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && backdrop.classList.contains("active")) {
      backdrop.classList.remove("active");
      backdrop.setAttribute("aria-hidden", "true");
    }
  });
}

// ==============================================================================
// 5. Alert Acknowledgement Modal Handling
// ==============================================================================
function openAckModal(patientId, patientName) {
  document.getElementById("ack-patient-id").value = patientId;
  document.getElementById("ack-clinical-notes").value = `Treating physician bedside review conducted for ${patientName}. Initiating proactive protocol stabilization.`;
  document.getElementById("ack-modal-overlay").classList.add("active");
}

function setupAlertModalEventListeners() {
  const modal = document.getElementById("ack-modal-overlay");
  const closeBtn = document.getElementById("ack-modal-close");
  const cancelBtn = document.getElementById("ack-btn-cancel");
  const submitBtn = document.getElementById("ack-btn-submit");

  const closeModal = () => modal.classList.remove("active");
  closeBtn.addEventListener("click", closeModal);
  cancelBtn.addEventListener("click", closeModal);

  submitBtn.addEventListener("click", async () => {
    const patientId = document.getElementById("ack-patient-id").value;
    const doctorName = document.getElementById("ack-doctor-name").value.trim();
    const notes = document.getElementById("ack-clinical-notes").value.trim();

    try {
      const res = await fetch(`/api/alerts/${patientId}/acknowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doctor_name: doctorName, notes: notes }),
      });
      const json = await res.json();
      if (json.success) {
        closeModal();
        await fetchDeteriorationAlerts();
        await fetchDashboardSummary();
      }
    } catch (err) {
      console.error("Failed to acknowledge alert:", err);
    }
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
