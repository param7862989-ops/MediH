/**
 * MediHaven Hospital Operations & Administration Controller
 * Manages executive KPI telemetry, high-risk inpatient surveillance,
 * patient-physician assignments/reassignments, and clinical escalation dispatches.
 */

let allPatients = [];
let allPhysicians = [];

document.addEventListener("DOMContentLoaded", () => {
  initAdminDashboard();
});

async function initAdminDashboard() {
  setupTabs();
  setupModals();
  setupSearch();
  initGeminiStudio();

  await Promise.all([
    fetchAdminSummary(),
    fetchAdminPatients(),
    fetchPhysiciansDirectory(),
    fetchActivityFeed(),
  ]);

  document.getElementById("btn-refresh-high-risk")?.addEventListener("click", fetchAdminPatients);
  document.getElementById("btn-refresh-activity")?.addEventListener("click", fetchActivityFeed);
}

// ==============================================================================
// 1. Tab Navigation
// ==============================================================================
let geminiStudioAutoLoaded = false;

function setupTabs() {
  const tabs = document.querySelectorAll(".admin-tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.dataset.tab;
      document.querySelectorAll(".admin-tab-content").forEach((c) => {
        c.style.display = c.id === targetId ? "block" : "none";
      });

      if (targetId === "tab-models") {
        document.querySelectorAll("#tab-models .model-stat-card").forEach((card) => {
          card.style.animation = "none";
          void card.offsetHeight;
          card.style.animation = "";
        });

        if (!geminiStudioAutoLoaded) {
          geminiStudioAutoLoaded = true;
          triggerGeminiSynthesis();
        }
      }
    });
  });
}

// ==============================================================================
// 2. Executive Operations KPI Summary
// ==============================================================================
async function fetchAdminSummary() {
  try {
    const res = await fetch("/api/dashboard/summary");
    const json = await res.json();
    if (!json.success || !json.data) return;

    const d = json.data;
    const dist = d.risk_distribution || {};

    document.getElementById("admin-total-patients").textContent = d.total_admitted || 100;
    document.getElementById("admin-critical-count").textContent = dist.Critical || 0;
    document.getElementById("admin-high-count").textContent = dist.High || 0;
    document.getElementById("admin-medium-count").textContent = dist.Medium || 0;
    document.getElementById("admin-low-count").textContent = dist.Low || 0;
    document.getElementById("admin-icu-count").textContent = `${d.icu_occupancy || 18} / 20`;

    const highRiskTotal = (dist.Critical || 0) + (dist.High || 0);
    const badgeEl = document.getElementById("tab-count-high-risk");
    if (badgeEl) badgeEl.textContent = highRiskTotal;
  } catch (err) {
    console.error("Failed to load admin summary:", err);
  }
}

// ==============================================================================
// 3. Patients Roster & High-Risk Surveillance
// ==============================================================================
async function fetchAdminPatients() {
  try {
    const res = await fetch("/api/patients");
    const json = await res.json();
    if (!json.success || !json.data) return;

    allPatients = json.data;
    renderHighRiskTable();
    renderAssignmentsTable();
  } catch (err) {
    console.error("Failed to load patients for admin:", err);
  }
}

function renderHighRiskTable() {
  const tbody = document.getElementById("high-risk-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const highRisk = allPatients.filter((p) => p.risk_tier === "Critical" || p.risk_tier === "High");

  if (highRisk.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align: center; padding: 32px; color: var(--text-muted);">
          No high-risk patients currently flagged across inpatient wards.
        </td>
      </tr>
    `;
    return;
  }

  highRisk.forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="risk-badge risk-badge-${p.risk_tier.toLowerCase()}">${escapeHtml(p.risk_tier)}</span></td>
      <td class="mono" style="font-weight: 700; color: #ffffff;">${p.risk_score.toFixed(2)}</td>
      <td>
        <div class="patient-name-cell">${escapeHtml(p.full_name)}</div>
        <div class="patient-mrn-cell">${escapeHtml(p.mrn)}</div>
      </td>
      <td>${p.age}y / ${escapeHtml(p.gender)}</td>
      <td>${escapeHtml(p.ward)} · <span class="mono">${escapeHtml(p.bed_number)}</span></td>
      <td>
        <span title="${escapeHtml(p.symptoms)}" style="color: #fbbf24; font-size: 11px; max-width: 180px; display: inline-block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(p.symptoms)}
        </span>
      </td>
      <td>
        <span style="color: #60a5fa; font-weight: 600; font-size: 12px;">${escapeHtml(p.assigned_physician || "Dr. Sarah Chen, MD")}</span>
      </td>
      <td style="text-align: right;">
        <div style="display: flex; gap: 6px; justify-content: flex-end; align-items: center;">
          <button class="btn btn-outline btn-sm btn-dispatch-row" data-id="${p.patient_id}" data-name="${escapeHtml(p.full_name)}" data-mrn="${escapeHtml(p.mrn)}" data-ward="${escapeHtml(p.ward)}" style="color: #f87171; border-color: rgba(239, 68, 68, 0.4);">
            ⚡ Dispatch Alert
          </button>
          <button class="btn btn-outline btn-sm btn-reassign-row" data-id="${p.patient_id}" data-name="${escapeHtml(p.full_name)}" data-mrn="${escapeHtml(p.mrn)}" data-ward="${escapeHtml(p.ward)}" data-current="${escapeHtml(p.assigned_physician || "")}">
            Reassign
          </button>
        </div>
      </td>
    `;

    tbody.appendChild(tr);
  });

  tbody.querySelectorAll(".btn-dispatch-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      openDispatchModal(btn.dataset.id, btn.dataset.name, btn.dataset.mrn, btn.dataset.ward);
    });
  });

  tbody.querySelectorAll(".btn-reassign-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      openReassignModal(btn.dataset.id, btn.dataset.name, btn.dataset.mrn, btn.dataset.ward, btn.dataset.current);
    });
  });
}

function renderAssignmentsTable() {
  const tbody = document.getElementById("assignments-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const query = document.getElementById("assignments-search-input")?.value.trim().toLowerCase() || "";

  const filtered = allPatients.filter((p) => {
    if (!query) return true;
    const matchName = p.full_name.toLowerCase().includes(query);
    const matchMrn = p.mrn.toLowerCase().includes(query);
    const matchDoc = (p.assigned_physician || "").toLowerCase().includes(query);
    return matchName || matchMrn || matchDoc;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 32px; color: var(--text-muted);">
          No patient records match the search query.
        </td>
      </tr>
    `;
    return;
  }

  filtered.forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(p.full_name)}</strong></td>
      <td class="mono">${escapeHtml(p.mrn)}</td>
      <td>${escapeHtml(p.ward)} · <span class="mono">${escapeHtml(p.bed_number)}</span></td>
      <td><span style="color: #60a5fa; font-weight: 600;">${escapeHtml(p.assigned_physician || "Dr. Sarah Chen, MD")}</span></td>
      <td><span class="risk-badge risk-badge-${p.risk_tier.toLowerCase()}">${escapeHtml(p.risk_tier)}</span></td>
      <td style="font-size: 12px; color: var(--text-secondary); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        ${escapeHtml(p.primary_diagnosis || "Inpatient Care")}
      </td>
      <td style="text-align: right;">
        <button class="btn btn-outline btn-sm btn-reassign-row" data-id="${p.patient_id}" data-name="${escapeHtml(p.full_name)}" data-mrn="${escapeHtml(p.mrn)}" data-ward="${escapeHtml(p.ward)}" data-current="${escapeHtml(p.assigned_physician || "")}">
          Reassign
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  tbody.querySelectorAll(".btn-reassign-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      openReassignModal(btn.dataset.id, btn.dataset.name, btn.dataset.mrn, btn.dataset.ward, btn.dataset.current);
    });
  });
}

function setupSearch() {
  const input = document.getElementById("assignments-search-input");
  if (input) {
    input.addEventListener("input", renderAssignmentsTable);
  }
}

// ==============================================================================
// 4. Physicians Directory & Caseload Grid
// ==============================================================================
async function fetchPhysiciansDirectory() {
  try {
    const res = await fetch("/api/patients/physicians");
    const json = await res.json();
    if (!json.success || !json.data) return;

    allPhysicians = json.data;
    renderPhysiciansCards();
  } catch (err) {
    console.error("Failed to load physicians directory:", err);
  }
}

function renderPhysiciansCards() {
  const grid = document.getElementById("physicians-cards-grid");
  if (!grid) return;
  grid.innerHTML = "";

  allPhysicians.forEach((doc) => {
    const card = document.createElement("div");
    card.className = "physician-card";
    card.innerHTML = `
      <div class="physician-card-header">
        <div>
          <div class="physician-name">${escapeHtml(doc.name)}</div>
          <div class="physician-dept">Department of Clinical Medicine</div>
        </div>
        <span class="badge badge-low">Active Rounds</span>
      </div>
      <div class="physician-stats-row">
        <div>
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Total Admitted</div>
          <div style="font-size: 18px; font-weight: 700; color: #ffffff;">${doc.active_patients}</div>
        </div>
        <div>
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">High Risk Watch</div>
          <div style="font-size: 18px; font-weight: 700; color: ${doc.high_risk_patients > 0 ? '#f87171' : '#34d399'};">
            ${doc.high_risk_patients}
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

// ==============================================================================
// 5. Activity Feed
// ==============================================================================
async function fetchActivityFeed() {
  try {
    const res = await fetch("/api/alerts/activity");
    const json = await res.json();
    if (!json.success || !json.data) return;

    const list = document.getElementById("activity-timeline-list");
    if (!list) return;
    list.innerHTML = "";

    json.data.forEach((ev) => {
      const item = document.createElement("div");
      const sev = ev.severity || "info";
      item.className = `activity-item severity-${sev}`;

      const dt = new Date(ev.timestamp);
      const timeStr = `${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}`;

      item.innerHTML = `
        <div style="font-size: 18px;">${sev === 'critical' ? '⚡' : sev === 'high' ? '⚠️' : 'ℹ️'}</div>
        <div style="flex: 1;">
          <div class="activity-title">${escapeHtml(ev.title)}</div>
          <div class="activity-desc">${escapeHtml(ev.description)}</div>
        </div>
        <div class="activity-time">${timeStr}</div>
      `;
      list.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load activity feed:", err);
  }
}

// ==============================================================================
// 6. Administrative Action Modals (Reassign & Dispatch)
// ==============================================================================
function setupModals() {
  // Reassign Modal
  const reassignModal = document.getElementById("modal-reassign-physician");
  const reassignClose = document.getElementById("modal-reassign-close");
  const reassignCancel = document.getElementById("reassign-btn-cancel");
  const reassignSubmit = document.getElementById("reassign-btn-submit");

  const closeReassign = () => reassignModal.classList.remove("active");
  reassignClose?.addEventListener("click", closeReassign);
  reassignCancel?.addEventListener("click", closeReassign);

  reassignSubmit?.addEventListener("click", async () => {
    const patientId = document.getElementById("reassign-patient-id").value;
    const newPhysician = document.getElementById("reassign-new-physician-select").value;
    const notes = document.getElementById("reassign-reason-notes").value.trim();

    reassignSubmit.disabled = true;
    reassignSubmit.textContent = "Updating Assignment...";

    try {
      const res = await fetch(`/api/patients/${patientId}/reassign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ assigned_physician: newPhysician, notes: notes }),
      });
      const data = await res.json();
      if (data.success) {
        closeReassign();
        await fetchAdminPatients();
        await fetchPhysiciansDirectory();
        await fetchActivityFeed();
      } else {
        alert(data.error?.message || "Failed to reassign patient.");
      }
    } catch (e) {
      console.error("Reassign request failed:", e);
    } finally {
      reassignSubmit.disabled = false;
      reassignSubmit.textContent = "Confirm Reassignment";
    }
  });

  // Dispatch Modal
  const dispatchModal = document.getElementById("modal-dispatch-alert");
  const dispatchClose = document.getElementById("modal-dispatch-close");
  const dispatchCancel = document.getElementById("dispatch-btn-cancel");
  const dispatchSubmit = document.getElementById("dispatch-btn-submit");

  const closeDispatch = () => dispatchModal.classList.remove("active");
  dispatchClose?.addEventListener("click", closeDispatch);
  dispatchCancel?.addEventListener("click", closeDispatch);

  dispatchSubmit?.addEventListener("click", async () => {
    const patientId = document.getElementById("dispatch-patient-id").value;
    const urgency = document.getElementById("dispatch-urgency-select").value;
    const notes = document.getElementById("dispatch-notes").value.trim();

    dispatchSubmit.disabled = true;
    dispatchSubmit.textContent = "Dispatching...";

    try {
      const res = await fetch("/api/alerts/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          urgency_level: urgency,
          notes: notes,
          dispatched_by: "Hospital Administrator",
        }),
      });
      const data = await res.json();
      if (data.success) {
        closeDispatch();
        await fetchAdminSummary();
        await fetchActivityFeed();
        alert(`✓ Escalation alert dispatched successfully for Patient #${patientId}.`);
      } else {
        alert(data.error?.message || "Failed to dispatch escalation alert.");
      }
    } catch (e) {
      console.error("Dispatch request failed:", e);
    } finally {
      dispatchSubmit.disabled = false;
      dispatchSubmit.textContent = "Dispatch Escalation to Attending Team →";
    }
  });
}

function openReassignModal(id, name, mrn, ward, currentPhysician) {
  document.getElementById("reassign-patient-id").value = id;
  document.getElementById("reassign-patient-name").textContent = name;
  document.getElementById("reassign-patient-meta").textContent = `MRN: ${mrn} · Ward: ${ward}`;
  document.getElementById("reassign-current-physician").textContent = `Current: ${currentPhysician || "Unassigned"}`;
  document.getElementById("reassign-reason-notes").value = "";
  document.getElementById("modal-reassign-physician").classList.add("active");
}

function openDispatchModal(id, name, mrn, ward) {
  document.getElementById("dispatch-patient-id").value = id;
  document.getElementById("dispatch-patient-name").textContent = name;
  document.getElementById("dispatch-patient-meta").textContent = `MRN: ${mrn} · Ward: ${ward}`;
  document.getElementById("dispatch-notes").value = "Rapid clinical deterioration suspected on administrative review. Immediate bedside evaluation recommended.";
  document.getElementById("modal-dispatch-alert").classList.add("active");
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

// ==============================================================================
// 7. Google Gemini Generative AI Studio
// ==============================================================================
function initGeminiStudio() {
  const btnRun = document.getElementById("btn-run-gemini-synth");
  const patientSelect = document.getElementById("gemini-patient-select");
  const modeSelect = document.getElementById("gemini-mode-select");
  const btnCopy = document.getElementById("btn-copy-gemini");

  if (btnRun) {
    btnRun.addEventListener("click", () => triggerGeminiSynthesis());
  }
  if (patientSelect) {
    patientSelect.addEventListener("change", () => triggerGeminiSynthesis());
  }
  if (modeSelect) {
    modeSelect.addEventListener("change", () => triggerGeminiSynthesis());
  }

  if (btnCopy) {
    btnCopy.addEventListener("click", () => {
      const text = document.getElementById("gemini-narrative-text")?.textContent?.trim() || "";
      if (text) {
        navigator.clipboard.writeText(text).then(() => {
          const original = btnCopy.textContent;
          btnCopy.textContent = "✓ Copied!";
          btnCopy.style.color = "#34d399";
          setTimeout(() => {
            btnCopy.textContent = original;
            btnCopy.style.color = "#94a3b8";
          }, 2000);
        });
      }
    });
  }
}

async function triggerGeminiSynthesis() {
  const patientSelect = document.getElementById("gemini-patient-select");
  const modeSelect = document.getElementById("gemini-mode-select");
  const narrativeEl = document.getElementById("gemini-narrative-text");
  const spinnerEl = document.getElementById("gemini-spinner");
  const modelBadge = document.getElementById("gemini-model-badge");
  const latencyBadge = document.getElementById("gemini-latency-badge");
  const btnRun = document.getElementById("btn-run-gemini-synth");

  const patientId = patientSelect ? patientSelect.value : 1;
  const mode = modeSelect ? modeSelect.value : "handover";

  if (spinnerEl) spinnerEl.style.display = "inline-block";
  if (btnRun) {
    btnRun.disabled = true;
    btnRun.style.opacity = "0.7";
  }
  if (narrativeEl) {
    narrativeEl.style.opacity = "0.45";
    narrativeEl.textContent = "✨ Google Gemini is synthesizing clinical biomarker trajectory, vitals, and decision tree rules...";
  }

  try {
    const res = await fetch("/api/admin/gemini/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, type: mode })
    });
    const json = await res.json();
    if (json.success && json.data) {
      const d = json.data;
      if (narrativeEl) {
        narrativeEl.style.opacity = "1";
        narrativeEl.textContent = d.narrative || "No narrative generated.";
      }
      if (modelBadge) {
        modelBadge.textContent = `Model: ${d.model || "gemini-3.5-flash-lite"}`;
      }
      if (latencyBadge) {
        latencyBadge.textContent = `⚡ ${d.latency_ms || 1800} ms (${d.is_live ? "Live Gemini API" : "Simulated Fallback"})`;
        latencyBadge.style.color = d.is_live ? "#34d399" : "#fbbf24";
      }
    } else {
      if (narrativeEl) {
        narrativeEl.style.opacity = "1";
        narrativeEl.textContent = json.error?.message || "Failed to generate clinical synthesis.";
      }
    }
  } catch (err) {
    console.error("Gemini synthesis error:", err);
    if (narrativeEl) {
      narrativeEl.style.opacity = "1";
      narrativeEl.textContent = "Clinical synthesis request error: " + err.message;
    }
  } finally {
    if (spinnerEl) spinnerEl.style.display = "none";
    if (btnRun) {
      btnRun.disabled = false;
      btnRun.style.opacity = "1";
    }
  }
}

