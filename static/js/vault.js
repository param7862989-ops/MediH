/**
 * MediHaven Patient Medical Vault Controller (Phase 7 Presentation Layer)
 * Manages categorized record viewing, dynamic QR token creation with selective scopes,
 * 1-click instant pass revocation, and immutable patient access audit logs.
 */

let currentPatientId = 1;
let currentCategoryFilter = "";
let allRecords = [];

document.addEventListener("DOMContentLoaded", () => {
  initVault();
});

async function initVault() {
  const patientSelect = document.getElementById("vault-patient-select");
  currentPatientId = parseInt(patientSelect.value, 10);

  patientSelect.addEventListener("change", async (e) => {
    currentPatientId = parseInt(e.target.value, 10);
    await refreshPatientVaultData();
  });

  setupCategoryFilterListeners();
  setupModalListeners();

  document.getElementById("btn-refresh-audit").addEventListener("click", async () => {
    await fetchAccessAuditLog();
  });

  await refreshPatientVaultData();
}

async function refreshPatientVaultData() {
  await Promise.all([
    fetchVaultRecords(),
    fetchActiveTokens(),
    fetchAccessAuditLog(),
  ]);
}

// ==============================================================================
// 1. Vault Records & Category Summary
// ==============================================================================
async function fetchVaultRecords() {
  try {
    const res = await fetch(`/api/vault/${currentPatientId}`);
    const json = await res.json();
    if (!json.success) return;

    allRecords = json.data.records || [];
    const summary = json.data.category_summary || {};

    // Update Category Summary Counts
    document.getElementById("count-cat-all").textContent = allRecords.length;
    document.getElementById("count-cat-allergy").textContent = summary.allergy || 0;
    document.getElementById("count-cat-medication").textContent = summary.medication || 0;
    document.getElementById("count-cat-lab").textContent = summary.lab_report || 0;
    document.getElementById("count-cat-surgery").textContent = summary.surgery || 0;
    document.getElementById("count-cat-condition").textContent = summary.chronic_condition || 0;
    document.getElementById("count-cat-immunization").textContent = summary.immunization || 0;
    document.getElementById("count-cat-notes").textContent = summary.clinical_note || 0;

    renderRecordsGrid();
  } catch (err) {
    console.error("Failed to load vault records:", err);
  }
}

function renderRecordsGrid() {
  const grid = document.getElementById("records-grid");
  grid.innerHTML = "";

  const filtered = currentCategoryFilter
    ? allRecords.filter((r) => r.category === currentCategoryFilter)
    : allRecords;

  document.getElementById("filtered-records-count").textContent = filtered.length;
  document.getElementById("current-category-label").textContent = currentCategoryFilter
    ? currentCategoryFilter.replace("_", " ").toUpperCase()
    : "ALL RECORDS";

  if (filtered.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-muted); background-color: var(--bg-surface); border-radius: var(--radius-md); border: 1px dashed var(--border-hairline);">
        No records found in this category. Click "+ Upload Record" to add clinical data.
      </div>
    `;
    return;
  }

  filtered.forEach((r) => {
    const card = document.createElement("div");
    card.className = "record-card";

    let jsonDisplay = "";
    if (r.structured_data && Object.keys(r.structured_data).length > 0) {
      jsonDisplay = `<pre class="record-json-view">${escapeHtml(JSON.stringify(r.structured_data, null, 2))}</pre>`;
    }

    card.innerHTML = `
      <div class="record-header">
        <span class="record-cat-pill">${escapeHtml(r.category.replace("_", " "))}</span>
        ${r.is_sensitive ? '<span title="Marked as Sensitive - Excluded from general sharing" style="font-size: 12px;">🔒 Sensitive</span>' : ''}
      </div>
      <div class="record-title">${escapeHtml(r.title)}</div>
      <div class="record-desc">${escapeHtml(r.description || "No specific details provided.")}</div>
      ${jsonDisplay}
      <div style="font-size: 11px; color: var(--text-muted); margin-top: auto; padding-top: 6px;">
        Vault ID #${r.vault_id} · Recorded: ${formatTimestamp(r.created_at)}
      </div>
    `;

    grid.appendChild(card);
  });
}

function setupCategoryFilterListeners() {
  document.querySelectorAll(".vault-cat-card").forEach((card) => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".vault-cat-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      currentCategoryFilter = card.dataset.category || "";
      renderRecordsGrid();
    });
  });
}

// ==============================================================================
// 2. Active Shared Passes Management (1-Click Revocation)
// ==============================================================================
async function fetchActiveTokens() {
  try {
    const res = await fetch(`/api/vault/${currentPatientId}/active-tokens`);
    const json = await res.json();
    const list = document.getElementById("active-tokens-list");
    list.innerHTML = "";

    const tokens = (json.success && json.data) ? json.data : [];
    document.getElementById("active-tokens-count-badge").textContent = `${tokens.length} Active Passes`;

    if (tokens.length === 0) {
      list.innerHTML = `
        <div style="color: var(--text-muted); font-size: 12px; padding: 12px 0;">
          No active sharing passes issued. Generate a pass above to share scoped health data with providers.
        </div>
      `;
      return;
    }

    tokens.forEach((t) => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.justifyContent = "space-between";
      row.style.alignItems = "center";
      row.style.padding = "10px 14px";
      row.style.backgroundColor = "var(--bg-surface-subtle)";
      row.style.border = "1px solid var(--border-hairline)";
      row.style.borderRadius = "var(--radius-sm)";
      row.style.flexWrap = "wrap";
      row.style.gap = "8px";

      const scopePills = (t.scope || [])
        .map((s) => `<span class="badge" style="background: rgba(67, 56, 202, 0.2); color: #a5b4fc; border: 1px solid rgba(67, 56, 202, 0.4);">${escapeHtml(s)}</span>`)
        .join(" ");

      row.innerHTML = `
        <div>
          <div style="font-weight: 600; color: #ffffff; font-size: 13px;">
            Pass #${t.token_id} · <span class="mono" style="color: var(--text-muted);">${escapeHtml(t.token_hash)}</span>
          </div>
          <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">
            Expires: <strong style="color: #ffffff;">${formatTimestamp(t.expires_at)}</strong> · Uses: <strong>${t.use_count} / ${t.max_uses ? t.max_uses : '∞'}</strong>
          </div>
          <div style="margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap;">${scopePills}</div>
        </div>
        <div>
          <button class="btn btn-danger btn-sm btn-revoke-pass" data-id="${t.token_id}">
            🛑 Revoke Access Now
          </button>
        </div>
      `;

      list.appendChild(row);
    });

    list.querySelectorAll(".btn-revoke-pass").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const tokenId = parseInt(e.target.dataset.id, 10);
        await revokeToken(tokenId);
      });
    });
  } catch (err) {
    console.error("Failed to load active tokens:", err);
  }
}

async function revokeToken(tokenId) {
  if (!confirm(`Are you sure you want to revoke Access Pass #${tokenId}? The consulting provider will be locked out immediately.`)) {
    return;
  }

  try {
    const res = await fetch(`/api/vault/revoke/${tokenId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: currentPatientId }),
    });
    const json = await res.json();
    if (json.success) {
      await fetchActiveTokens();
      await fetchAccessAuditLog();
    }
  } catch (err) {
    console.error("Failed to revoke pass:", err);
  }
}

// ==============================================================================
// 3. Immutable Access Audit Trail
// ==============================================================================
async function fetchAccessAuditLog() {
  try {
    const res = await fetch(`/api/vault/${currentPatientId}/access-log`);
    const json = await res.json();
    const tbody = document.getElementById("audit-tbody");
    tbody.innerHTML = "";

    const entries = (json.success && json.data) ? json.data : [];

    if (entries.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; padding: 24px; color: var(--text-muted);">
            No audit log entries recorded for this patient vault yet.
          </td>
        </tr>
      `;
      return;
    }

    entries.forEach((e) => {
      const tr = document.createElement("tr");

      let statusBadge = `<span class="risk-badge risk-badge-low">GRANTED</span>`;
      if (e.access_status === "REVOKED") {
        statusBadge = `<span class="risk-badge risk-badge-critical">REVOKED BLOCKED</span>`;
      } else if (e.access_status === "EXPIRED") {
        statusBadge = `<span class="risk-badge risk-badge-high">EXPIRED BLOCKED</span>`;
      } else if (e.access_status === "SCOPE_MISMATCH") {
        statusBadge = `<span class="risk-badge risk-badge-medium">SCOPE MISMATCH</span>`;
      }

      const scopes = (e.accessed_scope || []).join(", ") || "*";

      tr.innerHTML = `
        <td class="mono" style="font-size: 11px;">${formatTimestamp(e.accessed_at)}</td>
        <td>${statusBadge}</td>
        <td style="font-weight: 600; color: #ffffff;">${escapeHtml(e.accessed_by)}</td>
        <td style="font-size: 12px; color: #38bdf8;">${escapeHtml(scopes)}</td>
        <td class="mono" style="font-size: 11px; color: var(--text-muted);">${escapeHtml(e.ip_address)}</td>
        <td style="font-size: 11px; color: var(--text-muted); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(e.user_agent || "Scanner Web Client")}
        </td>
      `;

      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load audit trail:", err);
  }
}

// ==============================================================================
// 4. Modal Dialog Handlers (QR Share & Upload Record)
// ==============================================================================
function setupModalListeners() {
  // Share QR Modal
  const shareModal = document.getElementById("modal-qr-share");
  const openShareBtn = document.getElementById("btn-open-share-modal");
  const closeShareBtn = document.getElementById("modal-qr-close");
  const cancelShareBtn = document.getElementById("qr-btn-cancel");
  const generateShareBtn = document.getElementById("qr-btn-generate");
  const doneShareBtn = document.getElementById("qr-btn-done");

  const closeShare = () => {
    shareModal.classList.remove("active");
    document.getElementById("qr-form-step").style.display = "block";
    document.getElementById("qr-result-step").style.display = "none";
    generateShareBtn.style.display = "inline-flex";
    doneShareBtn.style.display = "none";
  };

  openShareBtn.addEventListener("click", () => {
    shareModal.classList.add("active");
  });

  closeShareBtn.addEventListener("click", closeShare);
  cancelShareBtn.addEventListener("click", closeShare);
  doneShareBtn.addEventListener("click", closeShare);

  // Wildcard toggle
  const wildcardCheck = document.getElementById("scope-wildcard");
  wildcardCheck.addEventListener("change", (e) => {
    document.querySelectorAll('input[name="scope"]').forEach((cb) => {
      cb.disabled = e.target.checked;
    });
  });

  // Generate QR Token
  generateShareBtn.addEventListener("click", async () => {
    let scopes = [];
    if (wildcardCheck.checked) {
      scopes = ["*"];
    } else {
      document.querySelectorAll('input[name="scope"]:checked').forEach((cb) => {
        scopes.push(cb.value);
      });
    }

    if (scopes.length === 0) {
      alert("Please select at least one clinical category to share.");
      return;
    }

    const ttl = parseInt(document.getElementById("qr-ttl-select").value, 10);
    const quotaVal = parseInt(document.getElementById("qr-max-uses-select").value, 10);
    const maxUses = quotaVal === 0 ? null : quotaVal;

    try {
      const res = await fetch(`/api/vault/${currentPatientId}/share`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scope: scopes,
          expires_in_minutes: ttl,
          max_uses: maxUses,
        }),
      });
      const json = await res.json();
      if (!json.success) {
        alert("Error generating token: " + (json.error ? json.error.message : "Unknown error"));
        return;
      }

      const tokenData = json.data;

      // Switch to result step
      document.getElementById("qr-form-step").style.display = "none";
      document.getElementById("qr-result-step").style.display = "block";
      generateShareBtn.style.display = "none";
      doneShareBtn.style.display = "inline-flex";

      // Render QR image
      const qrContainer = document.getElementById("qr-image-container");
      qrContainer.innerHTML = `<img src="${tokenData.qr_code_base64 || tokenData.qr_image_url}" alt="Cryptographic QR Access Pass" />`;

      const patientName = document.querySelector("#vault-patient-select option:checked").textContent;
      document.getElementById("qr-card-patient-name").textContent = patientName;
      document.getElementById("qr-card-exp").textContent = `Expires: ${formatTimestamp(tokenData.expires_at)}`;

      // Scopes list
      const scopesContainer = document.getElementById("qr-card-scopes");
      scopesContainer.innerHTML = (tokenData.scope || [])
        .map((s) => `<span class="badge badge-low" style="font-size: 10px;">${escapeHtml(s)}</span>`)
        .join("");

      document.getElementById("qr-token-text").textContent = tokenData.token || tokenData.qr_payload;

      await fetchActiveTokens();
    } catch (err) {
      console.error("Failed to generate pass:", err);
    }
  });

  // Upload Record Modal
  const uploadModal = document.getElementById("modal-upload-record");
  const openUploadBtn = document.getElementById("btn-open-upload-modal");
  const closeUploadBtn = document.getElementById("modal-upload-close");
  const cancelUploadBtn = document.getElementById("upload-btn-cancel");
  const submitUploadBtn = document.getElementById("upload-btn-submit");

  const closeUpload = () => uploadModal.classList.remove("active");
  openUploadBtn.addEventListener("click", () => uploadModal.classList.add("active"));
  closeUploadBtn.addEventListener("click", closeUpload);
  cancelUploadBtn.addEventListener("click", closeUpload);

  submitUploadBtn.addEventListener("click", async () => {
    const category = document.getElementById("record-category").value;
    const title = document.getElementById("record-title").value.trim();
    const desc = document.getElementById("record-description").value.trim();
    const isSensitive = document.getElementById("record-sensitive").checked;
    const jsonRaw = document.getElementById("record-json").value.trim();

    if (!title) {
      alert("Please provide a title for this medical record.");
      return;
    }

    let structuredData = null;
    if (jsonRaw) {
      try {
        structuredData = JSON.parse(jsonRaw);
      } catch (e) {
        alert("Invalid JSON format in structured data field.");
        return;
      }
    }

    try {
      const res = await fetch("/api/vault/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: currentPatientId,
          category: category,
          title: title,
          description: desc,
          structured_data: structuredData,
          is_sensitive: isSensitive,
        }),
      });
      const json = await res.json();
      if (json.success) {
        closeUpload();
        // Clear fields
        document.getElementById("record-title").value = "";
        document.getElementById("record-description").value = "";
        document.getElementById("record-json").value = "";
        document.getElementById("record-sensitive").checked = false;
        await fetchVaultRecords();
      } else {
        alert("Upload error: " + (json.error ? json.error.message : "Failed to add record"));
      }
    } catch (err) {
      console.error("Upload failed:", err);
    }
  });
}

function formatTimestamp(isoStr) {
  if (!isoStr) return "--";
  const d = new Date(isoStr);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
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
