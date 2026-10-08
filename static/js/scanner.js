/**
 * MediHaven Provider Optical QR Scanner Controller (Phase 7 Presentation Layer)
 * Manages live camera streaming, image QR uploads, direct JWT token decryption,
 * cryptographic signature validation, and scoped read-only medical record display.
 */

let videoStream = null;
let scanInterval = null;

document.addEventListener("DOMContentLoaded", () => {
  initScanner();
});

function initScanner() {
  const toggleCamBtn = document.getElementById("btn-toggle-camera");
  const fileInput = document.getElementById("qr-file-input");
  const submitTokenBtn = document.getElementById("btn-submit-token");

  toggleCamBtn.addEventListener("click", toggleCamera);

  // File upload change
  fileInput.addEventListener("change", handleFileUpload);

  // Submit token string
  submitTokenBtn.addEventListener("click", () => {
    const token = document.getElementById("qr-token-input").value.trim();
    if (!token) {
      alert("Please paste a cryptographic QR token payload.");
      return;
    }
    submitPayloadForVerification(token);
  });

  // Quick 1-Click Simulation Buttons
  const p1Btn = document.getElementById("btn-quick-scan-p1");
  if (p1Btn) {
    p1Btn.addEventListener("click", () => {
      loadDemoTokenForPatient(1, ["allergy", "medication"], "Dr. Sarah Chen, MD (Emergency Triage)");
    });
  }

  const p2Btn = document.getElementById("btn-quick-scan-p2");
  if (p2Btn) {
    p2Btn.addEventListener("click", () => {
      loadDemoTokenForPatient(2, ["*"], "Dr. Marcus Vance, MD (Critical Care)");
    });
  }
}

async function loadDemoTokenForPatient(patientId, scopes, clinician) {
  try {
    let res = await fetch(`/api/vault/${patientId}/share`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        scope: scopes,
        expires_in_minutes: 60,
        ttl_minutes: 60,
        max_uses: 10,
        created_by: "Bedside Optical Scanner",
      }),
    });
    let json = await res.json();
    if (!json.success) {
      res = await fetch("/api/vault/share", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          scope: scopes,
          expires_in_minutes: 60,
          ttl_minutes: 60,
          max_uses: 10,
        }),
      });
      json = await res.json();
    }

    if (json.success && json.data) {
      const token = json.data.token || json.data.qr_payload;
      document.getElementById("qr-token-input").value = token;
      if (clinician) document.getElementById("scanner-clinician-name").value = clinician;
      await submitPayloadForVerification(token);
    } else {
      alert("Failed to generate demo pass: " + (json.error ? json.error.message : "Server error"));
    }
  } catch (err) {
    console.error("Demo pass error:", err);
    alert("Error generating demo pass: " + err.message);
  }
}

// ==============================================================================
// 1. Camera Viewfinder Management
// ==============================================================================
async function toggleCamera() {
  const video = document.getElementById("scanner-video");
  const reticle = document.getElementById("reticle-box");
  const placeholder = document.getElementById("camera-placeholder");
  const btn = document.getElementById("btn-toggle-camera");

  if (videoStream) {
    // Stop camera
    stopCamera();
    btn.textContent = "📷 Start Camera";
    reticle.style.display = "none";
    placeholder.style.display = "block";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
    });

    videoStream = stream;
    video.srcObject = stream;
    await video.play();

    btn.textContent = "⏹ Stop Camera";
    reticle.style.display = "block";
    placeholder.style.display = "none";

    // Begin optical scanning capture loop (every 800ms)
    scanInterval = setInterval(captureAndScanFrame, 800);
  } catch (err) {
    console.warn("Camera access unavailable or denied:", err);
    alert("Camera access was not granted or is unavailable on this device. You can test optical QR verification using the file upload or text token option below.");
  }
}

function stopCamera() {
  if (scanInterval) {
    clearInterval(scanInterval);
    scanInterval = null;
  }
  if (videoStream) {
    videoStream.getTracks().forEach((track) => track.stop());
    videoStream = null;
  }
  const video = document.getElementById("scanner-video");
  video.srcObject = null;
}

function captureAndScanFrame() {
  const video = document.getElementById("scanner-video");
  const canvas = document.getElementById("qr-canvas");
  if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/png");
  // Submit frame to backend for pyzbar optical decode & verification
  submitPayloadForVerification(dataUrl, true);
}

// ==============================================================================
// 2. Optical File Upload Handling
// ==============================================================================
function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const base64Data = event.target.result;
    submitPayloadForVerification(base64Data);
  };
  reader.readAsDataURL(file);
}

// ==============================================================================
// 3. Verification & Access Submission
// ==============================================================================
async function submitPayloadForVerification(payload, isSilentScan = false) {
  const clinicianName = document.getElementById("scanner-clinician-name").value.trim() || "Consulting Clinician";

  try {
    const res = await fetch("/api/vault/access", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: payload,
        accessed_by: clinicianName,
      }),
    });

    const json = await res.json();

    if (!json.success) {
      if (!isSilentScan) {
        showErrorState(json.error ? json.error.message : "Access Denied: Invalid or Expired Pass");
      }
      return;
    }

    // Stop camera once verified
    if (videoStream) {
      stopCamera();
      document.getElementById("btn-toggle-camera").textContent = "📷 Start Camera";
      document.getElementById("reticle-box").style.display = "none";
      document.getElementById("camera-placeholder").style.display = "block";
    }

    renderVerificationResult(json.data);
  } catch (err) {
    if (!isSilentScan) {
      console.error("Verification error:", err);
      showErrorState("Network or server connection error during cryptographic verification.");
    }
  }
}

// ==============================================================================
// 4. Verification Result & Scoped Records Rendering
// ==============================================================================
function renderVerificationResult(data) {
  document.getElementById("idle-card").style.display = "none";
  document.getElementById("error-card").style.display = "none";

  // Show Badges
  const badgeCard = document.getElementById("verified-badge-card");
  badgeCard.style.display = "block";

  document.getElementById("verified-patient-name").textContent = data.patient_name || `Patient #${data.patient_id}`;
  document.getElementById("verified-patient-meta").textContent = `MRN: ${data.mrn || '--'} · Age: ${data.patient_age || '--'} · Sex: ${data.patient_gender || '--'}`;

  const scopesContainer = document.getElementById("verified-scopes-list");
  scopesContainer.innerHTML = (data.authorized_scope || [])
    .map((s) => `<span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4);">${escapeHtml(s.toUpperCase())}</span>`)
    .join("");

  document.getElementById("leakage-notice-card").style.display = "block";

  // Show Access Logged Confirmation
  const accessLoggedCard = document.getElementById("access-logged-notice-card");
  if (accessLoggedCard) {
    accessLoggedCard.style.display = "block";
    const accessText = document.getElementById("access-logged-text");
    if (accessText) {
      accessText.textContent = `Access-Logged Confirmation: Bedside consult session immutably logged to Patient #${data.patient_id} audit trail at ${new Date().toLocaleTimeString()} (Status: GRANTED, IP: 127.0.0.1).`;
    }
  }

  // Render Records
  const recordsContainer = document.getElementById("scoped-records-container");
  recordsContainer.style.display = "flex";
  recordsContainer.innerHTML = "";

  const records = data.records || [];
  if (records.length === 0) {
    recordsContainer.innerHTML = `
      <div class="card" style="padding: 24px; text-align: center; color: var(--text-muted);">
        Verified scope contains 0 current records for this patient.
      </div>
    `;
    return;
  }

  records.forEach((r) => {
    const card = document.createElement("div");
    card.className = "record-card";

    let borderStyle = "";
    if (r.category === "allergy") {
      borderStyle = "border-left: 4px solid var(--risk-critical);";
    } else if (r.category === "medication") {
      borderStyle = "border-left: 4px solid var(--color-brand);";
    }

    if (borderStyle) {
      card.style.cssText = borderStyle;
    }

    let jsonDisplay = "";
    if (r.structured_data && Object.keys(r.structured_data).length > 0) {
      jsonDisplay = `<pre class="record-json-view">${escapeHtml(JSON.stringify(r.structured_data, null, 2))}</pre>`;
    }

    card.innerHTML = `
      <div class="record-header">
        <span class="record-cat-pill">${escapeHtml(r.category.replace("_", " "))}</span>
        <span style="font-size: 11px; color: var(--text-muted);">Vault Record #${r.vault_id}</span>
      </div>
      <div class="record-title">${escapeHtml(r.title)}</div>
      <div class="record-desc">${escapeHtml(r.description || "No specific observations documented.")}</div>
      ${jsonDisplay}
    `;

    recordsContainer.appendChild(card);
  });
}

function showErrorState(msg) {
  document.getElementById("idle-card").style.display = "none";
  document.getElementById("verified-badge-card").style.display = "none";
  document.getElementById("leakage-notice-card").style.display = "none";
  const accessLoggedCard = document.getElementById("access-logged-notice-card");
  if (accessLoggedCard) accessLoggedCard.style.display = "none";
  document.getElementById("scoped-records-container").style.display = "none";

  const errorCard = document.getElementById("error-card");
  errorCard.style.display = "block";
  document.getElementById("error-card-msg").textContent = msg;
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
