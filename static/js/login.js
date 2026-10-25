/**
 * MediHaven login gateway: credential submission, demo quick-fill and redirect notices.
 */
const LOGIN_NOTICES = {
  signin: "Please sign in to access this portal.",
  role: "Your account role does not have access to that portal.",
  expired: "Your session has ended. Please sign in again.",
};

function fillCredentials(username, password) {
  document.getElementById("username-input").value = username;
  document.getElementById("password-input").value = password;
  document.getElementById("login-error-alert").style.display = "none";
}

/** Only same-site paths are followed after login (never "//host" or absolute URLs). */
function safeNextPath() {
  const next = new URLSearchParams(window.location.search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : null;
}

async function handleLoginSubmit(event) {
  event.preventDefault();
  const btn = document.getElementById("btn-login-submit");
  const errorBox = document.getElementById("login-error-alert");
  const username = document.getElementById("username-input").value.trim();
  const password = document.getElementById("password-input").value.trim();

  errorBox.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Authenticating...";

  const fail = (message) => {
    errorBox.textContent = message;
    errorBox.style.display = "block";
    btn.disabled = false;
    btn.textContent = "Sign In to Portal →";
  };

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.success) {
      fail(data.error?.message || "Invalid username or password.");
      return;
    }
    btn.textContent = "✓ Access Granted. Redirecting...";
    window.location.href = safeNextPath() || data.data?.redirect_url || "/";
  } catch (err) {
    fail("Connection error. Please try again.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("login-form").addEventListener("submit", handleLoginSubmit);
  document.querySelectorAll("[data-demo-username]").forEach((button) => {
    button.addEventListener("click", () => fillCredentials(button.dataset.demoUsername, button.dataset.demoPassword));
  });
  // Redirect notices come from a fixed list, so a crafted link cannot put its own text on this page
  const code = new URLSearchParams(window.location.search).get("error");
  if (code) {
    const errorBox = document.getElementById("login-error-alert");
    errorBox.textContent = LOGIN_NOTICES[code] || LOGIN_NOTICES.signin;
    errorBox.style.display = "block";
  }
});
