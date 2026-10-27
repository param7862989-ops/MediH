/**
 * Shared by every portal page. When the session ends (logout in another tab, expiry, server restart
 * with new secrets), API calls answer 401: send the user to the login page instead of leaving the
 * page silently frozen on stale data.
 */
(function () {
  const originalFetch = window.fetch.bind(window);
  let redirecting = false;
  window.fetch = async function (input, init) {
    const response = await originalFetch(input, init);
    const url = new URL(typeof input === "string" ? input : input.url, window.location.href);
    const sameOriginApi = url.origin === window.location.origin && url.pathname.startsWith("/api/");
    if (response.status === 401 && sameOriginApi && !url.pathname.startsWith("/api/auth/") && !redirecting) {
      redirecting = true;
      const next = encodeURIComponent(window.location.pathname);
      window.location.href = `/login?error=expired&next=${next}`;
    }
    return response;
  };
})();

/**
 * Six-level risk scale used by the research pages (same bands as src/training/stays.py).
 * riskBadge() returns a coloured badge element built with textContent only.
 */
const RISK_LEVELS = [
  { level: "Very low", from: 0.0, color: "#34d399" },
  { level: "Low", from: 0.025, color: "#a3e635" },
  { level: "Medium", from: 0.075, color: "#facc15" },
  { level: "High", from: 0.15, color: "#fb923c" },
  { level: "Very high", from: 0.3, color: "#f87171" },
  { level: "Extreme", from: 0.6, color: "#f472b6" },
];

function riskLevelColor(level) {
  const entry = RISK_LEVELS.find((l) => l.level === level);
  return entry ? entry.color : "#94a3b8";
}

function riskBadge(level, size = "normal") {
  const badge = document.createElement("span");
  const color = riskLevelColor(level);
  badge.textContent = level;
  badge.style.cssText =
    `display: inline-block; border-radius: 999px; font-weight: 700; white-space: nowrap; ` +
    `color: ${color}; border: 1px solid ${color}; background: ${color}22; ` +
    (size === "large" ? "font-size: 18px; padding: 4px 16px;" : size === "small" ? "font-size: 10px; padding: 1px 7px;" : "font-size: 12px; padding: 2px 10px;");
  return badge;
}

function riskRangeText(entry, next) {
  const pct = (x) => `${Math.round(x * 1000) / 10}%`;
  return next ? `${pct(entry.from)} – ${pct(next.from)}` : `≥ ${pct(entry.from)}`;
}
