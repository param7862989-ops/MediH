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
