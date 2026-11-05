/**
 * MediHaven theme switch (dark by default, optional light theme).
 * Loaded in <head> without defer so the saved theme is applied before the first paint.
 * The choice is remembered per browser in localStorage; nothing is sent to the server.
 */
(function () {
  var STORAGE_KEY = "medihaven-theme";
  var root = document.documentElement;

  function savedTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null; // private mode or blocked storage: fall back to the default
    }
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme === "light" ? "light" : "dark");
  }

  function updateToggles() {
    var isLight = root.getAttribute("data-theme") === "light";
    var buttons = document.querySelectorAll(".theme-toggle");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", isLight ? "true" : "false");
      buttons[i].setAttribute("aria-label", isLight ? "Switch to dark theme" : "Switch to light theme");
      buttons[i].title = isLight ? "Dark theme" : "Light theme";
    }
  }

  applyTheme(savedTheme());

  document.addEventListener("DOMContentLoaded", function () {
    updateToggles();
    var buttons = document.querySelectorAll(".theme-toggle");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function () {
        var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
        applyTheme(next);
        try {
          localStorage.setItem(STORAGE_KEY, next);
        } catch (e) {
          /* the theme still switches for this page */
        }
        updateToggles();
        document.dispatchEvent(new CustomEvent("medihaven:themechange", { detail: { theme: next } }));
      });
    }
  });
})();
