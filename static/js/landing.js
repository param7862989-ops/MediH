/**
 * MediHaven landing page. The contact form is a demo: nothing is sent anywhere.
 */
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("contact-inquiry-form");
  if (!form) return;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    document.getElementById("btn-submit-inquiry").style.display = "none";
    document.getElementById("contact-success-msg").style.display = "block";
    form.reset();
  });
});
