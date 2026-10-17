/**
 * Wattsmart 2.0 Frontend Client Logic
 */

// Helper: Escape HTML to prevent XSS
function escapeHtml(str) {
  if (typeof str !== "string") return str;
  return str.replace(/[&<>'"]/g, 
    tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag)
  );
}

// Helper: Show bootstrap dismissible alert in #alertContainer
function showAlert(message, type = "danger") {
  const container = document.getElementById("alertContainer");
  if (!container) return;

  const alertDiv = document.createElement("div");
  alertDiv.className = `alert alert-${type} alert-dismissible fade show shadow-sm`;
  alertDiv.role = "alert";
  alertDiv.innerHTML = `
    <div class="d-flex align-items-center gap-2">
      <i class="bi bi-exclamation-triangle-fill"></i>
      <div>${escapeHtml(message)}</div>
    </div>
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  container.appendChild(alertDiv);

  setTimeout(() => {
    try {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alertDiv);
      bsAlert.close();
    } catch (_) {}
  }, 6000);
}

// Dark Mode Toggle & Persistence
(function initTheme() {
  const htmlEl = document.documentElement;
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");

  const savedTheme = localStorage.getItem("wattsmart_theme") || "light";
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const currentTheme = htmlEl.getAttribute("data-bs-theme") || "light";
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(newTheme);
      localStorage.setItem("wattsmart_theme", newTheme);
    });
  }

  function applyTheme(theme) {
    htmlEl.setAttribute("data-bs-theme", theme);
    if (themeIcon && themeLabel) {
      if (theme === "dark") {
        themeIcon.className = "bi bi-sun-fill text-warning";
        themeLabel.textContent = "Light";
      } else {
        themeIcon.className = "bi bi-moon-stars-fill";
        themeLabel.textContent = "Dark";
      }
    }
  }
})();
