/**
 * AeroVision CV — AI Weather Classification Frontend
 *
 * The UI stays static and lightweight. All real predictions are made by
 * the FastAPI backend; this client never fabricates inference results.
 */

const API_OVERRIDE_KEY = "aerovision_api_override";
const isLocal = window.location.protocol === "file:" || window.location.port === "5500";
const DEFAULT_API_URL = isLocal ? "http://127.0.0.1:8000/predict" : "/api/predict";
const storedOverride = localStorage.getItem(API_OVERRIDE_KEY);
let API_URL = storedOverride || DEFAULT_API_URL;

const $ = (id) => document.getElementById(id);
const dropZone = $("dropZone");
const fileInput = $("fileInput");
const browseButton = $("browseButton");
const predictButton = $("predictButton");
const removeButton = $("removeButton");
const changeImageBtn = $("changeImageBtn");
const analyzeAnotherBtn = $("analyzeAnotherBtn");
const previewContainer = $("previewContainer");
const previewImage = $("previewImage");
const fileInfo = $("fileInfo");
const fileMetaSub = $("fileMetaSub");
const loading = $("loading");
const result = $("result");
const prediction = $("prediction");
const confidence = $("confidence");
const confidenceFill = $("confidenceFill");
const confidenceTier = $("confidenceTier");
const confidenceNoteVal = $("confidenceNoteVal");
const deviceInfo = $("deviceInfo");
const weatherIconContainer = $("weatherIconContainer");
const errorBox = $("error");
const errorMessage = $("errorMessage");
const errorHeading = $("errorHeading");
const errorSuggestion = $("errorSuggestion");
const configEndpointBtn = $("configEndpointBtn");
const configModal = $("configModal");
const closeModalBtn = $("closeModalBtn");
const cancelConfigBtn = $("cancelConfigBtn");
const saveConfigBtn = $("saveConfigBtn");
const apiUrlInput = $("apiUrlInput");
const statusDot = $("statusDot");
const statusText = $("statusText");

let selectedFile = null;
let previewUrl = null;

const weatherIcons = {
  rain: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15a7 7 0 1 1 13.7-2H19a3.5 3.5 0 1 1 0 7"/><path d="M8 18v3M12 18v3M16 18v3"/></svg>`,
  fog: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 9h14M4 13h16M6 17h12"/></svg>`,
  snow: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 3v18M4.2 7.5l15.6 9M19.8 7.5l-15.6 9"/><path d="M8 5l4 2 4-2M8 19l4-2 4 2M5.5 12l-1.5-4 4 .5M18.5 12l1.5 4-4-.5"/></svg>`,
  default: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>`
};

function isRelativeApi(url) {
  return url.startsWith("/");
}

function healthUrlFor(apiUrl) {
  return isRelativeApi(apiUrl) ? "/api/health" : `${new URL(apiUrl).origin}/health`;
}

function setStatus(text, connected = false) {
  statusText.textContent = text;
  statusDot.style.background = connected ? "#10b981" : "#38bdf8";
}

async function checkBackendHealth() {
  try {
    const response = await fetch(healthUrlFor(API_URL), {
      method: "GET",
      signal: AbortSignal.timeout(2500)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status === "degraded") throw new Error(data.model_error || "Backend unavailable");
    setStatus("Backend Connected", true);
  } catch {
    setStatus(isRelativeApi(API_URL) ? "API Unavailable" : "Local API Target", false);
  }
}

function showError(heading, message, suggestion = "") {
  errorHeading.textContent = heading || "Inference Error";
  errorMessage.textContent = message || "An unexpected error occurred.";
  errorSuggestion.textContent = suggestion;
  errorSuggestion.hidden = !suggestion;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

function validateFile(file) {
  if (!file) return false;
  const allowed = new Set(["image/jpeg", "image/png", "image/webp"]);
  if (!allowed.has((file.type || "").toLowerCase())) {
    showError("Invalid File Format", "Please upload a JPG, PNG, or WEBP image.", "Choose another supported image and try again.");
    return false;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError("File Too Large", "The maximum upload size is 15 MB.", "Compress or resize the image before uploading it again.");
    return false;
  }
  return true;
}

function handleFile(file) {
  hideError();
  if (!validateFile(file)) return;

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  fileInfo.textContent = file.name;

  const size = file.size >= 1024 * 1024
    ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
    : `${Math.max(file.size / 1024, 0.1).toFixed(1)} KB`;
  const format = file.type.split("/")[1]?.toUpperCase() || "IMAGE";
  fileMetaSub.textContent = `${format} • ${size} • Ready for classification`;

  dropZone.hidden = true;
  previewContainer.hidden = false;
  predictButton.disabled = false;
  result.hidden = true;
}

function resetUploadState() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  dropZone.hidden = false;
  previewContainer.hidden = true;
  predictButton.disabled = true;
  result.hidden = true;
  confidenceFill.style.width = "0%";
  hideError();
}

function loadSampleImage(type) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 420;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  if (type === "rain") {
    gradient.addColorStop(0, "#1e293b");
    gradient.addColorStop(1, "#020617");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 640, 420);
    ctx.strokeStyle = "rgba(186,230,253,.32)";
    for (let i = 0; i < 220; i += 1) {
      const x = Math.random() * 640;
      const y = Math.random() * 420;
      const len = 12 + Math.random() * 24;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 4, y + len); ctx.stroke();
    }
  } else if (type === "fog") {
    gradient.addColorStop(0, "#475569");
    gradient.addColorStop(.55, "#94a3b8");
    gradient.addColorStop(1, "#64748b");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 640, 420);
    ctx.fillStyle = "rgba(241,245,249,.34)";
    for (let i = 0; i < 6; i += 1) {
      ctx.beginPath(); ctx.ellipse(320, 135 + i * 44, 360, 38, 0, 0, Math.PI * 2); ctx.fill();
    }
  } else {
    gradient.addColorStop(0, "#334155");
    gradient.addColorStop(.45, "#64748b");
    gradient.addColorStop(1, "#f8fafc");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 640, 420);
    ctx.fillStyle = "rgba(255,255,255,.8)";
    for (let i = 0; i < 170; i += 1) {
      const x = Math.random() * 640;
      const y = Math.random() * 420;
      const radius = 1 + Math.random() * 3;
      ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
    }
  }

  canvas.toBlob((blob) => {
    if (!blob) return;
    handleFile(new File([blob], `sample_${type}_weather.png`, { type: "image/png" }));
  }, "image/png");
}

function renderResult(data) {
  const label = String(data.prediction || "UNKNOWN").toLowerCase();
  const score = Math.max(0, Math.min(100, Number(data.confidence) || 0));

  prediction.textContent = label.toUpperCase();
  weatherIconContainer.innerHTML = weatherIcons[label] || weatherIcons.default;
  confidence.textContent = `${score.toFixed(2)}%`;
  confidenceNoteVal.textContent = `${score.toFixed(2)}%`;
  confidenceFill.style.width = `${score}%`;
  confidenceFill.setAttribute("aria-valuenow", score.toFixed(2));

  confidenceTier.className = "confidence-tier-pill";
  if (score >= 85) {
    confidenceTier.classList.add("high");
    confidenceTier.textContent = "✓  High Confidence";
    confidenceFill.style.background = "#38bdf8";
  } else if (score >= 60) {
    confidenceTier.classList.add("moderate");
    confidenceTier.textContent = "ⓘ  Moderate Confidence";
    confidenceFill.style.background = "#fbbf24";
  } else {
    confidenceTier.classList.add("low");
    confidenceTier.textContent = "!  Low Confidence";
    confidenceFill.style.background = "#f87171";
  }

  deviceInfo.textContent = String(data.device || "unknown").toUpperCase();
  result.hidden = false;

  if (window.innerWidth < 768) result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function predictWeather() {
  if (!selectedFile) return;
  hideError();
  result.hidden = true;
  loading.hidden = false;
  predictButton.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(30000)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Server returned HTTP ${response.status}`);
    renderResult(data);
    setStatus("Backend Connected", true);
  } catch (error) {
    setStatus(isRelativeApi(API_URL) ? "API Unavailable" : "Local API Target", false);
    const suggestion = isLocal
      ? "Start FastAPI with: python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
      : "Check the deployed backend URL in API Config and verify that its /health endpoint is healthy.";
    showError("Backend Connection Failed", `Unable to reach the prediction API. ${error.message}`, suggestion);
  } finally {
    loading.hidden = true;
    predictButton.disabled = false;
  }
}

browseButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", (event) => { if (!event.target.closest("button")) fileInput.click(); });
dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); } });
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (event) => { event.preventDefault(); dropZone.classList.remove("drag-over"); handleFile(event.dataTransfer.files[0]); });
fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
[...document.querySelectorAll(".sample-chip")].forEach((chip) => chip.addEventListener("click", (event) => { event.stopPropagation(); loadSampleImage(chip.dataset.sample); }));
predictButton.addEventListener("click", predictWeather);
removeButton.addEventListener("click", resetUploadState);
analyzeAnotherBtn.addEventListener("click", resetUploadState);
changeImageBtn.addEventListener("click", () => fileInput.click());

function closeConfig() { configModal.hidden = true; }
configEndpointBtn.addEventListener("click", () => { apiUrlInput.value = API_URL; configModal.hidden = false; });
closeModalBtn.addEventListener("click", closeConfig);
cancelConfigBtn.addEventListener("click", closeConfig);
configModal.addEventListener("click", (event) => { if (event.target === configModal) closeConfig(); });

saveConfigBtn.addEventListener("click", () => {
  const nextUrl = apiUrlInput.value.trim();
  try {
    const parsed = new URL(nextUrl, window.location.origin);
    if (!isRelativeApi(nextUrl) && !/^https?:$/.test(parsed.protocol)) throw new Error("Invalid protocol");
  } catch {
    showError("Invalid API URL", "Enter a valid absolute URL or a relative endpoint such as /api/predict.", "Example: https://your-backend.example.com/predict");
    return;
  }

  API_URL = nextUrl;
  localStorage.setItem(API_OVERRIDE_KEY, API_URL);
  closeConfig();
  checkBackendHealth();
});

apiUrlInput.value = API_URL;
checkBackendHealth();
