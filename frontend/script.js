const API_URL_STORAGE_KEY = "weather_api_url";
const isStandaloneLocal = window.location.protocol === "file:" || window.location.port === "5500";
const DEFAULT_API_URL = isStandaloneLocal ? "http://127.0.0.1:8000/predict" : "/api/predict";
let API_URL = isStandaloneLocal
  ? (localStorage.getItem(API_URL_STORAGE_KEY) || DEFAULT_API_URL)
  : DEFAULT_API_URL;

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
const errorBox = $("error");
const errorHeading = $("errorHeading");
const errorMessage = $("errorMessage");
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

function setStatus(text, connected = false) {
  statusText.textContent = text;
  statusDot.style.background = connected ? "#10b981" : "#94a3b8";
}

async function checkBackendHealth() {
  try {
    const healthUrl = API_URL.startsWith("/")
      ? "/api/health"
      : `${new URL(API_URL).origin}/health`;
    const response = await fetch(healthUrl, { signal: AbortSignal.timeout(2500) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status === "degraded") throw new Error(data.model_error || "Backend is degraded");
    setStatus("Backend Connected", true);
  } catch {
    setStatus("Backend Offline");
  }
}

function showError(heading, message, suggestion) {
  errorHeading.textContent = heading;
  errorMessage.textContent = message;
  errorSuggestion.textContent = suggestion || "Please try again.";
  errorSuggestion.hidden = !suggestion;
  errorBox.hidden = false;
}
function hideError() { errorBox.hidden = true; }

function validateFile(file) {
  const allowed = new Set(["image/jpeg", "image/png", "image/webp"]);
  if (!allowed.has(file.type.toLowerCase())) {
    showError("Unsupported image", "Use a JPG, PNG, or WEBP image.", "Choose another image and try again.");
    return false;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError("Image is too large", "The maximum upload size is 15 MB.", "Resize or compress the image and upload it again.");
    return false;
  }
  return true;
}

function handleFile(file) {
  hideError();
  if (!file || !validateFile(file)) return;
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  fileInfo.textContent = file.name;
  const size = file.size >= 1048576 ? `${(file.size / 1048576).toFixed(2)} MB` : `${(file.size / 1024).toFixed(1)} KB`;
  fileMetaSub.textContent = `${file.type.split("/")[1].toUpperCase()} • ${size} • Ready for classification`;
  dropZone.hidden = true;
  previewContainer.hidden = false;
  predictButton.disabled = false;
  result.hidden = true;
}

function resetUploadState() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  dropZone.hidden = false;
  previewContainer.hidden = true;
  predictButton.disabled = true;
  result.hidden = true;
  hideError();
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
    const score = Number(data.confidence) || 0;
    const safeScore = Math.max(0, Math.min(100, score));
    prediction.textContent = String(data.prediction || "UNKNOWN").toUpperCase();
    confidence.textContent = `${safeScore.toFixed(2)}%`;
    if (confidenceNoteVal) confidenceNoteVal.textContent = `${safeScore.toFixed(2)}%`;
    confidenceFill.style.width = `${safeScore}%`;
    confidenceTier.textContent = safeScore >= 90 ? "High Confidence" : safeScore >= 70 ? "Moderate Confidence" : "Low Confidence";
    deviceInfo.textContent = data.device ? String(data.device).toUpperCase() : "UNKNOWN";
    result.hidden = false;
    setStatus("Backend Connected", true);
  } catch (error) {
    showError(
      "Inference failed",
      error.name === "TimeoutError" ? "The prediction service took too long to respond." : `Unable to reach the prediction API. ${error.message}`,
      API_URL.startsWith("/") ? "Check the deployment logs and make sure the model checkpoint is available." : "Start the service with: python -m uvicorn backend.main:app --reload"
    );
    setStatus("Backend Offline");
  } finally {
    loading.hidden = true;
    predictButton.disabled = false;
  }
}

browseButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", (event) => { if (!event.target.closest("button")) fileInput.click(); });
dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("drag"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", (event) => { event.preventDefault(); dropZone.classList.remove("drag"); handleFile(event.dataTransfer.files[0]); });
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
  try { new URL(nextUrl, window.location.origin); } catch { showError("Invalid API URL", "Enter a valid endpoint URL.", "For the deployed app, use /api/predict"); return; }
  API_URL = nextUrl;
  if (isStandaloneLocal) localStorage.setItem(API_URL_STORAGE_KEY, API_URL);
  closeConfig();
  checkBackendHealth();
});

apiUrlInput.value = API_URL;
checkBackendHealth();
