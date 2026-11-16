// =====================================================
// DocMind AI - frontend logic (Vanilla JavaScript)
// This file only talks to the backend using fetch().
// It never sees the Gemini API key - that stays on the server.
// =====================================================

// ---------- 1. Settings ----------

// Both backends expose the SAME API, so the frontend only needs a different base URL.
const BACKEND_URLS = {
  python: "http://localhost:8000",
  javascript: "http://localhost:3000",
};

const MAX_FILE_SIZE_MB = 10;

// Remember the last chosen backend, so a page refresh does not reset it.
let currentBackend = localStorage.getItem("docmind-backend");
if (!BACKEND_URLS[currentBackend]) {
  currentBackend = "javascript";
}

// ---------- 2. Grab the HTML elements we need ----------

const sidebar = document.getElementById("sidebar");
const backdrop = document.getElementById("backdrop");
const menuBtn = document.getElementById("menuBtn");
const backendSwitch = document.getElementById("backendSwitch");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadMessage = document.getElementById("uploadMessage");
const documentList = document.getElementById("documentList");
const documentsEmpty = document.getElementById("documentsEmpty");
const messages = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const errorBanner = document.getElementById("errorBanner");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");

// ---------- 3. Helper: talk to the backend ----------

// Sends a request to the currently selected backend and returns the JSON result.
// If anything goes wrong, it throws an Error with a message that is safe to show to the user.
async function apiRequest(path, options) {
  const url = BACKEND_URLS[currentBackend] + path;

  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    // fetch() fails like this when the server is not running (or blocked by CORS)
    throw new Error("Cannot reach the " + currentBackend + " backend at " + BACKEND_URLS[currentBackend] + ". Is the server running?");
  }

  let data = null;
  try {
    data = await response.json();
  } catch (error) {
    throw new Error("The server sent an unexpected response.");
  }

  if (!response.ok || data.success === false) {
    throw new Error(data.error || "Something went wrong.");
  }
  return data;
}

// Makes text safe before putting it inside HTML (prevents HTML injection).
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Very small formatter: **bold** and new lines.
function formatAnswer(text) {
  let safeText = escapeHtml(text);
  safeText = safeText.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return safeText.replace(/\n/g, "<br>");
}

// ---------- 4. Backend selector + health check ----------

function updateBackendButtons() {
  const buttons = backendSwitch.querySelectorAll("button");
  buttons.forEach(function (button) {
    button.classList.toggle("active", button.dataset.backend === currentBackend);
  });
}

async function checkHealth() {
  statusDot.className = "dot";
  statusText.textContent = "Checking...";
  try {
    const data = await apiRequest("/api/health");
    statusDot.className = "dot online";
    if (data.gemini_key_configured === false) {
      statusText.textContent = "Online, but GEMINI_API_KEY is missing";
    } else {
      statusText.textContent = "Online";
    }
  } catch (error) {
    statusDot.className = "dot offline";
    statusText.textContent = "Offline";
  }
}

async function switchBackend(name) {
  currentBackend = name;
  localStorage.setItem("docmind-backend", name);
  updateBackendButtons();
  hideError();
  await checkHealth();
  await loadDocuments();
}

backendSwitch.addEventListener("click", function (event) {
  const button = event.target.closest("button");
  if (button && button.dataset.backend !== currentBackend) {
    switchBackend(button.dataset.backend);
  }
});

// ---------- 5. Documents (sidebar) ----------

async function loadDocuments() {
  try {
    const data = await apiRequest("/api/documents");
    renderDocuments(data.documents);
  } catch (error) {
    renderDocuments([]);
  }
}

function renderDocuments(documents) {
  documentList.innerHTML = "";
  documentsEmpty.hidden = documents.length > 0;

  documents.forEach(function (doc) {
    const item = document.createElement("li");
    item.className = "document-item";

    const info = document.createElement("div");
    info.className = "document-info";

    const name = document.createElement("div");
    name.className = "document-name";
    name.textContent = doc.filename;
    name.title = doc.filename;

    const meta = document.createElement("div");
    meta.className = "document-meta";
    meta.innerHTML = '<span class="ready-tag">Ready</span> &middot; ' + doc.pages + " pages &middot; " + doc.chunks + " chunks";

    info.appendChild(name);
    info.appendChild(meta);

    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-btn";
    deleteButton.type = "button";
    deleteButton.title = "Delete document";
    deleteButton.innerHTML = "&times;";
    deleteButton.addEventListener("click", function () {
      deleteDocument(doc.id, doc.filename);
    });

    item.appendChild(info);
    item.appendChild(deleteButton);
    documentList.appendChild(item);
  });
}

async function deleteDocument(documentId, filename) {
  const sure = confirm('Delete "' + filename + '"?');
  if (!sure) return;

  try {
    await apiRequest("/api/documents/" + encodeURIComponent(documentId), { method: "DELETE" });
    await loadDocuments();
  } catch (error) {
    showError(error.message);
  }
}

// ---------- 6. Upload ----------

function showUploadMessage(text, type) {
  uploadMessage.hidden = false;
  uploadMessage.className = "upload-message " + (type || "");
  uploadMessage.innerHTML = "";

  if (type === "loading") {
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    uploadMessage.appendChild(spinner);
  }

  const span = document.createElement("span");
  span.textContent = text;
  uploadMessage.appendChild(span);
}

async function uploadFile(file) {
  // Quick checks in the browser (the backend checks again - never trust the client only)
  const looksLikePdf = file.name.toLowerCase().endsWith(".pdf");
  if (!looksLikePdf) {
    showUploadMessage("Only PDF files are allowed.", "error");
    return;
  }
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    showUploadMessage("File is too large. Maximum size is " + MAX_FILE_SIZE_MB + " MB.", "error");
    return;
  }

  showUploadMessage("Processing " + file.name + "... (reading, chunking, embedding)", "loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await apiRequest("/api/documents/upload", { method: "POST", body: formData });
    showUploadMessage(data.document.filename + " is ready.", "success");
    await loadDocuments();
  } catch (error) {
    showUploadMessage(error.message, "error");
  }
}

dropzone.addEventListener("click", function () { fileInput.click(); });
dropzone.addEventListener("keydown", function (event) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", function () {
  if (fileInput.files.length > 0) {
    uploadFile(fileInput.files[0]);
  }
  fileInput.value = ""; // allows uploading the same file again
});

dropzone.addEventListener("dragover", function (event) {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", function () {
  dropzone.classList.remove("dragover");
});
dropzone.addEventListener("drop", function (event) {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  if (event.dataTransfer.files.length > 0) {
    uploadFile(event.dataTransfer.files[0]);
  }
});

// ---------- 7. Chat ----------

function showError(text) {
  errorBanner.textContent = text;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// Adds one chat bubble. `sources` is only used for AI messages.
function addMessage(role, text, sources, isError) {
  emptyState.hidden = true;

  const row = document.createElement("div");
  row.className = "message " + role + (isError ? " is-error" : "");

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "user") {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = formatAnswer(text);
  }

  // Show where the answer came from: filename + page number
  if (sources && sources.length > 0) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";

    const label = document.createElement("span");
    label.className = "sources-label";
    label.textContent = "Sources";
    sourceBox.appendChild(label);

    sources.forEach(function (source) {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.title = source.snippet || "";
      chip.textContent = source.filename + " · ";
      const page = document.createElement("span");
      page.textContent = "Page " + source.page;
      chip.appendChild(page);
      sourceBox.appendChild(chip);
    });

    bubble.appendChild(sourceBox);
  }

  row.appendChild(bubble);
  messages.appendChild(row);
  scrollToBottom();
}

// Three bouncing dots shown while we wait for the AI.
function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message ai";
  row.innerHTML = '<div class="bubble"><div class="typing"><i></i><i></i><i></i></div></div>';
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

async function sendQuestion(question) {
  hideError();
  addMessage("user", question);
  questionInput.value = "";
  resizeInput();

  sendBtn.disabled = true;
  const typingRow = addTypingIndicator();

  try {
    const data = await apiRequest("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    typingRow.remove();
    addMessage("ai", data.answer, data.sources);
  } catch (error) {
    typingRow.remove();
    addMessage("ai", error.message, [], true);
  }

  sendBtn.disabled = false;
  questionInput.focus();
}

chatForm.addEventListener("submit", function (event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question === "" || sendBtn.disabled) return;
  sendQuestion(question);
});

// Enter = send, Shift+Enter = new line
questionInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

// Make the text box grow as the user types
function resizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
}
questionInput.addEventListener("input", resizeInput);

// Clear chat only clears the screen. Uploaded documents stay.
clearBtn.addEventListener("click", function () {
  messages.querySelectorAll(".message").forEach(function (element) { element.remove(); });
  emptyState.hidden = false;
  hideError();
});

// ---------- 8. Mobile sidebar ----------

function openSidebar() {
  sidebar.classList.add("open");
  backdrop.classList.add("show");
}
function closeSidebar() {
  sidebar.classList.remove("open");
  backdrop.classList.remove("show");
}
menuBtn.addEventListener("click", openSidebar);
backdrop.addEventListener("click", closeSidebar);

// ---------- 9. Start the app ----------

updateBackendButtons();
checkHealth();
loadDocuments();
