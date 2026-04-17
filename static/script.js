/**
 * MathMagic — Frontend JavaScript
 * Connects all input modalities to the Flask backend
 * Handles: Canvas drawing, File upload, Text input, Speech, Adaptive learning
 */

"use strict";

// ─── Config ────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;  // Same server (Flask serves both)

// ─── State ─────────────────────────────────────────────────────────────────
const state = {
  playerName: "Explorer",
  currentLevel: 1,
  correct: 0,
  total: 0,
  streak: 0,
  xp: 0,
  xpToNext: 10,
  currentProblem: null,
  brushColor: "#1a1a2e",
  brushSize: 16,
  isDrawing: false,
  canvasHasContent: false,
  isRecording: false,
  mediaRecorder: null,
  audioChunks: [],
  speechTranscript: "",
};

// ─── DOM Elements ──────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const canvas = $("drawing-canvas");
const ctx = canvas.getContext("2d");

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  initTabs();
  initControls();
  initUpload();
  initTypeTab();
  initSpeech();
  initPractice();
  initModal();
});

// ════════════════════════════════════════════════════════════════
// MODAL — Player Name
// ════════════════════════════════════════════════════════════════
function initModal() {
  const modal = $("name-modal");
  const nameInput = $("name-input");
  const startBtn = $("start-btn");

  // Check localStorage for returning player
  const savedName = localStorage.getItem("mathmagic_name");
  const savedLevel = parseInt(localStorage.getItem("mathmagic_level")) || 1;
  if (savedName) {
    state.playerName = savedName;
    state.currentLevel = savedLevel;
    modal.classList.add("hidden");
    updatePlayerBadge();
    generateProblem();
    return;
  }

  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") startBtn.click();
  });

  startBtn.addEventListener("click", () => {
    const name = nameInput.value.trim() || "Explorer";
    state.playerName = name;
    localStorage.setItem("mathmagic_name", name);
    localStorage.setItem("mathmagic_level", "1");
    modal.classList.add("hidden");
    updatePlayerBadge();
    generateProblem();
    showToast("🦄", "Welcome!", `Let's learn together, ${name}!`);
  });
}

function updatePlayerBadge() {
  $("player-name-display").textContent = state.playerName;
  $("player-level-display").textContent = `Level ${state.currentLevel} ⭐`;
}

// ════════════════════════════════════════════════════════════════
// TABS
// ════════════════════════════════════════════════════════════════
function initTabs() {
  document.querySelectorAll(".nav-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".nav-pill").forEach((p) => p.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((t) => t.classList.remove("active"));
      pill.classList.add("active");
      const tab = pill.dataset.tab;
      $(`tab-${tab}`).classList.add("active");
      resetOutput();
    });
  });
}

// ════════════════════════════════════════════════════════════════
// CANVAS — Drawing
// ════════════════════════════════════════════════════════════════
function initCanvas() {
  // Setup canvas background
  ctx.fillStyle = "#fff9f0";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Pointer events (works for mouse + touch + stylus)
  canvas.addEventListener("pointerdown", startDraw);
  canvas.addEventListener("pointermove", draw);
  canvas.addEventListener("pointerup", endDraw);
  canvas.addEventListener("pointerleave", endDraw);
  canvas.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });

  $("predict-btn").addEventListener("click", () => {
    if (!state.canvasHasContent) {
      showError("Please draw something first! ✏️");
      return;
    }
    const multiMode = $("multi-digit-mode").checked;
    if (multiMode) predictMultiDigit();
    else predictSingleDigit();
  });

  $("clear-btn").addEventListener("click", clearCanvas);
}

function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = e.clientX ?? e.touches?.[0]?.clientX;
  const clientY = e.clientY ?? e.touches?.[0]?.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function startDraw(e) {
  state.isDrawing = true;
  const pos = getCanvasPos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  // Hide hint on first draw
  const hint = document.querySelector(".canvas-hint");
  if (hint) hint.classList.add("hidden");
}

function draw(e) {
  if (!state.isDrawing) return;
  const pos = getCanvasPos(e);
  ctx.lineWidth = state.brushSize;
  ctx.strokeStyle = state.brushColor;
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  state.canvasHasContent = true;
}

function endDraw() {
  state.isDrawing = false;
  ctx.closePath();
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff9f0";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  state.canvasHasContent = false;
  const hint = document.querySelector(".canvas-hint");
  if (hint) hint.classList.remove("hidden");
  resetOutput();
}

function initControls() {
  // Brush size
  const brushSlider = $("brush-size");
  const brushPreview = $("brush-preview");
  brushSlider.addEventListener("input", () => {
    state.brushSize = parseInt(brushSlider.value);
    brushPreview.style.width = state.brushSize + "px";
    brushPreview.style.height = state.brushSize + "px";
  });

  // Color buttons
  document.querySelectorAll(".color-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.brushColor = btn.dataset.color;
      $("brush-preview").style.background = state.brushColor;
    });
  });
}

async function predictSingleDigit() {
  const imageData = canvas.toDataURL("image/png");
  showLoading();
  try {
    const res = await apiPost("/api/predict/digit", {
      image: imageData,
      name: state.playerName,
    });
    showDigitResult(res);
  } catch (err) {
    showError(err.message);
  }
}

async function predictMultiDigit() {
  const imageData = canvas.toDataURL("image/png");
  showLoading();
  try {
    // Try to get any typed equation context
    const expression = $("equation-input").value.trim() || null;
    const res = await apiPost("/api/predict/multidigit", {
      image: imageData,
      solve: true,
      expression: expression,
      name: state.playerName,
    });
    showMultiDigitResult(res);
  } catch (err) {
    showError(err.message);
  }
}

// ════════════════════════════════════════════════════════════════
// FILE UPLOAD — OCR
// ════════════════════════════════════════════════════════════════
function initUpload() {
  const uploadZone = $("upload-zone");
  const fileInput = $("file-input");
  const preview = $("upload-preview");
  const ocrBtn = $("ocr-btn");
  const clearBtn = $("clear-upload-btn");

  let uploadedFile = null;

  // Click to open file dialog
  uploadZone.addEventListener("click", () => fileInput.click());

  // Drag and drop
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (file.type.startsWith("image/")) {
        preview.src = e.target.result;
        preview.classList.remove("hidden");
      } else {
        preview.classList.add("hidden");
      }
      ocrBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  ocrBtn.addEventListener("click", async () => {
    if (!uploadedFile) return;
    showLoading();
    try {
      const formData = new FormData();
      formData.append("image", uploadedFile);
      const res = await fetch(`${API_BASE}/api/ocr`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      showOCRResult(data);
    } catch (err) {
      showError(err.message);
    }
  });

  clearBtn.addEventListener("click", () => {
    preview.src = "";
    preview.classList.add("hidden");
    ocrBtn.disabled = true;
    uploadedFile = null;
    fileInput.value = "";
    resetOutput();
  });
}

// ════════════════════════════════════════════════════════════════
// TEXT / TYPE TAB
// ════════════════════════════════════════════════════════════════
function initTypeTab() {
  $("solve-btn").addEventListener("click", solveTyped);
  $("clear-text-btn").addEventListener("click", () => {
    $("equation-input").value = "";
    resetOutput();
  });

  $("equation-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") solveTyped();
  });

  // Keypad buttons
  document.querySelectorAll(".keypad-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.key;
      const input = $("equation-input");
      if (key === "C") {
        input.value = "";
      } else if (key === "⌫") {
        input.value = input.value.slice(0, -1);
      } else if (key === "=") {
        solveTyped();
      } else {
        input.value += key;
      }
      input.focus();
    });
  });

  // Quick ops
  document.querySelectorAll(".op-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = $("equation-input");
      const op = btn.dataset.op;
      const pos = input.selectionStart;
      const before = input.value.substring(0, pos);
      const after = input.value.substring(pos);
      if (op === "( )") {
        input.value = before + "()" + after;
        input.setSelectionRange(pos + 1, pos + 1);
      } else {
        input.value = before + " " + op + " " + after;
        input.setSelectionRange(pos + 3, pos + 3);
      }
      input.focus();
    });
  });

  // Example chips
  document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("equation-input").value = chip.dataset.eq;
      solveTyped();
    });
  });
}

async function solveTyped() {
  const equation = $("equation-input").value.trim();
  if (!equation) { showError("Please enter an equation first!"); return; }
  showLoading();
  try {
    const res = await apiPost("/api/solve_equation", { equation, name: state.playerName });
    showEquationResult(res);
  } catch (err) {
    showError(err.message);
  }
}

// ════════════════════════════════════════════════════════════════
// SPEECH RECOGNITION
// ════════════════════════════════════════════════════════════════
function initSpeech() {
  const micBtn = $("mic-btn");
  const micStatus = $("mic-status");
  const transcriptBox = $("transcript-box");
  const transcriptText = $("transcript-text");
  const solveSpeechBtn = $("solve-speech-btn");

  // Try Web Speech API first (browser-native, no server needed)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    // Browser-native speech recognition
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    micBtn.addEventListener("click", () => {
      if (state.isRecording) {
        recognition.stop();
        return;
      }
      recognition.start();
    });

    recognition.onstart = () => {
      state.isRecording = true;
      micBtn.classList.add("recording");
      micStatus.textContent = "🔴 Listening... speak your math problem!";
    };

    recognition.onresult = (e) => {
      const transcript = Array.from(e.results)
        .map((r) => r[0].transcript).join("");
      transcriptText.textContent = transcript;
      transcriptBox.classList.remove("hidden");
      state.speechTranscript = transcript;
    };

    recognition.onend = () => {
      state.isRecording = false;
      micBtn.classList.remove("recording");
      micStatus.textContent = "Tap the mic to start speaking";
      if (state.speechTranscript) {
        solveSpeechBtn.classList.remove("hidden");
      }
    };

    recognition.onerror = (e) => {
      micStatus.textContent = `Error: ${e.error}. Please try again.`;
      state.isRecording = false;
      micBtn.classList.remove("recording");
    };

    solveSpeechBtn.addEventListener("click", () => {
      if (!state.speechTranscript) return;
      $("equation-input").value = state.speechTranscript;
      // Switch to type tab and solve
      document.querySelector('[data-tab="type"]').click();
      solveTyped();
    });

  } else {
    // Fallback: server-side Wav2Vec2
    micBtn.addEventListener("click", () => {
      if (state.isRecording) stopRecording();
      else startMediaRecording();
    });

    solveSpeechBtn.addEventListener("click", async () => {
      if (!state.audioChunks.length) return;
      showLoading();
      try {
        const audioBlob = new Blob(state.audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "speech.wav");
        const res = await fetch(`${API_BASE}/api/speech`, { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        transcriptText.textContent = data.transcription;
        transcriptBox.classList.remove("hidden");
        state.speechTranscript = data.transcription;
        showEquationResult(data);
      } catch (err) {
        showError(err.message);
      }
    });
  }
}

async function startMediaRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(stream);
    state.audioChunks = [];
    state.mediaRecorder.ondataavailable = (e) => state.audioChunks.push(e.data);
    state.mediaRecorder.start();
    state.isRecording = true;
    $("mic-btn").classList.add("recording");
    $("mic-status").textContent = "🔴 Recording...";
  } catch (err) {
    $("mic-status").textContent = "Microphone access denied.";
  }
}

function stopRecording() {
  if (state.mediaRecorder) {
    state.mediaRecorder.stop();
    state.mediaRecorder.stream.getTracks().forEach((t) => t.stop());
  }
  state.isRecording = false;
  $("mic-btn").classList.remove("recording");
  $("mic-status").textContent = "Processing...";
  $("solve-speech-btn").classList.remove("hidden");
}

// ════════════════════════════════════════════════════════════════
// OUTPUT RENDERING
// ════════════════════════════════════════════════════════════════
function resetOutput() {
  $("output-idle").classList.remove("hidden");
  $("output-loading").classList.add("hidden");
  $("output-result").classList.add("hidden");
  $("output-error").classList.add("hidden");
}

function showLoading() {
  $("output-idle").classList.add("hidden");
  $("output-result").classList.add("hidden");
  $("output-error").classList.add("hidden");
  $("output-loading").classList.remove("hidden");
}

function showError(msg) {
  $("output-idle").classList.add("hidden");
  $("output-loading").classList.add("hidden");
  $("output-result").classList.add("hidden");
  $("output-error").classList.remove("hidden");
  $("error-msg").textContent = msg;
}

function showDigitResult(data) {
  $("output-loading").classList.add("hidden");
  $("output-result").classList.remove("hidden");

  const { digit, confidence, probabilities, feedback } = data;

  // Digit badge
  $("result-digit").textContent = digit;
  $("result-confidence").textContent = `${(confidence * 100).toFixed(1)}% confidence`;
  $("confidence-bar").style.width = `${confidence * 100}%`;

  // Hide equation section
  $("equation-result").classList.add("hidden");

  // Feedback
  renderFeedback(feedback);

  // Prob bars
  if (probabilities) {
    $("probs-section").classList.remove("hidden");
    renderProbBars(probabilities, digit);
  }

  $("steps-section").classList.add("hidden");
}

function showMultiDigitResult(data) {
  $("output-loading").classList.add("hidden");
  $("output-result").classList.remove("hidden");

  const digitStr = data.digit_string || data.digits.map((d) => d.digit).join("");
  $("result-digit").textContent = digitStr;
  const avgConf = data.digits.length
    ? data.digits.reduce((s, d) => s + d.confidence, 0) / data.digits.length
    : 0;
  $("result-confidence").textContent = `${data.digits.length} digit(s) found · ${(avgConf * 100).toFixed(1)}% avg confidence`;
  $("confidence-bar").style.width = `${avgConf * 100}%`;

  if (data.result !== undefined && data.result !== null) {
    $("equation-result").classList.remove("hidden");
    $("eq-left").textContent = data.equation || digitStr;
    $("eq-answer").textContent = data.result;
  } else {
    $("equation-result").classList.add("hidden");
  }

  if (data.feedback) renderFeedback(data.feedback);
  if (data.steps) renderSteps(data.steps);
  $("probs-section").classList.add("hidden");
}

function showEquationResult(data) {
  $("output-loading").classList.add("hidden");
  $("output-result").classList.remove("hidden");

  const expr = data.equation || data.expression || "";
  const result = data.result;

  $("result-digit").textContent = result !== null ? "✓" : "?";
  $("result-confidence").textContent = result !== null ? "Solved!" : "Could not solve";
  $("confidence-bar").style.width = result !== null ? "100%" : "20%";

  if (result !== null && result !== undefined) {
    $("equation-result").classList.remove("hidden");
    $("eq-left").textContent = expr;
    $("eq-answer").textContent = result;
  } else {
    $("equation-result").classList.add("hidden");
  }

  if (data.feedback) renderFeedback(data.feedback);
  if (data.steps) renderSteps(data.steps);
  $("probs-section").classList.add("hidden");
}

function showOCRResult(data) {
  showEquationResult({
    ...data,
    feedback: {
      message: data.equation ? `Found: ${data.equation}` : "Text extracted!",
      encouragement: data.raw_text ? "Here's what I read:" : "",
      emoji: "📷",
    },
  });
}

function renderFeedback(feedback) {
  if (!feedback) return;
  $("feedback-card").classList.remove("hidden");
  $("feedback-emoji").textContent = feedback.emoji || "⭐";
  $("feedback-message").textContent = feedback.message || "";
  $("feedback-encouragement").textContent = feedback.encouragement || "";
  if (feedback.hint) {
    $("feedback-hint-wrap").classList.remove("hidden");
    $("feedback-hint-text").textContent = feedback.hint;
  } else {
    $("feedback-hint-wrap").classList.add("hidden");
  }
}

function renderSteps(steps) {
  if (!steps || !steps.length) return;
  $("steps-section").classList.remove("hidden");
  const list = $("steps-list");
  list.innerHTML = "";
  steps.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    list.appendChild(li);
  });
}

function renderProbBars(probs, topDigit) {
  const container = $("probs-bars");
  container.innerHTML = "";
  const sorted = probs.map((p, i) => ({ digit: i, prob: p }))
    .sort((a, b) => b.prob - a.prob).slice(0, 5);

  sorted.forEach(({ digit, prob }) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    const pct = (prob * 100).toFixed(1);
    row.innerHTML = `
      <span class="prob-label">${digit}</span>
      <div class="prob-bar-wrap">
        <div class="prob-bar ${digit === topDigit ? "top" : ""}" style="width:${pct}%"></div>
      </div>
      <span class="prob-pct">${pct}%</span>
    `;
    container.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════════════
// PRACTICE SECTION
// ════════════════════════════════════════════════════════════════
function initPractice() {
  $("new-problem-btn").addEventListener("click", generateProblem);
  $("check-answer-btn").addEventListener("click", checkAnswer);
  $("answer-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") checkAnswer();
  });
}

async function generateProblem() {
  try {
    const res = await apiPost("/api/problem/generate", {
      level: state.currentLevel,
      type: "auto",
    });
    state.currentProblem = res;
    $("problem-text").textContent = res.expression;
    $("problem-hint-text").textContent = `💡 ${res.hint}`;
    $("practice-level-label").textContent = `${res.level_name}`;
    $("answer-input").value = "";
    $("answer-input").focus();
    $("answer-feedback").classList.add("hidden");
    $("answer-feedback").textContent = "";
  } catch (err) {
    $("problem-text").textContent = "Could not load problem. Is the server running?";
  }
}

async function checkAnswer() {
  const input = $("answer-input").value.trim();
  if (!input || !state.currentProblem) return;

  const userAnswer = parseFloat(input);
  const correct = state.currentProblem.answer;
  const isCorrect = Math.abs(userAnswer - correct) < 0.01;

  const feedbackEl = $("answer-feedback");
  feedbackEl.classList.remove("hidden", "correct", "wrong");

  state.total++;

  if (isCorrect) {
    state.correct++;
    state.streak++;
    state.xp += 1;

    feedbackEl.classList.add("correct");
    feedbackEl.textContent = getPositiveMessage() + ` The answer is ${correct}! 🎉`;

    // XP / Level up
    updateStats();

    const accuracy = state.correct / state.total;
    if (state.xp >= state.xpToNext) {
      await tryLevelUp(accuracy);
    }

    // Next problem after 1.5s
    setTimeout(() => generateProblem(), 1500);

  } else {
    state.streak = 0;
    feedbackEl.classList.add("wrong");
    feedbackEl.textContent = `Almost! The answer is ${correct}. Try again next time! 💪`;
    updateStats();
    setTimeout(() => generateProblem(), 2500);
  }
}

async function tryLevelUp(accuracy) {
  try {
    const res = await apiPost("/api/adaptive", {
      accuracy,
      streak: state.streak,
      current_level: state.currentLevel,
      avg_time_sec: 15,
    });

    if (res.next_level !== state.currentLevel) {
      const oldLevel = state.currentLevel;
      state.currentLevel = res.next_level;
      state.xp = 0;
      state.xpToNext = 10 + state.currentLevel * 5;
      localStorage.setItem("mathmagic_level", state.currentLevel.toString());
      updatePlayerBadge();
      generateProblem();

      if (res.next_level > oldLevel) {
        showToast("🏆", "Level Up!", res.message);
      } else {
        showToast("💪", "Practice Mode", res.message);
      }
    } else {
      state.xp = 0;
      state.xpToNext = 10 + state.currentLevel * 5;
    }
    updateStats();
  } catch (e) {
    console.warn("Adaptive API error:", e);
  }
}

function updateStats() {
  $("stat-correct").textContent = state.correct;
  $("stat-streak").textContent = state.streak + (state.streak >= 3 ? " 🔥" : "");
  $("stat-accuracy").textContent = state.total
    ? `${Math.round((state.correct / state.total) * 100)}%`
    : "—";
  $("stat-level").textContent = state.currentLevel;
  const xpPct = Math.min((state.xp / state.xpToNext) * 100, 100);
  $("xp-bar").style.width = `${xpPct}%`;
  $("xp-label").textContent = `${state.xp} / ${state.xpToNext} XP to next level`;
}

function getPositiveMessage() {
  const messages = [
    "Amazing! 🌟", "Brilliant! ⭐", "Superstar! 🚀", "You got it! ✨",
    "Excellent! 🎯", "Fantastic! 🏆", "Well done! 💫", "Perfect! 🎉",
  ];
  return messages[Math.floor(Math.random() * messages.length)];
}

// ════════════════════════════════════════════════════════════════
// TOAST
// ════════════════════════════════════════════════════════════════
function showToast(emoji, title, msg) {
  const toast = $("achievement-toast");
  $("toast-emoji").textContent = emoji;
  $("toast-title").textContent = title;
  $("toast-msg").textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 4000);
}

// ════════════════════════════════════════════════════════════════
// API HELPER
// ════════════════════════════════════════════════════════════════
async function apiPost(endpoint, body) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}
