/**
 * Alzheimer's Disease MRI Classifier - Client JS Engine
 * Handles exact layout interactions, canvas rendering, zoom/pan controls, preset samples, API calls, and history.
 */

document.addEventListener("DOMContentLoaded", () => {
  const state = {
    currentImage: null,
    heatmapImage: null,
    viewMode: "original",
    zoom: 1.0,
    panX: 0,
    panY: 0,
    rotation: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    currentResult: null,
    samples: [],
    history: JSON.parse(localStorage.getItem("alzheimer_history") || "[]"),
  };

  // DOM Elements
  const canvas = document.getElementById("mriCanvas");
  const ctx = canvas.getContext("2d");
  const canvasViewport = document.getElementById("canvasViewport");
  const canvasEmptyMsg = document.getElementById("canvasEmptyMsg");
  const fileDropzone = document.getElementById("fileDropzone");
  const fileInput = document.getElementById("fileInput");

  // Initializing App
  init();

  function init() {
    setupCanvasControls();
    setupDropzone();
    setupNavTabs();
    fetchSamples();
    renderHistory();
  }

  // --- CANVAS RENDERING & CONTROLS ---
  function drawCanvas() {
    if (!state.currentImage) {
      canvasEmptyMsg.style.display = "flex";
      return;
    }
    canvasEmptyMsg.style.display = "none";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.translate(centerX + state.panX, centerY + state.panY);
    ctx.rotate((state.rotation * Math.PI) / 180);
    ctx.scale(state.zoom, state.zoom);

    const imgW = state.currentImage.width;
    const imgH = state.currentImage.height;
    const drawX = -imgW / 2;
    const drawY = -imgH / 2;

    if (state.viewMode === "original" || !state.heatmapImage) {
      ctx.drawImage(state.currentImage, drawX, drawY, imgW, imgH);
    } else if (state.viewMode === "heatmap") {
      ctx.drawImage(state.heatmapImage, drawX, drawY, imgW, imgH);
    } else if (state.viewMode === "overlay") {
      ctx.drawImage(state.currentImage, drawX, drawY, imgW, imgH);
      ctx.globalAlpha = 0.65;
      ctx.drawImage(state.heatmapImage, drawX, drawY, imgW, imgH);
      ctx.globalAlpha = 1.0;
    }

    ctx.restore();
  }

  function resetView() {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    state.rotation = 0;
    document.getElementById("zoomSlider").value = 100;
    document.getElementById("zoomValue").textContent = "100%";
    drawCanvas();
  }

  function setupCanvasControls() {
    const zoomSlider = document.getElementById("zoomSlider");
    const zoomValue = document.getElementById("zoomValue");

    zoomSlider.addEventListener("input", (e) => {
      state.zoom = e.target.value / 100;
      zoomValue.textContent = `${e.target.value}%`;
      drawCanvas();
    });

    document.getElementById("btnZoomIn").addEventListener("click", () => {
      let val = Math.min(parseInt(zoomSlider.value) + 25, 300);
      zoomSlider.value = val;
      state.zoom = val / 100;
      zoomValue.textContent = `${val}%`;
      drawCanvas();
    });

    document.getElementById("btnZoomOut").addEventListener("click", () => {
      let val = Math.max(parseInt(zoomSlider.value) - 25, 50);
      zoomSlider.value = val;
      state.zoom = val / 100;
      zoomValue.textContent = `${val}%`;
      drawCanvas();
    });

    document.getElementById("btnRotate").addEventListener("click", () => {
      state.rotation = (state.rotation + 90) % 360;
      drawCanvas();
    });

    document.getElementById("btnResetView").addEventListener("click", resetView);

    // View Mode Switcher
    ["btnModeOriginal", "btnModeHeatmap", "btnModeOverlay"].forEach((id) => {
      document.getElementById(id).addEventListener("click", (e) => {
        document.querySelectorAll(".mode-btn").forEach((btn) => btn.classList.remove("active"));
        e.target.classList.add("active");
        state.viewMode = e.target.getAttribute("data-mode");
        drawCanvas();
      });
    });

    // Mouse Pan Dragging
    canvasViewport.addEventListener("mousedown", (e) => {
      if (!state.currentImage) return;
      state.isDragging = true;
      state.dragStartX = e.clientX - state.panX;
      state.dragStartY = e.clientY - state.panY;
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.isDragging) return;
      state.panX = e.clientX - state.dragStartX;
      state.panY = e.clientY - state.dragStartY;
      drawCanvas();
    });

    window.addEventListener("mouseup", () => {
      state.isDragging = false;
    });
  }

  // --- DROPZONE UPLOAD ---
  function setupDropzone() {
    fileDropzone.addEventListener("click", () => fileInput.click());

    fileDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      fileDropzone.style.borderColor = "var(--green-pear)";
    });

    fileDropzone.addEventListener("dragleave", () => {
      fileDropzone.style.borderColor = "var(--green-border-active)";
    });

    fileDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      fileDropzone.style.borderColor = "var(--green-border-active)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const b64Data = e.target.result;
      const img = new Image();
      img.onload = () => {
        state.currentImage = img;
        canvas.width = img.width;
        canvas.height = img.height;
        resetView();

        runPredictionAPI({ image_data: b64Data });
      };
      img.src = b64Data;
    };
    reader.readAsDataURL(file);
  }

  // --- PRESET SAMPLE FETCHING ---
  function fetchSamples() {
    fetch("/api/samples")
      .then((res) => res.json())
      .then((data) => {
        state.samples = data.samples;
        renderSampleList(data.samples);
      })
      .catch((err) => console.error("Error fetching samples:", err));
  }

  function renderSampleList(samples) {
    const sampleList = document.getElementById("sampleList");
    const filmstripCarousel = document.getElementById("filmstripCarousel");
    sampleList.innerHTML = "";
    filmstripCarousel.innerHTML = "";

    // Group 1 sample per class for the left selection menu
    const classGroups = {};
    samples.forEach((s) => {
      if (!classGroups[s.class_key]) classGroups[s.class_key] = s;
      
      // Filmstrip Carousel Thumbnail
      const thumb = document.createElement("img");
      thumb.className = "filmstrip-thumb";
      thumb.src = s.path;
      thumb.alt = s.class_label;
      thumb.addEventListener("click", () => loadSample(s.path));
      filmstripCarousel.appendChild(thumb);
    });

    Object.values(classGroups).forEach((s) => {
      const row = document.createElement("div");
      row.className = "sample-row";

      let dotClass = "dot-non";
      if (s.class_key === "very_mild_demented") dotClass = "dot-verymild";
      if (s.class_key === "mild_demented") dotClass = "dot-mild";
      if (s.class_key === "moderate_demented") dotClass = "dot-moderate";

      row.innerHTML = `
        <img src="${s.path}" alt="${s.class_label}">
        <div class="sample-row-info">
          <div class="sample-row-name">
            <span class="${dotClass}"></span> ${s.class_label}
          </div>
          <div class="sample-row-desc">${getStageSub(s.class_label)}</div>
        </div>
      `;

      row.addEventListener("click", () => {
        document.querySelectorAll(".sample-row").forEach((r) => r.classList.remove("active"));
        row.classList.add("active");
        loadSample(s.path);
      });

      sampleList.appendChild(row);
    });

    // Auto-load first sample by default
    if (samples.length > 0) {
      loadSample(samples[0].path);
    }
  }

  function getStageSub(label) {
    if (label.includes("Non")) return "Healthy Control";
    if (label.includes("Very Mild")) return "Early Stage";
    if (label.includes("Mild")) return "Middle Stage";
    if (label.includes("Moderate")) return "Advanced Stage";
    return "";
  }

  function loadSample(path) {
    const img = new Image();
    img.onload = () => {
      state.currentImage = img;
      canvas.width = img.width;
      canvas.height = img.height;
      resetView();

      runPredictionAPI({ sample_path: path });
    };
    img.src = path;
  }

  // --- API PREDICTION RUNNER ---
  function runPredictionAPI(payload) {
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((res) => res.json())
      .then((data) => {
        state.currentResult = data;

        if (data.heatmap_base64) {
          const hm = new Image();
          hm.onload = () => {
            state.heatmapImage = hm;
            drawCanvas();
          };
          hm.src = data.heatmap_base64;
        }

        updatePredictionUI(data);
        addToHistory(data);
      })
      .catch((err) => console.error("Prediction Error:", err));
  }

  function updatePredictionUI(data) {
    document.getElementById("resStageName").textContent = data.predicted_class;
    document.getElementById("resStageSub").textContent = getStageSub(data.predicted_class);
    document.getElementById("resConfVal").textContent = `${data.confidence_percentage}%`;

    document.getElementById("confQTitle").textContent = `Confidence Quality: ${data.confidence_quality}`;
    document.getElementById("confQDesc").textContent = data.confidence_description;

    // Update Progress Bars
    data.decision_scores.forEach((item) => {
      if (item.class_key === "non_demented") {
        document.getElementById("pctNon").textContent = `${item.percentage}%`;
        document.getElementById("barNon").style.width = `${item.percentage}%`;
      } else if (item.class_key === "very_mild_demented") {
        document.getElementById("pctVeryMild").textContent = `${item.percentage}%`;
        document.getElementById("barVeryMild").style.width = `${item.percentage}%`;
      } else if (item.class_key === "mild_demented") {
        document.getElementById("pctMild").textContent = `${item.percentage}%`;
        document.getElementById("barMild").style.width = `${item.percentage}%`;
      } else if (item.class_key === "moderate_demented") {
        document.getElementById("pctModerate").textContent = `${item.percentage}%`;
        document.getElementById("barModerate").style.width = `${item.percentage}%`;
      }
    });
  }

  // --- HISTORY MANAGEMENT ---
  function addToHistory(res) {
    const item = {
      stage: res.predicted_class,
      confidence: res.confidence_percentage,
      quality: res.confidence_quality.includes("High") ? "High" : "Moderate",
      time: "Just Now",
    };
    state.history.unshift(item);
    if (state.history.length > 5) state.history.pop();
    localStorage.setItem("alzheimer_history", JSON.stringify(state.history));
    renderHistory();
  }

  function renderHistory() {
    const historyList = document.getElementById("historyList");
    if (state.history.length === 0) return;

    historyList.innerHTML = "";
    state.history.forEach((h) => {
      const item = document.createElement("div");
      item.className = "history-item";
      let nameClass = "name-mild";
      if (h.stage.includes("Non")) nameClass = "name-non";
      if (h.stage.includes("Very Mild")) nameClass = "name-verymild";
      if (h.stage.includes("Moderate")) nameClass = "name-moderate";

      item.innerHTML = `
        <div class="h-info">
          <span class="h-name ${nameClass}">${h.stage}</span>
          <span class="h-date">${h.time}</span>
        </div>
        <span class="h-pct">${h.confidence}%</span>
        <span class="h-badge ${h.quality === 'High' ? 'badge-high' : 'badge-mod'}">${h.quality}</span>
      `;
      historyList.appendChild(item);
    });
  }

  // --- TAB HELPERS ---
  function setupNavTabs() {
    document.querySelectorAll(".nav-tab").forEach((tab) => {
      tab.addEventListener("click", (e) => {
        document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
      });
    });

    document.querySelectorAll(".sidebar-item").forEach((item) => {
      item.addEventListener("click", () => {
        document.querySelectorAll(".sidebar-item").forEach((i) => i.classList.remove("active"));
        item.classList.add("active");
      });
    });

    document.getElementById("btnDownloadPDF").addEventListener("click", () => {
      window.print();
    });
  }
});
