// Drawing canvas panel — lets the user paint a digit and run it through LeNet.
// Exported function is called by main.js (worker-7 wires it up).
// Inference routing:
//   live mode  → POST /sample-image {image: number[784]}
//   static mode → window.staticInferImage(arr) exposed by static-mode-v2
//                 Falls back silently if hook not yet available.

const CANVAS_SIZE = 280;
const MODEL_SIZE  = 28;
const BRUSH_RADIUS = 11; // px on the 280×280 surface

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "cls") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}

function txt(s) { return document.createTextNode(s); }

export function setupDrawingCanvas(rootEl) {
  const titleSpan  = el("span", { cls: "draw-title" });
  titleSpan.textContent = "DRAW A DIGIT";
  const header = el("div", { cls: "draw-header" }, titleSpan);

  const cvs = el("canvas", {
    id: "draw-canvas",
    width: String(CANVAS_SIZE),
    height: String(CANVAS_SIZE),
    "aria-label": "Draw a digit here",
  });

  const clearBtn   = el("button", { id: "draw-clear",   cls: "btn-secondary draw-btn" });
  clearBtn.textContent = "CLEAR";
  const predictBtn = el("button", { id: "draw-predict", cls: "btn-primary draw-btn" });
  predictBtn.textContent = "PREDICT";
  const controls = el("div", { cls: "draw-controls" }, clearBtn, predictBtn);

  const resultEl = el("div", { cls: "draw-result" });
  resultEl.setAttribute("aria-live", "polite");

  const body  = el("div", { cls: "draw-body" }, cvs, controls, resultEl);
  const panel = el("div", { cls: "draw-panel" }, header, body);

  rootEl.appendChild(panel);

  // ── Canvas drawing setup ────────────────────────────────────────────────────

  const ctx = cvs.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  let painting = false;
  let lastX = 0, lastY = 0;

  function getPos(ev) {
    const rect = cvs.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    const src = ev.touches ? ev.touches[0] : ev;
    return [
      (src.clientX - rect.left) * scaleX,
      (src.clientY - rect.top)  * scaleY,
    ];
  }

  function paint(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    // Soft glow for anti-aliased feel
    const grad = ctx.createRadialGradient(x, y, BRUSH_RADIUS * 0.4, x, y, BRUSH_RADIUS * 1.4);
    grad.addColorStop(0, "rgba(255,255,255,0.25)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS * 1.4, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
  }

  function stroke(x, y) {
    const dx = x - lastX, dy = y - lastY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.max(1, Math.ceil(dist / 4));
    for (let i = 0; i <= steps; i++) {
      paint(lastX + (dx * i) / steps, lastY + (dy * i) / steps);
    }
    lastX = x; lastY = y;
  }

  cvs.addEventListener("pointerdown", (ev) => {
    painting = true;
    cvs.setPointerCapture(ev.pointerId);
    const [x, y] = getPos(ev);
    lastX = x; lastY = y;
    paint(x, y);
  });
  cvs.addEventListener("pointermove", (ev) => {
    if (!painting) return;
    const [x, y] = getPos(ev);
    stroke(x, y);
  });
  cvs.addEventListener("pointerup",     () => { painting = false; });
  cvs.addEventListener("pointercancel", () => { painting = false; });

  // Prevent page scroll while drawing on touch
  cvs.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
  cvs.addEventListener("touchmove",  (e) => e.preventDefault(), { passive: false });

  clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    resultEl.textContent = "";
    resultEl.className = "draw-result";
  });

  predictBtn.addEventListener("click", () => {
    runPredict(cvs, resultEl);
  });
}

// ── Inference helpers ─────────────────────────────────────────────────────────

function downsample28(srcCanvas) {
  const tmp  = document.createElement("canvas");
  tmp.width  = MODEL_SIZE;
  tmp.height = MODEL_SIZE;
  const tctx = tmp.getContext("2d");
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = "high";
  tctx.drawImage(srcCanvas, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const px  = tctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const arr = new Array(MODEL_SIZE * MODEL_SIZE);
  for (let i = 0; i < arr.length; i++) arr[i] = px[i * 4]; // R byte 0-255
  return arr;
}

async function runPredict(srcCanvas, resultEl) {
  resultEl.textContent = "\u2026"; // ellipsis
  resultEl.className = "draw-result draw-result--pending";

  const image28 = downsample28(srcCanvas);

  try {
    let digit;
    if (window.twinMode === "live") {
      digit = await inferLive(image28);
    } else {
      digit = await inferStatic(image28);
    }
    if (digit !== null && digit !== undefined) {
      resultEl.textContent = "\u2192 " + digit;
      resultEl.className = "draw-result draw-result--ok";
    } else {
      resultEl.textContent = "no result";
      resultEl.className = "draw-result draw-result--err";
    }
  } catch (err) {
    console.error("[draw] predict error:", err);
    resultEl.textContent = "error";
    resultEl.className = "draw-result draw-result--err";
  }
}

async function inferLive(image28) {
  const resp = await fetch("/sample-image", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ image: image28 }),
  });
  if (!resp.ok) throw new Error("/sample-image " + resp.status);
  const data = await resp.json();
  return data.digit ?? null;
}

async function inferStatic(image28) {
  if (typeof window.staticInferImage === "function") {
    return await window.staticInferImage(image28);
  }
  // Hook not yet installed — try live endpoint as best-effort fallback
  try { return await inferLive(image28); } catch { return null; }
}
