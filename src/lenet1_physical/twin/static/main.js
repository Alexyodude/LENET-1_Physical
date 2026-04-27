import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { setupArchOverlay } from "./arch.js";
import { setupArchPanel } from "./panel.js";
import { setupPhysicalBox } from "./box.js";
import { setupFaultControls } from "./faults.js";
import { createHistoryStore } from "./history-store.js";
import { setupHistoryUI } from "./history-ui.js";
import { setupDrawingCanvas } from "./draw.js";
import { setupCameraPresets } from "./camera-presets.js";
import { setupMobileDrawer } from "./mobile-drawer.js";
import { setupFrameDebounce, setupLazyMobile } from "./lazy-load.js";
import { paintSlice, markDirty, flushDirtyTo } from "./slice-render.js";
import { setupPerfOverlay } from "./perf-overlay.js";

window.twinEvents = window.twinEvents || new EventTarget();

// ── Constants ────────────────────────────────────────────────────────────────

const LAYER_META = {
  L1: { label: "L1 INPUT",  desc: "28×28 ×1 • 784 LEDs",   color: "#ffdc9a", threeColor: new THREE.Color(1.0, 0.86, 0.60) },
  L2: { label: "L2 CONV1",  desc: "24×24 ×4 • 2304 LEDs",  color: "#4488ff", threeColor: new THREE.Color(0.0, 0.0,  1.0 ) },
  L3: { label: "L3 POOL1",  desc: "12×12 ×4 • 576 LEDs",   color: "#00ffff", threeColor: new THREE.Color(0.0, 1.0,  1.0 ) },
  L4: { label: "L4 CONV2",  desc: "8×8 ×12 • 768 LEDs",    color: "#00e676", threeColor: new THREE.Color(0.0, 1.0,  0.0 ) },
  L5: { label: "L5 POOL2",  desc: "4×4 ×12 • 192 LEDs",    color: "#ffee58", threeColor: new THREE.Color(1.0, 0.93, 0.0 ) },
  L6: { label: "L6 OUTPUT", desc: "1×10 • 10 LEDs",         color: "#ff4444", threeColor: new THREE.Color(1.0, 0.0,  0.0 ) },
};

const LERP_SPEED = 0.15; // fraction per 16ms frame → ~150ms to reach target

// ── State ────────────────────────────────────────────────────────────────────

let mappingData = null;

// Per-LED color state: key = "chain:pos"
const currentColor = new Map();   // Map<string, THREE.Color>  — live animated value
const targetColor  = new Map();   // Map<string, THREE.Color>  — latest commanded value

// Per-LED metadata for tooltip: key = "chain:pos" → {chain, pos, layer, fmap, row, col}
const ledMeta = new Map();

// Three.js LED objects: key = "chain:pos" → Mesh
const ledMeshes = new Map();

// Slice canvases per layer+fmap
const sliceCtx = new Map(); // "L2:1" → {canvas, cols, rows}

let lastSeq = 0;
let fpsCounter = 0;
let fpsLast = performance.now();
let currentLayer = "--";
let predictedDigit = null;

// ── Three.js setup ───────────────────────────────────────────────────────────

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080c10);

// Subtle grid plane for depth
const gridHelper = new THREE.GridHelper(2000, 40, 0x1e2d3d, 0x111820);
gridHelper.position.y = -40;
scene.add(gridHelper);

const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 12000);
camera.position.set(1300, 500, 1800);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 100;
controls.maxDistance = 4000;
controls.target.set(0, 0, 850);

// Ambient + directional for the grid/labels; LEDs are emissive so unaffected
scene.add(new THREE.AmbientLight(0xffffff, 0.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.3);
dirLight.position.set(500, 800, 600);
scene.add(dirLight);

// Shared LED geometry (small sphere)
const LED_RADIUS = 3.2;
const ledGeom = new THREE.SphereGeometry(LED_RADIUS, 6, 5);

// Label sprites pool (reuse canvas textures)
const labelSprites = [];

// ── Legend ───────────────────────────────────────────────────────────────────

function buildLegend() {
  const container = document.getElementById("layer-legend");
  for (const [key, meta] of Object.entries(LAYER_META)) {
    const row = document.createElement("div");
    row.className = "legend-row";
    const swatch = document.createElement("div");
    swatch.className = "legend-swatch";
    swatch.style.background = meta.color;
    const label = document.createElement("div");
    label.className = "legend-label";
    label.textContent = `${meta.label} — ${meta.desc}`;
    row.appendChild(swatch);
    row.appendChild(label);
    container.appendChild(row);
  }
}

// ── Mapping → LED placement ──────────────────────────────────────────────────

function mmToScene(x, y, z) {
  // Map mm coords to Three.js scene units (1mm = 1 unit)
  // z_mm is the depth axis (front-to-back), map to Three.js Z
  return new THREE.Vector3(x, y, z);
}

function snakePosition(row, col, cols, rows = null, order = "row_major_snake") {
  if (order === "column_major_snake") {
    // Up-and-down zigzag: col 0 top-to-bottom, col 1 bottom-to-top, ...
    if (col % 2 === 0) return col * rows + row;
    return col * rows + (rows - 1 - row);
  }
  if (row % 2 === 0) return row * cols + col;
  return row * cols + (cols - 1 - col);
}

function placeLEDsFromMapping(mapping) {
  mappingData = mapping;

  for (const [layerName, layerData] of Object.entries(mapping.layers || {})) {
    const meta = LAYER_META[layerName];
    if (!meta) continue;

    for (const fm of layerData.feature_maps || []) {
      const ox = fm.origin_mm[0];
      const oy = fm.origin_mm[1] || 0;
      const oz = fm.origin_mm[2];
      const px = fm.pitch_mm[0];
      const py = fm.pitch_mm[1];
      const rows = fm.rows;
      const cols = fm.cols;
      const chainId = fm.chain_id;
      const offset = fm.offset_in_chain;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const snakeIdx = snakePosition(r, c, cols, rows, fm.order);
          const pos = offset + snakeIdx;
          const key = `${chainId}:${pos}`;

          const wx = ox + c * px;
          const wy = oy + r * py;
          const wz = oz;

          const mat = new THREE.MeshBasicMaterial({
            color: new THREE.Color(0x050810),
          });
          const mesh = new THREE.Mesh(ledGeom, mat);
          mesh.position.copy(mmToScene(wx, wy, wz));
          mesh.userData = { key, chain: chainId, position: pos, layer: layerName, fmap: fm.id, row: r, col: c };
          scene.add(mesh);
          ledMeshes.set(key, mesh);

          currentColor.set(key, new THREE.Color(0x050810));
          targetColor.set(key, new THREE.Color(0x050810));
          ledMeta.set(key, { chain: chainId, position: pos, layer: layerName, fmap: fm.id, row: r, col: c });
        }
      }
    }

    // Add a floating layer label above the slab
    addLayerLabel(layerName, mapping.layers[layerName]);
  }
}

function placeLEDsFallback() {
  // Fallback: chain-position grid when no mapping available
  // Place by chain (Z) and position (X)
  // We still need to render whatever comes in via WS
}

function addLayerLabel(layerName, layerData) {
  const meta = LAYER_META[layerName];
  if (!meta || !layerData.feature_maps?.length) return;

  // Find bounding box of this layer's LEDs
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (const fm of layerData.feature_maps) {
    const ox = fm.origin_mm[0];
    const oy = fm.origin_mm[1] || 0;
    const oz = fm.origin_mm[2];
    const endX = ox + (fm.cols - 1) * fm.pitch_mm[0];
    const endY = oy + (fm.rows - 1) * fm.pitch_mm[1];
    minX = Math.min(minX, ox); maxX = Math.max(maxX, endX);
    minY = Math.min(minY, oy); maxY = Math.max(maxY, endY);
    minZ = Math.min(minZ, oz); maxZ = Math.max(maxZ, oz);
  }

  const cx = (minX + maxX) / 2;
  const topY = maxY + 25;
  const cz = (minZ + maxZ) / 2;

  // Canvas sprite for the label
  const c = document.createElement("canvas");
  c.width = 512; c.height = 80;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, 512, 80);

  // Color bar
  ctx.fillStyle = meta.color;
  ctx.fillRect(0, 0, 512, 4);

  ctx.fillStyle = "rgba(8,12,16,0.75)";
  ctx.fillRect(0, 4, 512, 76);

  ctx.fillStyle = meta.color;
  ctx.font = "bold 22px 'Courier New', monospace";
  ctx.fillText(meta.label, 12, 32);

  ctx.fillStyle = "rgba(200,220,240,0.6)";
  ctx.font = "16px 'Courier New', monospace";
  ctx.fillText(meta.desc, 12, 56);

  const texture = new THREE.CanvasTexture(c);
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(120, 19, 1);
  sprite.position.set(cx, topY, cz);
  scene.add(sprite);
  labelSprites.push(sprite);
}

// ── Slice panels ─────────────────────────────────────────────────────────────

function ensureSlice(layer, fmapId, rows, cols) {
  const key = `${layer}:${fmapId}`;
  if (sliceCtx.has(key)) return sliceCtx.get(key);

  let group = document.getElementById(`slice-group-${layer}`);
  if (!group) {
    const slicesEl = document.getElementById("slices");
    group = document.createElement("div");
    group.className = "slice-group";
    group.id = `slice-group-${layer}`;

    const meta = LAYER_META[layer] || {};
    const groupLabel = document.createElement("div");
    groupLabel.className = "slice-group-label";
    groupLabel.textContent = meta.label || layer;
    groupLabel.style.color = meta.color || "#c9d8e8";
    group.appendChild(groupLabel);

    const fmapsWrap = document.createElement("div");
    fmapsWrap.className = "slice-fmaps";
    fmapsWrap.id = `slice-fmaps-${layer}`;
    group.appendChild(fmapsWrap);

    slicesEl.appendChild(group);
  }

  const fmapsWrap = document.getElementById(`slice-fmaps-${layer}`);
  const item = document.createElement("div");
  item.className = "slice-item";

  const pixelSize = Math.max(3, Math.min(8, Math.floor(96 / Math.max(rows, cols))));
  const cvs = document.createElement("canvas");
  cvs.width = cols * pixelSize;
  cvs.height = rows * pixelSize;
  const ctx2d = cvs.getContext("2d");
  ctx2d.fillStyle = "#050810";
  ctx2d.fillRect(0, 0, cvs.width, cvs.height);

  const fmLabel = document.createElement("div");
  fmLabel.className = "slice-fmap-label";
  fmLabel.textContent = `FM${fmapId}`;

  item.appendChild(cvs);
  item.appendChild(fmLabel);
  fmapsWrap.appendChild(item);

  const entry = { canvas: cvs, ctx: ctx2d, w: cvs.width, h: cvs.height, cols, rows, pixelSize };
  sliceCtx.set(key, entry);
  return entry;
}

// ── WebSocket ────────────────────────────────────────────────────────────────

const wsStatusEl  = document.getElementById("ws-status");
const seqEl       = document.getElementById("seq");
const fpsEl       = document.getElementById("fps");
const layerEl     = document.getElementById("current-layer");
const predOverlay = document.getElementById("prediction-overlay");

// Track per-(layer,fmap) pixel data for slice updates
// key "L2:1" → Uint8ClampedArray(rows*cols*3) of latest RGB
const slicePixelData = new Map();

let staticModeActive = false;

async function probeBackend() {
  // GitHub Pages and similar static hosts have no FastAPI server. Probe /healthz
  // with a short timeout; if it doesn't respond, switch to in-browser inference.
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const r = await fetch("./healthz", { signal: ctrl.signal, cache: "no-store" });
    clearTimeout(t);
    return r.ok;
  } catch (_) {
    return false;
  }
}

async function startStaticFallback() {
  if (staticModeActive) return;
  staticModeActive = true;
  wsStatusEl.textContent = "STATIC";
  wsStatusEl.className = "status-val status-connected";
  try {
    const mod = await import("./static-mode-v2.js?v=cms");
    // Bridge: route static-mode frames through handleFrame too so seq/fps update.
    window.twinEvents.addEventListener("frame", (ev) => handleFrame(ev.detail));
    await mod.startStaticMode();
  } catch (e) {
    console.error("[main] static fallback failed:", e);
    wsStatusEl.textContent = "OFFLINE";
    wsStatusEl.className = "status-val status-disconnected";
  }
}

function connectWS() {
  if (staticModeActive) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  let failed = false;

  ws.addEventListener("open", () => {
    wsStatusEl.textContent = "CONNECTED";
    wsStatusEl.className = "status-val status-connected";
  });

  ws.addEventListener("close", () => {
    if (staticModeActive) return;
    wsStatusEl.textContent = "DISCONNECTED";
    wsStatusEl.className = "status-val status-disconnected";
    if (failed) {
      probeBackend().then((alive) => alive ? setTimeout(connectWS, 2000) : startStaticFallback());
    } else {
      setTimeout(connectWS, 2000);
    }
  });

  ws.addEventListener("error", () => { failed = true; ws.close(); });

  ws.addEventListener("message", (ev) => {
    const f = JSON.parse(ev.data);
    handleFrame(f);
    window.twinEvents.dispatchEvent(new CustomEvent("frame", { detail: f }));
  });
}

async function startNetworking() {
  const alive = await probeBackend();
  if (alive) connectWS();
  else startStaticFallback();
}

function handleFrame(f) {
  lastSeq = f.seq;
  currentLayer = f.layer;
  seqEl.textContent = f.seq;
  layerEl.textContent = f.layer;

  fpsCounter++;

  // Update target colors for each delta
  for (const [chain, position, r, g, b] of f.deltas) {
    const key = `${chain}:${position}`;

    // Ensure mesh exists (fallback: place by chain/pos if no mapping loaded)
    if (!ledMeshes.has(key)) {
      ensureFallbackLED(chain, position);
    }

    let tc = targetColor.get(key);
    if (!tc) { tc = new THREE.Color(); targetColor.set(key, tc); }
    tc.setRGB(r / 255, g / 255, b / 255);

    if (!currentColor.has(key)) {
      currentColor.set(key, new THREE.Color(0x050810));
    }

    // Update slice pixel buffers
    const meta = ledMeta.get(key);
    if (meta) {
      const sliceKey = `${meta.layer}:${meta.fmap}`;
      // We need rows/cols — get from mapping if available
      if (mappingData) {
        const layerFms = mappingData.layers?.[meta.layer]?.feature_maps;
        if (layerFms) {
          const fm = layerFms.find(fm => fm.id === meta.fmap);
          if (fm) {
            ensureSlice(meta.layer, meta.fmap, fm.rows, fm.cols);
            let buf = slicePixelData.get(sliceKey);
            if (!buf) {
              buf = new Uint8ClampedArray(fm.rows * fm.cols * 3);
              slicePixelData.set(sliceKey, buf);
            }
            // Slice canvas is rendered row-major; index by (row, col), NOT
            // the physical chain order. (Decoupling slice layout from wiring
            // means slice panels stay readable regardless of snake direction.)
            const idx = meta.row * fm.cols + meta.col;
            buf[idx * 3]     = r;
            buf[idx * 3 + 1] = g;
            buf[idx * 3 + 2] = b;
            markDirty(meta.layer, meta.fmap);
          }
        }
      }
    }
  }

  // L6 prediction display
  if (f.layer === "L6") {
    updatePrediction(f.deltas);
  }

  // Flush dirty slice canvases via slice-render. The slice canvas entry
  // (sliceCtx) and pixel buffer (slicePixelData) live in separate maps, so
  // build a merged view per dirty key.
  const merged = new Map();
  for (const [key, entry] of sliceCtx) {
    const buf = slicePixelData.get(key);
    if (buf) merged.set(key, { canvas: entry.canvas, rows: entry.rows, cols: entry.cols, buf });
  }
  flushDirtyTo(merged);
}

function ensureFallbackLED(chain, position) {
  const key = `${chain}:${position}`;
  const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0x050810) });
  const mesh = new THREE.Mesh(ledGeom, mat);
  mesh.position.set(position * 5, 0, chain * 25);
  mesh.userData = { key, chain, position, layer: "?", fmap: 0, row: 0, col: position };
  scene.add(mesh);
  ledMeshes.set(key, mesh);
  currentColor.set(key, new THREE.Color(0x050810));
  targetColor.set(key, new THREE.Color(0x050810));
  ledMeta.set(key, { chain, position, layer: "?", fmap: 0, row: 0, col: position });
}

// flushSlices replaced by flushDirtyTo(sliceCtx) from slice-render.js

function updatePrediction(deltas) {
  if (!deltas.length) return;
  let maxVal = -1, digit = 0;
  for (const [chain, position, r, g, b] of deltas) {
    const brightness = (r + g + b) / 3;
    if (brightness > maxVal) { maxVal = brightness; digit = position; }
  }
  predictedDigit = digit;
  predOverlay.textContent = digit;
  predOverlay.classList.add("visible");
}

// ── FPS counter ───────────────────────────────────────────────────────────────

function updateFPS() {
  const now = performance.now();
  const elapsed = now - fpsLast;
  if (elapsed >= 1000) {
    fpsEl.textContent = Math.round(fpsCounter * 1000 / elapsed);
    fpsCounter = 0;
    fpsLast = now;
  }
}

// ── Tooltip via raycasting ────────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-9999, -9999);
const tooltipEl = document.getElementById("tooltip");
let hoveredKey = null;

canvas.addEventListener("mousemove", (ev) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((ev.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;

  tooltipEl.style.left = `${ev.clientX + 14}px`;
  tooltipEl.style.top  = `${ev.clientY - 10}px`;
});

canvas.addEventListener("mouseleave", () => {
  tooltipEl.classList.remove("visible");
  hoveredKey = null;
});

function checkHover() {
  raycaster.setFromCamera(mouse, camera);
  const meshArr = Array.from(ledMeshes.values());
  const hits = raycaster.intersectObjects(meshArr, false);

  if (hits.length > 0) {
    const hit = hits[0];
    const { key, chain, position, layer, fmap, row, col } = hit.object.userData;

    if (key !== hoveredKey) {
      hoveredKey = key;
      const cur = currentColor.get(key) || new THREE.Color();
      const ri = Math.round(cur.r * 255);
      const gi = Math.round(cur.g * 255);
      const bi = Math.round(cur.b * 255);
      tooltipEl.textContent =
        `chain    ${chain}\n` +
        `position ${position}\n` +
        `layer    ${layer}   fmap ${fmap}\n` +
        `row      ${row}   col  ${col}\n` +
        `RGB      ${ri} ${gi} ${bi}`;
    }
    tooltipEl.classList.add("visible");
  } else {
    if (hoveredKey !== null) {
      tooltipEl.classList.remove("visible");
      hoveredKey = null;
    }
  }
}

// ── Animation loop ────────────────────────────────────────────────────────────

function resize() {
  const wrap = document.getElementById("scene-wrap");
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);

// ── Panel drag-resize ────────────────────────────────────────────────────
// Right column width is driven by the CSS variable --panel-width.
// Persisted in localStorage. Hidden on mobile via CSS.
function setupPanelResize() {
  const handle = document.getElementById("panel-resizer");
  if (!handle) return;
  const root = document.documentElement;
  const MIN = 180, MAX = 480;

  const stored = parseInt(localStorage.getItem("twinPanelWidth") || "", 10);
  if (Number.isFinite(stored) && stored >= MIN && stored <= MAX) {
    root.style.setProperty("--panel-width", stored + "px");
  }

  let dragging = false;
  let startX = 0;
  let startWidth = 0;

  function getPanelWidth() {
    const v = getComputedStyle(root).getPropertyValue("--panel-width").trim();
    return parseInt(v, 10) || 240;
  }

  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    handle.classList.add("dragging");
    handle.setPointerCapture(e.pointerId);
    startX = e.clientX;
    startWidth = getPanelWidth();
    document.body.style.cursor = "col-resize";
  });

  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    // Dragging left increases width (handle is on the LEFT of the panel).
    const dx = startX - e.clientX;
    let w = Math.max(MIN, Math.min(MAX, startWidth + dx));
    root.style.setProperty("--panel-width", w + "px");
    resize();
  });

  function endDrag() {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    localStorage.setItem("twinPanelWidth", String(getPanelWidth()));
  }
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
  handle.addEventListener("lostpointercapture", endDrag);

  // Double-click resets to default
  handle.addEventListener("dblclick", () => {
    root.style.setProperty("--panel-width", "240px");
    localStorage.removeItem("twinPanelWidth");
    resize();
  });
}
setupPanelResize();

function animate() {
  requestAnimationFrame(animate);

  // Lerp all LED colors toward targets
  for (const [key, cur] of currentColor.entries()) {
    const tgt = targetColor.get(key);
    if (!tgt) continue;
    cur.lerp(tgt, LERP_SPEED);
    const mesh = ledMeshes.get(key);
    if (mesh) mesh.material.color.copy(cur);
  }

  checkHover();
  controls.update();
  updateFPS();
  renderer.render(scene, camera);
}

// ── Controls ──────────────────────────────────────────────────────────────────

document.getElementById("btn-sample").addEventListener("click", () => {
  predOverlay.classList.remove("visible");
  predictedDigit = null;
  fetch("/sample", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
});

document.getElementById("btn-step").addEventListener("click", () => {
  fetch("/step", { method: "POST" });
});

const brightEl  = document.getElementById("brightness");
const brightVal = document.getElementById("brightness-value");
brightEl.addEventListener("input", () => {
  const pct = Number(brightEl.value);
  brightVal.textContent = `${pct}%`;
  fetch("/brightness", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ value: pct / 100 }),
  });
});

document.getElementById("btn-test-pixel").addEventListener("click", () => {
  const chain = Number(document.getElementById("test-chain").value);
  const pos   = Number(document.getElementById("test-pos").value);
  fetch("/test-pixel", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ chain, pos, r: 255, g: 0, b: 0 }),
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  buildLegend();
  resize();

  // Probe the backend ONCE at startup. Modules that depend on backend-only
  // endpoints (history scrubber, fault sim, brightness sync) skip their setup
  // when running statically.
  const backendAlive = await probeBackend();
  window.twinMode = backendAlive ? "live" : "static";

  // Fetch mapping and place LEDs at real mm positions
  let mapping = { layers: {}, chains: [] };
  try {
    let resp = backendAlive ? await fetch("/mapping").catch(() => null) : null;
    if (!resp || !resp.ok) resp = await fetch("./mapping.json");
    if (resp.ok) {
      const data = await resp.json();
      mapping = data;
      if (data.layers && Object.keys(data.layers).length > 0) {
        placeLEDsFromMapping(data);
      } else {
        placeLEDsFallback();
      }
    }
  } catch {
    placeLEDsFallback();
  }

  window.twin = { scene, camera, ledMeshes, mapping };

  if (backendAlive) {
    try {
      const resp = await fetch("/brightness");
      if (resp.ok) {
        const data = await resp.json();
        const pct = Math.round(data.brightness * 100);
        brightEl.value = pct;
        brightVal.textContent = `${pct}%`;
      }
    } catch { /* ignore */ }
  }

  // Always-on visual modules
  setupArchOverlay(scene, mapping);
  setupPhysicalBox(scene, mapping);

  const archPanelHost = document.getElementById("arch-panel-host");
  if (archPanelHost) setupArchPanel(archPanelHost);

  // Drawing canvas
  const drawHost = document.getElementById("draw-host");
  if (drawHost) setupDrawingCanvas(drawHost);

  // Camera presets
  const cameraPresetsHost = document.getElementById("camera-presets-host");
  if (cameraPresetsHost) setupCameraPresets(cameraPresetsHost, camera, controls);

  // Performance overlay
  setupPerfOverlay(document.body);

  // Frame debounce
  const { dispatchCoalescedFrame } = setupFrameDebounce();
  window._dispatchCoalescedFrame = dispatchCoalescedFrame;

  // Backend-dependent modules: only when live
  if (backendAlive) {
    const faultHost = document.getElementById("fault-panel-host");
    if (faultHost) setupFaultControls(faultHost);

    const historyHost = document.getElementById("history-panel-host");
    if (historyHost) {
      const store = createHistoryStore();
      store.listRecords = store.refreshList;
      setupHistoryUI(historyHost, store, ledMeshes);
    }

    setupLazyMobile({
      loadFaults: () => {
        const h = document.getElementById("fault-panel-host");
        if (h) setupFaultControls(h);
        return Promise.resolve();
      },
      loadHistory: () => {
        const h = document.getElementById("history-panel-host");
        if (h) {
          const store = createHistoryStore();
          store.listRecords = store.refreshList;
          setupHistoryUI(h, store, ledMeshes);
        }
        return Promise.resolve();
      },
    });
  } else {
    // Hide hosts so they don't claim layout space.
    for (const id of ["fault-panel-host", "history-panel-host"]) {
      const el = document.getElementById(id);
      if (el) el.style.display = "none";
    }
    // Disable buttons that POST to nonexistent endpoints.
    for (const id of ["btn-sample", "btn-step", "btn-test-pixel"]) {
      const el = document.getElementById(id);
      if (el) { el.disabled = true; el.title = "disabled in static mode"; }
    }

    setupLazyMobile({
      loadFaults: () => Promise.resolve(),
      loadHistory: () => Promise.resolve(),
    });
  }

  if (backendAlive) connectWS();
  else startStaticFallback();
  animate();

  // Mobile drawer must be called last, after all panels exist
  setupMobileDrawer();
}

init();
