// In-browser inference fallback. Activates when no FastAPI backend is reachable
// (e.g. when this site is hosted on GitHub Pages).
//
// Loads lenet5.onnx, runs ONNX Runtime Web inference, and dispatches the same
// `window.twinEvents.frame` CustomEvents the WebSocket would have produced —
// so arch.js, panel.js, history-ui.js, etc. all keep working unchanged.

// ort is loaded as a UMD <script> in index.html, exposing globalThis.ort.
// Fall back to a graceful error if missing.
const ort = globalThis.ort;
if (!ort) {
  console.error("[static-mode] onnxruntime-web global 'ort' not found");
}

const LAYER_THEMES = {
  L1: [255, 220, 180],
  L2: [0,   0,   255],
  L3: [0,   255, 255],
  L4: [0,   255, 0  ],
  L5: [255, 255, 0  ],
  L6: [255, 0,   0  ],
};
const LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6"];
const MNIST_MEAN = 0.1307;
const MNIST_STD  = 0.3081;
const BRIGHTNESS_CAP = 0.3;

let session = null;
let mapping = null;
let samples = null;
let seq = 0;
let stopped = false;

function logicalToPhysical(layer, fmIdx, row, col) {
  const fm = mapping.layers[layer].feature_maps[fmIdx];
  const inFmap = (row % 2 === 0)
    ? row * fm.cols + col
    : row * fm.cols + (fm.cols - 1 - col);
  return [fm.chain_id, fm.offset_in_chain + inFmap];
}

function activationsToFrame(layer, data, dims) {
  // dims: e.g. [1, 4, 24, 24] for L2; reduce batch dim.
  const F = dims[1] ?? 1;
  const H = dims[dims.length - 2] ?? 1;
  const W = dims[dims.length - 1];
  const theme = LAYER_THEMES[layer];
  const deltas = [];

  for (let f = 0; f < F; f++) {
    // Per-fmap max for normalization.
    let max = 0;
    const off = f * H * W;
    for (let i = 0; i < H * W; i++) {
      const v = Math.max(0, data[off + i]);
      if (v > max) max = v;
    }
    if (max === 0) max = 1;

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const v = Math.max(0, data[off + r * W + c]);
        const brightness = (v / max) * BRIGHTNESS_CAP;
        const rByte = Math.round(theme[0] * brightness);
        const gByte = Math.round(theme[1] * brightness);
        const bByte = Math.round(theme[2] * brightness);
        const [chain, pos] = logicalToPhysical(layer, f, r, c);
        deltas.push([chain, pos, rByte, gByte, bByte]);
      }
    }
  }
  return { seq: ++seq, layer, deltas };
}

function dispatchFrame(frame) {
  window.twinEvents.dispatchEvent(new CustomEvent("frame", { detail: frame }));
}

async function inferOnce() {
  const sample = samples[Math.floor(Math.random() * samples.length)];
  const px = sample.image; // 784 uint8 values
  const norm = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    norm[i] = ((px[i] / 255.0) - MNIST_MEAN) / MNIST_STD;
  }
  const inputTensor = new ort.Tensor("float32", norm, [1, 1, 28, 28]);
  const result = await session.run({ input: inputTensor });

  for (const layer of LAYER_ORDER) {
    if (stopped) return;
    const out = result[layer];
    if (!out) continue;
    let dims = out.dims;
    let data = out.data;
    if (layer === "L6") {
      // L6 comes back as (1, 10) — reshape to (1, 1, 10).
      dims = [1, 1, 10];
    }
    const frame = activationsToFrame(layer, data, dims);
    dispatchFrame(frame);
    await sleep(600);
  }
  await sleep(2000);
}

async function loop() {
  while (!stopped) {
    try {
      await inferOnce();
    } catch (e) {
      console.error("[static-mode] inference failed:", e);
      await sleep(2000);
    }
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

export async function startStaticMode() {
  if (session) return;
  try {
    console.log("[static-mode] step 1: starting");
    console.log("[static-mode] step 2: typeof ort =", typeof ort);
    if (typeof ort === "undefined") {
      console.error("[static-mode] ort global is missing entirely");
      return;
    }
    console.log("[static-mode] step 3: ort.env exists?", !!ort.env);
    console.log("[static-mode] step 4: ort.InferenceSession exists?", !!ort.InferenceSession);
    console.log("[static-mode] step 5: ort.env.wasm exists?", !!(ort.env && ort.env.wasm));
  } catch (e) {
    console.error("[static-mode] early diag failed:", e, "stack:", e && e.stack);
    return;
  }

  // Tell ONNX runtime where its WASM lives (CDN).
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
  } catch (e) {
    console.error("[static-mode] env config failed:", e);
  }

  console.log("[static-mode] fetching mapping + samples...");
  const [m, s] = await Promise.all([
    fetch("./mapping.json").then(r => r.json()),
    fetch("./mnist-samples.json").then(r => r.json()),
  ]);
  mapping = m;
  samples = s;
  console.log("[static-mode] loaded", samples.length, "samples; creating ORT session");

  try {
    session = await ort.InferenceSession.create("./lenet5.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    // Print the most useful information available about a thrown emscripten error.
    let msg = "";
    if (e instanceof Error) msg = `${e.name}: ${e.message}\n${e.stack || ""}`;
    else if (typeof e === "number") msg = `bare number ${e} (likely emscripten exception)`;
    else msg = String(e);
    console.error("[static-mode] InferenceSession.create failed:", msg, "raw:", e);
    throw new Error(`InferenceSession.create failed: ${msg}`);
  }
  console.log("[static-mode] ORT session created; inputs:", session.inputNames, "outputs:", session.outputNames);

  // Decorate the page so visitors know they're in static mode.
  const banner = document.createElement("div");
  banner.id = "static-mode-banner";
  banner.textContent = "STATIC DEMO — model running entirely in your browser via ONNX";
  banner.style.cssText = `
    position: fixed; top: 8px; left: 50%; transform: translateX(-50%);
    background: #0c0e12; color: #6cf; border: 1px solid #2a3340;
    padding: 6px 14px; border-radius: 4px; font: 11px ui-monospace, monospace;
    z-index: 1000; pointer-events: none; opacity: 0.85;
  `;
  document.body.appendChild(banner);

  loop();
}

export function stopStaticMode() {
  stopped = true;
}
