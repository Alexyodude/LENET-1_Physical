// In-browser inference fallback. Activates when no FastAPI backend is reachable
// (e.g. when this site is hosted on GitHub Pages).
//
// Loads lenet5.onnx, runs ONNX Runtime Web inference, and dispatches the same
// `window.twinEvents.frame` CustomEvents the WebSocket would have produced —
// so arch.js, panel.js, history-ui.js, etc. all keep working unchanged.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/+esm";

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
  console.log("[static-mode] starting browser-side LeNet inference");
  // Tell ONNX runtime where its WASM lives (CDN).
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
  ort.env.wasm.numThreads = 1;

  const [m, s] = await Promise.all([
    fetch("./mapping.json").then(r => r.json()),
    fetch("./mnist-samples.json").then(r => r.json()),
  ]);
  mapping = m;
  samples = s;
  session = await ort.InferenceSession.create("./lenet5.onnx");

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
