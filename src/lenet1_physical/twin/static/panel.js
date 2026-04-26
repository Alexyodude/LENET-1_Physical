// ES module: MNIST input preview (L1) and confidence bar panel (L6)

const COLS_L1 = 28;
const ROWS_L1 = 28;
const L1_CHAIN = 0;
const L6_CHAIN = 16;
const NUM_CLASSES = 10;

// Unsnake a linear snake-order position back to (row, col)
function unsnake(snakeIdx, cols) {
  const row = Math.floor(snakeIdx / cols);
  const col = row % 2 === 0 ? snakeIdx % cols : cols - 1 - (snakeIdx % cols);
  return { row, col };
}

function buildInputPreview(container) {
  const PIXEL = 7; // px per LED pixel
  const wrap = document.createElement("div");
  wrap.className = "panel-input-wrap";

  const label = document.createElement("div");
  label.className = "panel-input-label";
  label.textContent = "INPUT (L1)";
  wrap.appendChild(label);

  const canvas = document.createElement("canvas");
  canvas.width  = COLS_L1 * PIXEL;
  canvas.height = ROWS_L1 * PIXEL;
  canvas.className = "panel-input-canvas";
  wrap.appendChild(canvas);

  container.appendChild(wrap);

  const ctx = canvas.getContext("2d");
  // Fill black initially
  ctx.fillStyle = "#050810";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const pixelBuf = new Uint8ClampedArray(ROWS_L1 * COLS_L1 * 3);

  return { ctx, pixelBuf, PIXEL };
}

function buildConfidencePanel(container) {
  const wrap = document.createElement("div");
  wrap.className = "panel-confidence-wrap";

  const label = document.createElement("div");
  label.className = "panel-confidence-label";
  label.textContent = "OUTPUT (L6)";
  wrap.appendChild(label);

  const bars = [];
  for (let d = 0; d < NUM_CLASSES; d++) {
    const row = document.createElement("div");
    row.className = "conf-row";

    const digit = document.createElement("span");
    digit.className = "conf-digit";
    digit.textContent = d;

    const track = document.createElement("div");
    track.className = "conf-track";

    const fill = document.createElement("div");
    fill.className = "conf-fill";
    track.appendChild(fill);

    const val = document.createElement("span");
    val.className = "conf-val";
    val.textContent = "0%";

    row.appendChild(digit);
    row.appendChild(track);
    row.appendChild(val);
    wrap.appendChild(row);

    bars.push({ row, fill, val });
  }

  container.appendChild(wrap);
  return bars;
}

export function setupArchPanel(container) {
  // Ensure twinEvents exists
  if (!window.twinEvents) {
    window.twinEvents = new EventTarget();
  }

  const { ctx, pixelBuf, PIXEL } = buildInputPreview(container);
  const bars = buildConfidencePanel(container);

  window.twinEvents.addEventListener("frame", (ev) => {
    const { layer, deltas } = ev.detail;

    if (layer === "L1") {
      for (const [chain, position, r, g, b] of deltas) {
        if (chain !== L1_CHAIN) continue;
        const { row, col } = unsnake(position, COLS_L1);
        if (row < 0 || row >= ROWS_L1 || col < 0 || col >= COLS_L1) continue;
        const idx = (row * COLS_L1 + col) * 3;
        pixelBuf[idx]     = r;
        pixelBuf[idx + 1] = g;
        pixelBuf[idx + 2] = b;
      }

      // Flush canvas
      for (let row = 0; row < ROWS_L1; row++) {
        for (let col = 0; col < COLS_L1; col++) {
          const idx = (row * COLS_L1 + col) * 3;
          const r = pixelBuf[idx], g = pixelBuf[idx + 1], b = pixelBuf[idx + 2];
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(col * PIXEL, row * PIXEL, PIXEL, PIXEL);
        }
      }
    }

    if (layer === "L6") {
      const values = new Array(NUM_CLASSES).fill(0);
      for (const [chain, position, r, g, b] of deltas) {
        if (chain !== L6_CHAIN) continue;
        if (position < 0 || position >= NUM_CLASSES) continue;
        values[position] = Math.max(r, g, b) / 255;
      }

      const maxVal = Math.max(...values);
      for (let d = 0; d < NUM_CLASSES; d++) {
        const pct = Math.round(values[d] * 100);
        bars[d].fill.style.width = `${pct}%`;
        bars[d].val.textContent  = `${pct}%`;
        const isMax = values[d] === maxVal && maxVal > 0;
        bars[d].row.classList.toggle("conf-row--max", isMax);
      }
    }
  });
}
