// dirty-tracking map: "layer:fmap" -> bool
const _dirty = new Map();

export function paintSlice(canvas, rows, cols, buf) {
  const img = new ImageData(cols, rows);
  const d = img.data;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const src = (r * cols + c) * 3;
      const dst = (r * cols + c) * 4;
      d[dst]     = buf[src];
      d[dst + 1] = buf[src + 1];
      d[dst + 2] = buf[src + 2];
      d[dst + 3] = 255;
    }
  }
  const offscreen = new OffscreenCanvas(cols, rows);
  const octx = offscreen.getContext("2d");
  octx.putImageData(img, 0, 0);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

export function markDirty(layer, fmap) {
  _dirty.set(`${layer}:${fmap}`, true);
}

export function flushDirtyTo(canvases) {
  for (const [key, isDirty] of _dirty) {
    if (!isDirty) continue;
    const entry = canvases.get(key);
    if (entry) {
      paintSlice(entry.canvas, entry.rows, entry.cols, entry.buf);
    }
    _dirty.set(key, false);
  }
}
