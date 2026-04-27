export function setupPerfOverlay(rootEl) {
  const params = new URLSearchParams(window.location.search);
  let enabled = params.get("perf") === "1";

  const pill = document.createElement("div");
  pill.style.cssText = [
    "position:fixed", "top:10px", "left:10px", "z-index:9999",
    "background:rgba(0,0,0,0.72)", "color:#0f0", "font:12px/1.4 monospace",
    "padding:6px 10px", "border-radius:12px", "pointer-events:none",
    "display:none", "white-space:pre",
  ].join(";");
  (rootEl || document.body).appendChild(pill);

  document.addEventListener("keydown", e => {
    if (e.key === "p" || e.key === "P") {
      enabled = !enabled;
      pill.style.display = enabled ? "block" : "none";
    }
  });

  if (enabled) pill.style.display = "block";

  let frameTimes = [];
  let inferenceFrames = 0;
  let framesThisInference = 0;
  let lastInferenceTime = performance.now();
  let last = performance.now();

  // expose hook so callers can signal an inference completed
  const overlay = {
    markInference() {
      inferenceFrames = framesThisInference;
      framesThisInference = 0;
    },
  };

  function tick(now) {
    requestAnimationFrame(tick);
    const dt = now - last;
    last = now;
    framesThisInference++;
    frameTimes.push(dt);

    // keep a 1-second rolling window
    const cutoff = now - 1000;
    let windowMs = 0;
    let keep = [];
    for (let i = frameTimes.length - 1; i >= 0; i--) {
      windowMs += frameTimes[i];
      keep.unshift(frameTimes[i]);
      if (windowMs >= 1000) break;
    }
    frameTimes = keep;

    if (!enabled) return;

    const fps = frameTimes.length > 0 ? (frameTimes.length / (windowMs / 1000)).toFixed(1) : "–";
    const avg = frameTimes.length > 0 ? (windowMs / frameTimes.length).toFixed(1) : "–";
    const fpi = inferenceFrames > 0 ? inferenceFrames : "–";
    pill.textContent = `FPS  ${fps}\nFrame ${avg} ms\nF/inf ${fpi}`;
  }

  requestAnimationFrame(tick);
  return overlay;
}
