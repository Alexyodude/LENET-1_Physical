// ES module: camera view preset buttons with smooth rAF tween

const PRESETS = [
  { label: "FRONT", pos: [0, 0, 2400],    target: [0, 0, 850] },
  { label: "SIDE",  pos: [2400, 0, 850],  target: [0, 0, 850] },
  { label: "TOP",   pos: [0, 2400, 850],  target: [0, 0, 850] },
  { label: "ISO",   pos: [1300, 500, 1800], target: [0, 0, 850] },
];

const DURATION = 600; // ms

function cubicInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function tween(camera, controls, toPos, toTarget) {
  const fromPos    = camera.position.clone();
  const fromTarget = controls.target.clone();
  const start = performance.now();

  function step(now) {
    const raw = Math.min((now - start) / DURATION, 1);
    const t   = cubicInOut(raw);

    camera.position.set(
      fromPos.x + (toPos[0] - fromPos.x) * t,
      fromPos.y + (toPos[1] - fromPos.y) * t,
      fromPos.z + (toPos[2] - fromPos.z) * t,
    );
    controls.target.set(
      fromTarget.x + (toTarget[0] - fromTarget.x) * t,
      fromTarget.y + (toTarget[1] - fromTarget.y) * t,
      fromTarget.z + (toTarget[2] - fromTarget.z) * t,
    );
    controls.update();

    if (raw < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

export function setupCameraPresets(rootEl, camera, controls) {
  const section = document.createElement("div");
  section.className = "panel-section";

  const label = document.createElement("div");
  label.className = "section-label";
  label.textContent = "VIEW";
  section.appendChild(label);

  const grid = document.createElement("div");
  grid.className = "camera-presets-grid";

  for (const preset of PRESETS) {
    const btn = document.createElement("button");
    btn.className = "btn-secondary camera-preset-btn";
    btn.textContent = preset.label;
    btn.addEventListener("click", () => tween(camera, controls, preset.pos, preset.target));
    grid.appendChild(btn);
  }

  section.appendChild(grid);
  rootEl.appendChild(section);
}
