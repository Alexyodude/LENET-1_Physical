import * as THREE from "three";

// Layer Z positions (mm) from mapping fixture
const LAYER_Z = { L1: 0, L2: 30, L3: 60, L4: 90, L5: 120, L6: 150 };

const OPS = [
  { from: "L1", to: "L2", label: "CONV 5×5 (×4)",  type: "conv",  kernels: 4  },
  { from: "L2", to: "L3", label: "AVG-POOL 2×2",    type: "pool",  kernels: null },
  { from: "L3", to: "L4", label: "CONV 5×5 (×12)", type: "conv",  kernels: 12 },
  { from: "L4", to: "L5", label: "AVG-POOL 2×2",    type: "pool",  kernels: null },
  { from: "L5", to: "L6", label: "FULLY CONNECTED",  type: "fc",    kernels: null },
];

const LAYER_COLORS = {
  L1: new THREE.Color(1.0, 0.86, 0.60),
  L2: new THREE.Color(0.0, 0.0,  1.0),
  L3: new THREE.Color(0.0, 1.0,  1.0),
  L4: new THREE.Color(0.0, 1.0,  0.0),
  L5: new THREE.Color(1.0, 0.93, 0.0),
  L6: new THREE.Color(1.0, 0.0,  0.0),
};

// Pulse state: layer → {opacity, decayTo}
const pulseState = new Map();

// All overlay objects grouped by operation index
const opGroups = [];

export function setupArchOverlay(scene, mapping) {
  _buildOverlay(scene, mapping);
  _startPulseLoop(scene);
  _listenFrames();
}

function _layerBounds(mapping, layerName) {
  const fms = mapping?.layers?.[layerName]?.feature_maps;
  if (!fms?.length) return null;
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const fm of fms) {
    const ox = fm.origin_mm[0];
    const oy = fm.origin_mm[1] || 0;
    const ex = ox + (fm.cols - 1) * fm.pitch_mm[0];
    const ey = oy + (fm.rows - 1) * fm.pitch_mm[1];
    minX = Math.min(minX, ox); maxX = Math.max(maxX, ex);
    minY = Math.min(minY, oy); maxY = Math.max(maxY, ey);
  }
  return { minX, maxX, minY, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
}

function _buildOverlay(scene, mapping) {
  for (let i = 0; i < OPS.length; i++) {
    const op = OPS[i];
    const group = new THREE.Group();
    group.name = `arch-op-${op.from}-${op.to}`;

    const fromZ = LAYER_Z[op.from];
    const toZ   = LAYER_Z[op.to];
    const midZ  = (fromZ + toZ) / 2;

    const fromBounds = _layerBounds(mapping, op.from);
    const toBounds   = _layerBounds(mapping, op.to);

    const fromColor = LAYER_COLORS[op.from] || new THREE.Color(1, 1, 1);
    const toColor   = LAYER_COLORS[op.to]   || new THREE.Color(1, 1, 1);
    const blendColor = fromColor.clone().lerp(toColor, 0.5);

    const mat = new THREE.LineBasicMaterial({
      color: blendColor,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });

    if (op.type === "conv") {
      _addConvWireframes(group, mat, op, fromBounds, fromZ, toBounds, toZ);
    } else if (op.type === "pool") {
      _addPoolWireframes(group, mat, op, fromBounds, fromZ, toBounds, toZ);
    } else if (op.type === "fc") {
      _addFCLines(group, mat, mapping, fromZ, toZ);
    }

    _addOpLabel(group, scene, op, fromBounds, toBounds, midZ, blendColor);

    scene.add(group);
    opGroups.push({ group, mat, op, baseMat: mat });
    pulseState.set(op.to, { opacity: 0.35, target: 0.35 });
  }
}

function _addConvWireframes(group, mat, op, fromBounds, fromZ, toBounds, toZ) {
  if (!fromBounds || !toBounds) return;

  const kernelCount = op.kernels || 4;
  // Draw representative kernel footprints as boxes between layers
  const kernelSize = 5; // 5×5 in LED units
  const pitch = 10;     // 10mm pitch

  // Space kernels evenly across the from-layer X range
  const span = fromBounds.maxX - fromBounds.minX;
  const step = kernelCount > 1 ? span / (kernelCount - 1) : 0;

  for (let k = 0; k < kernelCount; k++) {
    const kx = fromBounds.minX + k * step;
    const ky = fromBounds.cy;

    // Kernel footprint box at fromZ
    const hw = ((kernelSize - 1) * pitch) / 2;
    const hh = ((kernelSize - 1) * pitch) / 2;
    const boxPoints = [
      new THREE.Vector3(kx - hw, ky - hh, fromZ),
      new THREE.Vector3(kx + hw, ky - hh, fromZ),
      new THREE.Vector3(kx + hw, ky + hh, fromZ),
      new THREE.Vector3(kx - hw, ky + hh, fromZ),
      new THREE.Vector3(kx - hw, ky - hh, fromZ),
    ];
    const boxGeom = new THREE.BufferGeometry().setFromPoints(boxPoints);
    group.add(new THREE.Line(boxGeom, mat));

    // Arrow line from kernel center to corresponding output region
    const tx = toBounds.minX + k * (toBounds.maxX - toBounds.minX) / Math.max(kernelCount - 1, 1);
    const ty = toBounds.cy;
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(kx, ky, fromZ),
      new THREE.Vector3(tx, ty, toZ),
    ]);
    group.add(new THREE.Line(lineGeom, mat));

    // Small box at toZ output
    const ohw = 10, ohh = 10;
    const outBox = [
      new THREE.Vector3(tx - ohw, ty - ohh, toZ),
      new THREE.Vector3(tx + ohw, ty - ohh, toZ),
      new THREE.Vector3(tx + ohw, ty + ohh, toZ),
      new THREE.Vector3(tx - ohw, ty + ohh, toZ),
      new THREE.Vector3(tx - ohw, ty - ohh, toZ),
    ];
    const outGeom = new THREE.BufferGeometry().setFromPoints(outBox);
    group.add(new THREE.Line(outGeom, mat));
  }
}

function _addPoolWireframes(group, mat, op, fromBounds, fromZ, toBounds, toZ) {
  if (!fromBounds || !toBounds) return;

  // Pool 2×2 — draw 4 sample input windows collapsing to single output points
  const pitch = 10;
  const windowSize = 2;
  const hw = (windowSize * pitch) / 2;

  const sampleCount = 4;
  const xStep = (fromBounds.maxX - fromBounds.minX) / (sampleCount + 1);
  const yStep = (fromBounds.maxY - fromBounds.minY) / 3;

  for (let i = 0; i < sampleCount; i++) {
    const fx = fromBounds.minX + xStep * (i + 1);
    const fy = fromBounds.minY + yStep;

    // 2×2 window box
    const winPts = [
      new THREE.Vector3(fx,      fy,      fromZ),
      new THREE.Vector3(fx + hw, fy,      fromZ),
      new THREE.Vector3(fx + hw, fy + hw, fromZ),
      new THREE.Vector3(fx,      fy + hw, fromZ),
      new THREE.Vector3(fx,      fy,      fromZ),
    ];
    const winGeom = new THREE.BufferGeometry().setFromPoints(winPts);
    group.add(new THREE.Line(winGeom, mat));

    // Line to output
    const tx = toBounds.minX + (toBounds.maxX - toBounds.minX) * (i / (sampleCount - 1));
    const ty = toBounds.cy;
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(fx + hw / 2, fy + hw / 2, fromZ),
      new THREE.Vector3(tx, ty, toZ),
    ]);
    group.add(new THREE.Line(lineGeom, mat));
  }
}

function _addFCLines(group, mat, mapping, fromZ, toZ) {
  const l5fms = mapping?.layers?.L5?.feature_maps || [];
  const l6fms = mapping?.layers?.L6?.feature_maps || [];

  // Collect sample L5 positions (every Nth to stay under 200 lines)
  const l5pts = [];
  for (const fm of l5fms) {
    const stride = Math.max(1, Math.ceil((fm.rows * fm.cols) / 20));
    for (let r = 0; r < fm.rows; r += stride) {
      for (let c = 0; c < fm.cols; c += stride) {
        l5pts.push(new THREE.Vector3(
          fm.origin_mm[0] + c * fm.pitch_mm[0],
          (fm.origin_mm[1] || 0) + r * fm.pitch_mm[1],
          fromZ,
        ));
      }
    }
  }

  // Collect L6 positions
  const l6pts = [];
  for (const fm of l6fms) {
    for (let c = 0; c < fm.cols; c++) {
      l6pts.push(new THREE.Vector3(
        fm.origin_mm[0] + c * fm.pitch_mm[0],
        fm.origin_mm[1] || 0,
        toZ,
      ));
    }
  }

  // Draw up to 200 lines
  const fcMat = mat.clone();
  fcMat.opacity = 0.15;

  let lineCount = 0;
  const maxLines = 200;
  outer: for (const from of l5pts) {
    for (const to of l6pts) {
      if (lineCount >= maxLines) break outer;
      const geom = new THREE.BufferGeometry().setFromPoints([from, to]);
      group.add(new THREE.Line(geom, fcMat));
      lineCount++;
    }
  }
}

function _addOpLabel(group, scene, op, fromBounds, toBounds, midZ, color) {
  const cx = fromBounds ? (fromBounds.cx + (toBounds?.cx ?? fromBounds.cx)) / 2 : 300;
  const topY = (fromBounds?.maxY ?? 0) + 45;

  const c = document.createElement("canvas");
  c.width = 400; c.height = 56;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "rgba(8,12,16,0.82)";
  ctx.fillRect(0, 0, 400, 56);

  ctx.fillStyle = `#${color.getHexString()}`;
  ctx.fillRect(0, 0, 400, 3);

  ctx.fillStyle = `#${color.getHexString()}`;
  ctx.font = "bold 18px 'Courier New', monospace";
  ctx.fillText(op.label, 10, 26);

  ctx.fillStyle = "rgba(180,210,240,0.6)";
  ctx.font = "13px 'Courier New', monospace";
  ctx.fillText(`${op.from} → ${op.to}`, 10, 46);

  const texture = new THREE.CanvasTexture(c);
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(90, 13, 1);
  sprite.position.set(cx, topY, midZ);
  scene.add(sprite);
}

function _listenFrames() {
  window.twinEvents = window.twinEvents || new EventTarget();
  window.twinEvents.addEventListener("frame", (ev) => {
    const f = ev.detail;
    if (!f?.layer) return;

    // Find the op that terminates at this layer and pulse it
    for (let i = 0; i < OPS.length; i++) {
      if (OPS[i].to === f.layer) {
        pulseState.set(f.layer, { opacity: 0.9, target: 0.35 });
        break;
      }
    }
  });
}

let _lastPulse = 0;

function _startPulseLoop() {
  function tick(now) {
    requestAnimationFrame(tick);

    const dt = Math.min(now - _lastPulse, 50);
    _lastPulse = now;
    const decay = dt * 0.003; // ~0.003 opacity units per ms → ~300ms full decay

    for (let i = 0; i < OPS.length; i++) {
      const op = OPS[i];
      const state = pulseState.get(op.to);
      if (!state) continue;

      if (state.opacity > state.target) {
        state.opacity = Math.max(state.target, state.opacity - decay);
      }

      const { mat } = opGroups[i];
      if (mat) mat.opacity = state.opacity;
    }
  }
  requestAnimationFrame(tick);
}
