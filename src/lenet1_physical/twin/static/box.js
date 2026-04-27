import * as THREE from "three";

// Layer theme colors matching main.js LAYER_META threeColor values
const LAYER_THEMES = {
  L1: new THREE.Color(1.0,  0.86, 0.60),
  L2: new THREE.Color(0.0,  0.0,  1.0),
  L3: new THREE.Color(0.0,  1.0,  1.0),
  L4: new THREE.Color(0.0,  1.0,  0.0),
  L5: new THREE.Color(1.0,  0.93, 0.0),
  L6: new THREE.Color(1.0,  0.0,  0.0),
};

const SLAB_PAD   = 6;    // mm padding around LED extents for each fmap plane
const SLAB_Z_OFF = -1;   // mm behind LEDs (lower Z)
const INJ_DIST   = 1000; // mm accumulated chain length before injection marker

// Build snake-order LED positions for a feature map. Iterates the chain in
// physical order so the routing curve stitches LEDs in the order the wire
// actually visits them.
function snakePositions(fm) {
  const { origin_mm, pitch_mm, rows, cols, chain_id, offset_in_chain, order } = fm;
  const ox = origin_mm[0];
  const oy = origin_mm[1] || 0;
  const oz = origin_mm[2];
  const px = pitch_mm[0];
  const py = pitch_mm[1];
  const positions = [];
  // MNIST row 0 is the TOP of the image; Three.js +Y is up.
  // Map row 0 → max Y so the digit reads right-side-up.
  if (order === "column_major_snake") {
    // Up-and-down zigzag: col 0 top-to-bottom, col 1 bottom-to-top, ...
    for (let c = 0; c < cols; c++) {
      for (let i = 0; i < rows; i++) {
        const r = c % 2 === 0 ? i : (rows - 1 - i);
        positions.push({
          x: ox + c * px,
          y: oy + (rows - 1 - r) * py,
          z: oz,
          chainId: chain_id,
          pos: offset_in_chain + c * rows + i,
        });
      }
    }
  } else {
    // Default: row-major snake
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < cols; i++) {
        const c = r % 2 === 0 ? i : (cols - 1 - i);
        positions.push({
          x: ox + c * px,
          y: oy + (rows - 1 - r) * py,
          z: oz,
          chainId: chain_id,
          pos: offset_in_chain + r * cols + i,
        });
      }
    }
  }
  return positions;
}

function makeGpioSprite(gpioNum) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(8,12,16,0.82)";
  ctx.fillRect(0, 0, 256, 64);
  ctx.strokeStyle = "#2a2f36";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, 254, 62);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 24px monospace";
  ctx.fillText(`GPIO ${gpioNum}`, 14, 42);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(50, 12.5, 1);
  return sprite;
}

function makePowerMarker() {
  const geo = new THREE.SphereGeometry(2, 8, 6);
  const mat = new THREE.MeshBasicMaterial({ color: 0xff2222 });
  return new THREE.Mesh(geo, mat);
}

// Called by integration layer (worker-7 / main.js) with an existing Three.js scene.
// Also auto-invoked below if window.__lenet1Scene is available (set by main.js).
export function setupPhysicalBox(scene, mapping) {
  if (!mapping || !mapping.layers) return;

  // Build a chain→gpio lookup from mapping.chains
  const chainGpio = {};
  for (const ch of mapping.chains || []) {
    chainGpio[ch.id] = ch.gpio;
  }

  // Group feature maps by layer
  for (const [layerName, layerData] of Object.entries(mapping.layers)) {
    const theme = LAYER_THEMES[layerName];
    if (!theme) continue;
    const fmaps = layerData.feature_maps || [];
    if (!fmaps.length) continue;

    // ── One plane per feature map (each fmap is its own physical slab) ─────
    const slabMatBase = new THREE.MeshBasicMaterial({
      color: 0x0c0e12,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    // Frame color is the layer theme at low intensity so each layer's stack
    // is colour-coded at a glance.
    const frameColor = theme.clone().multiplyScalar(0.45);
    const frameMat = new THREE.LineBasicMaterial({
      color: frameColor,
      transparent: true,
      opacity: 0.85,
    });

    for (const fm of fmaps) {
      const ox = fm.origin_mm[0];
      const oy = fm.origin_mm[1] || 0;
      const oz = fm.origin_mm[2];
      const endX = ox + (fm.cols - 1) * fm.pitch_mm[0];
      const endY = oy + (fm.rows - 1) * fm.pitch_mm[1];

      const fmW = (endX - ox) + SLAB_PAD * 2;
      const fmH = (endY - oy) + SLAB_PAD * 2;
      const cx  = (ox + endX) / 2;
      const cy  = (oy + endY) / 2;
      const cz  = oz + SLAB_Z_OFF;

      const slab = new THREE.Mesh(new THREE.PlaneGeometry(fmW, fmH), slabMatBase);
      slab.position.set(cx, cy, cz);
      scene.add(slab);

      const hw = fmW / 2;
      const hh = fmH / 2;
      const verts = new Float32Array([
        cx - hw, cy - hh, cz,  cx + hw, cy - hh, cz,
        cx + hw, cy - hh, cz,  cx + hw, cy + hh, cz,
        cx + hw, cy + hh, cz,  cx - hw, cy + hh, cz,
        cx - hw, cy + hh, cz,  cx - hw, cy - hh, cz,
      ]);
      const buf = new THREE.BufferGeometry();
      buf.setAttribute("position", new THREE.BufferAttribute(verts, 3));
      scene.add(new THREE.LineSegments(buf, frameMat));
    }

    // ── Per-chain routing curves + GPIO labels + power injection ──────────
    // Group fmaps by chain_id
    const byChain = {};
    for (const fm of fmaps) {
      const cid = fm.chain_id;
      if (!byChain[cid]) byChain[cid] = [];
      byChain[cid].push(fm);
    }

    for (const [chainIdStr, chainFmaps] of Object.entries(byChain)) {
      const chainId = Number(chainIdStr);

      // Gather all LED positions in chain order (by offset_in_chain then snake)
      const allPts = [];
      const sortedFmaps = [...chainFmaps].sort((a, b) => a.offset_in_chain - b.offset_in_chain);
      for (const fm of sortedFmaps) {
        for (const p of snakePositions(fm)) {
          allPts.push(new THREE.Vector3(p.x, p.y, p.z));
        }
      }

      if (allPts.length < 2) continue;

      // ── BezierCurve routing (piecewise CatmullRom as smooth proxy) ──────
      // Use CatmullRomCurve3 which gracefully handles many points
      const curve = new THREE.CatmullRomCurve3(allPts);
      const segments = Math.min(64, allPts.length * 4);
      const curvePts = curve.getPoints(segments);

      const routeGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
      const routeMat = new THREE.LineBasicMaterial({
        color: theme,
        transparent: true,
        opacity: 0.25,
      });
      scene.add(new THREE.Line(routeGeo, routeMat));

      // ── GPIO label sprite 30mm above first LED ─────────────────────────
      const gpio = chainGpio[chainId];
      if (gpio !== undefined) {
        const first = allPts[0];
        const sprite = makeGpioSprite(gpio);
        sprite.position.set(first.x, first.y + 30, first.z);
        scene.add(sprite);
      }

      // ── Power injection markers every ~1000mm of chain length ──────────
      let accumulated = 0;
      let nextThreshold = INJ_DIST;
      for (let i = 1; i < allPts.length; i++) {
        accumulated += allPts[i].distanceTo(allPts[i - 1]);
        if (accumulated >= nextThreshold) {
          const marker = makePowerMarker();
          marker.position.copy(allPts[i]);
          scene.add(marker);
          nextThreshold += INJ_DIST;
        }
      }
    }
  }
}

// Auto-init: wait for main.js to expose its scene, then layer the physical box on top.
// main.js (or worker-7 integration) sets window.__lenet1Scene after placeLEDsFromMapping.
async function autoInit() {
  // Poll up to 5s for the scene reference
  let scene = null;
  for (let i = 0; i < 50; i++) {
    if (window.__lenet1Scene) { scene = window.__lenet1Scene; break; }
    await new Promise(r => setTimeout(r, 100));
  }
  if (!scene) return;

  try {
    let resp = await fetch("/mapping").catch(() => null);
    if (!resp || !resp.ok) resp = await fetch("./mapping.json").catch(() => null);
    if (!resp || !resp.ok) return;
    const mapping = await resp.json();
    if (mapping.layers && Object.keys(mapping.layers).length > 0) {
      setupPhysicalBox(scene, mapping);
    }
  } catch (e) {
    console.warn("[box.js] auto-init failed:", e);
  }
}

autoInit();
