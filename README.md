# LENET-1 Physical

Physical illuminated LeNet-5 visualization on a Raspberry Pi 4, with a
browser-based digital twin (3D scene, per-fmap planes, architecture overlay,
fault simulation, history scrubber).

## Quick start (dev machine, mock LEDs)

```
uv sync --extra dev
uv run python -m lenet1_physical.model.train --epochs 10
uv run python -m lenet1_physical.main \
   --mapping config/mapping.example.yaml \
   --weights weights/lenet5.pt \
   --mode simulate --demo
```

Open http://127.0.0.1:8080. With `--demo` the server auto-cycles MNIST
samples; without it, click **Sample** then **Step** in the UI (or wire a
GPIO button on a Pi).

## Public web demo

A `Dockerfile` is included so the twin can be deployed to any container
host. Recommended free options:

- **Hugging Face Spaces** — `huggingface.co/new-space` → SDK: Docker → set
  Space hardware to CPU basic → push this repo to the Space's git remote.
  Use port 7860 (already the Dockerfile default).
- **Render** — `render.com/new/web-service` → connect this repo → environment
  Docker → keep the default port. Free tier sleeps after 15 min of inactivity.
- **Fly.io** — `fly launch` from a clone of this repo and accept the Docker
  detection.

The deployed instance runs `--mode simulate --demo`, so any visitor sees
inferences cycling through automatically.

## Run on hardware (Raspberry Pi 4)

1. Wire LEDs per the mapping config.
2. Bring up one chain at a time: `sudo uv run python -m lenet1_physical.scripts.single_chain_walk --gpio 18 --count 30`
3. Bring up all chains in one layer: `sudo uv run python -m lenet1_physical.scripts.single_layer_walk --mapping config/mapping.example.yaml --layer L2`
4. Power-soak: `sudo uv run python -m lenet1_physical.scripts.power_stress --mapping config/mapping.example.yaml --duration 600`
5. Run for real: `sudo uv run python -m lenet1_physical.main --mapping config/mapping.example.yaml --weights weights/lenet5.pt --mode hardware`

## Twin v2 features

The digital twin UI ships a second-generation feature set on top of the base 3D LED viewer:

- **Physical box realism** — per-layer PCB slabs, chain routing curves, GPIO labels and power-injection markers rendered in the 3D scene.
- **Architecture overlay** — animated 3D connectors and labels showing CONV/POOL/FC operations between layers; pulses on frame arrival.
- **Input preview + confidence** — live 28×28 canvas of L1 pixel data and a 10-class confidence bar chart from L6 output, rendered in the right panel.
- **Fault simulation** — kill individual LEDs, break entire chains, or apply undervoltage scaling via the fault panel; faults are applied in the server before data reaches the hardware driver.
- **History scrubber** — every inference cycle is recorded server-side (up to 50); select any past inference from the list, scrub frame-by-frame, or play it back at 80 ms/frame.
- **Expected-vs-commanded diff** — toggle DIFF mode in the history scrubber to highlight LEDs whose commanded colour diverges from expected due to active faults (shown as red wireframe).

## Polish features

The v2 twin now ships six additional polish modules wired into the UI:

- **Draw canvas** — paint a digit on a 280×280 canvas and run it through LeNet live (`/sample-image`) or via the in-browser ONNX static fallback. Housed in `#draw-host` in the right panel.
- **Camera presets** — FRONT / SIDE / TOP / ISO buttons with a smooth 600 ms cubic-in-out tween to preset viewpoints. Rendered inside `#camera-presets-host` below the TEST PIXEL section.
- **Mobile drawer** — on viewports ≤ 768 px a slide-up drawer with CONTROLS / FAULTS / HISTORY / SLICES tabs replaces the desktop panel layout. Activated at the end of `init()` after all panels exist.
- **Performance overlay** — press `P` to toggle a fixed HUD showing FPS, avg frame time, and frames-per-inference. Activated unconditionally on `document.body`.
- **Lazy mobile loaders** — on mobile, advanced controls (faults + history) are gated behind a single "Load advanced controls" button to reduce initial JS work; on desktop they load eagerly.
- **Slice render optimisation** — feature-map slice canvases now use `ImageData` + `OffscreenCanvas` via `paintSlice` / `markDirty` / `flushDirtyTo` from `slice-render.js`, replacing the per-pixel `fillRect` loop.

## Tests

```
uv run pytest -q
npx playwright test --config=tests/e2e/playwright.config.ts
```

## Spec & plan

- [Design spec](docs/superpowers/specs/2026-04-26-lenet-physical-design.md)
- [Implementation plan](docs/superpowers/plans/2026-04-26-lenet-physical.md)
