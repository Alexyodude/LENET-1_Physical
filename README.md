# LENET-1 Physical

Physical illuminated LeNet-5 visualization on a Raspberry Pi 4.

## Quick start (dev machine, mock LEDs)

```
uv sync --extra dev
uv run python -m lenet1_physical.model.train --epochs 10
uv run python -m lenet1_physical.main \
   --mapping config/mapping.example.yaml \
   --weights weights/lenet5.pt \
   --mode mock
```

Open http://127.0.0.1:8080. Click **Sample**, then **Step** repeatedly.

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

## Tests

```
uv run pytest -q
npx playwright test --config=tests/e2e/playwright.config.ts
```

## Spec & plan

- [Design spec](docs/superpowers/specs/2026-04-26-lenet-physical-design.md)
- [Implementation plan](docs/superpowers/plans/2026-04-26-lenet-physical.md)
