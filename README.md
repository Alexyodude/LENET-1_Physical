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

## Tests

```
uv run pytest -q
npx playwright test --config=tests/e2e/playwright.config.ts
```

## Spec & plan

- [Design spec](docs/superpowers/specs/2026-04-26-lenet-physical-design.md)
- [Implementation plan](docs/superpowers/plans/2026-04-26-lenet-physical.md)
