# LENET-1 Physical — Design Spec

**Date:** 2026-04-26
**Status:** Draft, awaiting user review

## 1. Goal

A physical, illuminated visualization of a LeNet-style convolutional neural
network performing live MNIST digit classification. A Raspberry Pi 4 runs the
trained model, drives ~4634 WS2812B addressable LEDs arranged into 6 layers,
and serves a digital twin (3D + per-layer 2D slices) over a web UI for
real-time debugging.

The viewer presses a button, the Pi picks an MNIST sample, and the network's
internal activations propagate through the physical layers in step mode — input
image lights up first, then convolutional features, then pooled features, etc.,
ending on the predicted digit.

## 2. Network topology and LED counts

LeNet variant used:

| Layer | Shape | Feature maps × pixels | LEDs |
|-------|-------|----------------------|------|
| L1 input | 28×28×1 | 1 × 784 | 784 |
| L2 conv1 (C1) | 24×24×4 | 4 × 576 | 2304 |
| L3 pool1 (S2) | 12×12×4 | 4 × 144 | 576 |
| L4 conv2 (C3) | 8×8×12 | 12 × 64 | 768 |
| L5 pool2 (S4) | 4×4×12 | 12 × 16 | 192 |
| L6 output | 1×10 | 10 × 1 | 10 |
| **Total** | | | **4634** |

## 3. Hardware

### 3.1 LED partitioning into chains

17 WS2812B chains driven directly from Pi 4 GPIOs, partitioned per-feature-map
where reasonable:

| Chain group | Count | LEDs/chain | Maps per chain |
|-------------|-------|------------|----------------|
| L1 | 1 | 784 | 1 (whole input) |
| L2 | 4 | 576 | 1 (one feature map per chain) |
| L3 | 4 | 144 | 1 (one feature map per chain) |
| L4 | 4 | 192 | 3 (3 of 12 feature maps per chain) |
| L5 | 3 | 64 | 4 (4 of 12 feature maps per chain) |
| L6 | 1 | 10 | 10 outputs in one chain |

GPIO assignments are TBD — chosen at wiring time. All 17 GPIOs must be on the
same BCM bank to support DMA-bit-bang parallel output.

### 3.2 LED driver strategy

WS2812B requires ±150ns timing at 800kHz. Linux is not real-time, so reliable
output requires DMA. The Pi 4's standard `rpi_ws281x` library only supports
2 PWM channels — insufficient for 17. The chosen approach:

**DMA-to-GPIO-bank bit-banging** — a single DMA stream writes 32-bit words
directly to the GPIO output register, where each bit position is a different
GPIO. This drives up to 16+ parallel WS2812 chains off any GPIO pins on the
same bank. Library candidates (selection deferred to plan phase):

- A maintained fork of `rpi_ws281x` with multi-channel patches
- `Pi5Neo`-style approach ported to Pi 4
- A purpose-built thin C extension (~few hundred lines) wrapped in Python

This approach is less battle-tested than 2-PWM. Mitigations: a single-chain
prototype validates timing before wiring all 17; a sequence-numbered frame
log surfaces any flushes the driver flagged as unclean; the digital twin's
chain-walk and layer-walk diagnostics catch wiring/addressing bugs without
needing physical readback.

### 3.3 Power

Worst-case current is ~278A at full white across all 4634 LEDs. v1 caps
**global brightness in software at 30%**, giving a working maximum of
**~80A at 5V**. The build budgets for **5V / 30A regulated PSU** with
**power injection every ~1m of chain**, sized to typical-pattern current
draw rather than impossible worst case (most LEDs are dim/off most of the
time during inference visualization).

The Pi 4 monitors `vcgencmd get_throttled` on a slow loop. If undervoltage
is detected, the global brightness cap drops and the digital twin surfaces a
visible warning. Heavy-gauge wire and proper grounding are wiring-time
concerns covered in the build doc, not the spec.

### 3.4 Physical layout

Six layer-slabs arranged front-to-back like a deck of cards. Each slab is a
flat panel containing that layer's feature-map grids. The viewer looks at
the stack edge-on or at an angle so all six are visible simultaneously.
Exact slab spacing, panel material, and per-pixel mm coordinates are
deferred to the mapping config (see §4.3) and the build doc.

### 3.5 Inputs

Two physical buttons wired to GPIO inputs:

- **Button 1 — Sample.** Picks an MNIST test sample (random by default).
- **Button 2 — Step.** Advances one layer in step mode.

Buttons use software debouncing (e.g., 30ms). The web UI exposes equivalent
controls plus index-pick, delay tuning, replay, and a single-LED test panel.

## 4. Software architecture

### 4.1 Modules

| Module | Responsibility |
|--------|----------------|
| `model/` | LeNet-5 PyTorch definition, training script, ONNX/TorchScript export, inference wrapper that returns activations for every layer (not just final prediction). |
| `leds/` | WS2812B driver. Wraps the chosen DMA-bit-bang library. Exposes `write(chain_id, pixel_index, rgb)` and `flush()`. Includes a mock backend that publishes commanded frames to the digital twin without touching hardware (for off-Pi development). |
| `mapping/` | Loads a YAML config that maps logical pixels `(layer, feature_map, row, col)` → physical `(chain_id, position_in_chain, x_mm, y_mm, z_mm)`. Validates chain lengths at startup. The 3D coordinates feed the digital twin. |
| `twin/` | FastAPI server on the Pi. Serves a single-page Three.js app: 3D scene + per-layer 2D slice panels. WebSocket pushes pixel-frame deltas to all connected clients. HTTP endpoints handle controls (sample, step, replay, single-LED test, brightness cap). |
| `control/` | Button GPIO handling with debounce. State machine: `idle → selected → animating → done`. Inference-and-paint orchestrator that ties model + leds + mapping + twin together. |

### 4.2 Per-layer color theme

Brightness encodes activation magnitude (0..1, ReLU'd, normalized
per-feature-map by max). Color theme is per-layer:

- L1 input — warm white
- L2 conv1 — blue
- L3 pool1 — cyan
- L4 conv2 — green
- L5 pool2 — yellow
- L6 output — red, with the predicted digit pulsed brighter

### 4.3 Mapping config (YAML)

Single source of truth for both the LED driver and the digital twin. Schema:

```yaml
layers:
  L1:
    feature_maps:
      - id: 0
        chain_id: 0
        offset_in_chain: 0
        rows: 28
        cols: 28
        origin_mm: [0, 0, 0]
        pitch_mm: [10, 10]
        # row-major snake fill; toggle if hardware wired column-major
        order: row_major_snake
  L2:
    feature_maps:
      - id: 0
        chain_id: 1
        offset_in_chain: 0
        rows: 24
        cols: 24
        origin_mm: [0, 0, 30]
        pitch_mm: [10, 10]
        order: row_major_snake
      # ...3 more for L2
  # ...etc
```

The validator checks that `sum(feature_maps[*].rows × cols on chain N) ==
expected chain length for chain N`. Mismatch refuses to start with an error
that names the offending chain.

## 5. Data flow (one inference cycle)

```
Button 1 press (or web UI "sample")
  → control picks MNIST index
  → model.forward_with_activations(image)
    returns dict {L1: tensor[1,28,28], L2: tensor[4,24,24], ...}
  → for each layer L1..L6 in step order:
      - normalize activation tensor → 0..1 brightness scalar per pixel
      - apply per-layer color theme → RGB triplet per pixel
      - lookup mapping table → write to (chain, position) on leds backend
      - publish frame delta to twin via WebSocket (sequence-numbered)
      - wait for step trigger (button-2, web UI "step", or auto-timer)
  → hold final L6 state for N seconds
  → idle, await next sample
```

## 6. Digital twin

### 6.1 What it shows

- **3D view (Three.js)**: the physical box. Six layer-slabs in their actual
  positions. Each LED rendered as a small emissive sphere whose color matches
  what the driver is currently commanding for that physical pixel.
- **2D slices**: one panel per layer, rendering each feature map as a flat
  heatmap grid. Classic CNN-visualizer view.
- **Debug overlays**: chain boundaries highlighted with thin lines/colors,
  GPIO/chain index labels, frame sequence counter, FPS, brightness cap
  status, undervoltage warning if surfaced.
- **Single-LED test panel**: pick `(chain, position)` → light it red. For
  bring-up wiring verification.

### 6.2 What it can NOT show

WS2812B is one-way. The twin shows what the **driver commanded**, not what
the physical LEDs are actually displaying. A broken solder joint that skips
an LED is invisible to the twin. This is acceptable: the twin's value is
sanity-checking addressing, mapping, animation, and color logic without
having to stare at the physical box.

### 6.3 Transport

- WebSocket for frame deltas (only changed pixels per frame).
- HTTP for control commands (sample, step, replay, test, config).
- Frames carry monotonic sequence numbers; the twin highlights any frame
  the driver flagged as not flushed cleanly.

## 7. Error handling (what will actually go wrong)

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| DMA driver glitches | Driver flags unclean flush; sequence numbers visible in twin | Single-LED test panel isolates bad chains; rerun frame |
| Mapping config drift from physical wiring | Startup validator | Refuses to start; names the offending chain |
| Power brownout | `vcgencmd get_throttled` polled on slow loop | Auto-cap global brightness; surface warning in twin |
| Bad MNIST sample / model load failure | Try/except at startup and on inference | Surface error in twin; fall back to a hardcoded test pattern |
| No Pi available for dev | Mock LED backend | All logic, twin, and tests run on any machine |
| Button bounce | Software debounce | 30ms debounce window per button |

## 8. Testing

**Unit:**

- Model output shape per layer (against a fixed input).
- Mapping config validation (good config loads; bad config raises with
  named offending chain).
- Color/brightness encoder (activation tensor → RGB pixel).
- Chain-frame builder (logical pixels → driver write calls).

**Integration (mock backend):**

- Full inference cycle against a fixed MNIST sample produces a deterministic
  sequence of frames (snapshot test on the JSON the twin receives).
- Step mode pauses correctly; web UI step advances the state machine.

**Hardware bring-up scripts (run-on-Pi only):**

- `single-chain-walk` — light each LED in chain N in sequence, slowly.
- `single-layer-walk` — light each feature map's center pixel in turn.
- `power-stress` — fixed pattern at increasing brightness while watching
  `vcgencmd get_throttled` for undervoltage events.

**Twin (Playwright):**

- Connect to WebSocket, send scripted frames, assert the 3D scene's
  emissive material on the corresponding sphere updates.
- Single-LED test panel issues correct HTTP command.

## 9. Tech stack

- **Language:** Python 3.11+ on the Pi.
- **ML:** PyTorch for training, exported to TorchScript for inference (avoids
  PyTorch overhead at runtime; ONNX-runtime is an alternative if measured
  faster on Pi 4).
- **Web:** FastAPI + uvicorn. Single-page frontend in TypeScript with
  Three.js for 3D and a small framework-free 2D canvas for slice panels.
- **LED driver:** chosen DMA-bit-bang library, decision deferred to plan phase.
- **GPIO:** `gpiozero` or `RPi.GPIO` for buttons (small attack surface).
- **Tests:** `pytest` for Python, `playwright` for the twin.
- **Build/dev:** `uv` or `pip-tools` for deterministic deps.

## 10. Out of scope for v1

- Camera input (point a Pi cam at hand-drawn digits). Add as a later mode.
- Training UX (you train once on a beefy machine, ship the weights).
- Multiple-network support. LeNet only.
- Per-LED readback / closed-loop verification (would require RGB sensors).
- Audio cues, animations beyond per-layer fade.
- Mobile-optimized twin UI (desktop browser only in v1).

## 11. Open items deferred to plan phase

- Concrete BCM GPIO pin assignment for the 17 chains.
- Specific DMA-bit-bang library / source decision.
- Exact slab spacing and per-pixel mm coordinates for the mapping config.
- LED-strip vs LED-matrix-panel sourcing decisions.
- Power-injection wiring topology (star vs bus).
- Build doc / wiring diagrams.

## 12. Repo

Private GitHub repo named **`LENET-1_Physical`** under the user's account.
Initial commit contains this spec and a stub `README.md` linking to it.
Implementation plan is produced separately by `superpowers:writing-plans`
and committed before any code is written.
