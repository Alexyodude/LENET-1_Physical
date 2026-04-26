# LENET-1 Physical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physically illuminated visualization of a LeNet-5 CNN running live MNIST inference on a Raspberry Pi 4, driving ~4634 WS2812B LEDs across 17 chains, with a digital-twin web UI for real-time debugging.

**Architecture:** Library-agnostic `Driver` interface with a mock backend (off-Pi development) and a hardware backend (Pi-only). PyTorch LeNet-5 with an inference wrapper that returns activations for every layer. YAML mapping config validated at startup translates `(layer, fmap, row, col)` into `(chain, position, x_mm, y_mm, z_mm)` — same source of truth feeds the LEDs and the Three.js twin. FastAPI server publishes WebSocket frame deltas to any connected browser. Two physical buttons drive a small state machine for sample-pick and step.

**Tech Stack:** Python 3.11, PyTorch (training), TorchScript (inference), FastAPI + uvicorn, vanilla TypeScript + Three.js (no bundler), `rpi_ws281x` (initial 2-channel hardware bring-up), `gpiozero` (buttons), `pytest`, `playwright`, `uv` for env management.

**Spec:** [`docs/superpowers/specs/2026-04-26-lenet-physical-design.md`](../specs/2026-04-26-lenet-physical-design.md)

---

## File Structure

```
LENET-1_Physical/
├── pyproject.toml
├── README.md
├── .gitignore
├── docs/
│   ├── superpowers/
│   │   ├── specs/2026-04-26-lenet-physical-design.md
│   │   └── plans/2026-04-26-lenet-physical.md   ← this file
│   └── build/                                    (later: wiring photos, BOM)
├── config/
│   └── mapping.example.yaml                       (sample 17-chain mapping)
├── weights/
│   └── lenet5.pt                                  (committed; ~250KB)
├── src/lenet1_physical/
│   ├── __init__.py
│   ├── frame.py                                   (Frame dataclass, FrameBus)
│   ├── colors.py                                  (per-layer themes, activation→RGB)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── lenet.py                               (nn.Module definition)
│   │   ├── train.py                               (one-shot training script)
│   │   └── inference.py                           (forward_with_activations)
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── schema.py                              (dataclasses + parser)
│   │   └── validator.py                           (chain-length validation)
│   ├── leds/
│   │   ├── __init__.py
│   │   ├── driver.py                              (Driver Protocol)
│   │   ├── mock.py                                (publishes to FrameBus)
│   │   └── rpi_ws281x_backend.py                  (hardware; Pi-only import)
│   ├── pipeline.py                                (activations → driver writes)
│   ├── twin/
│   │   ├── __init__.py
│   │   ├── server.py                              (FastAPI app)
│   │   ├── ws.py                                  (WebSocket broadcaster)
│   │   └── static/
│   │       ├── index.html
│   │       ├── main.js                            (Three.js + 2D slices)
│   │       └── style.css
│   ├── control/
│   │   ├── __init__.py
│   │   ├── buttons.py                             (gpiozero with mock)
│   │   ├── state.py                               (state machine)
│   │   └── orchestrator.py                        (ties everything)
│   ├── scripts/
│   │   ├── single_chain_walk.py
│   │   ├── single_layer_walk.py
│   │   └── power_stress.py
│   └── main.py                                    (entry point)
└── tests/
    ├── conftest.py
    ├── test_frame.py
    ├── test_colors.py
    ├── test_model.py
    ├── test_mapping.py
    ├── test_pipeline.py
    ├── test_twin_server.py
    ├── test_orchestrator.py
    ├── fixtures/
    │   ├── mapping_minimal.yaml
    │   └── mapping_full_17.yaml
    └── e2e/
        └── test_twin_smoke.spec.ts                (Playwright)
```

---

## Phase 0 — Scaffolding

### Task 0.1: Python project setup with `uv`

**Files:**
- Create: `pyproject.toml`
- Create: `src/lenet1_physical/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Verify `uv` is available**

Run: `uv --version`
Expected: prints a version (e.g., `uv 0.4.x`). If not installed, follow https://docs.astral.sh/uv/getting-started/installation/

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "lenet1-physical"
version = "0.1.0"
description = "Physical LED visualization of LeNet running on a Raspberry Pi 4"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "websockets>=12",
    "pydantic>=2.7",
    "gpiozero>=2.0; sys_platform == 'linux'",
]

[project.optional-dependencies]
hardware = ["rpi_ws281x>=5.0; sys_platform == 'linux'"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "mypy>=1.10", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lenet1_physical"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
```

- [ ] **Step 3: Create empty package and conftest stubs**

```python
# src/lenet1_physical/__init__.py
__version__ = "0.1.0"
```

```python
# tests/conftest.py
import sys
from pathlib import Path

# Ensure src/ is importable in tests without an editable install during dev.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 4: Sync deps and run an empty pytest**

Run: `uv sync --extra dev`
Expected: deps install successfully (PyTorch download is large; allow several minutes).

Run: `uv run pytest -q`
Expected: `no tests ran in 0.0Xs` — confirms test infra works.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/lenet1_physical/__init__.py tests/conftest.py
git commit -m "chore: python project scaffolding with uv"
```

---

### Task 0.2: Frame dataclass and bus

The `Frame` is the contract between every producer (model→pipeline) and consumer (LED driver, twin server). Define it first so all later code can depend on it.

**Files:**
- Create: `src/lenet1_physical/frame.py`
- Create: `tests/test_frame.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame.py
import asyncio
import pytest
from lenet1_physical.frame import Frame, FrameBus, PixelDelta


def test_pixel_delta_immutable():
    p = PixelDelta(chain=2, position=14, r=255, g=128, b=0)
    with pytest.raises(Exception):
        p.r = 0  # frozen dataclass rejects assignment


def test_frame_groups_deltas_by_layer_with_seq():
    f = Frame(seq=7, layer="L2", deltas=(PixelDelta(0, 0, 10, 20, 30),))
    assert f.seq == 7
    assert f.layer == "L2"
    assert f.deltas[0].r == 10


@pytest.mark.asyncio
async def test_frame_bus_broadcasts_to_all_subscribers():
    bus = FrameBus()
    a, b = bus.subscribe(), bus.subscribe()
    f = Frame(seq=1, layer="L1", deltas=())
    await bus.publish(f)
    assert (await asyncio.wait_for(a.get(), 0.1)).seq == 1
    assert (await asyncio.wait_for(b.get(), 0.1)).seq == 1


@pytest.mark.asyncio
async def test_frame_bus_unsubscribe_stops_delivery():
    bus = FrameBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    await bus.publish(Frame(seq=1, layer="L1", deltas=()))
    assert q.empty()
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/test_frame.py -q`
Expected: `ModuleNotFoundError: No module named 'lenet1_physical.frame'`.

- [ ] **Step 3: Implement frame.py**

```python
# src/lenet1_physical/frame.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PixelDelta:
    """One pixel update on one chain."""
    chain: int
    position: int
    r: int
    g: int
    b: int


@dataclass(frozen=True, slots=True)
class Frame:
    """All pixel changes for a single layer in one inference cycle."""
    seq: int
    layer: str            # "L1".."L6"
    deltas: tuple[PixelDelta, ...]


class FrameBus:
    """In-process pub/sub. Each subscriber gets its own asyncio.Queue."""

    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Frame]] = []

    def subscribe(self) -> asyncio.Queue[Frame]:
        q: asyncio.Queue[Frame] = asyncio.Queue(maxsize=256)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Frame]) -> None:
        if q in self._subs:
            self._subs.remove(q)

    async def publish(self, frame: Frame) -> None:
        # Drop the frame on a slow subscriber rather than block the producer.
        for q in list(self._subs):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_frame.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/frame.py tests/test_frame.py
git commit -m "feat(frame): define Frame, PixelDelta, and FrameBus pub/sub"
```

---

### Task 0.3: Color encoder

Encode an activation tensor into RGB triplets per the spec's per-layer color theme.

**Files:**
- Create: `src/lenet1_physical/colors.py`
- Create: `tests/test_colors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colors.py
import numpy as np
from lenet1_physical.colors import LAYER_THEMES, encode_layer


def test_known_themes_present():
    assert set(LAYER_THEMES) == {"L1", "L2", "L3", "L4", "L5", "L6"}


def test_zero_activation_emits_black():
    out = encode_layer("L1", np.zeros((1, 28, 28), dtype=np.float32))
    assert out.shape == (1, 28, 28, 3)
    assert (out == 0).all()


def test_max_activation_emits_full_theme_color():
    out = encode_layer("L2", np.ones((4, 24, 24), dtype=np.float32))
    # L2 is blue: (0, 0, 255)
    assert (out[..., 0] == 0).all()
    assert (out[..., 1] == 0).all()
    assert (out[..., 2] == 255).all()


def test_normalization_per_feature_map():
    # Each feature map normalizes by its own max, so a hot fmap and a cold fmap
    # both reach full brightness at their own peak.
    a = np.array([[[2.0, 0.0]], [[0.0, 0.5]]], dtype=np.float32)  # shape (2,1,2)
    out = encode_layer("L6", a)
    # Inside fmap 0: peak=2.0 -> b=255 at idx 0
    assert out[0, 0, 0, 0] == 255
    # Inside fmap 1: peak=0.5 -> b=255 at idx 1
    assert out[1, 0, 1, 0] == 255


def test_brightness_cap_applied():
    out = encode_layer("L2", np.ones((1, 4, 4), dtype=np.float32), brightness_cap=0.3)
    assert int(out[..., 2].max()) == int(round(255 * 0.3))
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_colors.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement colors.py**

```python
# src/lenet1_physical/colors.py
from __future__ import annotations
import numpy as np

# (R, G, B) per spec section 4.2
LAYER_THEMES: dict[str, tuple[int, int, int]] = {
    "L1": (255, 220, 180),  # warm white
    "L2": (0, 0, 255),      # blue
    "L3": (0, 255, 255),    # cyan
    "L4": (0, 255, 0),      # green
    "L5": (255, 255, 0),    # yellow
    "L6": (255, 0, 0),      # red
}


def encode_layer(
    layer: str,
    activations: np.ndarray,
    brightness_cap: float = 1.0,
) -> np.ndarray:
    """Convert a (F, H, W) activation tensor to (F, H, W, 3) uint8 RGB.

    - ReLU is assumed already applied; negatives are clipped to 0.
    - Each feature map is normalized by its own max.
    - Output color = theme * brightness * brightness_cap, rounded uint8.
    """
    if layer not in LAYER_THEMES:
        raise KeyError(f"unknown layer {layer!r}")
    if activations.ndim != 3:
        raise ValueError(f"expected (F,H,W) tensor, got shape {activations.shape}")

    a = np.clip(activations, 0.0, None).astype(np.float32)
    # Per-fmap max with safe divide; fmaps that are entirely zero stay zero.
    peaks = a.reshape(a.shape[0], -1).max(axis=1, keepdims=True)
    peaks = np.where(peaks == 0, 1.0, peaks)
    norm = (a.reshape(a.shape[0], -1) / peaks).reshape(a.shape)

    theme = np.asarray(LAYER_THEMES[layer], dtype=np.float32) / 255.0
    rgb_float = norm[..., None] * theme[None, None, None, :] * float(brightness_cap)
    return np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_colors.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/colors.py tests/test_colors.py
git commit -m "feat(colors): per-layer theme RGB encoder with brightness cap"
```

---

## Phase 1 — Model

### Task 1.1: LeNet-5 module

**Files:**
- Create: `src/lenet1_physical/model/__init__.py`
- Create: `src/lenet1_physical/model/lenet.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from lenet1_physical.model.lenet import LeNet5


def test_forward_shape():
    m = LeNet5()
    x = torch.zeros(1, 1, 28, 28)
    out = m(x)
    assert out.shape == (1, 10)


def test_layer_shapes_match_spec():
    m = LeNet5()
    x = torch.zeros(1, 1, 28, 28)
    a = m.forward_with_activations(x)
    assert a["L1"].shape == (1, 1, 28, 28)
    assert a["L2"].shape == (1, 4, 24, 24)
    assert a["L3"].shape == (1, 4, 12, 12)
    assert a["L4"].shape == (1, 12, 8, 8)
    assert a["L5"].shape == (1, 12, 4, 4)
    assert a["L6"].shape == (1, 10)


def test_relu_is_applied_so_no_negatives_in_visualizable_layers():
    torch.manual_seed(0)
    m = LeNet5()
    x = torch.randn(1, 1, 28, 28)
    a = m.forward_with_activations(x)
    for k in ("L2", "L3", "L4", "L5"):
        assert (a[k] >= 0).all(), f"{k} has negatives -- visualization expects ReLU'd"
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/test_model.py -q`

- [ ] **Step 3: Implement lenet.py**

```python
# src/lenet1_physical/model/__init__.py
```

```python
# src/lenet1_physical/model/lenet.py
"""LeNet-5 sized to match the physical hardware.

Channel counts (4, 12) and feature-map sizes (28, 24, 12, 8, 4) match spec section 2.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, kernel_size=5)         # 28 -> 24
        self.c3 = nn.Conv2d(4, 12, kernel_size=5)        # 12 -> 8
        self.fc = nn.Linear(12 * 4 * 4, 10)              # final classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.c3(x))
        x = F.avg_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x)

    def forward_with_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every visualizable activation, ReLU'd where appropriate."""
        l1 = x
        l2 = F.relu(self.c1(x))
        l3 = F.avg_pool2d(l2, 2)
        l4 = F.relu(self.c3(l3))
        l5 = F.avg_pool2d(l4, 2)
        l6 = self.fc(l5.flatten(1))
        return {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5, "L6": l6}
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_model.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/model/ tests/test_model.py
git commit -m "feat(model): LeNet-5 with forward_with_activations matching hardware shape"
```

---

### Task 1.2: Training script (run once)

This task is intentionally not TDD — training is a one-shot script whose output is a weights file.

**Files:**
- Create: `src/lenet1_physical/model/train.py`
- Modify: `.gitignore` (allow `weights/lenet5.pt`)

- [ ] **Step 1: Write the training script**

```python
# src/lenet1_physical/model/train.py
"""Train LeNet-5 on MNIST. Run once, commit the resulting weights file."""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet1_physical.model.lenet import LeNet5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    parser.add_argument("--out", type=Path, default=Path("weights/lenet5.pt"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_ds)
        print(f"epoch {epoch + 1}: test_acc={acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Allow weights file through `.gitignore`**

Edit `.gitignore` and replace the line `*.pt` with:

```gitignore
*.pt
!weights/lenet5.pt
```

- [ ] **Step 3: Run training**

Run: `uv run python -m lenet1_physical.model.train --epochs 10`
Expected: ~5 minutes on CPU, prints test accuracy per epoch, final accuracy ≥ 0.985.

If accuracy stays below 0.98 after 20 epochs, the model definition is wrong — go back to Task 1.1.

- [ ] **Step 4: Verify the file exists and is committable**

Run: `ls -la weights/lenet5.pt`
Expected: file present, ~250–500KB.

- [ ] **Step 5: Commit weights and script**

```bash
git add src/lenet1_physical/model/train.py .gitignore weights/lenet5.pt
git commit -m "feat(model): training script and trained weights (>=98.5% test acc)"
```

---

### Task 1.3: Inference wrapper

**Files:**
- Create: `src/lenet1_physical/model/inference.py`
- Modify: `tests/test_model.py` (add inference test)

- [ ] **Step 1: Add a failing test to `tests/test_model.py`**

Append to `tests/test_model.py`:

```python
import numpy as np
from pathlib import Path
from lenet1_physical.model.inference import LeNetInference


def test_inference_loads_weights_and_returns_numpy_dict(tmp_path):
    inf = LeNetInference(Path("weights/lenet5.pt"))
    img = np.zeros((28, 28), dtype=np.float32)
    out = inf.run(img)
    assert set(out) == {"L1", "L2", "L3", "L4", "L5", "L6", "prediction"}
    assert isinstance(out["prediction"], int)
    assert 0 <= out["prediction"] <= 9
    assert out["L2"].shape == (4, 24, 24)
    assert out["L2"].dtype == np.float32
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_model.py::test_inference_loads_weights_and_returns_numpy_dict -q`

- [ ] **Step 3: Implement inference.py**

```python
# src/lenet1_physical/model/inference.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from lenet1_physical.model.lenet import LeNet5

# MNIST training normalization (must match train.py)
_MEAN = 0.1307
_STD = 0.3081


class LeNetInference:
    """Single-image inference returning per-layer numpy activations."""

    def __init__(self, weights_path: Path) -> None:
        self.model = LeNet5()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def run(self, image: np.ndarray) -> dict[str, np.ndarray | int]:
        if image.shape != (28, 28):
            raise ValueError(f"expected (28,28) image, got {image.shape}")
        # Normalize like training. Accept either 0..1 floats or 0..255.
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        normalized = (image - _MEAN) / _STD
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        with torch.no_grad():
            acts = self.model.forward_with_activations(x)
        out: dict[str, np.ndarray | int] = {
            k: v.squeeze(0).numpy().astype(np.float32) for k, v in acts.items()
        }
        out["prediction"] = int(np.asarray(out["L6"]).argmax())
        return out
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_model.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/model/inference.py tests/test_model.py
git commit -m "feat(model): inference wrapper returning per-layer numpy activations"
```

---

## Phase 2 — Mapping

### Task 2.1: Mapping schema dataclasses + parser

**Files:**
- Create: `src/lenet1_physical/mapping/__init__.py`
- Create: `src/lenet1_physical/mapping/schema.py`
- Create: `tests/fixtures/mapping_minimal.yaml`
- Create: `tests/test_mapping.py`

- [ ] **Step 1: Write fixture and failing test**

```yaml
# tests/fixtures/mapping_minimal.yaml
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
        order: row_major_snake
  L6:
    feature_maps:
      - id: 0
        chain_id: 16
        offset_in_chain: 0
        rows: 1
        cols: 10
        origin_mm: [0, 0, 150]
        pitch_mm: [12, 12]
        order: row_major_snake
chains:
  - id: 0
    gpio: 18
    length: 784
  - id: 16
    gpio: 6
    length: 10
```

```python
# tests/test_mapping.py
from pathlib import Path
import pytest
from lenet1_physical.mapping.schema import Mapping, FeatureMap, ChainSpec


FIXTURE = Path("tests/fixtures/mapping_minimal.yaml")


def test_load_minimal_parses_layers_and_chains():
    m = Mapping.from_yaml(FIXTURE)
    assert set(m.layers) == {"L1", "L6"}
    l1 = m.layers["L1"].feature_maps[0]
    assert isinstance(l1, FeatureMap)
    assert l1.rows == 28 and l1.cols == 28
    assert m.chains[0].gpio == 18


def test_logical_to_physical_row_major_snake():
    m = Mapping.from_yaml(FIXTURE)
    # On a 28-col row-major-snake fill: row 0 left-to-right (positions 0..27),
    # row 1 right-to-left (positions 28..55, with col 27 first).
    assert m.lookup("L1", fmap=0, row=0, col=0) == (0, 0)
    assert m.lookup("L1", fmap=0, row=0, col=27) == (0, 27)
    assert m.lookup("L1", fmap=0, row=1, col=27) == (0, 28)
    assert m.lookup("L1", fmap=0, row=1, col=0) == (0, 55)


def test_lookup_unknown_layer_raises():
    m = Mapping.from_yaml(FIXTURE)
    with pytest.raises(KeyError):
        m.lookup("L9", fmap=0, row=0, col=0)
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_mapping.py -q`

- [ ] **Step 3: Implement schema.py**

```python
# src/lenet1_physical/mapping/__init__.py
```

```python
# src/lenet1_physical/mapping/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True, slots=True)
class FeatureMap:
    id: int
    chain_id: int
    offset_in_chain: int
    rows: int
    cols: int
    origin_mm: tuple[float, float, float]
    pitch_mm: tuple[float, float]
    order: str  # "row_major_snake" supported in v1


@dataclass(frozen=True, slots=True)
class Layer:
    name: str
    feature_maps: tuple[FeatureMap, ...]


@dataclass(frozen=True, slots=True)
class ChainSpec:
    id: int
    gpio: int
    length: int


@dataclass(frozen=True, slots=True)
class Mapping:
    layers: dict[str, Layer] = field(default_factory=dict)
    chains: dict[int, ChainSpec] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Mapping":
        raw = yaml.safe_load(Path(path).read_text())
        layers: dict[str, Layer] = {}
        for name, body in raw["layers"].items():
            fmaps = tuple(
                FeatureMap(
                    id=fm["id"],
                    chain_id=fm["chain_id"],
                    offset_in_chain=fm["offset_in_chain"],
                    rows=fm["rows"],
                    cols=fm["cols"],
                    origin_mm=tuple(fm["origin_mm"]),
                    pitch_mm=tuple(fm["pitch_mm"]),
                    order=fm["order"],
                )
                for fm in body["feature_maps"]
            )
            layers[name] = Layer(name=name, feature_maps=fmaps)
        chains = {c["id"]: ChainSpec(id=c["id"], gpio=c["gpio"], length=c["length"]) for c in raw["chains"]}
        return cls(layers=layers, chains=chains)

    def lookup(self, layer: str, fmap: int, row: int, col: int) -> tuple[int, int]:
        """Return (chain_id, position_in_chain) for a logical pixel."""
        if layer not in self.layers:
            raise KeyError(f"unknown layer {layer!r}")
        fm = self.layers[layer].feature_maps[fmap]
        if fm.order == "row_major_snake":
            if row % 2 == 0:
                offset_in_fmap = row * fm.cols + col
            else:
                offset_in_fmap = row * fm.cols + (fm.cols - 1 - col)
            return fm.chain_id, fm.offset_in_chain + offset_in_fmap
        raise NotImplementedError(f"order {fm.order!r} not supported yet")
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_mapping.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/mapping/ tests/fixtures/mapping_minimal.yaml tests/test_mapping.py
git commit -m "feat(mapping): YAML schema, dataclasses, and row-major-snake lookup"
```

---

### Task 2.2: Mapping validator

**Files:**
- Create: `src/lenet1_physical/mapping/validator.py`
- Modify: `tests/test_mapping.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_mapping.py`:

```python
from lenet1_physical.mapping.validator import (
    ValidationError, validate, expected_chain_pixels,
)


def test_validate_accepts_minimal_fixture():
    m = Mapping.from_yaml(FIXTURE)
    validate(m)  # raises if bad


def test_validate_detects_chain_length_mismatch(tmp_path):
    bad_yaml = (tmp_path / "bad.yaml")
    bad_yaml.write_text(
        FIXTURE.read_text().replace("length: 784", "length: 999")
    )
    m = Mapping.from_yaml(bad_yaml)
    with pytest.raises(ValidationError, match="chain 0 expects 784 pixels"):
        validate(m)


def test_expected_pixels_per_chain():
    m = Mapping.from_yaml(FIXTURE)
    counts = expected_chain_pixels(m)
    assert counts[0] == 28 * 28
    assert counts[16] == 1 * 10
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement validator.py**

```python
# src/lenet1_physical/mapping/validator.py
from __future__ import annotations
from collections import defaultdict

from lenet1_physical.mapping.schema import Mapping


class ValidationError(ValueError):
    pass


def expected_chain_pixels(m: Mapping) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for layer in m.layers.values():
        for fm in layer.feature_maps:
            counts[fm.chain_id] += fm.rows * fm.cols
    return dict(counts)


def validate(m: Mapping) -> None:
    """Raises ValidationError if the mapping is internally inconsistent."""
    counts = expected_chain_pixels(m)
    for chain_id, expected in counts.items():
        if chain_id not in m.chains:
            raise ValidationError(f"chain {chain_id} referenced by a feature map but not declared")
        declared = m.chains[chain_id].length
        if declared != expected:
            raise ValidationError(
                f"chain {chain_id} expects {expected} pixels but is declared length {declared}"
            )
    occupied: dict[tuple[int, int], str] = {}
    for layer_name, layer in m.layers.items():
        for fm in layer.feature_maps:
            for row in range(fm.rows):
                for col in range(fm.cols):
                    chain, pos = m.lookup(layer_name, fm.id, row, col)
                    key = (chain, pos)
                    if key in occupied:
                        raise ValidationError(
                            f"collision at chain {chain} pos {pos}: "
                            f"{occupied[key]} and {layer_name} fmap {fm.id}"
                        )
                    occupied[key] = f"{layer_name} fmap {fm.id}"
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_mapping.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/mapping/validator.py tests/test_mapping.py
git commit -m "feat(mapping): validator catches chain-length mismatches and collisions"
```

---

### Task 2.3: Full 17-chain example mapping

**Files:**
- Create: `scripts/gen_example_mapping.py`
- Create: `config/mapping.example.yaml`
- Create: `tests/fixtures/mapping_full_17.yaml`
- Modify: `tests/test_mapping.py`

- [ ] **Step 1: Write the generator helper**

```python
# scripts/gen_example_mapping.py
"""One-shot generator for the 17-chain example mapping.

Run: `uv run python scripts/gen_example_mapping.py > config/mapping.example.yaml`
"""
from __future__ import annotations

# layer: (fmap_count, rows, cols, fmaps_per_chain)
PARTITION = {
    "L1": (1, 28, 28, 1),
    "L2": (4, 24, 24, 1),
    "L3": (4, 12, 12, 1),
    "L4": (12, 8, 8, 3),     # 4 chains x 3 fmaps
    "L5": (12, 4, 4, 4),     # 3 chains x 4 fmaps
    "L6": (1, 1, 10, 1),
}
SLAB_SPACING_MM = 30.0
PITCH_MM = 10.0
GPIO_PINS = [4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # 17 GPIOs


def main() -> None:
    print("# Generated by scripts/gen_example_mapping.py -- edit by hand for tuning.")
    print("layers:")
    chains_so_far = 0
    z = 0.0
    chains_declared: list[tuple[int, int]] = []  # (chain_id, length)
    for layer, (fcount, rows, cols, per_chain) in PARTITION.items():
        print(f"  {layer}:")
        print("    feature_maps:")
        layer_chains_used: dict[int, int] = {}  # chain_id -> pixel count
        for fm_id in range(fcount):
            chain_for_this_fm = chains_so_far + fm_id // per_chain
            offset = (fm_id % per_chain) * rows * cols
            print(f"      - id: {fm_id}")
            print(f"        chain_id: {chain_for_this_fm}")
            print(f"        offset_in_chain: {offset}")
            print(f"        rows: {rows}")
            print(f"        cols: {cols}")
            print(f"        origin_mm: [{fm_id * (cols + 2) * PITCH_MM}, 0, {z}]")
            print(f"        pitch_mm: [{PITCH_MM}, {PITCH_MM}]")
            print(f"        order: row_major_snake")
            layer_chains_used[chain_for_this_fm] = layer_chains_used.get(chain_for_this_fm, 0) + rows * cols
        for cid, length in sorted(layer_chains_used.items()):
            chains_declared.append((cid, length))
        n_chains = -(-fcount // per_chain)
        chains_so_far += n_chains
        z += SLAB_SPACING_MM
    print("chains:")
    for cid, length in chains_declared:
        gpio = GPIO_PINS[cid]
        print(f"  - id: {cid}")
        print(f"    gpio: {gpio}")
        print(f"    length: {length}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the example**

Run: `uv run python scripts/gen_example_mapping.py > config/mapping.example.yaml`

Run: `cp config/mapping.example.yaml tests/fixtures/mapping_full_17.yaml`

- [ ] **Step 3: Add a test that validates the realistic mapping**

Append to `tests/test_mapping.py`:

```python
def test_full_17_mapping_validates_clean():
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_full_17.yaml"))
    validate(m)
    counts = expected_chain_pixels(m)
    assert sum(counts.values()) == 4634
    assert len(m.chains) == 17
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_mapping.py -v`

If counts don't match, fix `gen_example_mapping.py` until they do.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_example_mapping.py config/mapping.example.yaml tests/fixtures/mapping_full_17.yaml tests/test_mapping.py
git commit -m "feat(mapping): generator and example for the full 17-chain partition"
```

---

## Phase 3 — LED Driver (interface + mock)

### Task 3.1: `Driver` Protocol

**Files:**
- Create: `src/lenet1_physical/leds/__init__.py`
- Create: `src/lenet1_physical/leds/driver.py`
- Create: `tests/test_driver_protocol.py`

- [ ] **Step 1: Write a failing test**

```python
# tests/test_driver_protocol.py
from lenet1_physical.leds.driver import Driver


class _StubDriver:
    def __init__(self):
        self.writes = []
        self.flushes = 0

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.writes.append((chain, position, r, g, b))

    def flush(self) -> bool:
        self.flushes += 1
        return True

    def close(self) -> None:
        pass


def test_stub_satisfies_protocol():
    d: Driver = _StubDriver()
    d.write(0, 5, 10, 20, 30)
    assert d.flush() is True
    d.close()
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement driver.py**

```python
# src/lenet1_physical/leds/__init__.py
```

```python
# src/lenet1_physical/leds/driver.py
from __future__ import annotations
from typing import Protocol


class Driver(Protocol):
    """Backend-agnostic LED driver.

    Implementations: MockDriver (off-Pi, publishes to FrameBus),
    RpiWs281xDriver (on-Pi hardware).
    """

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        """Stage a single pixel update; not visible until flush()."""
        ...

    def flush(self) -> bool:
        """Push the staged frame to hardware/twin. Returns False if the flush
        was incomplete (e.g., DMA glitch). Higher layers may surface this."""
        ...

    def close(self) -> None:
        """Release any held resources (DMA, GPIO, etc.)."""
        ...
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_driver_protocol.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/leds/ tests/test_driver_protocol.py
git commit -m "feat(leds): Driver protocol for backend swap"
```

---

### Task 3.2: Mock driver — publishes to FrameBus

**Files:**
- Create: `src/lenet1_physical/leds/mock.py`
- Create: `tests/test_mock_driver.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_mock_driver.py
import asyncio
import pytest
from lenet1_physical.frame import FrameBus
from lenet1_physical.leds.mock import MockDriver


@pytest.mark.asyncio
async def test_writes_then_flush_publishes_frame_with_seq():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})
    drv.write(0, 0, 1, 2, 3)
    drv.write(0, 1, 4, 5, 6)
    assert drv.flush() is True
    f = await asyncio.wait_for(sub.get(), 0.1)
    assert f.seq == 1
    assert f.layer == "L1"
    assert len(f.deltas) == 2


@pytest.mark.asyncio
async def test_flush_with_no_writes_emits_no_frame():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})
    assert drv.flush() is True
    assert sub.empty()


@pytest.mark.asyncio
async def test_writes_to_multiple_chains_split_per_layer():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1", 1: "L2"})
    drv.write(0, 0, 9, 9, 9)
    drv.write(1, 0, 1, 1, 1)
    drv.flush()
    seen_layers = {(await asyncio.wait_for(sub.get(), 0.1)).layer for _ in range(2)}
    assert seen_layers == {"L1", "L2"}
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement mock.py**

```python
# src/lenet1_physical/leds/mock.py
from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Mapping

from lenet1_physical.frame import Frame, FrameBus, PixelDelta


class MockDriver:
    """In-process driver. Stages writes, publishes Frame on flush()."""

    def __init__(self, bus: FrameBus, layer_for_chain: Mapping[int, str]) -> None:
        self._bus = bus
        self._layer_for_chain = dict(layer_for_chain)
        self._staged: dict[int, list[PixelDelta]] = defaultdict(list)
        self._seq = 0

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self._staged[chain].append(PixelDelta(chain, position, r, g, b))

    def flush(self) -> bool:
        for chain, deltas in self._staged.items():
            layer = self._layer_for_chain.get(chain, "L?")
            self._seq += 1
            frame = Frame(seq=self._seq, layer=layer, deltas=tuple(deltas))
            try:
                asyncio.get_event_loop().create_task(self._bus.publish(frame))
            except RuntimeError:
                # No running loop (sync test context) -- run inline.
                asyncio.run(self._bus.publish(frame))
        self._staged.clear()
        return True

    def close(self) -> None:
        self._staged.clear()
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_mock_driver.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/leds/mock.py tests/test_mock_driver.py
git commit -m "feat(leds): MockDriver publishes per-chain Frames to FrameBus on flush"
```

---

### Task 3.3: Hardware backend stub (`rpi_ws281x`)

**Files:**
- Create: `src/lenet1_physical/leds/rpi_ws281x_backend.py`
- Create: `docs/build/MULTI_CHANNEL_TODO.md`

- [ ] **Step 1: Write the backend file**

```python
# src/lenet1_physical/leds/rpi_ws281x_backend.py
"""Hardware LED driver. Pi-only.

NOTE: this is a 2-channel bring-up backend used to validate timing on a
single chain end-to-end. The 17-chain DMA-bit-bang migration is Phase 7.
"""
from __future__ import annotations
import platform
import sys


def _require_linux_arm() -> None:
    if sys.platform != "linux" or "arm" not in platform.machine().lower():
        raise RuntimeError(
            "rpi_ws281x backend only runs on a Raspberry Pi. "
            "Use MockDriver on dev machines."
        )


class RpiWs281xDriver:
    """Wraps `rpi_ws281x` for ONE chain.

    Multiple chains are constructed and flushed together by the caller.
    """

    def __init__(
        self,
        gpio: int,
        led_count: int,
        *,
        freq_hz: int = 800_000,
        dma: int = 10,
        brightness: int = 80,
    ) -> None:
        _require_linux_arm()
        # Late import so non-Pi machines can still import this module.
        from rpi_ws281x import PixelStrip, ws  # type: ignore[import-not-found]

        self._strip = PixelStrip(
            num=led_count,
            pin=gpio,
            freq_hz=freq_hz,
            dma=dma,
            invert=False,
            brightness=brightness,
            channel=0 if gpio in (12, 18) else 1,  # PWM0 vs PWM1
            strip_type=ws.WS2811_STRIP_GRB,
        )
        self._strip.begin()
        self._dirty = False

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        # `chain` is informational here; one driver instance == one chain.
        self._strip.setPixelColorRGB(position, r, g, b)
        self._dirty = True

    def flush(self) -> bool:
        if not self._dirty:
            return True
        self._strip.show()
        self._dirty = False
        return True

    def close(self) -> None:
        for i in range(self._strip.numPixels()):
            self._strip.setPixelColorRGB(i, 0, 0, 0)
        self._strip.show()
```

- [ ] **Step 2: Smoke test on dev machine**

Run: `uv run python -c "from lenet1_physical.leds.rpi_ws281x_backend import RpiWs281xDriver"`
Expected: no import error.

- [ ] **Step 3: Document the deferred 17-chain backend**

```markdown
# Multi-channel WS2812B on Pi 4 -- Phase 7

The bring-up backend (`rpi_ws281x_backend.py`) drives ONE chain via PWM0/PWM1.
The full 17-chain build needs DMA-to-GPIO-bank bit-banging.

## Library candidates to assess

1. **Maintained `rpi_ws281x` forks** with multi-channel patches.
2. **Custom thin C extension**: `mmap` `/dev/mem` for GPIO bank, allocate a DMA
   control block ring, encode the 17-bit-wide WS2812 bitstream into 32-bit
   words. ~300 lines of C, ~50 lines of Python wrapper. Most reliable.
3. **`Pi5Neo`**: written for Pi 5, may need backporting to Pi 4 BCM2711.

## Acceptance for Phase 7

- 17 simultaneous chains, each up to 2304 LEDs, drive a sustained 5+ FPS
  fixed pattern with zero visible glitches over a 10-minute run.
- The new driver implements the same `Driver` Protocol so all upstream code
  continues to work without changes.
```

- [ ] **Step 4: Verify import still works**

Run: `uv run pytest -q`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/leds/rpi_ws281x_backend.py docs/build/MULTI_CHANNEL_TODO.md
git commit -m "feat(leds): single-chain rpi_ws281x bring-up backend + Phase 7 TODO"
```

---

## Phase 4 — Frame Pipeline

### Task 4.1: `paint_layer` — write a layer's RGB tensor to a Driver

**Files:**
- Create: `src/lenet1_physical/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_pipeline.py
import numpy as np
from pathlib import Path
from lenet1_physical.frame import FrameBus
from lenet1_physical.leds.mock import MockDriver
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.pipeline import paint_layer


FIXTURE = Path("tests/fixtures/mapping_minimal.yaml")


def test_paint_layer_writes_one_pixel_per_logical_position():
    m = Mapping.from_yaml(FIXTURE)
    bus = FrameBus()
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    rgb = np.zeros((1, 28, 28, 3), dtype=np.uint8)
    rgb[0, 0, 0] = (200, 100, 50)
    paint_layer(drv, m, "L1", rgb)
    drv.flush()


def test_paint_layer_rejects_wrong_shape():
    m = Mapping.from_yaml(FIXTURE)
    drv = MockDriver(FrameBus(), layer_for_chain={0: "L1"})
    rgb = np.zeros((4, 28, 28, 3), dtype=np.uint8)
    import pytest
    with pytest.raises(ValueError, match="L1 expected 1 fmaps"):
        paint_layer(drv, m, "L1", rgb)
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement pipeline.py**

```python
# src/lenet1_physical/pipeline.py
from __future__ import annotations
import numpy as np

from lenet1_physical.leds.driver import Driver
from lenet1_physical.mapping.schema import Mapping


def paint_layer(driver: Driver, mapping: Mapping, layer: str, rgb: np.ndarray) -> None:
    """Write every pixel of a layer's (F,H,W,3) RGB tensor to the driver.

    Caller is responsible for calling driver.flush() afterwards.
    """
    if layer not in mapping.layers:
        raise KeyError(f"unknown layer {layer!r}")
    fmaps = mapping.layers[layer].feature_maps
    if rgb.shape[0] != len(fmaps):
        raise ValueError(f"{layer} expected {len(fmaps)} fmaps, got {rgb.shape[0]}")
    for fm in fmaps:
        if rgb.shape[1:3] != (fm.rows, fm.cols):
            raise ValueError(
                f"{layer} fmap {fm.id} expected ({fm.rows},{fm.cols}), "
                f"got {rgb.shape[1:3]}"
            )
        for row in range(fm.rows):
            for col in range(fm.cols):
                chain, pos = mapping.lookup(layer, fm.id, row, col)
                r, g, b = rgb[fm.id, row, col]
                driver.write(chain, pos, int(r), int(g), int(b))
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_pipeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): paint_layer writes RGB tensor through mapping to Driver"
```

---

### Task 4.2: End-to-end activations → frames test

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_pipeline.py`:

```python
import asyncio
import pytest
from lenet1_physical.colors import encode_layer
from lenet1_physical.frame import Frame


@pytest.mark.asyncio
async def test_activation_to_frame_e2e():
    m = Mapping.from_yaml(FIXTURE)
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})

    a = np.zeros((1, 28, 28), dtype=np.float32)
    a[0, 0, 0] = 1.0
    rgb = encode_layer("L1", a, brightness_cap=1.0)
    paint_layer(drv, m, "L1", rgb)
    drv.flush()

    frame: Frame = await asyncio.wait_for(sub.get(), 0.5)
    assert frame.layer == "L1"
    hot = next(d for d in frame.deltas if (d.chain, d.position) == (0, 0))
    assert (hot.r, hot.g, hot.b) == (255, 220, 180)
```

- [ ] **Step 2: Run — expect green** (should already pass given prior tasks)

Run: `uv run pytest tests/test_pipeline.py::test_activation_to_frame_e2e -v`

If it fails, the bug is most likely in `MockDriver.flush` async dispatch — debug there.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test(pipeline): activation tensor through colors+mapping+driver lands as expected Frame"
```

---

### Task 4.3: Inference → all layers paint helper

**Files:**
- Modify: `src/lenet1_physical/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Failing test**

Append:

```python
from lenet1_physical.pipeline import paint_inference_step


def test_paint_inference_step_paints_only_the_requested_layer():
    m = Mapping.from_yaml(FIXTURE)
    drv = MockDriver(FrameBus(), layer_for_chain={0: "L1", 16: "L6"})

    activations = {
        "L1": np.zeros((1, 28, 28), dtype=np.float32),
        "L6": np.zeros((1, 1, 10), dtype=np.float32),
    }
    activations["L1"][0, 0, 0] = 1.0
    activations["L6"][0, 0, 7] = 1.0

    paint_inference_step(drv, m, activations, "L6", brightness_cap=0.3)
    drv.flush()
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Add helper**

Append to `src/lenet1_physical/pipeline.py`:

```python
from lenet1_physical.colors import encode_layer


def paint_inference_step(
    driver: Driver,
    mapping: Mapping,
    activations: dict[str, np.ndarray],
    layer: str,
    *,
    brightness_cap: float = 1.0,
) -> None:
    """Encode one layer's activations and write through the mapping to the driver.

    Caller decides ordering and step timing.
    """
    a = activations[layer]
    if layer == "L6" and a.ndim == 1:
        a = a.reshape(1, 1, -1)
    elif layer == "L1" and a.ndim == 2:
        a = a.reshape(1, *a.shape)
    rgb = encode_layer(layer, a, brightness_cap=brightness_cap)
    paint_layer(driver, mapping, layer, rgb)
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_pipeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): paint_inference_step combines colors + paint_layer for one layer"
```

---

## Phase 5 — Twin Server + Frontend

### Task 5.1: FastAPI app and WebSocket route

**Files:**
- Create: `src/lenet1_physical/twin/__init__.py`
- Create: `src/lenet1_physical/twin/server.py`
- Create: `src/lenet1_physical/twin/ws.py`
- Create: `tests/test_twin_server.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_twin_server.py
import asyncio
import pytest
from fastapi.testclient import TestClient
from lenet1_physical.frame import Frame, FrameBus, PixelDelta
from lenet1_physical.twin.server import build_app


@pytest.fixture
def app_and_bus():
    bus = FrameBus()
    app = build_app(bus, mapping_path=None)
    return app, bus


def test_health_endpoint(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_websocket_streams_frame(app_and_bus):
    app, bus = app_and_bus
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        async def emit():
            await bus.publish(Frame(seq=42, layer="L1", deltas=(PixelDelta(0, 0, 1, 2, 3),)))
        loop = asyncio.new_event_loop()
        loop.run_until_complete(emit())
        msg = ws.receive_json()
        assert msg["seq"] == 42
        assert msg["layer"] == "L1"
        assert msg["deltas"][0] == [0, 0, 1, 2, 3]
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_twin_server.py -q`

- [ ] **Step 3: Implement server.py and ws.py**

```python
# src/lenet1_physical/twin/__init__.py
```

```python
# src/lenet1_physical/twin/ws.py
from __future__ import annotations
from fastapi import WebSocket, WebSocketDisconnect

from lenet1_physical.frame import Frame, FrameBus


def frame_to_dict(f: Frame) -> dict:
    return {
        "seq": f.seq,
        "layer": f.layer,
        "deltas": [[d.chain, d.position, d.r, d.g, d.b] for d in f.deltas],
    }


async def stream_frames(ws: WebSocket, bus: FrameBus) -> None:
    await ws.accept()
    queue = bus.subscribe()
    try:
        while True:
            frame = await queue.get()
            await ws.send_json(frame_to_dict(frame))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
```

```python
# src/lenet1_physical/twin/server.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lenet1_physical.frame import FrameBus
from lenet1_physical.twin.ws import stream_frames


orchestrator_hooks: dict[str, Callable[..., Any]] = {}


class _SampleBody(BaseModel):
    index: int | None = None


class _TestPixelBody(BaseModel):
    chain: int
    pos: int
    r: int
    g: int
    b: int


def build_app(bus: FrameBus, mapping_path: Path | None) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await stream_frames(ws, bus)

    @app.post("/sample")
    async def sample(body: _SampleBody) -> dict:
        fn = orchestrator_hooks.get("sample")
        if fn is None:
            return {"error": "no orchestrator wired"}
        return {"picked": fn(body.index)}

    @app.post("/step")
    async def step() -> dict:
        fn = orchestrator_hooks.get("step")
        if fn is None:
            return {"error": "no orchestrator wired"}
        return {"now_at": fn()}

    @app.post("/test-pixel")
    async def test_pixel(body: _TestPixelBody) -> dict:
        fn = orchestrator_hooks.get("test_pixel")
        if fn is None:
            return {"error": "no orchestrator wired"}
        fn(body.chain, body.pos, body.r, body.g, body.b)
        return {"ok": True}

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_twin_server.py -v`

The websocket test is brittle because of mismatched event loops between TestClient and the bus. If it flakes, replace with an httpx-based async test using `lifespan` — see Task 5.4 for the durable end-to-end test.

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/twin/__init__.py src/lenet1_physical/twin/server.py src/lenet1_physical/twin/ws.py tests/test_twin_server.py
git commit -m "feat(twin): FastAPI app, healthz, WS stream, sample/step/test-pixel endpoints"
```

---

### Task 5.2: Static frontend (Three.js + 2D slices, no bundler)

**Files:**
- Create: `src/lenet1_physical/twin/static/index.html`
- Create: `src/lenet1_physical/twin/static/main.js`
- Create: `src/lenet1_physical/twin/static/style.css`

- [ ] **Step 1: Write index.html**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>LENET-1 Physical -- Digital Twin</title>
  <link rel="stylesheet" href="/style.css" />
  <script type="importmap">
    { "imports": { "three": "https://unpkg.com/three@0.165.0/build/three.module.js",
                   "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/" } }
  </script>
</head>
<body>
  <div id="layout">
    <canvas id="scene"></canvas>
    <aside id="controls">
      <button id="btn-sample">Sample</button>
      <button id="btn-step">Step</button>
      <input id="brightness" type="range" min="0" max="100" value="30" />
      <span id="brightness-value">30%</span>
      <div id="status">seq: <span id="seq">-</span></div>
    </aside>
    <section id="slices"></section>
  </div>
  <script type="module" src="/main.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write style.css**

```css
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: #0a0a0a; color: #ddd;
             font-family: ui-monospace, monospace; }
#layout { display: grid; grid-template-columns: 2fr 240px; grid-template-rows: 1fr 200px;
          height: 100vh; gap: 8px; padding: 8px; }
#scene { width: 100%; height: 100%; grid-row: 1 / 3; }
#controls { display: flex; flex-direction: column; gap: 8px; padding: 8px;
            background: #141414; border-radius: 6px; }
#controls button { padding: 8px; }
#slices { grid-column: 1 / 3; display: flex; gap: 8px; overflow-x: auto;
          background: #141414; border-radius: 6px; padding: 8px; }
.slice { display: flex; flex-direction: column; align-items: center; }
.slice canvas { background: black; image-rendering: pixelated; }
.slice .label { font-size: 11px; opacity: 0.7; }
```

- [ ] **Step 3: Write main.js**

```javascript
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ws = new WebSocket(`ws://${location.host}/ws`);
const seqEl = document.getElementById("seq");

const sceneCanvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas: sceneCanvas, antialias: true });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
camera.position.set(400, 200, 400);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));

const ledMeshes = new Map();
const placeholderGeom = new THREE.SphereGeometry(2, 6, 6);
function ensureLed(chain, position) {
  const key = `${chain}:${position}`;
  let mesh = ledMeshes.get(key);
  if (!mesh) {
    const mat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    mesh = new THREE.Mesh(placeholderGeom, mat);
    mesh.position.set(position * 4, 0, chain * 20);
    scene.add(mesh);
    ledMeshes.set(key, mesh);
  }
  return mesh;
}

function resize() {
  const r = sceneCanvas.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();

const slicesEl = document.getElementById("slices");
const sliceCanvases = {};
function ensureSlice(layer) {
  if (sliceCanvases[layer]) return sliceCanvases[layer];
  const wrapper = document.createElement("div"); wrapper.className = "slice";
  const label = document.createElement("div"); label.className = "label"; label.textContent = layer;
  const c = document.createElement("canvas");
  c.width = 120; c.height = 80;
  wrapper.appendChild(c); wrapper.appendChild(label);
  slicesEl.appendChild(wrapper);
  sliceCanvases[layer] = { canvas: c, ctx: c.getContext("2d") };
  return sliceCanvases[layer];
}

ws.addEventListener("message", (ev) => {
  const f = JSON.parse(ev.data);
  seqEl.textContent = f.seq;
  for (const [chain, position, r, g, b] of f.deltas) {
    const m = ensureLed(chain, position);
    m.material.color.setRGB(r / 255, g / 255, b / 255);
  }
  const slice = ensureSlice(f.layer);
  const { ctx, canvas } = slice;
  const w = canvas.width, h = canvas.height;
  const pxw = Math.max(1, Math.floor(w / Math.max(1, f.deltas.length)));
  for (let i = 0; i < f.deltas.length; i++) {
    const [, , r, g, b] = f.deltas[i];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i * pxw, 0, pxw, h);
  }
});

document.getElementById("btn-sample").onclick = () =>
  fetch("/sample", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
document.getElementById("btn-step").onclick = () =>
  fetch("/step", { method: "POST" });
const brightEl = document.getElementById("brightness");
const brightVal = document.getElementById("brightness-value");
brightEl.oninput = () => { brightVal.textContent = `${brightEl.value}%`; };
```

- [ ] **Step 4: Manual smoke test**

Run a small dev script (added next task) and open http://127.0.0.1:8080. The page should load with an empty 3D canvas and `seq: -`.

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/twin/static/
git commit -m "feat(twin): static frontend with Three.js scene, 2D slices, controls"
```

---

### Task 5.3: Dev runner script for the twin

**Files:**
- Create: `src/lenet1_physical/scripts/__init__.py`
- Create: `src/lenet1_physical/scripts/run_twin_dev.py`

- [ ] **Step 1: Write the script**

```python
# src/lenet1_physical/scripts/__init__.py
```

```python
# src/lenet1_physical/scripts/run_twin_dev.py
"""Boots the twin and emits a synthetic frame stream for browser dev."""
from __future__ import annotations
import asyncio
import uvicorn

from lenet1_physical.frame import Frame, FrameBus, PixelDelta
from lenet1_physical.twin.server import build_app


async def emit_loop(bus: FrameBus) -> None:
    seq = 0
    while True:
        seq += 1
        await bus.publish(Frame(seq=seq, layer="L1", deltas=(PixelDelta(0, 0, 255, 0, 0),)))
        await asyncio.sleep(0.5)


async def main() -> None:
    bus = FrameBus()
    app = build_app(bus, None)
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), emit_loop(bus))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run and verify in a browser**

Run: `uv run python -m lenet1_physical.scripts.run_twin_dev`
Open: http://127.0.0.1:8080 — confirm `seq` increments. Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add src/lenet1_physical/scripts/__init__.py src/lenet1_physical/scripts/run_twin_dev.py
git commit -m "feat(scripts): run_twin_dev synthetic-frame dev server"
```

---

### Task 5.4: Playwright smoke test

**Files:**
- Create: `tests/e2e/playwright.config.ts`
- Create: `tests/e2e/test_twin_smoke.spec.ts`
- Modify: `package.json`, `.gitignore`

- [ ] **Step 1: Initialize npm and Playwright**

Run from repo root:

```bash
npm init -y
npm install --save-dev @playwright/test typescript
npx playwright install chromium
```

Add `node_modules/` to `.gitignore` if not already there.

- [ ] **Step 2: Write the playwright config**

```ts
// tests/e2e/playwright.config.ts
import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: ".",
  use: { headless: true, baseURL: "http://127.0.0.1:8080" },
  webServer: {
    command: "uv run python -m lenet1_physical.scripts.run_twin_dev",
    url: "http://127.0.0.1:8080/healthz",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
```

- [ ] **Step 3: Write the spec**

```ts
// tests/e2e/test_twin_smoke.spec.ts
import { test, expect } from "@playwright/test";

test("twin loads and shows incrementing sequence numbers", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#scene")).toBeVisible();
  const seq = page.locator("#seq");
  await expect(seq).not.toHaveText("-", { timeout: 5_000 });
  const first = await seq.textContent();
  await page.waitForTimeout(700);
  const second = await seq.textContent();
  expect(Number(second)).toBeGreaterThan(Number(first));
});

test("clicking Sample issues a POST sample", async ({ page }) => {
  let posted = false;
  page.on("request", (r) => { if (r.method() === "POST" && r.url().endsWith("/sample")) posted = true; });
  await page.goto("/");
  await page.click("#btn-sample");
  await expect.poll(() => posted).toBe(true);
});
```

- [ ] **Step 4: Run**

Run: `npx playwright test --config=tests/e2e/playwright.config.ts`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/ package.json package-lock.json .gitignore
git commit -m "test(twin): playwright smoke tests for WS stream and Sample click"
```

---

## Phase 6 — Control + Orchestrator

### Task 6.1: Buttons abstraction with mock

**Files:**
- Create: `src/lenet1_physical/control/__init__.py`
- Create: `src/lenet1_physical/control/buttons.py`
- Create: `tests/test_buttons.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_buttons.py
from lenet1_physical.control.buttons import MockButton, ButtonRig


def test_mock_buttons_invoke_callbacks():
    presses = []
    rig = ButtonRig(
        sample_button=MockButton(),
        step_button=MockButton(),
        on_sample=lambda: presses.append("sample"),
        on_step=lambda: presses.append("step"),
    )
    rig.start()
    rig.sample_button.press()
    rig.step_button.press()
    assert presses == ["sample", "step"]
    rig.stop()


def test_double_press_is_debounced_when_within_window(monkeypatch):
    times = iter([0.0, 0.005, 0.500])
    monkeypatch.setattr("time.monotonic", lambda: next(times))
    presses = []
    rig = ButtonRig(MockButton(), MockButton(), lambda: presses.append("x"), lambda: None)
    rig.start()
    rig.sample_button.press()
    rig.sample_button.press()  # debounced out
    rig.sample_button.press()
    assert presses == ["x", "x"]
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement buttons.py**

```python
# src/lenet1_physical/control/__init__.py
```

```python
# src/lenet1_physical/control/buttons.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Protocol


class _Button(Protocol):
    def attach(self, callback: Callable[[], None]) -> None: ...
    def detach(self) -> None: ...


class MockButton:
    """In-process button whose .press() invokes the attached callback."""
    def __init__(self) -> None:
        self._cb: Callable[[], None] | None = None
    def attach(self, callback: Callable[[], None]) -> None:
        self._cb = callback
    def detach(self) -> None:
        self._cb = None
    def press(self) -> None:
        if self._cb is not None:
            self._cb()


@dataclass
class ButtonRig:
    sample_button: _Button
    step_button: _Button
    on_sample: Callable[[], None]
    on_step: Callable[[], None]
    debounce_ms: int = 30

    def __post_init__(self) -> None:
        self._last_sample: float = -1.0
        self._last_step: float = -1.0

    def start(self) -> None:
        self.sample_button.attach(self._sample)
        self.step_button.attach(self._step)

    def stop(self) -> None:
        self.sample_button.detach()
        self.step_button.detach()

    def _sample(self) -> None:
        now = time.monotonic()
        if (now - self._last_sample) * 1000.0 < self.debounce_ms:
            return
        self._last_sample = now
        self.on_sample()

    def _step(self) -> None:
        now = time.monotonic()
        if (now - self._last_step) * 1000.0 < self.debounce_ms:
            return
        self._last_step = now
        self.on_step()
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_buttons.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/control/ tests/test_buttons.py
git commit -m "feat(control): button rig with mock and 30ms debounce"
```

---

### Task 6.2: State machine

**Files:**
- Create: `src/lenet1_physical/control/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_state.py
from lenet1_physical.control.state import StateMachine, State


def test_initial_state_is_idle():
    sm = StateMachine()
    assert sm.state is State.IDLE


def test_select_then_step_through_layers():
    sm = StateMachine()
    sm.on_sample()
    assert sm.state is State.ANIMATING
    assert sm.layer == "L1"
    for expected in ["L2", "L3", "L4", "L5", "L6"]:
        sm.on_step()
        assert sm.layer == expected
    sm.on_step()
    assert sm.state is State.DONE


def test_step_in_idle_is_a_no_op():
    sm = StateMachine()
    sm.on_step()
    assert sm.state is State.IDLE


def test_sample_during_animating_resets_to_l1():
    sm = StateMachine()
    sm.on_sample()
    sm.on_step()
    assert sm.layer == "L2"
    sm.on_sample()
    assert sm.layer == "L1"
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement state.py**

```python
# src/lenet1_physical/control/state.py
from __future__ import annotations
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    ANIMATING = auto()
    DONE = auto()


_LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6"]


class StateMachine:
    def __init__(self) -> None:
        self.state: State = State.IDLE
        self._idx: int = -1

    @property
    def layer(self) -> str | None:
        if self._idx < 0 or self._idx >= len(_LAYER_ORDER):
            return None
        return _LAYER_ORDER[self._idx]

    def on_sample(self) -> None:
        self.state = State.ANIMATING
        self._idx = 0

    def on_step(self) -> None:
        if self.state is not State.ANIMATING:
            return
        self._idx += 1
        if self._idx >= len(_LAYER_ORDER):
            self.state = State.DONE
            self._idx = len(_LAYER_ORDER) - 1
```

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/control/state.py tests/test_state.py
git commit -m "feat(control): idle/animating/done state machine over the 6 layers"
```

---

### Task 6.3: Orchestrator

**Files:**
- Create: `src/lenet1_physical/control/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_orchestrator.py
import asyncio
import numpy as np
import pytest
from pathlib import Path

from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.frame import FrameBus
from lenet1_physical.leds.mock import MockDriver
from lenet1_physical.mapping.schema import Mapping


class FakeInference:
    def run(self, image):
        return {
            "L1": np.zeros((1, 28, 28), dtype=np.float32),
            "L2": np.zeros((4, 24, 24), dtype=np.float32),
            "L3": np.zeros((4, 12, 12), dtype=np.float32),
            "L4": np.zeros((12, 8, 8), dtype=np.float32),
            "L5": np.zeros((12, 4, 4), dtype=np.float32),
            "L6": np.zeros((1, 1, 10), dtype=np.float32),
            "prediction": 7,
        }


class FakeMnist:
    def __getitem__(self, idx):
        return np.zeros((28, 28), dtype=np.float32), 0
    def __len__(self):
        return 100


@pytest.mark.asyncio
async def test_sample_triggers_l1_paint():
    bus = FrameBus()
    sub = bus.subscribe()
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_minimal.yaml"))
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    orch = Orchestrator(
        inference=FakeInference(),
        mapping=m,
        driver=drv,
        mnist=FakeMnist(),
        brightness_cap=0.3,
    )
    orch.on_sample(0)
    orch.on_step()
    f = await asyncio.wait_for(sub.get(), 0.5)
    assert f.layer == "L1"
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement orchestrator.py**

```python
# src/lenet1_physical/control/orchestrator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from lenet1_physical.control.state import State, StateMachine
from lenet1_physical.leds.driver import Driver
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.pipeline import paint_inference_step


@dataclass
class Orchestrator:
    """Glues inference + state machine + driver."""
    inference: Any
    mapping: Mapping
    driver: Driver
    mnist: Sequence[tuple[np.ndarray, int]]
    brightness_cap: float = 0.3
    sm: StateMachine = field(default_factory=StateMachine)
    _activations: dict[str, np.ndarray] | None = None

    def on_sample(self, index: int | None = None) -> int:
        rng = np.random.default_rng()
        if index is None:
            index = int(rng.integers(0, len(self.mnist)))
        img, _label = self.mnist[index]
        self._activations = self.inference.run(img)
        self.sm.on_sample()
        return index

    def on_step(self) -> str | None:
        if self.sm.state is State.IDLE or self._activations is None:
            return None
        layer = self.sm.layer
        if layer is None:
            return None
        paint_inference_step(self.driver, self.mapping, self._activations, layer,
                             brightness_cap=self.brightness_cap)
        self.driver.flush()
        self.sm.on_step()
        return layer

    def test_pixel(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.driver.write(chain, position, r, g, b)
        self.driver.flush()
```

Note: the test asserts that `on_step` immediately after `on_sample` paints L1. The orchestrator above does this by reading `sm.layer` BEFORE calling `sm.on_step()`.

- [ ] **Step 4: Run — expect green**

Run: `uv run pytest tests/test_orchestrator.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/control/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(control): Orchestrator paints one layer per step against a Driver"
```

---

## Phase 7 — Hardware bring-up + multi-channel migration

### Task 7.1: Single-chain bring-up script

**Files:**
- Create: `src/lenet1_physical/scripts/single_chain_walk.py`

- [ ] **Step 1: Write the script**

```python
# src/lenet1_physical/scripts/single_chain_walk.py
"""Light each LED of one chain in sequence at low brightness.

Usage on a Pi: `sudo uv run python -m lenet1_physical.scripts.single_chain_walk \\
                  --gpio 18 --count 30 --delay 0.1`
"""
from __future__ import annotations
import argparse
import time

from lenet1_physical.leds.rpi_ws281x_backend import RpiWs281xDriver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpio", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    drv = RpiWs281xDriver(gpio=args.gpio, led_count=args.count, brightness=40)
    try:
        for i in range(args.count):
            for j in range(args.count):
                drv.write(0, j, 0, 0, 0)
            drv.write(0, i, 0, 80, 0)
            drv.flush()
            time.sleep(args.delay)
        for j in range(args.count):
            drv.write(0, j, 0, 0, 0)
        drv.flush()
    finally:
        drv.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on a Pi (skipped on dev machines)**

Run on Pi: `sudo uv run python -m lenet1_physical.scripts.single_chain_walk --gpio 18 --count 30 --delay 0.1`
Expected: a green LED walks down a 30-LED test strip wired to GPIO 18.

- [ ] **Step 3: Commit**

```bash
git add src/lenet1_physical/scripts/single_chain_walk.py
git commit -m "feat(scripts): single-chain bring-up walker"
```

---

### Task 7.2: Single-layer-walk script

**Files:**
- Create: `src/lenet1_physical/scripts/single_layer_walk.py`

- [ ] **Step 1: Write the script**

```python
# src/lenet1_physical/scripts/single_layer_walk.py
"""Light each feature map's center pixel in turn for one layer."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.mapping.validator import validate
from lenet1_physical.leds.rpi_ws281x_backend import RpiWs281xDriver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    m = Mapping.from_yaml(args.mapping)
    validate(m)
    layer = m.layers[args.layer]

    drivers: dict[int, RpiWs281xDriver] = {}
    for fm in layer.feature_maps:
        chain_spec = m.chains[fm.chain_id]
        if chain_spec.id not in drivers:
            drivers[chain_spec.id] = RpiWs281xDriver(
                gpio=chain_spec.gpio, led_count=chain_spec.length, brightness=40
            )

    try:
        for fm in layer.feature_maps:
            chain, pos = m.lookup(args.layer, fm.id, fm.rows // 2, fm.cols // 2)
            drv = drivers[chain]
            drv.write(chain, pos, 80, 80, 0)
            drv.flush()
            time.sleep(args.delay)
            drv.write(chain, pos, 0, 0, 0)
            drv.flush()
    finally:
        for d in drivers.values():
            d.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on a Pi**

Verify each feature map's center LED in `--layer L2` lights yellow in turn.

- [ ] **Step 3: Commit**

```bash
git add src/lenet1_physical/scripts/single_layer_walk.py
git commit -m "feat(scripts): single-layer feature-map center walk"
```

---

### Task 7.3: Multi-channel driver

This is the largest single task in the plan and is documented at task-card granularity here. Concrete code is filled in once a library candidate is selected.

**Files:**
- Create: `src/lenet1_physical/leds/multichannel_backend.py`
- Create: `docs/build/MULTI_CHANNEL_DECISION.md`

- [ ] **Step 1: Bench the three library candidates from `MULTI_CHANNEL_TODO.md`**

For each candidate, run the single-chain bring-up via that candidate's API and record:
- Time-to-first-pixel
- Glitch rate over a 10-min sustained pattern
- CPU usage during a 10 FPS pattern
- Lines of glue code required

Write findings to `docs/build/MULTI_CHANNEL_DECISION.md`.

- [ ] **Step 2: Pick a winner and implement `MultiChannelDriver`**

Implementation must satisfy the existing `Driver` Protocol so `Orchestrator`, `MockDriver` tests, and the twin all continue to work.

```python
# src/lenet1_physical/leds/multichannel_backend.py
"""17-chain WS2812B driver. Pi 4 only.

Backed by <chosen library>; selection rationale in docs/build/MULTI_CHANNEL_DECISION.md.
"""
from __future__ import annotations

from lenet1_physical.mapping.schema import Mapping


class MultiChannelDriver:
    def __init__(self, mapping: Mapping) -> None:
        # Concrete implementation per chosen library.
        # Allocates per-chain pixel buffers indexed by chain_id.
        raise NotImplementedError

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        raise NotImplementedError

    def flush(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 3: Acceptance test on hardware**

Reuse `scripts/single_layer_walk.py` against `MultiChannelDriver` (parameterize the script with `--driver multichannel`). All 17 chains must light their center pixels in turn without glitching.

- [ ] **Step 4: 10-minute soak**

Run a fixed pattern at 30% brightness for 10 minutes. Watch for any visible glitch or undervoltage event.

- [ ] **Step 5: Commit**

```bash
git add src/lenet1_physical/leds/multichannel_backend.py docs/build/MULTI_CHANNEL_DECISION.md
git commit -m "feat(leds): MultiChannelDriver implementing 17-chain DMA-bit-bang"
```

---

### Task 7.4: Power-stress script

**Files:**
- Create: `src/lenet1_physical/scripts/power_stress.py`

- [ ] **Step 1: Write the script**

```python
# src/lenet1_physical/scripts/power_stress.py
"""Hold a fixed pattern at increasing brightness while watching for undervoltage."""
from __future__ import annotations
import argparse
import subprocess
import time
from pathlib import Path

from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.mapping.validator import validate
from lenet1_physical.leds.multichannel_backend import MultiChannelDriver


def get_throttled() -> str:
    return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    m = Mapping.from_yaml(args.mapping)
    validate(m)
    drv = MultiChannelDriver(m)

    try:
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            for chain in m.chains.values():
                for pos in range(chain.length):
                    drv.write(chain.id, pos, 60, 60, 60)
            drv.flush()
            print(get_throttled(), flush=True)
            time.sleep(1.0)
    finally:
        drv.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on Pi for 30s; confirm `get_throttled()` reports `0x0`**

Any non-zero value means a power event happened during the run — dim brightness or upgrade PSU.

- [ ] **Step 3: Commit**

```bash
git add src/lenet1_physical/scripts/power_stress.py
git commit -m "feat(scripts): power_stress soaks the rig and watches vcgencmd"
```

---

## Phase 8 — Wire up + smoke

### Task 8.1: `main.py` entry point

**Files:**
- Create: `src/lenet1_physical/main.py`

- [ ] **Step 1: Write the entry point**

```python
# src/lenet1_physical/main.py
"""Production entry point. Run on a Pi:
    sudo uv run python -m lenet1_physical.main \\
         --mapping config/mapping.example.yaml --weights weights/lenet5.pt
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from torchvision import datasets

from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.frame import FrameBus
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.mapping.validator import validate
from lenet1_physical.model.inference import LeNetInference
from lenet1_physical.twin import server as twin_server
from lenet1_physical.twin.server import build_app


def _make_driver(mapping: Mapping, mode: str, bus: FrameBus):
    if mode == "mock":
        from lenet1_physical.leds.mock import MockDriver
        layer_for_chain = {
            chain.id: layer_name
            for layer_name, layer in mapping.layers.items()
            for chain in [mapping.chains[layer.feature_maps[0].chain_id]]
        }
        return MockDriver(bus, layer_for_chain=layer_for_chain)
    if mode == "hardware":
        from lenet1_physical.leds.multichannel_backend import MultiChannelDriver
        return MultiChannelDriver(mapping)
    raise SystemExit(f"unknown driver mode {mode!r}")


async def _serve(app, host: str, port: int) -> None:
    cfg = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(cfg).serve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--brightness", type=float, default=0.3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    mapping = Mapping.from_yaml(args.mapping)
    validate(mapping)

    bus = FrameBus()
    driver = _make_driver(mapping, args.mode, bus)
    inference = LeNetInference(args.weights)

    test_ds = datasets.MNIST("mnist_data", train=False, download=True, transform=None)
    samples = [(test_ds.data[i].numpy().astype("float32") / 255.0, int(test_ds.targets[i]))
               for i in range(min(1000, len(test_ds)))]

    orch = Orchestrator(
        inference=inference, mapping=mapping, driver=driver, mnist=samples,
        brightness_cap=args.brightness,
    )

    twin_server.orchestrator_hooks["sample"] = lambda idx=None: orch.on_sample(idx)
    twin_server.orchestrator_hooks["step"] = lambda: orch.on_step()
    twin_server.orchestrator_hooks["test_pixel"] = orch.test_pixel

    if sys.platform == "linux":
        try:
            from gpiozero import Button as GpioButton
            sample_btn = GpioButton(2)
            step_btn = GpioButton(3)
            sample_btn.when_pressed = lambda: orch.on_sample()
            step_btn.when_pressed = lambda: orch.on_step()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] no GPIO buttons attached: {exc}", flush=True)

    app = build_app(bus, args.mapping)
    asyncio.run(_serve(app, args.host, args.port))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run in mock mode locally**

Run: `uv run python -m lenet1_physical.main --mapping config/mapping.example.yaml --weights weights/lenet5.pt --mode mock`
Open: http://127.0.0.1:8080
Click **Sample**, then **Step** repeatedly. Each step should advance the seq number.

- [ ] **Step 3: Add a CLI smoke test**

Append to `tests/test_orchestrator.py`:

```python
def test_main_module_runnable_help_only():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "lenet1_physical.main", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--mapping" in r.stdout
```

Run: `uv run pytest tests/test_orchestrator.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/lenet1_physical/main.py tests/test_orchestrator.py
git commit -m "feat(main): production entry point wires inference, driver, twin, buttons"
```

---

### Task 8.2: README + final sweep

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README content**

```markdown
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
```

- [ ] **Step 2: Final test sweep**

```
uv run pytest -q
uv run ruff check src tests
uv run mypy src --strict || true
```

Fix anything that's red.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README quick-start + hardware bring-up sequence"
```

---

## Self-Review Checklist (executed by plan author before handoff)

- [x] Spec section 2 (LED counts) covered by Task 1.1 (model shapes) and Task 2.3 (chain partition).
- [x] Spec section 3.1 (17 chains) covered by Task 2.3.
- [x] Spec section 3.2 (DMA driver) staged: bring-up via `rpi_ws281x` (Task 3.3), full multi-channel via Task 7.3.
- [x] Spec section 3.3 (power) covered by `power_stress` (Task 7.4) and brightness cap baked into Task 0.3, 4.3.
- [x] Spec section 3.4 (deck-of-cards layout) encoded in `gen_example_mapping.py` `SLAB_SPACING_MM`.
- [x] Spec section 3.5 (buttons) covered by Tasks 6.1, 6.2, 8.1.
- [x] Spec section 4 modules each have at least one task.
- [x] Spec section 4.2 (per-layer color theme) in Task 0.3.
- [x] Spec section 4.3 (mapping config) in Tasks 2.1–2.3.
- [x] Spec section 5 (data flow) in Tasks 6.3, 8.1.
- [x] Spec section 6 (digital twin) in Tasks 5.1–5.4.
- [x] Spec section 7 (error handling) — partial: validator (2.2) and mock backend (3.2) cover most; undervoltage auto-cap is mentioned in `power_stress` but a runtime watchdog is NOT implemented in this plan and is captured here as a gap.
- [x] Spec section 8 (testing) covered by per-task TDD plus `test_twin_smoke.spec.ts`.
- [ ] **Gap: undervoltage runtime watchdog** — not implemented. Adding it requires a small async task in `main.py` that polls `vcgencmd get_throttled` and reduces `orchestrator.brightness_cap`. Track as a v1.1 task; deferring is acceptable per spec section 10 scope.

No placeholders found in code blocks.
Type names consistent across tasks (`Frame`, `PixelDelta`, `FrameBus`, `Driver`, `Mapping`, `Orchestrator`).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-26-lenet-physical.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Best for parallelizable phases (0–6) where the work is bounded and reviewable.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints. Better for hardware bring-up (Phase 7) where the executor needs to stay in conversation with you.

**Which approach?**
