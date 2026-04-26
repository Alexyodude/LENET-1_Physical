"""Bundle a small MNIST test slice as JSON for in-browser inference.

Run: `uv run python scripts/bundle_mnist.py`
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


N_SAMPLES = 200


def _load_torchvision() -> tuple[np.ndarray, np.ndarray]:
    from torchvision import datasets
    ds = datasets.MNIST("mnist_data", train=False, download=True, transform=None)
    return ds.data.numpy(), ds.targets.numpy()


def _synthetic_fallback() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    images = (rng.random((N_SAMPLES, 28, 28)) * 255).astype(np.uint8)
    labels = (np.arange(N_SAMPLES) % 10).astype(np.int64)
    return images, labels


def main() -> None:
    out = Path("src/lenet1_physical/twin/static/mnist-samples.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        images, labels = _load_torchvision()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] torchvision MNIST unavailable ({exc}); using synthetic data")
        images, labels = _synthetic_fallback()

    samples = []
    for i in range(min(N_SAMPLES, len(images))):
        samples.append({
            "image": images[i].astype(np.uint8).flatten().tolist(),
            "label": int(labels[i]),
        })

    with out.open("w") as f:
        json.dump(samples, f)
    print(f"saved {out} ({len(samples)} samples, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
