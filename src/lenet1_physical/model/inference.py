from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from lenet1_physical.model.lenet import LeNet5

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
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        normalized = (image - _MEAN) / _STD
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            acts = self.model.forward_with_activations(x)
        out: dict[str, np.ndarray | int] = {
            k: v.squeeze(0).numpy().astype(np.float32) for k, v in acts.items()
        }
        out["prediction"] = int(np.asarray(out["L6"]).argmax())
        return out
