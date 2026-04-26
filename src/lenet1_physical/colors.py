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
