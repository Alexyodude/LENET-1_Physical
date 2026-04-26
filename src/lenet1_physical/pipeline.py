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
