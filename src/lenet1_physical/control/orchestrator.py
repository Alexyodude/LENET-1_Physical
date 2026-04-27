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
    _activations: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def set_brightness(self, value: float) -> None:
        self.brightness_cap = max(0.0, min(1.0, value))

    def on_sample_with_image(self, image: np.ndarray) -> int:
        if image.shape != (28, 28):
            raise ValueError(f"Expected image shape (28, 28), got {image.shape}")
        self._activations = self.inference.run(image)
        self.sm.on_sample()
        return -1

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
