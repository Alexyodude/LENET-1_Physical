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
