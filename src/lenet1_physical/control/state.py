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
