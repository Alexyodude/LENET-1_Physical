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
                loop = asyncio.get_running_loop()
                loop.create_task(self._bus.publish(frame))
            except RuntimeError:
                # No running loop (sync test context) -- run inline.
                asyncio.run(self._bus.publish(frame))
        self._staged.clear()
        return True

    def close(self) -> None:
        self._staged.clear()
