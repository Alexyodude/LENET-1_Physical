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
