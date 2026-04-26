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
