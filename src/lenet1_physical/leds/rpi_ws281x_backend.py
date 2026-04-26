"""Hardware LED driver. Pi-only.

NOTE: this is a 2-channel bring-up backend used to validate timing on a
single chain end-to-end. The 17-chain DMA-bit-bang migration is Phase 7.
"""
from __future__ import annotations
import platform
import sys


def _require_linux_arm() -> None:
    if sys.platform != "linux" or "arm" not in platform.machine().lower():
        raise RuntimeError(
            "rpi_ws281x backend only runs on a Raspberry Pi. "
            "Use MockDriver on dev machines."
        )


class RpiWs281xDriver:
    """Wraps `rpi_ws281x` for ONE chain.

    Multiple chains are constructed and flushed together by the caller.
    """

    def __init__(
        self,
        gpio: int,
        led_count: int,
        *,
        freq_hz: int = 800_000,
        dma: int = 10,
        brightness: int = 80,
    ) -> None:
        _require_linux_arm()
        # Late import so non-Pi machines can still import this module.
        from rpi_ws281x import PixelStrip, ws  # type: ignore[import-not-found]

        self._strip = PixelStrip(
            num=led_count,
            pin=gpio,
            freq_hz=freq_hz,
            dma=dma,
            invert=False,
            brightness=brightness,
            channel=0 if gpio in (12, 18) else 1,  # PWM0 vs PWM1
            strip_type=ws.WS2811_STRIP_GRB,
        )
        self._strip.begin()
        self._dirty = False

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        # `chain` is informational here; one driver instance == one chain.
        self._strip.setPixelColorRGB(position, r, g, b)
        self._dirty = True

    def flush(self) -> bool:
        if not self._dirty:
            return True
        self._strip.show()
        self._dirty = False
        return True

    def close(self) -> None:
        for i in range(self._strip.numPixels()):
            self._strip.setPixelColorRGB(i, 0, 0, 0)
        self._strip.show()
