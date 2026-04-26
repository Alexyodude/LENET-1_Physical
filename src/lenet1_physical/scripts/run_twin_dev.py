"""Boots the twin and emits a synthetic frame stream for browser dev."""
from __future__ import annotations
import asyncio
import math
import uvicorn

from lenet1_physical.frame import Frame, FrameBus, PixelDelta
from lenet1_physical.twin.server import build_app

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]
# (chain_id, led_count, color_rgb)
LAYER_DEMO = [
    (0,  784, (255, 220, 180)),
    (1,  576, (0,   0,   255)),
    (2,  144, (0,   255, 255)),
    (3,  192, (0,   255, 0  )),
    (4,   64, (255, 255, 0  )),
    (16,  10, (255, 0,   0  )),
]


async def emit_loop(bus: FrameBus) -> None:
    seq = 0
    layer_idx = 0
    while True:
        seq += 1
        layer = LAYERS[layer_idx]
        chain, count, (r, g, b) = LAYER_DEMO[layer_idx]
        t = (seq % 60) / 60.0
        brightness = 0.3 + 0.7 * abs(math.sin(t * math.pi))
        deltas = tuple(
            PixelDelta(chain, i, int(r * brightness), int(g * brightness), int(b * brightness))
            for i in range(min(count, 100))
        )
        await bus.publish(Frame(seq=seq, layer=layer, deltas=deltas))
        await asyncio.sleep(0.08)
        if seq % 30 == 0:
            layer_idx = (layer_idx + 1) % len(LAYERS)


async def main() -> None:
    bus = FrameBus()
    app = build_app(bus, None)
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), emit_loop(bus))


if __name__ == "__main__":
    asyncio.run(main())
