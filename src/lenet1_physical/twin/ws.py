from __future__ import annotations
from fastapi import WebSocket, WebSocketDisconnect

from lenet1_physical.frame import Frame, FrameBus


def frame_to_dict(f: Frame) -> dict:
    return {
        "seq": f.seq,
        "layer": f.layer,
        "deltas": [[d.chain, d.position, d.r, d.g, d.b] for d in f.deltas],
    }


async def stream_frames(ws: WebSocket, bus: FrameBus) -> None:
    await ws.accept()
    queue = bus.subscribe()
    try:
        while True:
            frame = await queue.get()
            await ws.send_json(frame_to_dict(frame))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
