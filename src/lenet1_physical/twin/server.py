from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lenet1_physical.frame import FrameBus
from lenet1_physical.twin.ws import stream_frames

orchestrator_hooks: dict[str, Callable[..., Any]] = {}

_mapping_data: dict | None = None
_brightness_cap: float = 0.30


class _SampleBody(BaseModel):
    index: int | None = None


class _TestPixelBody(BaseModel):
    chain: int
    pos: int
    r: int
    g: int
    b: int


class _BrightnessBody(BaseModel):
    value: float


class _SampleImageBody(BaseModel):
    image: list[float]


def build_app(bus: FrameBus, mapping_path: Path | None, fault_store=None, history_recorder=None) -> FastAPI:
    global _mapping_data
    if mapping_path is not None:
        raw = yaml.safe_load(Path(mapping_path).read_text())
        _mapping_data = raw
    else:
        _mapping_data = None

    app = FastAPI()

    if fault_store is not None:
        from lenet1_physical.twin.faults import register_fault_routes
        register_fault_routes(app, fault_store)

    if history_recorder is not None:
        from lenet1_physical.twin.history import register_history_routes
        register_history_routes(app, history_recorder)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await stream_frames(ws, bus)

    @app.get("/mapping")
    async def mapping() -> dict:
        if _mapping_data is None:
            return {"layers": {}, "chains": []}
        return _mapping_data

    @app.post("/brightness")
    async def brightness(body: _BrightnessBody) -> dict:
        global _brightness_cap
        value = max(0.0, min(1.0, body.value))
        _brightness_cap = value
        fn = orchestrator_hooks.get("brightness")
        if fn is not None:
            fn(value)
        return {"brightness": value}

    @app.get("/brightness")
    async def get_brightness() -> dict:
        return {"brightness": _brightness_cap}

    @app.post("/sample")
    async def sample(body: _SampleBody) -> dict:
        fn = orchestrator_hooks.get("sample")
        if fn is None:
            return {"error": "no orchestrator wired"}
        return {"picked": fn(body.index)}

    @app.post("/step")
    async def step() -> dict:
        fn = orchestrator_hooks.get("step")
        if fn is None:
            return {"error": "no orchestrator wired"}
        return {"now_at": fn()}

    @app.post("/test-pixel")
    async def test_pixel(body: _TestPixelBody) -> dict:
        fn = orchestrator_hooks.get("test_pixel")
        if fn is None:
            return {"error": "no orchestrator wired"}
        fn(body.chain, body.pos, body.r, body.g, body.b)
        return {"ok": True}

    @app.post("/sample-image")
    async def sample_image(body: _SampleImageBody) -> JSONResponse:
        if len(body.image) != 784:
            return JSONResponse(status_code=400, content={"error": "image must have exactly 784 values"})
        fn = orchestrator_hooks.get("sample_image")
        if fn is None:
            return JSONResponse(status_code=200, content={"error": "no orchestrator wired"})
        arr = np.array(body.image, dtype=np.float32).reshape(28, 28)
        predicted = fn(arr)
        return JSONResponse(status_code=200, content={"predicted_digit": predicted})

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
