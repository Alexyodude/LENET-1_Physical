import asyncio
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from lenet1_physical.frame import Frame, FrameBus, PixelDelta
from lenet1_physical.twin.server import build_app


MAPPING_FIXTURE = Path(__file__).parent.parent / "config" / "mapping.example.yaml"


@pytest.fixture
def app_and_bus():
    bus = FrameBus()
    app = build_app(bus, mapping_path=None)
    return app, bus


@pytest.fixture
def app_with_mapping():
    bus = FrameBus()
    app = build_app(bus, mapping_path=MAPPING_FIXTURE)
    return app, bus


def test_health_endpoint(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_websocket_streams_frame(app_and_bus):
    app, bus = app_and_bus
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        async def emit():
            await bus.publish(Frame(seq=42, layer="L1", deltas=(PixelDelta(0, 0, 1, 2, 3),)))
        loop = asyncio.new_event_loop()
        loop.run_until_complete(emit())
        msg = ws.receive_json()
        assert msg["seq"] == 42
        assert msg["layer"] == "L1"
        assert msg["deltas"][0] == [0, 0, 1, 2, 3]


def test_mapping_endpoint_no_mapping(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.get("/mapping")
    assert r.status_code == 200
    data = r.json()
    assert "layers" in data
    assert "chains" in data


def test_mapping_endpoint_with_mapping(app_with_mapping):
    app, _ = app_with_mapping
    client = TestClient(app)
    r = client.get("/mapping")
    assert r.status_code == 200
    data = r.json()
    assert "L1" in data["layers"]
    assert "L6" in data["layers"]
    assert len(data["chains"]) == 17


def test_brightness_endpoint_set_and_get(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.post("/brightness", json={"value": 0.5})
    assert r.status_code == 200
    assert r.json()["brightness"] == pytest.approx(0.5)
    r2 = client.get("/brightness")
    assert r2.json()["brightness"] == pytest.approx(0.5)


def test_brightness_clamped_to_valid_range(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.post("/brightness", json={"value": 2.0})
    assert r.json()["brightness"] == pytest.approx(1.0)
    r2 = client.post("/brightness", json={"value": -1.0})
    assert r2.json()["brightness"] == pytest.approx(0.0)


def test_brightness_hook_called(app_and_bus):
    from lenet1_physical.twin import server as srv
    app, _ = app_and_bus
    called_with = []
    srv.orchestrator_hooks["brightness"] = lambda v: called_with.append(v)
    client = TestClient(app)
    client.post("/brightness", json={"value": 0.25})
    assert called_with == [pytest.approx(0.25)]
    del srv.orchestrator_hooks["brightness"]


def test_sample_endpoint_no_hook(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.post("/sample", json={})
    assert r.status_code == 200
    assert "error" in r.json()


def test_test_pixel_endpoint_no_hook(app_and_bus):
    app, _ = app_and_bus
    client = TestClient(app)
    r = client.post("/test-pixel", json={"chain": 0, "pos": 0, "r": 255, "g": 0, "b": 0})
    assert r.status_code == 200
    assert "error" in r.json()
