from __future__ import annotations
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from lenet1_physical.frame import FrameBus
from lenet1_physical.twin.server import build_app
from lenet1_physical.twin import server as srv
from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.leds.mock import MockDriver
from lenet1_physical.mapping.schema import Mapping


class FakeInference:
    def run(self, image):
        return {
            "L1": np.zeros((1, 28, 28), dtype=np.float32),
            "L2": np.zeros((4, 24, 24), dtype=np.float32),
            "L3": np.zeros((4, 12, 12), dtype=np.float32),
            "L4": np.zeros((12, 8, 8), dtype=np.float32),
            "L5": np.zeros((12, 4, 4), dtype=np.float32),
            "L6": np.zeros((1, 1, 10), dtype=np.float32),
            "prediction": 3,
        }


class FakeMnist:
    def __getitem__(self, idx):
        return np.zeros((28, 28), dtype=np.float32), 0

    def __len__(self):
        return 100


@pytest.fixture
def app_client():
    bus = FrameBus()
    app = build_app(bus, mapping_path=None)
    return TestClient(app), bus


@pytest.fixture
def app_client_with_hook():
    bus = FrameBus()
    app = build_app(bus, mapping_path=None)
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_minimal.yaml"))
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    orch = Orchestrator(
        inference=FakeInference(),
        mapping=m,
        driver=drv,
        mnist=FakeMnist(),
        brightness_cap=0.3,
    )
    srv.orchestrator_hooks["sample_image"] = orch.on_sample_with_image
    client = TestClient(app)
    yield client, orch
    del srv.orchestrator_hooks["sample_image"]


def test_sample_image_valid_returns_200_with_predicted_digit(app_client_with_hook):
    client, _ = app_client_with_hook
    pixels = [0.0] * 784
    r = client.post("/sample-image", json={"image": pixels})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_digit" in data
    assert data["predicted_digit"] == -1


def test_sample_image_bad_shape_returns_400(app_client):
    client, _ = app_client
    pixels = [0.0] * 100
    r = client.post("/sample-image", json={"image": pixels})
    assert r.status_code == 400
    assert "error" in r.json()


def test_orchestrator_on_sample_with_image_updates_state():
    bus = FrameBus()
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_minimal.yaml"))
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    orch = Orchestrator(
        inference=FakeInference(),
        mapping=m,
        driver=drv,
        mnist=FakeMnist(),
        brightness_cap=0.3,
    )
    image = np.zeros((28, 28), dtype=np.float32)
    result = orch.on_sample_with_image(image)
    assert result == -1
    assert orch._activations is not None
    assert "L1" in orch._activations
