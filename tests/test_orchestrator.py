import asyncio
import numpy as np
import pytest
from pathlib import Path

from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.frame import FrameBus
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
            "prediction": 7,
        }


class FakeMnist:
    def __getitem__(self, idx):
        return np.zeros((28, 28), dtype=np.float32), 0

    def __len__(self):
        return 100


@pytest.mark.asyncio
async def test_sample_triggers_l1_paint():
    bus = FrameBus()
    sub = bus.subscribe()
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_minimal.yaml"))
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    orch = Orchestrator(
        inference=FakeInference(),
        mapping=m,
        driver=drv,
        mnist=FakeMnist(),
        brightness_cap=0.3,
    )
    orch.on_sample(0)
    orch.on_step()
    f = await asyncio.wait_for(sub.get(), 0.5)
    assert f.layer == "L1"


def test_set_brightness_clamps():
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
    orch.set_brightness(1.5)
    assert orch.brightness_cap == 1.0
    orch.set_brightness(-0.5)
    assert orch.brightness_cap == 0.0
    orch.set_brightness(0.7)
    assert orch.brightness_cap == 0.7


def test_main_module_runnable_help_only():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "lenet1_physical.main", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--mapping" in r.stdout
