import numpy as np
from pathlib import Path
from lenet1_physical.frame import FrameBus
from lenet1_physical.leds.mock import MockDriver
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.pipeline import paint_layer


FIXTURE = Path("tests/fixtures/mapping_minimal.yaml")


def test_paint_layer_writes_one_pixel_per_logical_position():
    m = Mapping.from_yaml(FIXTURE)
    bus = FrameBus()
    drv = MockDriver(bus, layer_for_chain={0: "L1", 16: "L6"})
    rgb = np.zeros((1, 28, 28, 3), dtype=np.uint8)
    rgb[0, 0, 0] = (200, 100, 50)
    paint_layer(drv, m, "L1", rgb)
    drv.flush()


def test_paint_layer_rejects_wrong_shape():
    m = Mapping.from_yaml(FIXTURE)
    drv = MockDriver(FrameBus(), layer_for_chain={0: "L1"})
    rgb = np.zeros((4, 28, 28, 3), dtype=np.uint8)
    import pytest
    with pytest.raises(ValueError, match="L1 expected 1 fmaps"):
        paint_layer(drv, m, "L1", rgb)


import asyncio
import pytest
from lenet1_physical.colors import encode_layer
from lenet1_physical.frame import Frame


@pytest.mark.asyncio
async def test_activation_to_frame_e2e():
    m = Mapping.from_yaml(FIXTURE)
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})

    a = np.zeros((1, 28, 28), dtype=np.float32)
    a[0, 0, 0] = 1.0
    rgb = encode_layer("L1", a, brightness_cap=1.0)
    paint_layer(drv, m, "L1", rgb)
    drv.flush()

    frame: Frame = await asyncio.wait_for(sub.get(), 0.5)
    assert frame.layer == "L1"
    hot = next(d for d in frame.deltas if (d.chain, d.position) == (0, 0))
    assert (hot.r, hot.g, hot.b) == (255, 220, 180)


from lenet1_physical.pipeline import paint_inference_step


def test_paint_inference_step_paints_only_the_requested_layer():
    m = Mapping.from_yaml(FIXTURE)
    drv = MockDriver(FrameBus(), layer_for_chain={0: "L1", 16: "L6"})

    activations = {
        "L1": np.zeros((1, 28, 28), dtype=np.float32),
        "L6": np.zeros((1, 1, 10), dtype=np.float32),
    }
    activations["L1"][0, 0, 0] = 1.0
    activations["L6"][0, 0, 7] = 1.0

    paint_inference_step(drv, m, activations, "L6", brightness_cap=0.3)
    drv.flush()
