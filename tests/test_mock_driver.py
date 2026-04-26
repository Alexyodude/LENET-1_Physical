import asyncio
import pytest
from lenet1_physical.frame import FrameBus
from lenet1_physical.leds.mock import MockDriver


@pytest.mark.asyncio
async def test_writes_then_flush_publishes_frame_with_seq():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})
    drv.write(0, 0, 1, 2, 3)
    drv.write(0, 1, 4, 5, 6)
    assert drv.flush() is True
    f = await asyncio.wait_for(sub.get(), 0.1)
    assert f.seq == 1
    assert f.layer == "L1"
    assert len(f.deltas) == 2


@pytest.mark.asyncio
async def test_flush_with_no_writes_emits_no_frame():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1"})
    assert drv.flush() is True
    assert sub.empty()


@pytest.mark.asyncio
async def test_writes_to_multiple_chains_split_per_layer():
    bus = FrameBus()
    sub = bus.subscribe()
    drv = MockDriver(bus, layer_for_chain={0: "L1", 1: "L2"})
    drv.write(0, 0, 9, 9, 9)
    drv.write(1, 0, 1, 1, 1)
    drv.flush()
    seen_layers = {(await asyncio.wait_for(sub.get(), 0.1)).layer for _ in range(2)}
    assert seen_layers == {"L1", "L2"}
