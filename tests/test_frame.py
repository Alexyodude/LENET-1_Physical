import asyncio
import pytest
from lenet1_physical.frame import Frame, FrameBus, PixelDelta


def test_pixel_delta_immutable():
    p = PixelDelta(chain=2, position=14, r=255, g=128, b=0)
    with pytest.raises(Exception):
        p.r = 0  # frozen dataclass rejects assignment


def test_frame_groups_deltas_by_layer_with_seq():
    f = Frame(seq=7, layer="L2", deltas=(PixelDelta(0, 0, 10, 20, 30),))
    assert f.seq == 7
    assert f.layer == "L2"
    assert f.deltas[0].r == 10


@pytest.mark.asyncio
async def test_frame_bus_broadcasts_to_all_subscribers():
    bus = FrameBus()
    a, b = bus.subscribe(), bus.subscribe()
    f = Frame(seq=1, layer="L1", deltas=())
    await bus.publish(f)
    assert (await asyncio.wait_for(a.get(), 0.1)).seq == 1
    assert (await asyncio.wait_for(b.get(), 0.1)).seq == 1


@pytest.mark.asyncio
async def test_frame_bus_unsubscribe_stops_delivery():
    bus = FrameBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    await bus.publish(Frame(seq=1, layer="L1", deltas=()))
    assert q.empty()
