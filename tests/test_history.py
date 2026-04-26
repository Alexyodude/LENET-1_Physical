from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lenet1_physical.frame import Frame, FrameBus, PixelDelta
from lenet1_physical.twin.history import HistoryRecorder, register_history_routes


def make_frame(seq: int, layer: str, deltas: list[tuple] | None = None) -> Frame:
    if deltas is None:
        deltas = [(0, 0, 10, 20, 30)]
    return Frame(
        seq=seq,
        layer=layer,
        deltas=tuple(PixelDelta(c, p, r, g, b) for c, p, r, g, b in deltas),
    )


async def publish_cycle(bus: FrameBus, seq_start: int) -> None:
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]
    for i, layer in enumerate(layers):
        await bus.publish(make_frame(seq_start + i, layer))


@pytest.mark.asyncio
async def test_frames_grouped_into_cycles():
    bus = FrameBus()
    recorder = HistoryRecorder(bus, max_records=50)
    recorder.start()

    # Publish two full cycles
    await publish_cycle(bus, seq_start=0)
    await publish_cycle(bus, seq_start=10)
    # Third L1 triggers save of second cycle
    await bus.publish(make_frame(seq=20, layer="L1"))
    await asyncio.sleep(0.05)

    records = recorder.list_records()
    assert len(records) == 2
    # Each cycle should have 6 frames
    assert records[0]["n_frames"] == 6
    assert records[1]["n_frames"] == 6


@pytest.mark.asyncio
async def test_eviction_at_max_records():
    bus = FrameBus()
    recorder = HistoryRecorder(bus, max_records=3)
    recorder.start()

    # Publish 4 cycles (need 5 L1s to flush 4 cycles)
    for i in range(5):
        await bus.publish(make_frame(seq=i * 10, layer="L1"))
        for j, layer in enumerate(["L2", "L3", "L4", "L5", "L6"]):
            await bus.publish(make_frame(seq=i * 10 + j + 1, layer=layer))

    # Sixth L1 triggers save of fifth cycle
    await bus.publish(make_frame(seq=60, layer="L1"))
    await asyncio.sleep(0.05)

    records = recorder.list_records()
    assert len(records) == 3  # oldest evicted


@pytest.mark.asyncio
async def test_routes_return_correct_shape():
    bus = FrameBus()
    recorder = HistoryRecorder(bus, max_records=50)
    recorder.start()

    await publish_cycle(bus, seq_start=0)
    await bus.publish(make_frame(seq=10, layer="L1"))
    await asyncio.sleep(0.05)

    app = FastAPI()
    register_history_routes(app, recorder)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/history")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        summary = data[0]
        assert set(summary.keys()) == {"id", "started_at", "predicted_digit", "sample_index", "n_frames"}

        record_id = summary["id"]
        resp2 = await client.get(f"/history/{record_id}")
        assert resp2.status_code == 200
        full = resp2.json()
        assert "frames" in full
        assert isinstance(full["frames"], list)
        first_frame = full["frames"][0]
        assert set(first_frame.keys()) == {"seq", "layer", "deltas"}
        assert isinstance(first_frame["deltas"], list)
        assert isinstance(first_frame["deltas"][0], list)


@pytest.mark.asyncio
async def test_predicted_digit_from_l6():
    bus = FrameBus()
    recorder = HistoryRecorder(bus, max_records=50)
    recorder.start()

    # L6 frame: position 7 has highest RGB sum
    l6_deltas = [(0, i, i * 10, i * 10, i * 10) for i in range(10)]
    # position 9 will have sum 9*30=270, position 7 = 7*30=210
    # Let's make position 5 the winner with a big delta
    l6_deltas = [(0, 5, 100, 100, 100), (0, 3, 10, 10, 10)]

    cycle_frames = [
        make_frame(0, "L1"),
        make_frame(1, "L2"),
        make_frame(2, "L3"),
        make_frame(3, "L4"),
        make_frame(4, "L5"),
        Frame(seq=5, layer="L6", deltas=tuple(PixelDelta(c, p, r, g, b) for c, p, r, g, b in l6_deltas)),
    ]
    for f in cycle_frames:
        await bus.publish(f)

    # Next L1 triggers save
    await bus.publish(make_frame(10, "L1"))
    await asyncio.sleep(0.05)

    records = recorder.list_records()
    assert len(records) == 1
    assert records[0]["predicted_digit"] == 5
