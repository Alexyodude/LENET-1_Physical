from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI

from lenet1_physical.frame import Frame, FrameBus


@dataclass
class InferenceRecord:
    id: int
    started_at: str
    predicted_digit: int | None
    sample_index: int | None
    frames: list[Frame]


def _digit_from_l6(frame: Frame) -> int:
    """Return the position (0-9) with the highest total RGB sum across all deltas."""
    sums: dict[int, int] = {}
    for d in frame.deltas:
        sums[d.position] = sums.get(d.position, 0) + d.r + d.g + d.b
    if not sums:
        return 0
    return max(sums, key=lambda k: sums[k])


def _frame_to_dict(frame: Frame) -> dict:
    return {
        "seq": frame.seq,
        "layer": frame.layer,
        "deltas": [[d.chain, d.position, d.r, d.g, d.b] for d in frame.deltas],
    }


class HistoryRecorder:
    def __init__(self, bus: FrameBus, max_records: int = 50) -> None:
        self._bus = bus
        self._max_records = max_records
        self._queue: asyncio.Queue[Frame] = bus.subscribe()
        self._records: deque[InferenceRecord] = deque()
        self._next_id = 0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        current_frames: list[Frame] = []
        current_start: str = datetime.now(timezone.utc).isoformat()
        current_sample: int | None = None
        current_predicted: int | None = None

        while True:
            frame: Frame = await self._queue.get()

            if frame.layer == "L1":
                # Save previous cycle if it has frames
                if current_frames:
                    record = InferenceRecord(
                        id=self._next_id,
                        started_at=current_start,
                        predicted_digit=current_predicted,
                        sample_index=current_sample,
                        frames=current_frames,
                    )
                    self._records.append(record)
                    self._next_id += 1
                    if len(self._records) > self._max_records:
                        self._records.popleft()

                # Start new cycle
                current_frames = [frame]
                current_start = datetime.now(timezone.utc).isoformat()
                current_sample = None
                current_predicted = None
            else:
                current_frames.append(frame)
                if frame.layer == "L6":
                    current_predicted = _digit_from_l6(frame)

    def list_records(self) -> list[dict]:
        return [
            {
                "id": r.id,
                "started_at": r.started_at,
                "predicted_digit": r.predicted_digit,
                "sample_index": r.sample_index,
                "n_frames": len(r.frames),
            }
            for r in self._records
        ]

    def get_record(self, record_id: int) -> InferenceRecord | None:
        for r in self._records:
            if r.id == record_id:
                return r
        return None


def register_history_routes(app: FastAPI, recorder: HistoryRecorder) -> None:
    @app.get("/history")
    async def list_history() -> list:
        return recorder.list_records()

    @app.get("/history/{record_id}")
    async def get_history(record_id: int) -> dict | None:
        record = recorder.get_record(record_id)
        if record is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Record not found")
        return {
            "id": record.id,
            "started_at": record.started_at,
            "predicted_digit": record.predicted_digit,
            "sample_index": record.sample_index,
            "frames": [_frame_to_dict(f) for f in record.frames],
        }
