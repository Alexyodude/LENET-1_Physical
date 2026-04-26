from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class FaultStore:
    dead_leds: set[tuple[int, int]] = field(default_factory=set)
    broken_chains: set[int] = field(default_factory=set)
    undervoltage_cap: float = 1.0


class FaultedDriver:
    def __init__(self, underlying: Any, store: FaultStore) -> None:
        self._underlying = underlying
        self._store = store

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        if (chain, position) in self._store.dead_leds:
            return
        if chain in self._store.broken_chains:
            return
        cap = self._store.undervoltage_cap
        if cap != 1.0:
            r = int(r * cap)
            g = int(g * cap)
            b = int(b * cap)
        self._underlying.write(chain, position, r, g, b)

    def flush(self) -> bool:
        return self._underlying.flush()

    def close(self) -> None:
        self._underlying.close()


class _DeadLedBody(BaseModel):
    chain: int
    pos: int
    active: bool


class _BrokenChainBody(BaseModel):
    chain: int
    active: bool


class _UndervoltageBody(BaseModel):
    value: float


def register_fault_routes(app: FastAPI, store: FaultStore) -> None:
    @app.post("/fault/dead-led")
    async def dead_led(body: _DeadLedBody) -> dict:
        key = (body.chain, body.pos)
        if body.active:
            store.dead_leds.add(key)
        else:
            store.dead_leds.discard(key)
        return {"dead_leds": list(store.dead_leds)}

    @app.post("/fault/broken-chain")
    async def broken_chain(body: _BrokenChainBody) -> dict:
        if body.active:
            store.broken_chains.add(body.chain)
        else:
            store.broken_chains.discard(body.chain)
        return {"broken_chains": list(store.broken_chains)}

    @app.post("/fault/undervoltage")
    async def undervoltage(body: _UndervoltageBody) -> dict:
        store.undervoltage_cap = max(0.0, min(1.0, body.value))
        return {"undervoltage_cap": store.undervoltage_cap}

    @app.post("/fault/clear")
    async def clear_faults() -> dict:
        store.dead_leds.clear()
        store.broken_chains.clear()
        store.undervoltage_cap = 1.0
        return {"ok": True}

    @app.get("/fault/state")
    async def fault_state() -> dict:
        return {
            "dead_leds": [list(led) for led in store.dead_leds],
            "broken_chains": list(store.broken_chains),
            "undervoltage_cap": store.undervoltage_cap,
        }
