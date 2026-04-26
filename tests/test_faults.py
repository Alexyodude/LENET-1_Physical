from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lenet1_physical.twin.faults import FaultStore, FaultedDriver, register_fault_routes


class StubDriver:
    def __init__(self):
        self.writes: list[tuple[int, int, int, int, int]] = []
        self.flushed = 0
        self.closed = 0

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.writes.append((chain, position, r, g, b))

    def flush(self) -> bool:
        self.flushed += 1
        return True

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def store():
    return FaultStore()


@pytest.fixture
def stub():
    return StubDriver()


@pytest.fixture
def client(store):
    app = FastAPI()
    register_fault_routes(app, store)
    return TestClient(app)


# --- FaultStore toggle tests ---

def test_dead_led_toggle(store, stub):
    driver = FaultedDriver(stub, store)
    store.dead_leds.add((0, 5))
    driver.write(0, 5, 255, 255, 255)
    assert stub.writes == []


def test_dead_led_removed_allows_write(store, stub):
    driver = FaultedDriver(stub, store)
    store.dead_leds.add((0, 5))
    store.dead_leds.discard((0, 5))
    driver.write(0, 5, 100, 100, 100)
    assert stub.writes == [(0, 5, 100, 100, 100)]


def test_broken_chain_suppresses_all_positions(store, stub):
    driver = FaultedDriver(stub, store)
    store.broken_chains.add(2)
    driver.write(2, 0, 255, 0, 0)
    driver.write(2, 10, 0, 255, 0)
    assert stub.writes == []


def test_healthy_led_passes_through(store, stub):
    driver = FaultedDriver(stub, store)
    driver.write(0, 0, 10, 20, 30)
    assert stub.writes == [(0, 0, 10, 20, 30)]


def test_undervoltage_scales_rgb(store, stub):
    driver = FaultedDriver(stub, store)
    store.undervoltage_cap = 0.5
    driver.write(0, 0, 200, 100, 50)
    assert stub.writes == [(0, 0, 100, 50, 25)]


def test_flush_and_close_passthrough(store, stub):
    driver = FaultedDriver(stub, store)
    result = driver.flush()
    driver.close()
    assert result is True
    assert stub.flushed == 1
    assert stub.closed == 1


# --- HTTP route tests ---

def test_route_dead_led_activate(client, store):
    r = client.post("/fault/dead-led", json={"chain": 1, "pos": 3, "active": True})
    assert r.status_code == 200
    assert (1, 3) in store.dead_leds


def test_route_dead_led_deactivate(client, store):
    store.dead_leds.add((1, 3))
    r = client.post("/fault/dead-led", json={"chain": 1, "pos": 3, "active": False})
    assert r.status_code == 200
    assert (1, 3) not in store.dead_leds


def test_route_broken_chain(client, store):
    r = client.post("/fault/broken-chain", json={"chain": 4, "active": True})
    assert r.status_code == 200
    assert 4 in store.broken_chains
    r2 = client.post("/fault/broken-chain", json={"chain": 4, "active": False})
    assert r2.status_code == 200
    assert 4 not in store.broken_chains


def test_route_undervoltage(client, store):
    r = client.post("/fault/undervoltage", json={"value": 0.75})
    assert r.status_code == 200
    assert r.json()["undervoltage_cap"] == pytest.approx(0.75)
    assert store.undervoltage_cap == pytest.approx(0.75)


def test_route_undervoltage_clamped(client, store):
    client.post("/fault/undervoltage", json={"value": 2.0})
    assert store.undervoltage_cap == pytest.approx(1.0)
    client.post("/fault/undervoltage", json={"value": -0.5})
    assert store.undervoltage_cap == pytest.approx(0.0)


def test_route_clear(client, store):
    store.dead_leds.add((0, 0))
    store.broken_chains.add(1)
    store.undervoltage_cap = 0.3
    r = client.post("/fault/clear")
    assert r.status_code == 200
    assert store.dead_leds == set()
    assert store.broken_chains == set()
    assert store.undervoltage_cap == pytest.approx(1.0)


def test_route_state(client, store):
    store.dead_leds.add((2, 7))
    store.broken_chains.add(3)
    store.undervoltage_cap = 0.6
    r = client.get("/fault/state")
    assert r.status_code == 200
    data = r.json()
    assert [2, 7] in data["dead_leds"]
    assert 3 in data["broken_chains"]
    assert data["undervoltage_cap"] == pytest.approx(0.6)


def test_passthrough_after_clear(store, stub):
    driver = FaultedDriver(stub, store)
    store.dead_leds.add((0, 0))
    store.broken_chains.add(1)
    store.undervoltage_cap = 0.5
    store.dead_leds.clear()
    store.broken_chains.clear()
    store.undervoltage_cap = 1.0
    driver.write(0, 0, 100, 100, 100)
    driver.write(1, 5, 50, 50, 50)
    assert (0, 0, 100, 100, 100) in stub.writes
    assert (1, 5, 50, 50, 50) in stub.writes
