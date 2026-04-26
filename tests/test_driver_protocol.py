from lenet1_physical.leds.driver import Driver


class _StubDriver:
    def __init__(self):
        self.writes = []
        self.flushes = 0

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.writes.append((chain, position, r, g, b))

    def flush(self) -> bool:
        self.flushes += 1
        return True

    def close(self) -> None:
        pass


def test_stub_satisfies_protocol():
    d: Driver = _StubDriver()
    d.write(0, 5, 10, 20, 30)
    assert d.flush() is True
    d.close()
