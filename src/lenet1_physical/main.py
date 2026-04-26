"""Production entry point. Run on a Pi:
    sudo uv run python -m lenet1_physical.main \
         --mapping config/mapping.example.yaml --weights weights/lenet5.pt
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from torchvision import datasets

from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.frame import FrameBus
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.mapping.validator import validate
from lenet1_physical.model.inference import LeNetInference
from lenet1_physical.twin import server as twin_server
from lenet1_physical.twin.server import build_app


def _make_driver(mapping: Mapping, mode: str, bus: FrameBus):
    if mode == "mock":
        from lenet1_physical.leds.mock import MockDriver
        layer_for_chain = {
            chain.id: layer_name
            for layer_name, layer in mapping.layers.items()
            for chain in [mapping.chains[layer.feature_maps[0].chain_id]]
        }
        return MockDriver(bus, layer_for_chain=layer_for_chain)
    if mode == "hardware":
        from lenet1_physical.leds.multichannel_backend import MultiChannelDriver
        return MultiChannelDriver(mapping)
    raise SystemExit(f"unknown driver mode {mode!r}")


async def _serve(app, host: str, port: int) -> None:
    cfg = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(cfg).serve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--brightness", type=float, default=0.3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    mapping = Mapping.from_yaml(args.mapping)
    validate(mapping)

    bus = FrameBus()
    driver = _make_driver(mapping, args.mode, bus)
    inference = LeNetInference(args.weights)

    test_ds = datasets.MNIST("mnist_data", train=False, download=True, transform=None)
    samples = [(test_ds.data[i].numpy().astype("float32") / 255.0, int(test_ds.targets[i]))
               for i in range(min(1000, len(test_ds)))]

    orch = Orchestrator(
        inference=inference, mapping=mapping, driver=driver, mnist=samples,
        brightness_cap=args.brightness,
    )

    twin_server.orchestrator_hooks["sample"] = lambda idx=None: orch.on_sample(idx)
    twin_server.orchestrator_hooks["step"] = lambda: orch.on_step()
    twin_server.orchestrator_hooks["test_pixel"] = orch.test_pixel
    twin_server.orchestrator_hooks["brightness"] = orch.set_brightness

    if sys.platform == "linux":
        try:
            from gpiozero import Button as GpioButton
            sample_btn = GpioButton(2)
            step_btn = GpioButton(3)
            sample_btn.when_pressed = lambda: orch.on_sample()
            step_btn.when_pressed = lambda: orch.on_step()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] no GPIO buttons attached: {exc}", flush=True)

    app = build_app(bus, args.mapping)
    asyncio.run(_serve(app, args.host, args.port))


if __name__ == "__main__":
    main()
