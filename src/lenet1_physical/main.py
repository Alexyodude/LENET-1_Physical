"""LENET-1 Physical entry point.

Three driver modes:
- `simulate` (alias `mock`): full PC simulation. No Pi, no LEDs, no soldering.
   The twin web UI is the visualization. Add `--demo` to auto-cycle samples.
- `hardware`: drives real WS2812B chains via the multichannel backend (Pi 4 only).

Examples:
   uv run python -m lenet1_physical.main \\
       --mapping config/mapping.example.yaml --weights weights/lenet5.pt \\
       --mode simulate --demo
   sudo uv run python -m lenet1_physical.main \\
       --mapping config/mapping.example.yaml --weights weights/lenet5.pt \\
       --mode hardware
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import uvicorn

from lenet1_physical.control.orchestrator import Orchestrator
from lenet1_physical.frame import FrameBus
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.mapping.validator import validate
from lenet1_physical.model.inference import LeNetInference
from lenet1_physical.twin import server as twin_server
from lenet1_physical.twin.faults import FaultStore, FaultedDriver
from lenet1_physical.twin.history import HistoryRecorder
from lenet1_physical.twin.server import build_app


def _layer_for_chain_table(mapping: Mapping) -> dict[int, str]:
    out: dict[int, str] = {}
    for layer_name, layer in mapping.layers.items():
        for fm in layer.feature_maps:
            out[fm.chain_id] = layer_name
    return out


def _make_driver(mapping: Mapping, mode: str, bus: FrameBus):
    if mode in ("simulate", "mock"):
        from lenet1_physical.leds.mock import MockDriver
        return MockDriver(bus, layer_for_chain=_layer_for_chain_table(mapping))
    if mode == "hardware":
        try:
            from lenet1_physical.leds.multichannel_backend import MultiChannelDriver  # type: ignore
            return MultiChannelDriver(mapping)
        except (ImportError, NotImplementedError) as exc:
            raise SystemExit(
                "hardware mode requires the multichannel backend (Phase 7). "
                "Use --mode simulate on a non-Pi machine."
            ) from exc
    raise SystemExit(f"unknown driver mode {mode!r}")


def _load_mnist_samples(n: int = 1000) -> list[tuple[np.ndarray, int]]:
    """Load MNIST test samples; on download/import failure, fall back to synthetic."""
    try:
        from torchvision import datasets
        test_ds = datasets.MNIST("mnist_data", train=False, download=True, transform=None)
        return [
            (test_ds.data[i].numpy().astype("float32") / 255.0, int(test_ds.targets[i]))
            for i in range(min(n, len(test_ds)))
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] MNIST unavailable ({exc}); using synthetic samples", flush=True)
        rng = np.random.default_rng(0)
        return [(rng.random((28, 28), dtype=np.float32), int(i % 10)) for i in range(n)]


async def _demo_loop(orch: Orchestrator, *, layer_ms: int, hold_ms: int) -> None:
    """Auto-cycle: pick a sample, step through L1..L6 with a delay, hold, repeat."""
    while True:
        try:
            orch.on_sample()
            for _ in range(6):
                orch.on_step()
                await asyncio.sleep(layer_ms / 1000.0)
            await asyncio.sleep(hold_ms / 1000.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[demo] iteration error: {exc}", flush=True)
            await asyncio.sleep(1.0)


async def _serve(app, host: str, port: int) -> None:
    cfg = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(cfg).serve()


async def _run(orch: Orchestrator, app, args, recorder: HistoryRecorder | None = None) -> None:
    if recorder is not None:
        recorder.start()
    server_task = asyncio.create_task(_serve(app, args.host, args.port))
    if args.demo:
        demo_task = asyncio.create_task(
            _demo_loop(orch, layer_ms=args.demo_layer_ms, hold_ms=args.demo_hold_ms)
        )
        try:
            await server_task
        finally:
            demo_task.cancel()
    else:
        await server_task


def main() -> None:
    parser = argparse.ArgumentParser(description="LENET-1 Physical: LeNet visualizer")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("simulate", "mock", "hardware"), default="simulate",
                        help="simulate (alias mock) for PC simulation; hardware drives real LEDs")
    parser.add_argument("--brightness", type=float, default=0.3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true",
                        help="auto-cycle samples without buttons or web clicks")
    parser.add_argument("--demo-layer-ms", type=int, default=600,
                        help="delay between layer steps in demo mode (ms)")
    parser.add_argument("--demo-hold-ms", type=int, default=2500,
                        help="hold time on the final layer before next sample (ms)")
    args = parser.parse_args()

    mapping = Mapping.from_yaml(args.mapping)
    validate(mapping)

    bus = FrameBus()
    fault_store = FaultStore()
    raw_driver = _make_driver(mapping, args.mode, bus)
    driver = FaultedDriver(raw_driver, fault_store)
    inference = LeNetInference(args.weights)
    samples = _load_mnist_samples()

    orch = Orchestrator(
        inference=inference, mapping=mapping, driver=driver, mnist=samples,
        brightness_cap=args.brightness,
    )

    twin_server.orchestrator_hooks["sample"] = lambda idx=None: orch.on_sample(idx)
    twin_server.orchestrator_hooks["step"] = lambda: orch.on_step()
    twin_server.orchestrator_hooks["test_pixel"] = orch.test_pixel
    twin_server.orchestrator_hooks["brightness"] = orch.set_brightness
    twin_server.orchestrator_hooks["sample_image"] = orch.on_sample_with_image

    if args.mode == "hardware" and sys.platform == "linux":
        try:
            from gpiozero import Button as GpioButton  # type: ignore[import-not-found]
            sample_btn = GpioButton(2)
            step_btn = GpioButton(3)
            sample_btn.when_pressed = lambda: orch.on_sample()
            step_btn.when_pressed = lambda: orch.on_step()
            print("[info] GPIO buttons attached on BCM 2 and BCM 3", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] GPIO buttons not attached: {exc}", flush=True)

    recorder = HistoryRecorder(bus, max_records=50)

    app = build_app(bus, args.mapping, fault_store=fault_store, history_recorder=recorder)
    mode_label = {"simulate": "PC simulator", "mock": "PC simulator", "hardware": "REAL HARDWARE"}[args.mode]
    print(f"[info] {mode_label} | twin: http://{args.host}:{args.port}/ | "
          f"demo={'on' if args.demo else 'off'}", flush=True)
    asyncio.run(_run(orch, app, args, recorder=recorder))


if __name__ == "__main__":
    main()
