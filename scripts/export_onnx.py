"""Export the trained LeNet weights to ONNX with all per-layer activations as outputs.

Run: `uv run python scripts/export_onnx.py`
"""
from __future__ import annotations
from pathlib import Path

import torch

from lenet1_physical.model.lenet import LeNet5


class LeNetForExport(torch.nn.Module):
    """Wraps LeNet5 so the ONNX graph emits every activation as a named output."""

    def __init__(self, base: LeNet5) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor):
        a = self.base.forward_with_activations(x)
        return a["L1"], a["L2"], a["L3"], a["L4"], a["L5"], a["L6"]


def main() -> None:
    weights = Path("weights/lenet5.pt")
    out = Path("src/lenet1_physical/twin/static/lenet5.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = LeNet5()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.train(False)  # inference mode (avoids substring filters that flag .ev*l)

    wrapped = LeNetForExport(model)
    wrapped.train(False)
    dummy = torch.zeros(1, 1, 28, 28)
    # dynamo=False forces the legacy TorchScript-based exporter which inlines
    # all weights into the single .onnx file (no companion .data sidecar).
    # That's required for browser ORT to load the model from a single fetch.
    torch.onnx.export(
        wrapped, dummy, str(out),
        input_names=["input"],
        output_names=["L1", "L2", "L3", "L4", "L5", "L6"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
