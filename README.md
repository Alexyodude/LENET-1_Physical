# LENET-1 Physical

Physical, illuminated visualization of a LeNet-style convolutional neural
network performing live MNIST digit classification on a Raspberry Pi 4.

~4634 WS2812B addressable LEDs arranged into 6 layer-slabs (28×28×1 →
24×24×4 → 12×12×4 → 8×8×12 → 4×4×12 → 1×10), driven directly from Pi 4
GPIOs via DMA-bit-bang multi-channel output across 17 parallel chains.
A digital twin (3D + per-layer 2D slices) runs on a small web server on
the Pi for real-time debugging.

## Status

Design phase. See [the spec](docs/superpowers/specs/2026-04-26-lenet-physical-design.md).

Implementation plan and code will land after the spec is reviewed.
