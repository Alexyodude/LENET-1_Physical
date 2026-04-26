# Multi-channel WS2812B on Pi 4 -- Phase 7

The bring-up backend (`rpi_ws281x_backend.py`) drives ONE chain via PWM0/PWM1.
The full 17-chain build needs DMA-to-GPIO-bank bit-banging.

## Library candidates to assess

1. **Maintained `rpi_ws281x` forks** with multi-channel patches.
2. **Custom thin C extension**: `mmap` `/dev/mem` for GPIO bank, allocate a DMA
   control block ring, encode the 17-bit-wide WS2812 bitstream into 32-bit
   words. ~300 lines of C, ~50 lines of Python wrapper. Most reliable.
3. **`Pi5Neo`**: written for Pi 5, may need backporting to Pi 4 BCM2711.

## Acceptance for Phase 7

- 17 simultaneous chains, each up to 2304 LEDs, drive a sustained 5+ FPS
  fixed pattern with zero visible glitches over a 10-minute run.
- The new driver implements the same `Driver` Protocol so all upstream code
  continues to work without changes.
