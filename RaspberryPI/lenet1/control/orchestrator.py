from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

<<<<<<< Updated upstream
from lenet1_physical.control.state import State, StateMachine
from lenet1_physical.leds.driver import Driver
from lenet1_physical.mapping.schema import Mapping
from lenet1_physical.pipeline import paint_inference_step
=======
from lenet1.control.state import State, StateMachine
from debug.leds.driver import Driver
from debug.mapping.schema import Mapping
from debug.pipeline import paint_inference_fmap, paint_inference_step


VALID_STYLES = ("color", "white", "front", "fade", "solo")


class RenderTransform:
    """Wraps a Driver to apply user-facing render toggles (mirror / style).

    - mirror: reverses each chain end-to-end (position -> length-1-position)
    - style:
        - "color": pass through per-layer colors (default)
        - "white": flatten every write to white at the source value's luma
        - "front": the orchestrator's paint methods handle this by also
          repainting non-active layers in dim white at each step
        - "fade":  similar to front but blends previously-active layer in
          half-white as a one-step gradient
    """

    def __init__(self, inner: Driver, mapping: Mapping) -> None:
        self._inner = inner
        self._mapping = mapping
        self.mirror = False
        self.style = "color"
        # Per-chain ranges so mirror flips only WITHIN each feature map,
        # not across the whole daisy chain. Sorted by offset so we can
        # linear-scan (fmap counts per chain are small).
        self._chain_fmap_ranges: dict[int, list[tuple[int, int]]] = {}
        for layer in mapping.layers.values():
            for fm in layer.feature_maps:
                size = fm.rows * fm.cols
                self._chain_fmap_ranges.setdefault(fm.chain_id, []).append(
                    (fm.offset_in_chain, size))
        for cid in self._chain_fmap_ranges:
            self._chain_fmap_ranges[cid].sort()

    def _mirror_pos(self, chain: int, position: int) -> int:
        for offset, size in self._chain_fmap_ranges.get(chain, ()):
            if offset <= position < offset + size:
                return offset + (size - 1) - (position - offset)
        # No fmap claims this position — fall back to chain-wide flip so the
        # toggle at least does something for raw test pixels.
        chain_obj = self._mapping.chains.get(chain)
        if chain_obj is None:
            return position
        return chain_obj.length - 1 - position

    def write(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        if self.style == "white":
            v = max(r, g, b)
            r = g = b = v
        if self.mirror:
            position = self._mirror_pos(chain, position)
        self._inner.write(chain, position, r, g, b)

    def flush(self) -> bool:
        return self._inner.flush()

    def close(self) -> None:
        self._inner.close()
>>>>>>> Stashed changes


@dataclass
class Orchestrator:
    """Glues inference + state machine + driver."""
    inference: Any
    mapping: Mapping
    driver: Driver
    mnist: Sequence[tuple[np.ndarray, int]]
    brightness_cap: float = 0.3
    sm: StateMachine = field(default_factory=StateMachine)
    _activations: dict[str, np.ndarray] | None = field(default=None, repr=False)
<<<<<<< Updated upstream
=======
    # Per-feature-map walk cursor: index into _fmap_walk_order, or None when
    # the walk has not started (no sample) or has finished a full pass.
    _fmap_cursor: int | None = field(default=None, repr=False)
    _fmap_walk_order: list[tuple[str, int]] | None = field(default=None, repr=False)
    # Layers explicitly disabled by the user — used as a power-debug switch
    # so the operator can take whole sections of the display dark to test
    # whether brown-out symptoms shift when current draw drops.
    _disabled_layers: set[str] = field(default_factory=set, repr=False)
    _last_painted_layer: str | None = field(default=None, repr=False)
>>>>>>> Stashed changes

    def set_brightness(self, value: float) -> None:
        self.brightness_cap = max(0.0, min(1.0, value))

<<<<<<< Updated upstream
=======
    def _build_fmap_walk_order(self) -> list[tuple[str, int]]:
        """Flatten the mapping into a sequence of (layer, list_index) entries
        following the YAML fmap order.  Reordering the YAML reorders the
        propagation visually without changing fm.id semantics (id stays bound
        to the activation tensor index)."""
        order: list[tuple[str, int]] = []
        for layer_name in ("L1", "L2", "L3", "L4", "L5", "L6"):
            layer = self.mapping.layers.get(layer_name)
            if layer is None:
                continue
            for list_idx, _fm in enumerate(layer.feature_maps):
                order.append((layer_name, list_idx))
        return order

    def _reset_fmap_cursor(self) -> None:
        """Reset the per-feature-map walk to the start.  Called on each
        on_sample so the next on_step_fmap begins at L1 fmap 0."""
        if self._fmap_walk_order is None:
            self._fmap_walk_order = self._build_fmap_walk_order()
        self._fmap_cursor = 0

    def _layer_chains(self, layer: str) -> set[int]:
        l = self.mapping.layers.get(layer)
        if l is None:
            return set()
        return {fm.chain_id for fm in l.feature_maps}

    def _blackout_layer(self, layer: str) -> None:
        chain_ids = self._layer_chains(layer)
        for cid in chain_ids:
            chain = self.mapping.chains.get(cid)
            if chain is None:
                continue
            for pos in range(chain.length):
                self.driver.write(cid, pos, 0, 0, 0)

    def _blackout_all(self) -> None:
        for chain in self.mapping.chains.values():
            for pos in range(chain.length):
                self.driver.write(chain.id, pos, 0, 0, 0)

    def set_layer_enabled(self, layer: str, enabled: bool) -> dict:
        if enabled:
            self._disabled_layers.discard(layer)
        else:
            self._disabled_layers.add(layer)
            self._blackout_layer(layer)
            self.driver.flush()
        return {"layer": layer, "enabled": enabled,
                "disabled_layers": sorted(self._disabled_layers)}

    def get_layer_states(self) -> dict[str, bool]:
        return {name: name not in self._disabled_layers
                for name in self.mapping.layers}

    def _render_transform(self) -> RenderTransform | None:
        d = self.driver
        while d is not None:
            if isinstance(d, RenderTransform):
                return d
            d = getattr(d, "_inner", None)
        return None

    def set_render_mode(self, mirror: bool | None = None,
                        white: bool | None = None,
                        style: str | None = None) -> dict:
        rt = self._render_transform()
        if rt is None:
            return {"error": "render transform not active"}
        if mirror is not None:
            rt.mirror = bool(mirror)
        if style is not None:
            if style not in VALID_STYLES:
                return {"error": f"style must be one of {VALID_STYLES}"}
            rt.style = style
        elif white is not None:
            # Backwards compat for the early white toggle.
            rt.style = "white" if white else "color"
        return {"mirror": rt.mirror, "style": rt.style}

    def get_render_mode(self) -> dict:
        rt = self._render_transform()
        if rt is None:
            return {"mirror": False, "style": "color"}
        return {"mirror": rt.mirror, "style": rt.style}

    def _paint_afterglow(self, active_layer: str) -> None:
        """For 'front' and 'fade' styles, repaint ONLY layers the propagation
        has already passed (strictly before active_layer in L1..L6 order) in
        white.  Future layers stay dark so the propagation front reads
        correctly from L1 toward L6."""
        rt = self._render_transform()
        if rt is None or rt.style not in ("front", "fade"):
            return
        if self._activations is None:
            return
        layer_order = ("L1", "L2", "L3", "L4", "L5", "L6")
        try:
            active_idx = layer_order.index(active_layer)
        except ValueError:
            return
        saved_style = rt.style
        rt.style = "white"
        try:
            for other in layer_order[:active_idx]:
                if other not in self.mapping.layers:
                    continue
                if other in self._disabled_layers:
                    continue
                if other not in self._activations:
                    continue
                paint_inference_step(self.driver, self.mapping, self._activations,
                                     other, brightness_cap=self.brightness_cap)
        finally:
            rt.style = saved_style
        self._last_painted_layer = active_layer

    def _paint_l1(self) -> None:
        """Paint the L1 input image immediately after inference so users
        see the sample appear without having to step through layers."""
        if self._activations is None:
            return
        rt = self._render_transform()
        if rt is not None and rt.style == "solo":
            self._blackout_all()
        if "L1" in self._disabled_layers:
            self._blackout_layer("L1")
            self.driver.flush()
            return
        paint_inference_step(self.driver, self.mapping, self._activations,
                             "L1", brightness_cap=self.brightness_cap)
        self._paint_afterglow("L1")
        self.driver.flush()

>>>>>>> Stashed changes
    def on_sample_with_image(self, image: np.ndarray) -> int:
        if image.shape != (28, 28):
            raise ValueError(f"Expected image shape (28, 28), got {image.shape}")
        self._activations = self.inference.run(image)
        self.sm.on_sample()
<<<<<<< Updated upstream
        return -1
=======
        self._reset_fmap_cursor()
        self._paint_l1()
        logits = self._activations["L6"].flatten()
        return int(np.argmax(logits))
>>>>>>> Stashed changes

    def on_sample(self, index: int | None = None) -> int:
        rng = np.random.default_rng()
        if index is None:
            index = int(rng.integers(0, len(self.mnist)))
        img, _label = self.mnist[index]
        self._activations = self.inference.run(img)
        self.sm.on_sample()
<<<<<<< Updated upstream
=======
        self._reset_fmap_cursor()
        self._paint_l1()
>>>>>>> Stashed changes
        return index

    def on_step(self) -> str | None:
        if self.sm.state is State.IDLE or self._activations is None:
            return None
        layer = self.sm.layer
        if layer is None:
            return None
<<<<<<< Updated upstream
        paint_inference_step(self.driver, self.mapping, self._activations, layer,
                             brightness_cap=self.brightness_cap)
=======
        rt = self._render_transform()
        if rt is not None and rt.style == "solo":
            self._blackout_all()
        if layer in self._disabled_layers:
            self._blackout_layer(layer)
        else:
            paint_inference_step(self.driver, self.mapping, self._activations, layer,
                                 brightness_cap=self.brightness_cap)
            self._paint_afterglow(layer)
>>>>>>> Stashed changes
        self.driver.flush()
        self.sm.on_step()
        return layer

<<<<<<< Updated upstream
    def test_pixel(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.driver.write(chain, position, r, g, b)
        self.driver.flush()
=======
    def on_step_fmap(self) -> dict | None:
        """Advance the per-feature-map walk by one step.

        Returns a dict {layer, fmap, index, total, done} describing the
        feature map that was just painted, or None if no sample is loaded.
        `done` is True when the call completed the final feature map of
        the pass (so the caller can schedule the next on_sample)."""
        if self._activations is None:
            return None
        if self._fmap_walk_order is None or self._fmap_cursor is None:
            self._reset_fmap_cursor()
        order = self._fmap_walk_order or []
        idx = self._fmap_cursor or 0
        if idx >= len(order):
            return {"layer": None, "fmap": None,
                    "index": idx, "total": len(order), "done": True}
        layer, list_idx = order[idx]
        fmap_id = self.mapping.layers[layer].feature_maps[list_idx].id
        rt = self._render_transform()
        if rt is not None and rt.style == "solo":
            self._blackout_all()
        if layer in self._disabled_layers:
            self._blackout_layer(layer)
        else:
            paint_inference_fmap(self.driver, self.mapping, self._activations,
                                 layer, list_idx, brightness_cap=self.brightness_cap)
            self._paint_afterglow(layer)
        self.driver.flush()
        self._fmap_cursor = idx + 1
        return {"layer": layer, "fmap": fmap_id, "index": idx,
                "total": len(order), "done": idx + 1 >= len(order)}

    def test_pixel(self, chain: int, position: int, r: int, g: int, b: int) -> None:
        self.driver.write(chain, position, r, g, b)
        self.driver.flush()

    def on_clear(self) -> None:
        """Write black to every LED on every chain in the mapping, then flush.
        Also resets the per-feature-map walk so the next sample starts fresh."""
        for chain in self.mapping.chains.values():
            for pos in range(chain.length):
                self.driver.write(chain.id, pos, 0, 0, 0)
        self.driver.flush()
        self._fmap_cursor = None
>>>>>>> Stashed changes
