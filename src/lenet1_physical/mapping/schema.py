from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True, slots=True)
class FeatureMap:
    id: int
    chain_id: int
    offset_in_chain: int
    rows: int
    cols: int
    origin_mm: tuple[float, float, float]
    pitch_mm: tuple[float, float]
    order: str  # "row_major_snake" supported in v1


@dataclass(frozen=True, slots=True)
class Layer:
    name: str
    feature_maps: tuple[FeatureMap, ...]


@dataclass(frozen=True, slots=True)
class ChainSpec:
    id: int
    gpio: int
    length: int


@dataclass(frozen=True, slots=True)
class Mapping:
    layers: dict[str, Layer] = field(default_factory=dict)
    chains: dict[int, ChainSpec] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Mapping":
        raw = yaml.safe_load(Path(path).read_text())
        layers: dict[str, Layer] = {}
        for name, body in raw["layers"].items():
            fmaps = tuple(
                FeatureMap(
                    id=fm["id"],
                    chain_id=fm["chain_id"],
                    offset_in_chain=fm["offset_in_chain"],
                    rows=fm["rows"],
                    cols=fm["cols"],
                    origin_mm=tuple(fm["origin_mm"]),
                    pitch_mm=tuple(fm["pitch_mm"]),
                    order=fm["order"],
                )
                for fm in body["feature_maps"]
            )
            layers[name] = Layer(name=name, feature_maps=fmaps)
        chains = {c["id"]: ChainSpec(id=c["id"], gpio=c["gpio"], length=c["length"]) for c in raw["chains"]}
        return cls(layers=layers, chains=chains)

    def lookup(self, layer: str, fmap: int, row: int, col: int) -> tuple[int, int]:
        """Return (chain_id, position_in_chain) for a logical pixel."""
        if layer not in self.layers:
            raise KeyError(f"unknown layer {layer!r}")
        fm = self.layers[layer].feature_maps[fmap]
        if fm.order == "row_major_snake":
            if row % 2 == 0:
                offset_in_fmap = row * fm.cols + col
            else:
                offset_in_fmap = row * fm.cols + (fm.cols - 1 - col)
        elif fm.order == "column_major_snake":
            # Up-and-down zigzag: col 0 top-to-bottom, col 1 bottom-to-top, etc.
            if col % 2 == 0:
                offset_in_fmap = col * fm.rows + row
            else:
                offset_in_fmap = col * fm.rows + (fm.rows - 1 - row)
        else:
            raise NotImplementedError(f"order {fm.order!r} not supported yet")
        return fm.chain_id, fm.offset_in_chain + offset_in_fmap
