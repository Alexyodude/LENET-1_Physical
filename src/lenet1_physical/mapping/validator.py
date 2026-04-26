from __future__ import annotations
from collections import defaultdict

from lenet1_physical.mapping.schema import Mapping


class ValidationError(ValueError):
    pass


def expected_chain_pixels(m: Mapping) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for layer in m.layers.values():
        for fm in layer.feature_maps:
            counts[fm.chain_id] += fm.rows * fm.cols
    return dict(counts)


def validate(m: Mapping) -> None:
    """Raises ValidationError if the mapping is internally inconsistent."""
    counts = expected_chain_pixels(m)
    for chain_id, expected in counts.items():
        if chain_id not in m.chains:
            raise ValidationError(f"chain {chain_id} referenced by a feature map but not declared")
        declared = m.chains[chain_id].length
        if declared != expected:
            raise ValidationError(
                f"chain {chain_id} expects {expected} pixels but is declared length {declared}"
            )
    occupied: dict[tuple[int, int], str] = {}
    for layer_name, layer in m.layers.items():
        for fm in layer.feature_maps:
            for row in range(fm.rows):
                for col in range(fm.cols):
                    chain, pos = m.lookup(layer_name, fm.id, row, col)
                    key = (chain, pos)
                    if key in occupied:
                        raise ValidationError(
                            f"collision at chain {chain} pos {pos}: "
                            f"{occupied[key]} and {layer_name} fmap {fm.id}"
                        )
                    occupied[key] = f"{layer_name} fmap {fm.id}"
