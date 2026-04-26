from pathlib import Path
import pytest
from lenet1_physical.mapping.schema import Mapping, FeatureMap, ChainSpec
from lenet1_physical.mapping.validator import (
    ValidationError, validate, expected_chain_pixels,
)


FIXTURE = Path("tests/fixtures/mapping_minimal.yaml")


def test_load_minimal_parses_layers_and_chains():
    m = Mapping.from_yaml(FIXTURE)
    assert set(m.layers) == {"L1", "L6"}
    l1 = m.layers["L1"].feature_maps[0]
    assert isinstance(l1, FeatureMap)
    assert l1.rows == 28 and l1.cols == 28
    assert m.chains[0].gpio == 18


def test_logical_to_physical_row_major_snake():
    m = Mapping.from_yaml(FIXTURE)
    # On a 28-col row-major-snake fill: row 0 left-to-right (positions 0..27),
    # row 1 right-to-left (positions 28..55, with col 27 first).
    assert m.lookup("L1", fmap=0, row=0, col=0) == (0, 0)
    assert m.lookup("L1", fmap=0, row=0, col=27) == (0, 27)
    assert m.lookup("L1", fmap=0, row=1, col=27) == (0, 28)
    assert m.lookup("L1", fmap=0, row=1, col=0) == (0, 55)


def test_lookup_unknown_layer_raises():
    m = Mapping.from_yaml(FIXTURE)
    with pytest.raises(KeyError):
        m.lookup("L9", fmap=0, row=0, col=0)


def test_validate_accepts_minimal_fixture():
    m = Mapping.from_yaml(FIXTURE)
    validate(m)  # raises if bad


def test_validate_detects_chain_length_mismatch(tmp_path):
    bad_yaml = (tmp_path / "bad.yaml")
    bad_yaml.write_text(
        FIXTURE.read_text().replace("length: 784", "length: 999")
    )
    m = Mapping.from_yaml(bad_yaml)
    with pytest.raises(ValidationError, match="chain 0 expects 784 pixels"):
        validate(m)


def test_expected_pixels_per_chain():
    m = Mapping.from_yaml(FIXTURE)
    counts = expected_chain_pixels(m)
    assert counts[0] == 28 * 28
    assert counts[16] == 1 * 10


def test_full_17_mapping_validates_clean():
    m = Mapping.from_yaml(Path("tests/fixtures/mapping_full_17.yaml"))
    validate(m)
    counts = expected_chain_pixels(m)
    assert sum(counts.values()) == 4634
    assert len(m.chains) == 17
