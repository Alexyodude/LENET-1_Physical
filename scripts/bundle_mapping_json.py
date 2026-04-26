"""Convert the mapping YAML into a JSON file shipped alongside the static frontend.

Run: `uv run python scripts/bundle_mapping_json.py`
"""
from __future__ import annotations
import json
from pathlib import Path

import yaml


def main() -> None:
    src = Path("config/mapping.example.yaml")
    dst = Path("src/lenet1_physical/twin/static/mapping.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as f:
        data = yaml.safe_load(f)
    with dst.open("w") as f:
        json.dump(data, f)
    print(f"saved {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
