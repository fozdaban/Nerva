"""Load the JSON-compatible YAML configuration without extra dependencies."""

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a fresh config; general YAML syntax is intentionally unsupported."""
    source = Path(path) if path else Path(__file__).with_name("config.yaml")
    with source.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    return config
