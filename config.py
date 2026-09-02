from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SECTION_KEYS = ("dataset", "preprocessing", "graph", "model", "training")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a sectioned YAML file and flatten it for the established runners."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    config: dict[str, Any] = {}
    for section in SECTION_KEYS:
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
        config.update(values)

    if "name" not in config:
        raise ValueError("Config must define dataset.name.")
    return config
