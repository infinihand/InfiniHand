"""Flat YAML configuration loader."""

from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict), path
    return config
