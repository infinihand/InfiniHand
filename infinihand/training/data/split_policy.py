"""Validate the dataset boundary before training reads any scene data."""
import json
import re
from pathlib import Path


def validate_training_manifest(manifest, source="manifest"):
    if manifest.get("split") != "train":
        raise ValueError(f"Training requires explicit split='train': {source}")
    held_out = re.compile(r"(?:^|[/_.-])(test|val|valid|validation|eval|evaluation)(?:$|[/_.-])", re.I)
    records = [manifest, *manifest.get("scenes", [])]
    for record in records:
        if record.get("split", "train") != "train":
            raise ValueError(f"Non-training scene split: {source}")
        for key in ("dataset_name", "sampling_name", "raw_s3_root", "prefix"):
            if held_out.search(str(record.get(key, ""))):
                raise ValueError(f"Held-out identifier in {key}: {source}")


def validate_training_paths(paths, split_weights=None):
    if split_weights and set(split_weights) != {"train"}:
        raise ValueError("Split budget must contain only train")
    if not paths:
        raise ValueError("At least one training manifest is required")
    for path in paths:
        validate_training_manifest(json.loads(Path(path).read_text()), str(path))
