"""Normalize dataset-specific MANO layout before a sample becomes a batch."""

from __future__ import annotations

import numpy as np


MANO21_CONNECTIVITY = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# HOT3D's official toolkit draws a 20-landmark skeleton (the extra/padded
# landmark is not connected).  These are the same official edges expressed in
# the canonical 21-point order below, so projection can keep the public batch
# shape while omitting the unused point from the skeleton.
HOT3D_LANDMARK_CONNECTIVITY = (
    (0, 13), (13, 14), (14, 15), (15, 20),
    (0, 17), (17, 18), (18, 19), (19, 16),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 2), (2, 3), (3, 4),
    (2, 5), (5, 9), (9, 17), (17, 13),
)

# Source fingers requested by the user:
# 0-13-14-15-16, 0-1-2-3-17, 0-4-5-6-18,
# 0-7-8-9-19, 0-10-11-12-20.
HOT3D_SOURCE_TO_MANO21 = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)

DATASET_SOURCE_TO_MANO21 = {
    "hot3d": HOT3D_SOURCE_TO_MANO21,
}


def canonicalize_mano_joints(dataset_name: str, joints: np.ndarray | None) -> np.ndarray | None:
    """Return camera-space joints in canonical MANO-21 index order."""
    if joints is None:
        return None
    value = np.asarray(joints, dtype=np.float32)
    mapping = DATASET_SOURCE_TO_MANO21.get(str(dataset_name).lower())
    if mapping is None:
        return value
    if value.shape[-2] != len(mapping):
        raise ValueError(
            f"{dataset_name}: expected {len(mapping)} source MANO joints, got {value.shape}"
        )
    return np.take(value, mapping, axis=-2)


def validate_public_mano_contract(
    dataset_name: str,
    joints: np.ndarray | None,
    vertices: np.ndarray | None,
) -> None:
    """Fail inside the loader rather than emitting a malformed batch."""
    if joints is not None and np.asarray(joints).shape[-2:] != (21, 3):
        raise ValueError(f"{dataset_name}: public MANO joints must end in [21, 3]")
    if vertices is not None and np.asarray(vertices).shape[-2:] != (778, 3):
        raise ValueError(f"{dataset_name}: public MANO vertices must end in [778, 3]")
