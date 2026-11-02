"""Restore SAM3 masks to original decoded-frame coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np

from .sam3_helpers import mask_to_box


SIDE_NAMES = {0: "left", 1: "right"}


def _archive_masks(sam3_raw_dir: Path, archive_mode: str) -> dict:
    if archive_mode == "off":
        return {}
    if archive_mode == "npz":
        archive = np.load(sam3_raw_dir / "framewise_masks.npz")
        return {
            key: archive[key].astype(bool)
            for key in archive.files
            if key.startswith("mask_")
        }
    assert archive_mode == "rle", archive_mode
    from pycocotools import mask as mask_utils

    payload = json.loads(
        (sam3_raw_dir / "framewise_masks_rle.json").read_text(encoding="utf-8")
    )
    masks = {}
    for row in payload["entries"]:
        encoded = {"size": row["size"], "counts": row["counts"].encode("ascii")}
        key = f"mask_{int(row['frame_idx']):06d}_{row['side_name']}"
        masks[key] = mask_utils.decode(encoded).astype(bool)
    return masks


def restore_masks(
    sam3_raw_dir: Path,
    output_dir: Path,
    num_frames: int,
    original_size_hw: tuple[int, int],
    sam3_size_hw: tuple[int, int],
    archive_mode: str,
) -> dict:
    masks_dir = output_dir / "masks"
    for side_name in SIDE_NAMES.values():
        (masks_dir / side_name).mkdir(parents=True)
    archived = _archive_masks(sam3_raw_dir, archive_mode)
    original_height, original_width = original_size_hw
    rows = []
    side_nonempty = {0: 0, 1: 0}
    for frame_idx in range(num_frames):
        for side, side_name in SIDE_NAMES.items():
            key = f"mask_{frame_idx:06d}_{side_name}"
            if archive_mode == "off":
                raw_path = (
                    sam3_raw_dir
                    / "framewise_masks"
                    / side_name
                    / f"{frame_idx:06d}.png"
                )
                mask = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE) > 0
            else:
                mask = archived.get(key, np.zeros(sam3_size_hw, dtype=bool))
            if tuple(mask.shape) != tuple(original_size_hw):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            destination = masks_dir / side_name / f"{frame_idx:06d}.png"
            written = cv2.imwrite(str(destination), mask.astype(np.uint8) * 255)
            assert written, destination
            area = int(mask.sum())
            side_nonempty[side] += int(area > 0)
            box = None if area == 0 else [float(value) for value in mask_to_box(mask)]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "side": side,
                    "side_name": side_name,
                    "status": "ok" if area > 0 else "empty_mask",
                    "mask_area": area,
                    "mask_box_xyxy": box,
                    "mask_relpath": destination.relative_to(output_dir).as_posix(),
                }
            )
    records_path = output_dir / "masks.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "num_frames": num_frames,
        "mask_record_count": len(rows),
        "frames_with_left_mask": side_nonempty[0],
        "frames_with_right_mask": side_nonempty[1],
        "left_mask_coverage": side_nonempty[0] / float(num_frames),
        "right_mask_coverage": side_nonempty[1] / float(num_frames),
        "mask_coordinate_space": "original_decoded_frame",
        "original_image_size_hw": list(original_size_hw),
        "sam3_image_size_hw": list(sam3_size_hw),
    }
    (output_dir / "mask_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
