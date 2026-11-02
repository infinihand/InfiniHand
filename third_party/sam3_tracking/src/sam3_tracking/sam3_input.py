"""Prepare the resized SAM3 frame folder and matching detection payload."""

import json
from pathlib import Path

import cv2
import torch


def resize_shape(image_size_hw: tuple[int, int], max_side: int):
    height, width = [int(value) for value in image_size_hw]
    if max_side <= 0 or max(height, width) <= max_side:
        return (height, width), 1.0, 1.0
    scale = float(max_side) / float(max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    return (
        (resized_height, resized_width),
        resized_width / float(width),
        resized_height / float(height),
    )


def scale_detection_payload(
    payload: dict,
    original_size_hw: tuple[int, int],
    resized_size_hw: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> dict:
    scaled = dict(payload)
    boxes = payload["boxes_xyxy"].clone()
    boxes[:, [0, 2]] *= float(scale_x)
    boxes[:, [1, 3]] *= float(scale_y)
    scaled["boxes_xyxy"] = boxes
    scaled["image_size_hw"] = torch.tensor(resized_size_hw, dtype=torch.long)
    scaled["prompt_image_size_hw"] = torch.tensor(
        resized_size_hw, dtype=torch.long
    )
    scaled["scale_xy"] = torch.tensor(
        [float(scale_x), float(scale_y)], dtype=torch.float32
    )
    scaled["coordinate_space"] = "sam3_resized_xyxy_abs_pixels"
    scaled["original_image_size_hw"] = torch.tensor(
        original_size_hw, dtype=torch.long
    )
    return scaled


def prepare_sam3_input(
    frame_paths: list[Path],
    detection_payload: dict,
    segment_dir: Path,
    image_size_hw: tuple[int, int],
    max_side: int,
    jpg_quality: int,
) -> dict:
    frame_dir = segment_dir / "frames"
    detection_dir = segment_dir / "detection"
    frame_dir.mkdir(parents=True)
    detection_dir.mkdir(parents=True)
    resized_size_hw, scale_x, scale_y = resize_shape(image_size_hw, int(max_side))
    resized_height, resized_width = resized_size_hw
    for frame_idx, source in enumerate(frame_paths):
        image = cv2.imread(str(source))
        assert image is not None, source
        if resized_size_hw != image_size_hw:
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        written = cv2.imwrite(
            str(frame_dir / f"{frame_idx:06d}.jpg"),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)],
        )
        assert written, source
    scaled_payload = scale_detection_payload(
        detection_payload,
        image_size_hw,
        resized_size_hw,
        scale_x,
        scale_y,
    )
    torch.save(scaled_payload, detection_dir / "bboxes_2d.pth")
    metadata = {
        "original_image_size_hw": list(image_size_hw),
        "sam3_image_size_hw": list(resized_size_hw),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "coordinate_contract": {
            "frames": "original decoded frame coordinates",
            "sam3_input": "resized coordinates",
            "final_masks": "restored to original decoded frame coordinates",
        },
    }
    (segment_dir / "resize_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
