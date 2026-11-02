"""Batched HaWoR YOLO hand detection with production area-max selection."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .detection_payload import (
    detection_payload_from_selected,
    save_detection_payload,
    select_area_max_by_side,
)


SIDE_NAMES = {0: "left", 1: "right"}


def detect_hands(
    frame_paths: list[Path],
    image_size_hw: tuple[int, int],
    detector_weight: Path,
    output_path: Path,
    detections_jsonl: Path,
    device: str,
    confidence: float,
    batch_size: int,
    imgsz: int,
    video_id: str,
) -> dict:
    from ultralytics import YOLO

    detector = YOLO(str(detector_weight))
    detector.to(device)
    selected_by_frame = []
    predict_kwargs = {"imgsz": int(imgsz)} if imgsz > 0 else {}
    for batch_start in tqdm(range(0, len(frame_paths), int(batch_size)), desc="YOLO"):
        paths = frame_paths[batch_start : batch_start + int(batch_size)]
        images = [cv2.imread(str(path)) for path in paths]
        assert all(image is not None for image in images), paths
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda", enabled=device.startswith("cuda")
            ):
                results = detector.predict(
                    images,
                    conf=float(confidence),
                    verbose=False,
                    **predict_kwargs,
                )
        counts = [int(result.boxes.xyxy.shape[0]) for result in results]
        rows_by_image = [
            torch.cat(
                [
                    result.boxes.xyxy,
                    result.boxes.conf[:, None],
                    result.boxes.cls[:, None],
                ],
                dim=1,
            )
            for result in results
        ]
        reference = next(
            (rows for rows in rows_by_image if rows.shape[0] > 0), rows_by_image[0]
        )
        stacked = torch.cat(
            [
                rows.to(device=reference.device, dtype=reference.dtype)
                for rows in rows_by_image
            ],
            dim=0,
        ).cpu().numpy()
        offset = 0
        for count in counts:
            rows = stacked[offset : offset + count]
            offset += count
            selected_by_frame.append(
                select_area_max_by_side(rows[:, :4], rows[:, 4], rows[:, 5])
            )
    payload = detection_payload_from_selected(
        selected_by_frame,
        image_size_hw,
        video_id=video_id,
        segment_id=output_path.parent.parent.name,
        conf=float(confidence),
    )
    output_path.parent.mkdir(parents=True)
    save_detection_payload(payload, output_path)
    with detections_jsonl.open("w", encoding="utf-8") as f:
        for frame_idx, selected in enumerate(selected_by_frame):
            for side in sorted(selected):
                area, score, box = selected[side]
                f.write(
                    json.dumps(
                        {
                            "frame_idx": frame_idx,
                            "side": int(side),
                            "side_name": SIDE_NAMES[int(side)],
                            "score": float(score),
                            "box_area": float(area),
                            "box_xyxy": [float(value) for value in box],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return payload


def detection_metrics(payload: dict, num_frames: int) -> dict:
    frame_idx = payload["frame_idx"].cpu().numpy()
    sides = payload["handedness"].cpu().numpy()
    left_frames = set(frame_idx[sides == 0].astype(np.int64).tolist())
    right_frames = set(frame_idx[sides == 1].astype(np.int64).tolist())
    return {
        "detection_count": int(len(frame_idx)),
        "frames_with_left_detection": len(left_frames),
        "frames_with_right_detection": len(right_frames),
        "frames_with_both_detections": len(left_frames & right_frames),
        "left_detection_coverage": len(left_frames) / float(num_frames),
        "right_detection_coverage": len(right_frames) / float(num_frames),
        "both_detection_coverage": len(left_frames & right_frames)
        / float(num_frames),
    }
