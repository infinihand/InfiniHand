"""Render original-coordinate detection and mask QA artifacts."""

import json
from pathlib import Path

import cv2
import numpy as np

from .sam3_helpers import detection_from_payload, detection_index


SIDE_NAMES = {0: "left", 1: "right"}
SIDE_COLORS_BGR = {0: (255, 0, 0), 1: (0, 0, 255)}


def draw_frame(
    frame: np.ndarray,
    frame_idx: int,
    masks_dir: Path,
    detections_by_frame: dict,
    alpha: float,
) -> np.ndarray:
    canvas = frame.copy()
    for side, side_name in SIDE_NAMES.items():
        mask = cv2.imread(
            str(masks_dir / side_name / f"{frame_idx:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert mask is not None, (frame_idx, side_name)
        foreground = mask > 0
        color = np.asarray(SIDE_COLORS_BGR[side], dtype=np.float32)
        canvas[foreground] = (
            canvas[foreground].astype(np.float32) * (1.0 - float(alpha))
            + color * float(alpha)
        ).astype(np.uint8)
    for detection in detections_by_frame.get(frame_idx, []):
        side = int(detection["side"])
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box_xyxy"]]
        color = SIDE_COLORS_BGR[side]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{SIDE_NAMES[side]} {float(detection['score']):.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"frame {frame_idx:06d}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_visualization(
    frame_paths: list[Path],
    masks_dir: Path,
    detection_payload: dict,
    output_dir: Path,
    fps: float,
    alpha: float,
    contact_frames: int,
) -> dict:
    first = cv2.imread(str(frame_paths[0]))
    assert first is not None, frame_paths[0]
    height, width = first.shape[:2]
    det_by_frame = detection_index(detection_from_payload(detection_payload))
    video_path = output_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    assert writer.isOpened(), video_path
    contact_indices = set(
        np.linspace(
            0,
            len(frame_paths) - 1,
            min(int(contact_frames), len(frame_paths)),
            dtype=np.int64,
        ).tolist()
    )
    contact_images = []
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        assert frame is not None, frame_path
        canvas = draw_frame(frame, frame_idx, masks_dir, det_by_frame, alpha)
        writer.write(canvas)
        if frame_idx in contact_indices:
            tile_width = min(480, width)
            tile_height = int(round(height * tile_width / float(width)))
            contact_images.append(
                cv2.resize(canvas, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            )
    writer.release()
    columns = min(4, len(contact_images))
    rows = int(np.ceil(len(contact_images) / float(columns)))
    tile_height, tile_width = contact_images[0].shape[:2]
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(contact_images):
        row = index // columns
        column = index % columns
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = image
    contact_path = output_dir / "contact_sheet.jpg"
    written = cv2.imwrite(str(contact_path), sheet)
    assert written, contact_path
    summary = {
        "overlay_path": str(video_path),
        "contact_sheet_path": str(contact_path),
        "frame_size_wh": [width, height],
        "coordinate_space": "original_decoded_frame",
        "mask_colors_bgr": {"left": [255, 0, 0], "right": [0, 0, 255]},
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
