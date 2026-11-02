"""Decode videos or normalize image folders into ordered JPEG frames."""

import json
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_video_frames(
    video_path: Path,
    frame_dir: Path,
    out_fps: float,
    jpg_quality: int,
    max_frames: int,
) -> dict:
    frame_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), video_path
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    assert source_fps > 0.0, source_fps
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_idx = 0
    sample_idx = 0
    next_sample_time = 0.0
    sample_dt = 1.0 / float(out_fps)
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        source_time = source_frame_idx / source_fps
        if source_time + 1e-9 >= next_sample_time:
            name = f"{sample_idx:06d}.jpg"
            written = cv2.imwrite(
                str(frame_dir / name),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)],
            )
            assert written, name
            rows.append(
                {
                    "frame_idx": sample_idx,
                    "source_frame_idx": source_frame_idx,
                    "source_time_sec": source_time,
                    "file": name,
                }
            )
            sample_idx += 1
            next_sample_time += sample_dt
            if max_frames > 0 and sample_idx >= max_frames:
                break
        source_frame_idx += 1
    cap.release()
    assert rows, video_path
    return {
        "input_type": "video",
        "input_path": str(video_path.resolve()),
        "source_fps": source_fps,
        "source_video_frame_count": source_frame_count,
        "output_fps": float(out_fps),
        "width": width,
        "height": height,
        "num_frames": len(rows),
        "frames": rows,
    }


def prepare_frame_folder(
    input_dir: Path,
    frame_dir: Path,
    fps: float,
    jpg_quality: int,
    max_frames: int,
) -> dict:
    frame_dir.mkdir(parents=True)
    inputs = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_frames > 0:
        inputs = inputs[:max_frames]
    assert inputs, input_dir
    rows = []
    width = None
    height = None
    for frame_idx, source in enumerate(inputs):
        name = f"{frame_idx:06d}.jpg"
        destination = frame_dir / name
        image = cv2.imread(str(source))
        assert image is not None, source
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(source, destination)
        else:
            written = cv2.imwrite(
                str(destination),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)],
            )
            assert written, destination
        current_height, current_width = image.shape[:2]
        if width is None:
            width, height = current_width, current_height
        assert (current_width, current_height) == (width, height), source
        rows.append(
            {
                "frame_idx": frame_idx,
                "source_frame_idx": frame_idx,
                "source_time_sec": frame_idx / float(fps),
                "source_file": source.name,
                "file": name,
            }
        )
    return {
        "input_type": "frames",
        "input_path": str(input_dir.resolve()),
        "source_fps": float(fps),
        "output_fps": float(fps),
        "width": width,
        "height": height,
        "num_frames": len(rows),
        "frames": rows,
    }


def write_frame_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
