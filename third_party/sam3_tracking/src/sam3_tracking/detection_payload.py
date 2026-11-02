import numpy as np
import torch

DETECTION_PAYLOAD_SCHEMA_VERSION = 1


def select_area_max_by_side(coords, confs, sides):
    kept = {}
    for coords_i, score_i, side_i in zip(coords, confs, sides):
        side = int(float(side_i) > 0.5)
        x1, y1, x2, y2 = [float(v) for v in coords_i[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        row = (area, float(score_i), [x1, y1, x2, y2])
        if side not in kept or row[:2] > kept[side][:2]:
            kept[side] = row
    return kept


def detection_payload_from_selected(
    selected_by_frame,
    image_size_hw,
    *,
    video_id="",
    segment_id="",
    frame_manifest=None,
    prompt_image_size_hw=None,
    scale_xy=(1.0, 1.0),
    coordinate_space="original_image_xyxy_abs_pixels",
    source="yolo_detector_predict_area_max_by_side",
    selection="per_frame_per_side_largest_area",
    conf=0.2,
):
    frame_idx = []
    side_id = []
    boxes_xyxy = []
    scores = []
    per_frame_counts = []
    for t, kept in enumerate(selected_by_frame):
        per_frame_counts.append({"frame_idx": int(t), "kept_sides": sorted(int(k) for k in kept)})
        for side in sorted(kept):
            _area, score, box = kept[side]
            frame_idx.append(int(t))
            side_id.append(int(side))
            boxes_xyxy.append(box)
            scores.append(float(score))
    if boxes_xyxy:
        box_tensor = torch.tensor(np.asarray(boxes_xyxy), dtype=torch.float32)  # (N, 4)
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)  # (N, 4)
    image_size_hw = tuple(int(x) for x in image_size_hw)
    if prompt_image_size_hw is None:
        prompt_image_size_hw = image_size_hw
    payload = {
        "schema_version": int(DETECTION_PAYLOAD_SCHEMA_VERSION),
        "video_id": str(video_id),
        "segment_id": str(segment_id),
        "frame_manifest": list(frame_manifest or []),
        "frame_idx": torch.tensor(frame_idx, dtype=torch.long),  # (N,)
        "side_id": torch.tensor(side_id, dtype=torch.long),  # (N,)
        "semantic_side": torch.tensor(side_id, dtype=torch.long),  # (N,)
        "track_id": torch.tensor(side_id, dtype=torch.long),  # legacy alias, semantic side only
        "boxes_xyxy": box_tensor,
        "scores": torch.tensor(scores, dtype=torch.float32),  # (N,)
        "handedness": torch.tensor(side_id, dtype=torch.long),  # legacy alias, 0 left / 1 right
        "handedness_labels": ["left", "right"],
        "image_size_hw": torch.tensor(image_size_hw, dtype=torch.long),
        "prompt_image_size_hw": torch.tensor(tuple(int(x) for x in prompt_image_size_hw), dtype=torch.long),
        "scale_xy": torch.tensor([float(scale_xy[0]), float(scale_xy[1])], dtype=torch.float32),
        "coordinate_space": str(coordinate_space),
        "box_format": "xyxy_abs_pixels",
        "source": str(source),
        "selection": str(selection),
        "conf": float(conf),
        "per_frame_counts": per_frame_counts,
    }
    return payload


def slice_detection_payload(
    full_selection_by_source_idx,
    frame_manifest,
    image_size_hw,
    *,
    video_id="",
    segment_id="",
    conf=0.2,
    source="yolo_detector_predict_area_max_by_side",
    selection="per_frame_per_side_largest_area",
):
    """25-14 detect-once: build a per-segment SAM3 seeding payload by slicing a
    full-video area-max-by-side selection on ``source_frame_idx``.

    ``full_selection_by_source_idx`` maps ``source_frame_idx -> select_area_max_by_side(...)``
    result (a ``{side: (area, score, box)}`` dict). For each frame in ``frame_manifest``
    (in segment order) we look up its selection by ``source_frame_idx`` and feed the
    ordered list to ``detection_payload_from_selected`` with the SAME signature
    ``save_yolo_detector_area_max_bboxes`` uses (no ``frame_manifest`` kwarg), so the
    resulting payload is byte-equivalent to the per-segment YOLO seed for the consumed
    fields when the full-video selection equals the per-segment one. Joins are exact:
    a segment frame whose ``source_frame_idx`` is absent fails loud (no silent empty box).
    """
    selected_by_frame = []
    for item in frame_manifest:
        src_idx = int(item["source_frame_idx"])
        assert src_idx in full_selection_by_source_idx, (
            f"detect-once join miss: segment frame source_frame_idx={src_idx} "
            f"is not present in the full-video detection selection"
        )
        selected_by_frame.append(full_selection_by_source_idx[src_idx])
    return detection_payload_from_selected(
        selected_by_frame,
        image_size_hw,
        video_id=video_id,
        segment_id=segment_id,
        conf=conf,
        source=source,
        selection=selection,
    )


def save_detection_payload(payload, out_path):
    torch.save(payload, out_path)
    return payload


def load_detection_payload(path):
    return torch.load(path, map_location="cpu")
