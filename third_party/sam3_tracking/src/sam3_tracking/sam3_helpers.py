"""Small SAM3 helper surface used by the production point-prompt path."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SIDE_NAMES = {0: "left", 1: "right"}


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def detection_from_payload(detection):
    return {
        "frame_idx": as_numpy(detection["frame_idx"]).astype(np.int64),
        "track_id": as_numpy(detection["track_id"]).astype(np.int64),
        "boxes_xyxy": as_numpy(detection["boxes_xyxy"]).astype(np.float32),
        "scores": as_numpy(detection["scores"]).astype(np.float32),
        "handedness": as_numpy(detection["handedness"]).astype(np.int64),
    }


def load_detection(path):
    return detection_from_payload(torch.load(path, map_location="cpu", weights_only=False))


def detection_index(det):
    by_frame = defaultdict(list)
    for frame, track_id, box, score, side in zip(
        det["frame_idx"],
        det["track_id"],
        det["boxes_xyxy"],
        det["scores"],
        det["handedness"],
    ):
        by_frame[int(frame)].append(
            {
                "track_id": int(track_id),
                "box_xyxy": box.astype(np.float32),
                "score": float(score),
                "side": int(side),
            }
        )
    return by_frame


def sorted_frames(segment_dir):
    frames = sorted((Path(segment_dir) / "frames").glob("*.jpg"))
    assert frames, f"no jpg frames in {Path(segment_dir) / 'frames'}"
    return frames


def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def xywh_rel_to_xyxy_abs(box_xywh, width, height):
    x, y, w, h = [float(v) for v in box_xywh]
    return np.array(
        [x * width, y * height, (x + w) * width, (y + h) * height],
        dtype=np.float32,
    )


def xyxy_abs_to_xywh_rel(box_xyxy, width, height):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return np.array(
        [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height],
        dtype=np.float32,
    )


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1.0, ys.max() + 1.0],
        dtype=np.float32,
    )


def normalize_outputs(outputs):
    obj_ids = as_numpy(outputs["out_obj_ids"]).astype(np.int64)
    boxes = as_numpy(outputs["out_boxes_xywh"]).astype(np.float32)
    masks = as_numpy(outputs["out_binary_masks"])
    probs = as_numpy(
        outputs.get("out_probs", np.ones((len(obj_ids),), dtype=np.float32))
    ).astype(np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return obj_ids, boxes, masks.astype(bool), probs


def output_box_xyxy(box_rel, mask, width, height):
    box = xywh_rel_to_xyxy_abs(box_rel, width, height)
    if int(mask.sum()) > 0 and (
        (box[2] - box[0]) <= 1 or (box[3] - box[1]) <= 1
    ):
        box = mask_to_box(mask)
    return box.astype(np.float32)


def side_probs_from_label_score(det):
    score = float(det["score"])
    if int(det["side"]) == 0:
        return score, 1.0 - score
    return 1.0 - score, score


def side_margin(det):
    prob_left, prob_right = side_probs_from_label_score(det)
    return abs(prob_left - prob_right)


def box_area_xyxy(box_xyxy):
    x1, y1, x2, y2 = [float(x) for x in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def build_predictor(checkpoint_path, default_output_prob_thresh, sam_version="sam3"):
    if sam_version == "sam3":
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            compile=False,
            async_loading_frames=False,
        )
        predictor.default_output_prob_thresh = float(default_output_prob_thresh)
        return predictor
    assert sam_version == "sam3.1", sam_version
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    return build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint_path,
        use_fa3=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        default_output_prob_thresh=float(default_output_prob_thresh),
    )
