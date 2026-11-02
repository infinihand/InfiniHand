#!/usr/bin/env python3
"""Production SAM3 point-prompt tight boxes from SAM mask.

This is the SAM mask point-prompt strategy without the comparison/baseline
rendering path. It consumes a segment directory containing frames/ and
raw detector side boxes in detection/bboxes_2d.pth, then writes SAM tight boxes
for HaWoR crop inference.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
SAM3_SRC = os.environ.get("SAM3_SRC") or os.environ.get("SAM_SRC") or str(REPO_ROOT / "thirdparty" / "sam3")
sys.path.insert(0, SAM3_SRC)

from . import sam3_helpers as simple  # noqa: E402
from .ego_candidate_score import (  # noqa: E402
    DEFAULT_EGO_GATE_CONFIG,
    EGO_GATE_CONFIG_KEYS,
    config_from_namespace,
    config_to_dict,
    cfg_value,
    score_ego_candidate,
    score_frame_candidates,
    select_window_ego_seed,
)
from .sam3_helpers import detection_from_payload  # noqa: E402
from .stage_events import phase19_stage_event, stage_event_scope, stage_events_enabled  # noqa: E402
from .unit_errors import RecoverableUnitError  # noqa: E402

SIDE_NAMES = simple.SIDE_NAMES

# 27-01 W0-c: per-run accumulators for mask PNG writes inside
# write_record_from_obj; emitted as ONE sam3_point.png_write_total event at the
# end of each export_framewise call, then reset.
_PNG_WRITE_SEC = 0.0
_PNG_WRITE_COUNT = 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--sam-version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--score-thresh", type=float, default=0.75)
    parser.add_argument("--bottom-y2-frac", type=float, default=0.98)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--point-expand", type=float, default=1.1)
    parser.add_argument("--prompt-geometry", choices=["ring", "below", "box", "box_below", "box_perimeter"], default="ring")
    parser.add_argument("--neg-below-count", type=int, default=5)
    parser.add_argument("--neg-below-frac", type=float, default=0.05)
    parser.add_argument("--sam-output-thresh", type=float, default=0.15)
    parser.add_argument("--default-output-prob-thresh", type=float, default=0.15)
    parser.add_argument("--prompt-text", default="hand")
    parser.add_argument("--left-prompt-text", default="")
    parser.add_argument("--right-prompt-text", default="")
    parser.add_argument("--tracking-mode", choices=["separate", "joint"], default="joint")
    parser.add_argument("--init-mode", choices=["point", "semantic"], default="point")
    parser.add_argument("--propagation-scheme", choices=["dual", "single_anchor"], default="dual")
    parser.add_argument("--mask-archive", choices=["off", "rle", "npz"], default="off", help="27-04 D-07: 'off' (default) = legacy per-frame mask PNGs, byte-identical. 'rle'/'npz' = keep masks in memory and pack ONE archive per segment (COCO RLE JSON / savez_compressed npz) instead of ~4000 PNGs; sam_tight_bboxes_2d.jsonl/.csv content is identical to the PNG path. 'rle' requires pycocotools and fails loud (ImportError) if missing — 'npz' is the operator-chosen fallback, never a silent one.")
    parser.add_argument("--exec-micro-opts", action="store_true", help="27-04 N4+C3: normalize propagation outputs exactly ONCE per streamed frame and share the (ids, boxes, masks, probs) tuple across sides (exact equality — same tensors, same values), and pass run_gc_collect=False on close_session (the pipeline owns memory hygiene via per-video cleanup + the 25-12 recycle valve). Flag off (default) = byte-identical legacy request dicts and per-side normalization.")
    parser.add_argument("--offload-video", action="store_true", help="27-04 N2 / D-02 OOM ladder step 3 ONLY: add offload_video_to_cpu=True to every start_session request (upstream sam3_base_predictor already honors it on the PIL-list branch). Costs ~5-10 percent propagation and +6.2 GB host RAM/worker; NOT for routine use. Flag off (default) = no key, byte-identical.")
    parser.add_argument("--offload-state", action="store_true", help="27-09 VRAM attribution follow-up: add offload_state_to_cpu=True to every start_session request — moves the SAM3 per-frame tracking memory bank (inference_state storage) to host RAM (upstream init_state native support; ~10-15 percent propagation cost). The memory bank, not the video frames, owns the propagation-stage VRAM peak. Flag off (default) = no key, byte-identical.")
    parser.add_argument("--chunk-trim", action="store_true", help="27-09: add trim_grounding_chunk=True to every start_session request — the batched grounding path caches a 16-frame chunk whose prev_encoder_out/encoder_hidden_states entries are dead weight (skip_keys in _slice_batched_output, never read after the chunk forward). Flag off (default) = no key, byte-identical.")
    parser.add_argument("--dedup-pos-enc", action="store_true", help="27-09: add dedup_image_pos_enc=True to every start_session request — store ONE shared image_pos_enc (content-identical sine grid every frame) in the session constants instead of one copy per propagated frame. Flag off (default) = no key, byte-identical.")
    parser.add_argument("--seed-window-ratio-min", type=float, default=0.5, help="v3 entry criteria (Kevin 2026-06-13): minimum per-side fraction of prompt windows that produced a valid seed; below this the unit is entry-skipped as a data-quality case (DATA_SKIP). 0 disables the ratio condition.")
    parser.add_argument("--ego-gate-enabled", action="store_true", help="Phase 31: select SAM3 prompt seeds with the bbox-only ego-hand score gate (bbox sqrt-area + bbox bottom position) before tracking.")
    for key in EGO_GATE_CONFIG_KEYS:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=float,
            default=float(getattr(DEFAULT_EGO_GATE_CONFIG, key)),
        )
    parser.add_argument("--close-empty-cache", action="store_true", help="27-09 2-up co-location: pass clear_cache_threshold=0 on every close_session request so the predictor-side empty_cache ALWAYS fires at session close (legacy gate: only at >=80 percent device pressure, which 1-up never crosses — so SLAM/DROID stacks on the retained propagation pool; attribution job 6412946: 26.75 GB returned when it fired). Forces run_gc_collect=True at close (the cache eviction is nested under it), overriding the --exec-micro-opts skip for the close call only. One CUDA sync per segment. Flag off (default) = byte-identical legacy pressure gate.")
    parser.add_argument("--cuda-invalid-retries", type=int, default=1)
    parser.add_argument("--text-inject-every", type=int, default=0, help="Step2 spike: if >0, set inference_state text_ids to TEXT on every Nth frame after point seeding (no add_prompt -> no reset); text rides along during propagation. 0 = off (pure point baseline).")
    parser.add_argument("--git-commit", default="unknown")
    return parser.parse_args(argv)


def now():
    return time.perf_counter()


def dump_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clean_retry_outputs(out_dir: Path):
    for name in ["masks", "framewise_masks"]:
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in [
        "run_config.json",
        "prompt_seed_debug.json",
        "point_prompt_records.jsonl",
        "prompt_points.json",
        "point_prompt_summary.json",
        "sam_tight_bboxes_2d.jsonl",
        "sam_tight_bboxes_2d.csv",
        "framewise_masks_rle.json",
        "framewise_masks.npz",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def side_margin(det):
    return simple.side_margin(det)


def prompt_text_for_side(args, side):
    text = args.left_prompt_text if int(side) == 0 else args.right_prompt_text
    return text if text else args.prompt_text


def ego_gate_enabled(args) -> bool:
    return bool(getattr(args, "ego_gate_enabled", False))


def resolve_image_size_hw(image_size_hw_or_height, args):
    if isinstance(image_size_hw_or_height, (tuple, list)):
        assert len(image_size_hw_or_height) >= 2, image_size_hw_or_height
        return int(image_size_hw_or_height[0]), int(image_size_hw_or_height[1])
    assert not ego_gate_enabled(args), "--ego-gate-enabled requires image_size_hw=(height,width)"
    return int(image_size_hw_or_height), 0


def ego_gate_config_from_args(args):
    return config_from_namespace(args)


def det_to_ego_candidate(side: int, frame_idx: int, det):
    return {
        "frame_idx": int(frame_idx),
        "side": int(side),
        "bbox_xyxy": [float(x) for x in det["box_xyxy"]],
        "score": float(det["score"]),
        "track_id": int(det["track_id"]),
    }


def seed_from_ego_row(row, window_start, window_end):
    seed = {
        "side": int(row["side"]),
        "side_name": SIDE_NAMES[int(row["side"])],
        "window_start": int(window_start),
        "window_end": int(window_end),
        "frame_idx": int(row["frame_idx"]),
        "score": float(row["score"]),
        "side_margin": 0.0,
        "track_id": int(row["track_id"]),
        "box_xyxy": [float(x) for x in row["bbox_xyxy"]],
        "det_box_area": float(simple.box_area_xyxy(row["bbox_xyxy"])),
        "ego_score": float(row["ego_score"]),
        "ego_margin": float(row["ego_margin"]),
        "ego_score_top1": float(row["ego_score_top1"]),
        "ego_score_top2": float(row["ego_score_top2"]),
        "area_score": float(row["area_score"]),
        "bottom_score": float(row["bottom_score"]),
        "size_ratio": float(row["size_ratio"]),
        "bottom_ratio": float(row["bottom_ratio"]),
        "frame_side_rank": int(row["frame_side_rank"]),
        "selection_status": str(row["selection_status"]),
    }
    return seed


def select_prompt_seeds(det_by_frame, num_frames: int, image_size_hw_or_height, args):
    height, width = resolve_image_size_hw(image_size_hw_or_height, args)
    window = int(round(float(args.window_sec) * float(args.fps)))
    assert window > 0, window
    bottom_y2 = float(args.bottom_y2_frac) * float(height)
    seed_by_side = {0: [], 1: []}
    debug = {
        "window_frames": window,
        "bottom_y2_px": bottom_y2,
        "score_thresh": float(args.score_thresh),
        "ego_gate_enabled": ego_gate_enabled(args),
        "sides": {},
    }
    cfg = ego_gate_config_from_args(args) if ego_gate_enabled(args) else DEFAULT_EGO_GATE_CONFIG
    if ego_gate_enabled(args):
        debug["image_size_hw"] = [int(height), int(width)]
        debug["ego_gate_config"] = config_to_dict(cfg)
    for side in [0, 1]:
        side_debug = {
            "valid_candidate_count": 0,
            "bottom_skip_count": 0,
            "low_score_skip_count": 0,
            "ego_gate_skip_count": 0,
            "ego_gate_selected_count": 0,
            "windows": [],
        }
        for start in range(0, int(num_frames), window):
            end = min(int(num_frames) - 1, start + window - 1)
            candidates = []
            for frame_idx in range(start, end + 1):
                for det in det_by_frame.get(frame_idx, []):
                    if int(det["side"]) != int(side):
                        continue
                    if float(det["score"]) <= float(args.score_thresh):
                        side_debug["low_score_skip_count"] += 1
                        continue
                    if float(det["box_xyxy"][3]) >= bottom_y2:
                        side_debug["bottom_skip_count"] += 1
                        continue
                    side_debug["valid_candidate_count"] += 1
                    candidates.append((int(frame_idx), det, det_to_ego_candidate(side, frame_idx, det)))
            if candidates and ego_gate_enabled(args):
                scored_rows = score_frame_candidates([item[2] for item in candidates], (height, width), cfg)
                gate = select_window_ego_seed(scored_rows, side, start, end, cfg)
                window_debug = dict(gate)
                window_debug["candidate_count"] = int(len(scored_rows))
                if bool(gate["selected"]):
                    seed = seed_from_ego_row(gate, start, end)
                    seed_by_side[side].append(seed)
                    side_debug["ego_gate_selected_count"] += 1
                    side_debug["windows"].append({**seed, "candidate_count": int(len(scored_rows)), "status": "selected"})
                else:
                    side_debug["ego_gate_skip_count"] += 1
                    side_debug["windows"].append(window_debug)
            elif candidates:
                seed_frame_idx, det, _cand = max(candidates, key=lambda item: (float(item[1]["score"]), -int(item[0])))
                seed = {
                    "side": int(side),
                    "side_name": SIDE_NAMES[int(side)],
                    "window_start": int(start),
                    "window_end": int(end),
                    "frame_idx": int(seed_frame_idx),
                    "score": float(det["score"]),
                    "side_margin": float(side_margin(det)),
                    "track_id": int(det["track_id"]),
                    "box_xyxy": [float(x) for x in det["box_xyxy"]],
                    "det_box_area": float(simple.box_area_xyxy(det["box_xyxy"])),
                }
                seed_by_side[side].append(seed)
                side_debug["windows"].append(seed)
            else:
                side_debug["windows"].append({"side": int(side), "window_start": int(start), "window_end": int(end), "status": "no_valid_seed"})
        debug["sides"][SIDE_NAMES[side]] = side_debug
    return seed_by_side, debug


def is_valid_prompt_det(det, height: int, args):
    if float(det["score"]) <= float(args.score_thresh):
        return False
    if ego_gate_enabled(args):
        if isinstance(height, (tuple, list)):
            image_size_hw = height
        else:
            raise AssertionError("--ego-gate-enabled select_joint_init_seeds must receive image_size_hw or use selected prompt seeds")
        cfg = ego_gate_config_from_args(args)
        score = score_ego_candidate(det["box_xyxy"], image_size_hw, cfg)
        return float(score["ego_score"]) > cfg_value(cfg, "ego_gate_keep_tau")
    return float(det["box_xyxy"][3]) < float(args.bottom_y2_frac) * float(height)


def det_to_seed(side: int, frame_idx: int, det):
    return {
        "side": int(side),
        "side_name": SIDE_NAMES[int(side)],
        "window_start": None,
        "window_end": None,
        "frame_idx": int(frame_idx),
        "score": float(det["score"]),
        "side_margin": float(side_margin(det)),
        "track_id": int(det["track_id"]),
        "box_xyxy": [float(x) for x in det["box_xyxy"]],
        "det_box_area": float(simple.box_area_xyxy(det["box_xyxy"])),
    }


def select_joint_init_seeds(det_by_frame, height: int, active_sides, args):
    if len(active_sides) == 1:
        side = int(active_sides[0])
        best = None
        for frame_idx in sorted(det_by_frame):
            candidates = [d for d in det_by_frame[frame_idx] if int(d["side"]) == side and is_valid_prompt_det(d, height, args)]
            if not candidates:
                continue
            det = max(candidates, key=lambda d: (float(d["score"]), simple.box_area_xyxy(d["box_xyxy"])))
            seed = det_to_seed(side, frame_idx, det)
            if best is None or (float(seed["score"]), float(seed["det_box_area"])) > (float(best["score"]), float(best["det_box_area"])):
                best = seed
        if best is None:
            raise RecoverableUnitError("no_valid_single_side_seed", {"side": SIDE_NAMES[side]})
        return {side: best}

    assert sorted(active_sides) == [0, 1], active_sides
    best_pair = None
    for frame_idx in sorted(det_by_frame):
        by_side = {}
        for side in [0, 1]:
            candidates = [d for d in det_by_frame[frame_idx] if int(d["side"]) == side and is_valid_prompt_det(d, height, args)]
            if candidates:
                by_side[side] = max(candidates, key=lambda d: (float(d["score"]), simple.box_area_xyxy(d["box_xyxy"])))
        if set(by_side) != {0, 1}:
            continue
        left = by_side[0]
        right = by_side[1]
        quality = (
            min(float(left["score"]), float(right["score"])),
            float(left["score"]) + float(right["score"]),
            simple.box_area_xyxy(left["box_xyxy"]) + simple.box_area_xyxy(right["box_xyxy"]),
            -int(frame_idx),
        )
        if best_pair is None or quality > best_pair[0]:
            best_pair = (quality, {0: det_to_seed(0, frame_idx, left), 1: det_to_seed(1, frame_idx, right)})
    if best_pair is None:
        raise RecoverableUnitError("no_covisible_hand_seed", {"active_sides": list(active_sides)})
    return best_pair[1]


def select_joint_init_seeds_from_prompt_seeds(seeds_by_side, active_sides):
    if len(active_sides) == 1:
        side = int(active_sides[0])
        seeds = list(seeds_by_side[side])
        assert seeds, side
        return {
            side: max(
                seeds,
                key=lambda s: (
                    float(s["score"]),
                    float(s.get("ego_score", 0.0)),
                    float(s["det_box_area"]),
                    -int(s["frame_idx"]),
                ),
            )
        }

    assert sorted(active_sides) == [0, 1], active_sides
    left_by_frame = defaultdict(list)
    right_by_frame = defaultdict(list)
    for seed in seeds_by_side[0]:
        left_by_frame[int(seed["frame_idx"])].append(seed)
    for seed in seeds_by_side[1]:
        right_by_frame[int(seed["frame_idx"])].append(seed)
    best_pair = None
    for frame_idx in sorted(set(left_by_frame) & set(right_by_frame)):
        left = max(left_by_frame[frame_idx], key=lambda s: (float(s["score"]), float(s.get("ego_score", 0.0)), float(s["det_box_area"])))
        right = max(right_by_frame[frame_idx], key=lambda s: (float(s["score"]), float(s.get("ego_score", 0.0)), float(s["det_box_area"])))
        quality = (
            min(float(left["score"]), float(right["score"])),
            float(left["score"]) + float(right["score"]),
            min(float(left.get("ego_score", 0.0)), float(right.get("ego_score", 0.0))),
            float(left.get("ego_score", 0.0)) + float(right.get("ego_score", 0.0)),
            float(left["det_box_area"]) + float(right["det_box_area"]),
            -int(frame_idx),
        )
        if best_pair is None or quality > best_pair[0]:
            best_pair = (quality, {0: left, 1: right})
    if best_pair is None:
        raise RecoverableUnitError("no_covisible_ego_gate_seed", {"active_sides": list(active_sides)})
    return best_pair[1]


def entry_skip_from_seeds(seeds_by_side, seed_debug, det_by_frame, height, args):
    """v3 entry criteria as a pure decision. Returns None to PROCESS the unit, or
    (reason, detail) to SKIP it. This is the SINGLE source of truth shared by the
    in-run check (run_point_prompt joint branch) and the unit-planning pre-skip
    (pipeline --early-entry-skip), so the two can never drift."""
    active_sides = [side for side in [0, 1] if seeds_by_side[side]]
    if len(active_sides) == 0:
        return ("missing_side_seed", {"active_sides": []})
    if float(args.seed_window_ratio_min) > 0.0:
        ratios = {}
        for side in [0, 1]:
            windows = seed_debug["sides"][SIDE_NAMES[side]]["windows"]
            ratios[SIDE_NAMES[side]] = len(seeds_by_side[side]) / max(1, len(windows))
        if max(ratios.values()) < float(args.seed_window_ratio_min):
            return ("low_seed_window_ratio", {"ratios": ratios, "min_required": float(args.seed_window_ratio_min)})
    try:
        if ego_gate_enabled(args):
            select_joint_init_seeds_from_prompt_seeds(seeds_by_side, active_sides)
        else:
            select_joint_init_seeds(det_by_frame, height, active_sides, args)
    except RecoverableUnitError as exc:
        return (exc.reason, exc.detail)
    return None


def entry_skip_reason_for_payload(detection_payload, num_frames, height, args):
    """Detection-only entry-skip decision for unit planning. Replicates the seed path
    run_point_prompt takes (detection_from_payload -> detection_index ->
    select_prompt_seeds -> entry_skip_from_seeds) from a sliced detection payload, with
    NO decode / resize / SAM3 session. Returns None to PROCESS or (reason, detail) to
    SKIP. Equivalent by construction: the planner feeds the SAME slice_detection_payload
    output the runtime feeds, over the unit's frame slice."""
    det = detection_from_payload(detection_payload)
    det_by_frame = simple.detection_index(det)
    image_size_hw = [int(x) for x in detection_payload["image_size_hw"].tolist()]
    seeds_by_side, seed_debug = select_prompt_seeds(det_by_frame, num_frames, image_size_hw, args)
    return entry_skip_from_seeds(seeds_by_side, seed_debug, det_by_frame, height, args)


def expand_box_xyxy(box, scale, width, height):
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1) * float(scale), (y2 - y1) * float(scale)
    return np.array([
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width - 1), cx + bw * 0.5),
        min(float(height - 1), cy + bh * 0.5),
    ], dtype=np.float32)


def bbox_to_instance_points(box, width, height, expand, geometry="ring", neg_below_count=5, neg_below_frac=0.05):
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    def below_negatives():
        # A row of negatives just below the hand box (forearm enters from below in
        # head-mounted video). x uniform across box width, y a fraction of box height
        # below the bottom edge. No side/top negatives so fingers are not clipped.
        bh = max(1.0, y2 - y1)
        ny = min(float(height - 1), y2 + float(neg_below_frac) * bh)
        return [[float(x) / width, ny / height] for x in np.linspace(x1, x2, int(neg_below_count))]

    if geometry == "below":
        # 1 positive center + N forearm negatives below.
        negs = below_negatives()
        points = [[cx / width, cy / height]] + negs
        labels = [1] + [0] * len(negs)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [x1, y1, x2, y2]

    if geometry in ("box", "box_below"):
        # SAM2-style box prompt: two corners encoded as points with labels 2 (top-left)
        # and 3 (bottom-right). Optionally add forearm negatives ("box_below").
        points = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
        labels = [2, 3]
        if geometry == "box_below":
            negs = below_negatives()
            points = points + negs
            labels = labels + [0] * len(negs)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [x1, y1, x2, y2]

    if geometry == "box_perimeter":
        # box_below box markers (labels 2,3) but negatives replaced by 8 box-perimeter
        # NEGATIVE points: 4 corners + 4 edge midpoints (no forearm-below row).
        points = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
        labels = [2, 3]
        for px, py in [[x1, y1], [cx, y1], [x2, y1], [x2, cy], [x2, y2], [cx, y2], [x1, y2], [x1, cy]]:
            points.append([px / width, py / height])
            labels.append(0)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [x1, y1, x2, y2]

    # geometry == "ring" (current default behavior): 1 positive center + 12 negatives on
    # the (optionally expanded) box edges.
    ex1, ey1, ex2, ey2 = [float(x) for x in expand_box_xyxy(box, expand, width, height)]
    points = [[cx / width, cy / height]]
    for x in np.linspace(ex1, ex2, 3):
        points.append([float(x) / width, ey1 / height])
    for x in np.linspace(ex1, ex2, 3):
        points.append([float(x) / width, ey2 / height])
    for y in np.linspace(ey1, ey2, 3):
        points.append([ex1 / width, float(y) / height])
    for y in np.linspace(ey1, ey2, 3):
        points.append([ex2 / width, float(y) / height])
    labels = [1] + [0] * 12
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [float(x) for x in [ex1, ey1, ex2, ey2]]


def choose_obj(outputs, prompt_box, width, height):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    rows = []
    for pos, obj_id in enumerate(ids):
        box = simple.output_box_xyxy(boxes_rel[pos], masks[pos], width, height)
        rows.append((simple.xyxy_iou(np.asarray(prompt_box, dtype=np.float32), box), int(masks[pos].sum()), float(probs[pos]), int(obj_id), pos, box))
    assert rows, "semantic init returned no SAM objects"
    return max(rows, key=lambda x: (x[0], x[1], x[2]))


def choose_semantic_objects(outputs, init_by_side, width, height):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    side_ids = sorted(init_by_side)
    assert len(ids) >= len(side_ids), f"semantic init returned {len(ids)} objects for {len(side_ids)} side prompts"
    rows_by_side = {}
    for side in side_ids:
        rows = []
        prompt_box = np.asarray(init_by_side[side]["box_xyxy"], dtype=np.float32)
        for pos, obj_id in enumerate(ids):
            box = simple.output_box_xyxy(boxes_rel[pos], masks[pos], width, height)
            rows.append(
                {
                    "side": int(side),
                    "obj_id": int(obj_id),
                    "pos": int(pos),
                    "prompt_iou": float(simple.xyxy_iou(prompt_box, box)),
                    "mask_area": int(masks[pos].sum()),
                    "prob": float(probs[pos]),
                    "box_xyxy": [float(x) for x in box],
                }
            )
        rows_by_side[side] = rows
    best = None
    for positions in permutations(range(len(ids)), len(side_ids)):
        picked = [rows_by_side[side][pos] for side, pos in zip(side_ids, positions)]
        obj_ids = [row["obj_id"] for row in picked]
        if len(set(obj_ids)) != len(obj_ids):
            continue
        quality = (sum(row["prompt_iou"] for row in picked), sum(row["mask_area"] for row in picked), sum(row["prob"] for row in picked))
        if best is None or quality > best[0]:
            best = (quality, picked)
    assert best is not None, f"could not assign unique semantic objects for sides {side_ids}"
    return {int(row["side"]): row for row in best[1]}, rows_by_side


def find_obj(outputs, obj_id: int):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    matches = np.where(ids == int(obj_id))[0]
    assert len(matches) == 1, f"expected obj_id={obj_id} once, got {ids.tolist()}"
    pos = int(matches[0])
    return boxes_rel[pos], masks[pos], float(probs[pos])


def find_obj_or_none(outputs, obj_id: int):
    return find_in_normalized(simple.normalize_outputs(outputs), obj_id)


def find_in_normalized(normalized, obj_id: int):
    # 27-04 N4: lookup against an already-normalized (ids, boxes_rel, masks, probs)
    # tuple so propagation consumers normalize each streamed frame exactly ONCE
    # instead of re-downloading the full GPU mask stack per side.
    ids, boxes_rel, masks, probs = normalized
    matches = np.where(ids == int(obj_id))[0]
    if len(matches) == 0:
        return None
    assert len(matches) == 1, f"expected obj_id={obj_id} once, got {ids.tolist()}"
    pos = int(matches[0])
    return boxes_rel[pos], masks[pos], float(probs[pos])


def start_session_request(sam3_resource, offload_video: bool, offload_state: bool = False, chunk_trim: bool = False, dedup_pos_enc: bool = False):
    request = {"type": "start_session", "resource_path": sam3_resource}
    if offload_video:
        # 27-04 N2 (D-02 OOM ladder step 3 only): upstream reads this key and
        # offloads decoded video frames to host RAM on the PIL-list branch.
        request["offload_video_to_cpu"] = True
    if offload_state:
        # 27-09: the per-frame tracking memory bank (inference_state) is the
        # propagation-stage VRAM peak owner — store it on host RAM.
        request["offload_state_to_cpu"] = True
    if chunk_trim:
        # 27-09: drop the dead prev_encoder_out/encoder_hidden_states entries from
        # each cached 16-frame grounding chunk after the backbone slices are taken.
        request["trim_grounding_chunk"] = True
    if dedup_pos_enc:
        # 27-09: per-frame image_pos_enc is the same sine grid every frame — store
        # one shared copy in the session constants instead of one per frame.
        request["dedup_image_pos_enc"] = True
    return request


def close_session_request(session_id, exec_micro_opts: bool, close_empty_cache: bool = False):
    request = {"type": "close_session", "session_id": session_id}
    if exec_micro_opts and not close_empty_cache:
        # 27-04 C3: skip the predictor-side gc.collect; the pipeline owns memory
        # hygiene (per-video cleanup + the 25-12 recycle valve).
        request["run_gc_collect"] = False
    if close_empty_cache:
        # 27-09 2-up: threshold 0 = empty_cache fires unconditionally at close.
        # run_gc_collect must stay True — the eviction branch is nested under it.
        request["clear_cache_threshold"] = 0
    return request


def pack_mask_payload(mask):
    # 27-04 D-07: in-memory mask record — bool (H, W) packed to bits (1D uint8),
    # ~8x smaller than bool so a full segment's records stay bounded. In-memory
    # only, never serialized to jsonl.
    return np.packbits(np.ascontiguousarray(mask))


def unpack_mask_payload(payload, height: int, width: int):
    return np.unpackbits(payload, count=int(height) * int(width)).reshape(int(height), int(width)).astype(bool)


def write_mask_archive(selected, out_dir: Path, width: int, height: int, mode: str):
    # 27-04 D-07: pack the selected (frame_idx, side) masks of the segment into ONE
    # archive file in the sam3 output tree (swept by enqueue_tree) instead of
    # ~4000 per-frame PNGs. No try/except: archive write failures crash loud.
    entries = sorted(selected.items())
    if mode == "rle":
        # A4: pycocotools is probed, never installed — ImportError propagates loud;
        # npz is the operator-chosen fallback via the flag value, never a silent one.
        from pycocotools import mask as mask_util

        packed = []
        for (frame_idx, side), rec in entries:
            mask = unpack_mask_payload(rec["mask_payload"], height, width)
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            packed.append({
                "frame_idx": int(frame_idx),
                "side": int(side),
                "side_name": SIDE_NAMES[int(side)],
                "obj_id": int(rec["obj_id"]),
                "size": [int(x) for x in rle["size"]],
                "counts": rle["counts"].decode("ascii"),
            })
        archive_path = out_dir / "framewise_masks_rle.json"
        archive_path.write_text(
            json.dumps({"format": "coco_rle", "height": int(height), "width": int(width), "entries": packed}, sort_keys=True),
            encoding="utf-8",
        )
        return archive_path
    assert mode == "npz", mode
    arrays = {}
    frame_idx_arr = []
    side_arr = []
    obj_id_arr = []
    for (frame_idx, side), rec in entries:
        arrays[f"mask_{int(frame_idx):06d}_{SIDE_NAMES[int(side)]}"] = unpack_mask_payload(rec["mask_payload"], height, width)
        frame_idx_arr.append(int(frame_idx))
        side_arr.append(int(side))
        obj_id_arr.append(int(rec["obj_id"]))
    archive_path = out_dir / "framewise_masks.npz"
    np.savez_compressed(
        archive_path,
        frame_idx=np.asarray(frame_idx_arr, dtype=np.int64),
        side=np.asarray(side_arr, dtype=np.int64),
        obj_id=np.asarray(obj_id_arr, dtype=np.int64),
        **arrays,
    )
    return archive_path


def write_record_from_obj(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, boxes_rel, mask, prob, width: int, height: int, prompt_frame: int, mask_archive: str = "off"):
    global _PNG_WRITE_SEC, _PNG_WRITE_COUNT
    box = simple.output_box_xyxy(boxes_rel, mask, width, height)
    mask_area = int(mask.sum())
    mask_path = None
    mask_payload = None
    if mask_area > 0:
        if mask_archive != "off":
            # 27-04 D-07: keep the bool mask in memory (packed bits) instead of a PNG;
            # mask_path stays None. Every other key keeps identical values and types.
            mask_payload = pack_mask_payload(mask)
        else:
            mask_path = mask_dir / SIDE_NAMES[int(side)] / f"{source}_obj{obj_id}_frame{frame_idx:06d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if stage_events_enabled():
                _t0 = time.perf_counter()
                ok = cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                _PNG_WRITE_SEC += time.perf_counter() - _t0
                _PNG_WRITE_COUNT += 1
            else:
                ok = cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            assert ok, f"failed to write {mask_path}"
    rec = {
        "frame_idx": int(frame_idx),
        "side": int(side),
        "side_name": SIDE_NAMES[int(side)],
        "obj_id": int(obj_id),
        "source": source,
        "prompt_frame_idx": int(prompt_frame),
        "prob": float(prob),
        "mask_area": int(mask_area),
        "box_xyxy": [float(x) for x in box],
        "mask_path": None if mask_path is None else str(mask_path),
    }
    if mask_payload is not None:
        rec["mask_payload"] = mask_payload
    return rec


def write_record(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, outputs, width: int, height: int, prompt_frame: int, mask_archive: str = "off"):
    boxes_rel, mask, prob = find_obj(outputs, obj_id)
    return write_record_from_obj(mask_dir, frame_idx, side, source, obj_id, boxes_rel, mask, prob, width, height, prompt_frame, mask_archive=mask_archive)


def write_record_if_present(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, outputs, width: int, height: int, prompt_frame: int, mask_archive: str = "off", normalized=None):
    # 27-04 N4: when the consumer loop already normalized this streamed frame
    # (--exec-micro-opts), reuse the shared tuple; default None = legacy per-call path.
    obj = find_obj_or_none(outputs, obj_id) if normalized is None else find_in_normalized(normalized, obj_id)
    if obj is None:
        return None
    boxes_rel, mask, prob = obj
    return write_record_from_obj(mask_dir, frame_idx, side, source, obj_id, boxes_rel, mask, prob, width, height, prompt_frame, mask_archive=mask_archive)


def merge_records(records):
    by_key = {}
    for rec in records:
        if int(rec["mask_area"]) <= 0:
            continue
        key = (int(rec["frame_idx"]), int(rec["side"]))
        quality = (float(rec["prob"]), -abs(int(rec["frame_idx"]) - int(rec["prompt_frame_idx"])), int(rec["mask_area"]))
        if key not in by_key or quality > by_key[key][0]:
            by_key[key] = (quality, rec)
    return {key: rec for key, (_, rec) in by_key.items()}


def export_framewise(frames, selected, out_dir: Path, width: int, height: int, mask_archive: str = "off"):
    global _PNG_WRITE_SEC, _PNG_WRITE_COUNT
    with stage_event_scope("sam3_point.export_framewise", frame_count=int(len(frames))):
        framewise_dir = out_dir / "framewise_masks"
        zero = np.zeros((height, width), dtype=np.uint8)
        rows = []
        if mask_archive == "off":
            for side in [0, 1]:
                (framewise_dir / SIDE_NAMES[side]).mkdir(parents=True, exist_ok=True)
        for frame_idx in range(len(frames)):
            row = {"frame_idx": int(frame_idx)}
            for side in [0, 1]:
                side_name = SIDE_NAMES[side]
                out_path = framewise_dir / side_name / f"{frame_idx:06d}.png"
                rec = selected.get((frame_idx, side))
                if rec is None:
                    if mask_archive == "off":
                        ok = cv2.imwrite(str(out_path), zero)
                        assert ok, f"failed to write {out_path}"
                    # 27-04 D-07: archive mode writes NO PNGs (not even all-zero frames);
                    # rows keep the same path strings so jsonl/csv stay content-identical.
                    row[f"{side_name}_mask_path"] = str(out_path)
                    row[f"{side_name}_mask_area"] = 0
                    row[f"{side_name}_sam_tight_box_xyxy"] = None
                    continue
                if mask_archive == "off":
                    mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
                    assert mask is not None, f"failed to read {rec['mask_path']}"
                    mask = (mask > 0)
                    ok = cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255)
                    assert ok, f"failed to write {out_path}"
                else:
                    mask = unpack_mask_payload(rec["mask_payload"], height, width)
                row[f"{side_name}_mask_path"] = str(out_path)
                row[f"{side_name}_mask_area"] = int(mask.sum())
                row[f"{side_name}_sam_tight_box_xyxy"] = [float(x) for x in simple.mask_to_box(mask)]
            rows.append(row)
        with (out_dir / "sam_tight_bboxes_2d.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with (out_dir / "sam_tight_bboxes_2d.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame_idx", "left_mask_path", "left_mask_area", "left_sam_tight_box_xyxy", "right_mask_path", "right_mask_area", "right_sam_tight_box_xyxy"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    phase19_stage_event("sam3_point.png_write_total", "aggregate", write_count=int(_PNG_WRITE_COUNT), write_sec=float(_PNG_WRITE_SEC))
    _PNG_WRITE_SEC = 0.0
    _PNG_WRITE_COUNT = 0
    return rows


def run_point_prompt(args, predictor=None, inmemory_frames=None, detection_payload=None):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "run_config.json", vars(args))
    segment_dir = Path(args.segment_dir)
    assert (segment_dir / "detection" / "bboxes_2d.pth").exists(), segment_dir / "detection" / "bboxes_2d.pth"
    if inmemory_frames is not None:
        # 25-12 in-memory SAM3: frames are a list[PIL.Image] (RGB, at the sam3 resized size);
        # start_session accepts the list directly (load_resource_as_video_frames list branch),
        # so there is no segment_dir/frames on disk and no JPEG re-encode.
        frames = inmemory_frames
        sam3_resource = inmemory_frames
        width, height = frames[0].size  # PIL size is (W, H)
    else:
        assert (segment_dir / "frames").exists(), segment_dir / "frames"
        frames = simple.sorted_frames(segment_dir)
        first = cv2.imread(str(frames[0]))
        assert first is not None, f"failed to read first frame: {frames[0]}"
        height, width = first.shape[:2]
        sam3_resource = str(segment_dir / "frames")
    if detection_payload is not None:
        # 27-04 C5: detection payload passed in memory — skip the final torch.load.
        # The file at segment_dir/detection/bboxes_2d.pth is still written upstream
        # (it is part of the uploaded sam3 output tree).
        det = detection_from_payload(detection_payload)
    else:
        det = simple.load_detection(segment_dir / "detection" / "bboxes_2d.pth")
    det_by_frame = simple.detection_index(det)
    seeds_by_side, seed_debug = select_prompt_seeds(det_by_frame, len(frames), (height, width), args)
    dump_json(out_dir / "prompt_seed_debug.json", seed_debug)

    t_total = now()
    runtime = defaultdict(float)
    t = now()
    if predictor is not None:
        # Injected warm predictor (persistent worker, D-02): reuse the checkpoint
        # loaded once at worker startup; no per-segment build / interpreter spawn.
        runtime["model_load_sec"] = 0.0
        runtime["predictor_reused"] = True
    else:
        # Standalone / subprocess entry: build exactly as before (numerics unchanged).
        predictor = simple.build_predictor(args.checkpoint_path, args.default_output_prob_thresh, args.sam_version)
        runtime["model_load_sec"] = now() - t
        runtime["predictor_reused"] = False
    mask_dir = out_dir / "masks"
    mask_archive = args.mask_archive  # 27-04 D-07: off = legacy PNGs, rle/npz = one packed archive
    exec_micro_opts = bool(args.exec_micro_opts)  # 27-04 N4+C3
    close_empty_cache = bool(args.close_empty_cache)  # 27-09 2-up pool release
    offload_video = bool(args.offload_video)  # 27-04 N2 (D-02 OOM ladder step 3 only)
    offload_state = bool(args.offload_state)  # 27-09 memory-bank offload
    chunk_trim = bool(args.chunk_trim)  # 27-09 grounding-chunk dead-weight trim
    dedup_pos_enc = bool(args.dedup_pos_enc)  # 27-09 shared image_pos_enc constant
    all_records = []
    all_prompt_rows = []
    side_summaries = {}

    if args.tracking_mode == "separate":
        for side in [0, 1]:
            side_name = SIDE_NAMES[side]
            side_seeds = seeds_by_side[side]
            if not side_seeds:
                side_summaries[side_name] = {"status": "no_valid_prompt_seed", "prompt_seed_count": 0, "tracking_mode": args.tracking_mode}
                continue
            init_seed = max(side_seeds, key=lambda s: (float(s["score"]), -int(s["frame_idx"])))
            t = now()
            session = predictor.handle_request(start_session_request(sam3_resource, offload_video, offload_state, chunk_trim, dedup_pos_enc))
            session_id = session["session_id"]
            runtime["session_init_sec"] += now() - t
            try:
                if args.init_mode == "semantic":
                    t = now()
                    prompt_box_rel = simple.xyxy_abs_to_xywh_rel(np.asarray(init_seed["box_xyxy"], dtype=np.float32), width, height)
                    semantic = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(init_seed["frame_idx"]),
                        "text": prompt_text_for_side(args, side),
                        "bounding_boxes": prompt_box_rel[None],
                        "bounding_box_labels": np.ones((1,), dtype=np.int64),
                        "output_prob_thresh": float(args.sam_output_thresh),
                    })
                    runtime["semantic_init_sec"] += now() - t
                    _iou, _area, _prob, obj_id, _pos, _box = choose_obj(semantic["outputs"], init_seed["box_xyxy"], width, height)
                    all_records.append(write_record(mask_dir, int(init_seed["frame_idx"]), side, "semantic_init", obj_id, semantic["outputs"], width, height, int(init_seed["frame_idx"]), mask_archive=mask_archive))
                else:
                    obj_id = int(side) + 1

                for seed_i, seed in enumerate(side_seeds):
                    points, labels, expanded_box = bbox_to_instance_points(seed["box_xyxy"], width, height, args.point_expand, geometry=args.prompt_geometry, neg_below_count=args.neg_below_count, neg_below_frac=args.neg_below_frac)
                    t = now()
                    resp = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(seed["frame_idx"]),
                        "points": points,
                        "point_labels": labels,
                        "obj_id": int(obj_id),
                        "rel_coordinates": True,
                        "output_prob_thresh": float(args.sam_output_thresh),
                    })
                    runtime["point_prompt_sec"] += now() - t
                    rec = write_record_if_present(mask_dir, int(seed["frame_idx"]), side, f"point_seed{seed_i:03d}", obj_id, resp["outputs"], width, height, int(seed["frame_idx"]), mask_archive=mask_archive)
                    if rec is None:
                        runtime[f"point_seed_missing_obj_frames_{side_name}"] += 1
                    else:
                        all_records.append(rec)
                    all_prompt_rows.append({**seed, "obj_id": int(obj_id), "points_rel": points.tolist(), "point_labels": labels.tolist(), "expanded_box_xyxy": expanded_box})

                min_frame = min(int(s["frame_idx"]) for s in side_seeds)
                max_frame = max(int(s["frame_idx"]) for s in side_seeds)
                if args.propagation_scheme == "single_anchor":
                    # One estimate per frame: forward B->end, backward B->0 from the init
                    # seed; window seeds stay cond frames in both directions.
                    anchor_frame = int(init_seed["frame_idx"])
                    forward_start, backward_start = anchor_frame, anchor_frame
                    forward_prompt, backward_prompt = anchor_frame, anchor_frame
                    runtime[f"propagation_anchor_frame_{side_name}"] = anchor_frame
                else:
                    forward_start, backward_start = min_frame, max_frame
                    forward_prompt, backward_prompt = min_frame, max_frame
                t = now()
                for stream_out in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": int(forward_start),
                    "max_frame_num_to_track": int(len(frames) - forward_start),
                    "output_prob_thresh": float(args.sam_output_thresh),
                }):
                    normalized = simple.normalize_outputs(stream_out["outputs"]) if exec_micro_opts else None  # N4: once per frame
                    rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "forward", obj_id, stream_out["outputs"], width, height, forward_prompt, mask_archive=mask_archive, normalized=normalized)
                    if rec is None:
                        runtime[f"forward_missing_obj_frames_{side_name}"] += 1
                        continue
                    all_records.append(rec)
                runtime["forward_propagate_sec"] += now() - t
                t = now()
                for stream_out in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "backward",
                    "start_frame_index": int(backward_start),
                    "max_frame_num_to_track": int(backward_start + 1),
                    "output_prob_thresh": float(args.sam_output_thresh),
                }):
                    normalized = simple.normalize_outputs(stream_out["outputs"]) if exec_micro_opts else None  # N4: once per frame
                    rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "backward", obj_id, stream_out["outputs"], width, height, backward_prompt, mask_archive=mask_archive, normalized=normalized)
                    if rec is None:
                        runtime[f"backward_missing_obj_frames_{side_name}"] += 1
                        continue
                    all_records.append(rec)
                runtime["backward_propagate_sec"] += now() - t
                side_summaries[side_name] = {"status": "ok", "obj_id": int(obj_id), "prompt_seed_count": len(side_seeds), "init_seed": init_seed, "tracking_mode": args.tracking_mode, "init_mode": args.init_mode}
            finally:
                with stage_event_scope("sam3_point.close_session"):
                    predictor.handle_request(close_session_request(session_id, exec_micro_opts, close_empty_cache))
    else:
        active_sides = [side for side in [0, 1] if seeds_by_side[side]]
        # v3 entry criteria (Kevin 2026-06-13). Extracted to entry_skip_from_seeds so the
        # unit-planning pre-skip (pipeline --early-entry-skip) and this in-run check share
        # ONE source of truth. With pre-skip on, a skipped unit is already 'skipped' in
        # units.csv and never reaches here, so this raise should ~never fire post-plan; it
        # stays as the correctness safety net (flag off, or a planner/runtime mismatch).
        _entry_skip = entry_skip_from_seeds(seeds_by_side, seed_debug, det_by_frame, height, args)
        if _entry_skip is not None:
            raise RecoverableUnitError(*_entry_skip)
        for side in [0, 1]:
            if not seeds_by_side[side]:
                side_summaries[SIDE_NAMES[side]] = {"status": "no_valid_prompt_seed", "prompt_seed_count": 0, "tracking_mode": args.tracking_mode}
        if active_sides:
            if ego_gate_enabled(args):
                init_by_side = select_joint_init_seeds_from_prompt_seeds(seeds_by_side, active_sides)
            else:
                init_by_side = select_joint_init_seeds(det_by_frame, height, active_sides, args)
            t = now()
            session = predictor.handle_request(start_session_request(sam3_resource, offload_video, offload_state, chunk_trim, dedup_pos_enc))
            session_id = session["session_id"]
            runtime["session_init_sec"] += now() - t
            try:
                if args.init_mode == "semantic":
                    # v5 joint+hand prepass (decision 260607-jhand): SAM3 semantic text+box
                    # accepts ONE box and resets session state, so each side runs its own
                    # throwaway-session semantic prepass; only the refined box survives and
                    # is injected as a point prompt into this fresh joint session.
                    semantic_assignment = {}
                    semantic_candidates = {}
                    obj_by_side = {}
                    for side in active_sides:
                        obj_by_side[side] = int(side) + 1
                        semantic_frame = int(init_by_side[side]["frame_idx"])
                        prompt_box_rel = simple.xyxy_abs_to_xywh_rel(np.asarray(init_by_side[side]["box_xyxy"], dtype=np.float32), width, height)[None, :]
                        prompt_text = prompt_text_for_side(args, side)
                        t = now()
                        sem_session = predictor.handle_request(start_session_request(sam3_resource, offload_video, offload_state, chunk_trim, dedup_pos_enc))
                        sem_session_id = sem_session["session_id"]
                        runtime["semantic_prepass_session_init_sec"] += now() - t
                        try:
                            t = now()
                            semantic = predictor.handle_request({
                                "type": "add_prompt",
                                "session_id": sem_session_id,
                                "frame_index": int(semantic_frame),
                                "text": prompt_text,
                                "bounding_boxes": prompt_box_rel,
                                "bounding_box_labels": np.ones((1,), dtype=np.int64),
                                "output_prob_thresh": float(args.sam_output_thresh),
                            })
                            runtime["semantic_init_sec"] += now() - t
                            prompt_iou, mask_area, prob, sem_obj_id, pos, box = choose_obj(semantic["outputs"], init_by_side[side]["box_xyxy"], width, height)
                        finally:
                            with stage_event_scope("sam3_point.close_session"):
                                predictor.handle_request(close_session_request(sem_session_id, exec_micro_opts, close_empty_cache))
                        semantic_assignment[side] = {
                            "side": int(side),
                            "obj_id": int(obj_by_side[side]),
                            "semantic_prepass_obj_id": int(sem_obj_id),
                            "pos": int(pos),
                            "prompt_iou": float(prompt_iou),
                            "mask_area": int(mask_area),
                            "prob": float(prob),
                            "box_xyxy": [float(x) for x in box],
                            "prompt_text": prompt_text,
                            "seed_source": "semantic_prepass_point_init",
                        }
                        semantic_candidates[side] = [semantic_assignment[side]]
                        semantic_seed = dict(init_by_side[side])
                        semantic_seed["box_xyxy"] = [float(x) for x in box]
                        semantic_seed["score"] = float(prob)
                        semantic_seed["semantic_prepass_obj_id"] = int(sem_obj_id)
                        semantic_seed["semantic_prompt_text"] = prompt_text
                        points, labels, expanded_box = bbox_to_instance_points(semantic_seed["box_xyxy"], width, height, args.point_expand, geometry=args.prompt_geometry, neg_below_count=args.neg_below_count, neg_below_frac=args.neg_below_frac)
                        t = now()
                        resp = predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": int(semantic_seed["frame_idx"]),
                            "points": points,
                            "point_labels": labels,
                            "obj_id": int(obj_by_side[side]),
                            "rel_coordinates": True,
                            "output_prob_thresh": float(args.sam_output_thresh),
                        })
                        runtime["point_prompt_sec"] += now() - t
                        rec = write_record_if_present(mask_dir, int(semantic_seed["frame_idx"]), side, "semantic_text_point_init_joint", obj_by_side[side], resp["outputs"], width, height, int(semantic_seed["frame_idx"]), mask_archive=mask_archive)
                        if rec is None:
                            runtime[f"semantic_text_point_missing_obj_frames_{SIDE_NAMES[side]}"] += 1
                        else:
                            all_records.append(rec)
                        all_prompt_rows.append({**semantic_seed, "obj_id": int(obj_by_side[side]), "points_rel": points.tolist(), "point_labels": labels.tolist(), "expanded_box_xyxy": expanded_box, "source": "semantic_text_point_init_joint"})
                    assert len(set(obj_by_side.values())) == len(obj_by_side), f"joint semantic init returned duplicate obj ids: {obj_by_side}"
                else:
                    semantic_assignment = {}
                    semantic_candidates = {}
                    obj_by_side = {side: int(side) + 1 for side in active_sides}

                for side in active_sides:
                    side_name = SIDE_NAMES[side]
                    for seed_i, seed in enumerate(seeds_by_side[side]):
                        points, labels, expanded_box = bbox_to_instance_points(seed["box_xyxy"], width, height, args.point_expand, geometry=args.prompt_geometry, neg_below_count=args.neg_below_count, neg_below_frac=args.neg_below_frac)
                        t = now()
                        resp = predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": int(seed["frame_idx"]),
                            "points": points,
                            "point_labels": labels,
                            "obj_id": int(obj_by_side[side]),
                            "rel_coordinates": True,
                            "output_prob_thresh": float(args.sam_output_thresh),
                        })
                        runtime["point_prompt_sec"] += now() - t
                        rec = write_record_if_present(mask_dir, int(seed["frame_idx"]), side, f"point_seed{seed_i:03d}", obj_by_side[side], resp["outputs"], width, height, int(seed["frame_idx"]), mask_archive=mask_archive)
                        if rec is None:
                            runtime[f"point_seed_missing_obj_frames_{SIDE_NAMES[side]}"] += 1
                        else:
                            all_records.append(rec)
                        all_prompt_rows.append({**seed, "obj_id": int(obj_by_side[side]), "points_rel": points.tolist(), "point_labels": labels.tolist(), "expanded_box_xyxy": expanded_box})
                    side_summaries[side_name] = {
                        "status": "ok",
                        "obj_id": int(obj_by_side[side]),
                        "prompt_seed_count": len(seeds_by_side[side]),
                        "init_seed": init_by_side[side],
                        "tracking_mode": args.tracking_mode,
                        "init_mode": args.init_mode,
                        "semantic_assignment": semantic_assignment.get(side),
                        "semantic_candidates": semantic_candidates.get(side),
                    }

                if args.text_inject_every > 0:
                    # Step2 spike: inject the text concept on every Nth frame directly into the
                    # session inference_state (no add_prompt -> no reset), so text grounding rides
                    # along with the point-tracked objects during propagation. 0=TEXT,1=VISUAL.
                    state = predictor._all_inference_states[session_id]["state"]
                    ib = state["input_batch"]
                    ib.find_text_batch[0] = prompt_text_for_side(args, active_sides[0])
                    for t_idx in range(int(state["num_frames"])):
                        ti = ib.find_inputs[t_idx].text_ids.clone()  # inference tensor -> normal
                        ti[...] = 0 if (t_idx % int(args.text_inject_every) == 0) else 1
                        ib.find_inputs[t_idx].text_ids = ti
                    runtime["text_inject_every"] = int(args.text_inject_every)

                min_frame_by_side = {side: min(int(s["frame_idx"]) for s in seeds_by_side[side]) for side in active_sides}
                max_frame_by_side = {side: max(int(s["frame_idx"]) for s in seeds_by_side[side]) for side in active_sides}
                min_frame = min(min_frame_by_side.values())
                max_frame = max(max_frame_by_side.values())
                if args.propagation_scheme == "single_anchor":
                    # One estimate per frame: forward B->end, backward B->0 from the joint
                    # init frame; window seeds stay cond frames in both directions.
                    anchor_frame = int(next(iter(init_by_side.values()))["frame_idx"])
                    forward_start, backward_start = anchor_frame, anchor_frame
                    forward_prompt_by_side = {side: anchor_frame for side in active_sides}
                    backward_prompt_by_side = {side: anchor_frame for side in active_sides}
                    runtime["propagation_anchor_frame"] = anchor_frame
                else:
                    forward_start, backward_start = min_frame, max_frame
                    forward_prompt_by_side = min_frame_by_side
                    backward_prompt_by_side = max_frame_by_side
                t = now()
                for stream_out in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": int(forward_start),
                    "max_frame_num_to_track": int(len(frames) - forward_start),
                    "output_prob_thresh": float(args.sam_output_thresh),
                }):
                    normalized = simple.normalize_outputs(stream_out["outputs"]) if exec_micro_opts else None  # N4: ONE normalize shared across sides
                    for side in active_sides:
                        side_name = SIDE_NAMES[side]
                        rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "forward", obj_by_side[side], stream_out["outputs"], width, height, forward_prompt_by_side[side], mask_archive=mask_archive, normalized=normalized)
                        if rec is None:
                            runtime[f"forward_missing_obj_frames_{side_name}"] += 1
                            continue
                        all_records.append(rec)
                runtime["forward_propagate_sec"] += now() - t
                t = now()
                for stream_out in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "backward",
                    "start_frame_index": int(backward_start),
                    "max_frame_num_to_track": int(backward_start + 1),
                    "output_prob_thresh": float(args.sam_output_thresh),
                }):
                    normalized = simple.normalize_outputs(stream_out["outputs"]) if exec_micro_opts else None  # N4: ONE normalize shared across sides
                    for side in active_sides:
                        side_name = SIDE_NAMES[side]
                        rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "backward", obj_by_side[side], stream_out["outputs"], width, height, backward_prompt_by_side[side], mask_archive=mask_archive, normalized=normalized)
                        if rec is None:
                            runtime[f"backward_missing_obj_frames_{side_name}"] += 1
                            continue
                        all_records.append(rec)
                runtime["backward_propagate_sec"] += now() - t
            finally:
                with stage_event_scope("sam3_point.close_session"):
                    predictor.handle_request(close_session_request(session_id, exec_micro_opts, close_empty_cache))

    selected = merge_records(all_records)
    rows = export_framewise(frames, selected, out_dir, width, height, mask_archive=mask_archive)
    archive_path = None
    if mask_archive != "off":
        archive_path = write_mask_archive(selected, out_dir, width, height, mask_archive)
        # T-27-13: free segment masks so a persistent worker never accumulates
        # mask payloads across segments (also keeps the records jsonl serializable).
        for rec in all_records:
            rec.pop("mask_payload", None)
    records_path = out_dir / "point_prompt_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    dump_json(out_dir / "prompt_points.json", {"prompts": all_prompt_rows})
    runtime["total_sec"] = now() - t_total
    summary = {
        "status": "ok",
        "bbox_source": "sam3_point_prompt",
        "tracking_mode": args.tracking_mode,
        "init_mode": args.init_mode,
        "propagation_scheme": args.propagation_scheme,
        "git_commit": args.git_commit,
        "segment_dir": str(segment_dir),
        "output_dir": str(out_dir),
        "total_frames": int(len(frames)),
        "image_size_hw": [int(height), int(width)],
        "side_summaries": side_summaries,
        "runtime": dict(runtime),
        "records_jsonl": str(records_path),
        "sam_tight_bboxes_jsonl": str(out_dir / "sam_tight_bboxes_2d.jsonl"),
        "sam_tight_bboxes_csv": str(out_dir / "sam_tight_bboxes_2d.csv"),
        "visible_rows": int(sum(1 for row in rows for side in ["left", "right"] if int(row.get(f"{side}_mask_area") or 0) > 0)),
    }
    if mask_archive != "off":
        # Only added in archive mode so the flag-off summary stays byte-identical.
        summary["mask_archive"] = mask_archive
        summary["mask_archive_path"] = str(archive_path)
    dump_json(out_dir / "point_prompt_summary.json", summary)
    print("POINT_PROMPT_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    args = parse_args()
    run_point_prompt(args)


if __name__ == "__main__":
    main()
