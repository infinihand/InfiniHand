#!/usr/bin/env python3
"""CPU-only bbox ego-hand scoring for Phase 31.

The v1 rule deliberately uses only bbox sqrt-area and bbox bottom position.
It keeps raw same-side candidates so score-only visualization and later SAM3
prompt gating can audit ambiguous/non-ego hands before expensive tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EgoGateConfig:
    ego_gate_size_ratio_min: float = 0.045
    ego_gate_size_ratio_good: float = 0.187
    ego_gate_bottom_ratio_min: float = 0.35
    ego_gate_bottom_ratio_good: float = 0.85
    ego_gate_area_weight: float = 0.7
    ego_gate_bottom_weight: float = 0.3
    ego_gate_keep_tau: float = 0.75
    ego_gate_flag_tau: float = 0.35
    ego_gate_margin_tau: float = 1.00


DEFAULT_EGO_GATE_CONFIG = EgoGateConfig()
EGO_GATE_CONFIG_KEYS = tuple(EgoGateConfig.__dataclass_fields__.keys())
SIDE_NAMES = {0: "left", 1: "right", "left": "left", "right": "right"}


def cfg_value(cfg: EgoGateConfig | dict[str, Any], key: str) -> float:
    if isinstance(cfg, dict):
        return float(cfg[key])
    return float(getattr(cfg, key))


def config_to_dict(cfg: EgoGateConfig | dict[str, Any]) -> dict[str, float]:
    return {key: cfg_value(cfg, key) for key in EGO_GATE_CONFIG_KEYS}


def config_from_namespace(args) -> EgoGateConfig:
    return EgoGateConfig(**{key: float(getattr(args, key)) for key in EGO_GATE_CONFIG_KEYS})


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _norm_score(value: float, low: float, good: float) -> float:
    return _clip01((float(value) - float(low)) / (float(good) - float(low)))


def side_name(side) -> str:
    return SIDE_NAMES[int(side)] if not isinstance(side, str) else SIDE_NAMES[side]


def score_decision(ego_score: float, cfg: EgoGateConfig | dict[str, Any] = DEFAULT_EGO_GATE_CONFIG) -> str:
    if float(ego_score) > cfg_value(cfg, "ego_gate_keep_tau"):
        return "keep"
    if float(ego_score) >= cfg_value(cfg, "ego_gate_flag_tau"):
        return "flag_review"
    return "skip_non_ego_hand"


def score_ego_candidate(
    box_xyxy,
    image_size_hw,
    cfg: EgoGateConfig | dict[str, Any] = DEFAULT_EGO_GATE_CONFIG,
) -> dict[str, float | str | None]:
    height, width = [float(v) for v in image_size_hw]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    bbox_area_ratio = (bbox_w * bbox_h) / (width * height)
    size_ratio = math.sqrt(bbox_area_ratio)
    bottom_ratio = y2 / height
    area_score = _norm_score(
        size_ratio,
        cfg_value(cfg, "ego_gate_size_ratio_min"),
        cfg_value(cfg, "ego_gate_size_ratio_good"),
    )
    bottom_score = _norm_score(
        bottom_ratio,
        cfg_value(cfg, "ego_gate_bottom_ratio_min"),
        cfg_value(cfg, "ego_gate_bottom_ratio_good"),
    )
    ego_score = (
        cfg_value(cfg, "ego_gate_area_weight") * area_score
        + cfg_value(cfg, "ego_gate_bottom_weight") * bottom_score
    )
    return {
        "size_ratio": float(size_ratio),
        "bottom_ratio": float(bottom_ratio),
        "area_score": float(area_score),
        "bottom_score": float(bottom_score),
        "ego_score": float(ego_score),
        "decision": score_decision(ego_score, cfg),
        "ego_score_top1": None,
        "ego_score_top2": None,
        "ego_margin": None,
        "selection_status": "unselected",
    }


def annotate_frame_margins(scored_rows, cfg: EgoGateConfig | dict[str, Any] = DEFAULT_EGO_GATE_CONFIG) -> list[dict[str, Any]]:
    rows = [dict(row) for row in scored_rows]
    grouped = {}
    for idx, row in enumerate(rows):
        key = (int(row["frame_idx"]), int(row["side"]))
        grouped.setdefault(key, []).append((idx, row))
    for (_frame_idx, _side), items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (
                float(item[1]["ego_score"]),
                float(item[1].get("score", 0.0)),
                -int(item[1].get("track_id", item[0])),
            ),
            reverse=True,
        )
        top1 = float(ordered[0][1]["ego_score"])
        top2 = float(ordered[1][1]["ego_score"]) if len(ordered) > 1 else 0.0
        margin = top1 - top2 if len(ordered) > 1 else 1.0
        for rank, (idx, row) in enumerate(ordered, start=1):
            rows[idx]["ego_score_top1"] = top1
            rows[idx]["ego_score_top2"] = top2
            rows[idx]["ego_margin"] = float(margin)
            rows[idx]["frame_side_rank"] = int(rank)
            if rank == 1 and float(row["ego_score"]) > cfg_value(cfg, "ego_gate_keep_tau") and margin >= cfg_value(cfg, "ego_gate_margin_tau"):
                rows[idx]["selection_status"] = "eligible"
            elif rank == 1 and float(row["ego_score"]) <= cfg_value(cfg, "ego_gate_keep_tau"):
                rows[idx]["selection_status"] = rows[idx]["decision"]
            elif rank == 1:
                rows[idx]["selection_status"] = "skip_ambiguous_ego_candidate"
            else:
                rows[idx]["selection_status"] = "same_side_lower_ego_score"
    return rows


def score_frame_candidates(
    candidates,
    image_size_hw,
    cfg: EgoGateConfig | dict[str, Any] = DEFAULT_EGO_GATE_CONFIG,
) -> list[dict[str, Any]]:
    rows = []
    for cand in candidates:
        row = dict(cand)
        row["frame_idx"] = int(row["frame_idx"])
        row["side"] = int(row["side"])
        row["side_name"] = side_name(row["side"])
        row["bbox_xyxy"] = [float(v) for v in row["bbox_xyxy"][:4]]
        row["score"] = float(row["score"])
        row.update(score_ego_candidate(row["bbox_xyxy"], image_size_hw, cfg))
        rows.append(row)
    return annotate_frame_margins(rows, cfg)


def select_window_ego_seed(
    scored_rows,
    side,
    window_start: int,
    window_end: int,
    cfg: EgoGateConfig | dict[str, Any] = DEFAULT_EGO_GATE_CONFIG,
) -> dict[str, Any]:
    side = int(side)
    rows = annotate_frame_margins(scored_rows, cfg)
    window_rows = [
        row for row in rows
        if int(row["side"]) == side and int(window_start) <= int(row["frame_idx"]) <= int(window_end)
    ]
    if not window_rows:
        return {
            "side": side,
            "side_name": side_name(side),
            "window_start": int(window_start),
            "window_end": int(window_end),
            "selection_status": "no_candidate",
            "selected": False,
            "skip_reason": "no_candidate",
            "ego_score_top1": 0.0,
            "ego_score_top2": 0.0,
            "ego_margin": 0.0,
        }

    top_rows = [row for row in window_rows if int(row["frame_side_rank"]) == 1]
    eligible = [
        row for row in top_rows
        if float(row["ego_score"]) > cfg_value(cfg, "ego_gate_keep_tau")
        and float(row["ego_margin"]) >= cfg_value(cfg, "ego_gate_margin_tau")
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["score"]),
                float(row["ego_score"]),
                -int(row["frame_idx"]),
            ),
        )
        out = dict(selected)
        out["window_start"] = int(window_start)
        out["window_end"] = int(window_end)
        out["selection_status"] = "selected"
        out["selected"] = True
        return out

    best = max(top_rows, key=lambda row: (float(row["ego_score"]), float(row["score"])))
    status = "skip_ambiguous_ego_candidate"
    if float(best["ego_score"]) < cfg_value(cfg, "ego_gate_flag_tau"):
        status = "skip_non_ego_hand"
    out = dict(best)
    out["window_start"] = int(window_start)
    out["window_end"] = int(window_end)
    out["selection_status"] = status
    out["selected"] = False
    out["skip_reason"] = status
    return out
