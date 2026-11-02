from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from omegaconf import OmegaConf


@dataclass
class HandDataLoaderConfig:
    ego3r_training_root: str = str(Path(__file__).resolve().parents[1])
    dataset_json: str = "/path/to/config/datasets/hand/ego_data_processed_dexycb.json"
    dataset_jsons: List[str] = field(default_factory=list)
    dataset_sampling_version: str = "dexycb_only"
    sampling_weights: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"dexycb_only": {"dexycb": 10000}}
    )
    sampling_split_weights: Dict[str, float] = field(default_factory=dict)
    dataset_len_train: int = 10000
    max_gap: int = 64
    seed: int = 123
    image_size: int = 518
    wilor_image_height: int = 256
    wilor_image_width: int = 192
    bbox_rescale_range: Tuple[float, float] = (1.5, 1.5)
    min_hand_frame_fraction: float | None = None
    num_views: int = 4
    annotation_task: str = "full"
    batch_size: int = 1
    max_img_per_gpu: int = 4
    num_workers: int = 2
    data_cache_mode: str = "metadata"
    metadata_cache_dir: str = ""
    preload_manifest_root: str = ""
    dataset_retry_attempts: int = 128
    require_hand_det_files: bool = False
    require_geometry_ready: bool = False
    required_geometry_pipeline: str = ""
    # The current metadata can contain legacy ``masks``/``detections`` paths;
    # training always resolves them to the mesh-rendered assets explicitly.
    annotation_mask_directory: str = "masks"
    annotation_detection_directory: str = "derived_from_masks"


def setup_data_imports(cfg: HandDataLoaderConfig) -> None:
    if cfg.ego3r_training_root not in sys.path:
        sys.path.insert(0, cfg.ego3r_training_root)


def resolve_dataset_manifest(path: str, training_root: str) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    # Release configs use paths relative to the repository, not the training
    # package directory. Retain the legacy training-root fallback explicitly.
    repo_path = Path(__file__).resolve().parents[3] / path
    if repo_path.is_file():
        return str(repo_path)
    return str(Path(training_root) / path)




FILTERED_FRAME_SAMPLING_VERSION = (
    "eight_eligible_f36_sqrt_frames_special2_supervision125"
)
SPECIAL_DATASET_MULTIPLIERS = {
    "arctic": 2.0,
    "hot3d": 2.0,
}
SUPERVISED_DATASET_MULTIPLIERS = {
    "arctic": 1.25,
    "dexycb": 1.25,
    "egodex": 1.25,
    "egoverse": 1.25,
    "hot3d": 1.25,
}


def _dataset_target(dataset_json_path: str) -> str:
    with open(dataset_json_path, "r") as handle:
        payload = json.load(handle)
    if not (str(payload.get("schema_version", "")).startswith("compact_virtual_camera_") or (payload.get("schema_version") == "compact_camera_mano_joints_v9_20261012" and payload.get("mano_basis_policy") == "egohandbench_right_mirror_v3_to_native_B")):
        raise ValueError("Use prepared ego_data_v2 virtual-camera manifests; legacy formats are unsupported")
    if payload.get("data_format") != "unified_preprocessed":
        raise ValueError("data_format must be unified_preprocessed")
    if int(payload.get("hand_num_points", 21)) != 21:
        raise ValueError("Expected 21 MANO joints")
    if payload.get("hand_keypoint_layout", "mano21").lower() not in ("mano21", "mano_21"):
        raise ValueError("Expected MANO21 joint ordering")
    return "infinihand.training.data.datasets.unified_hand_dataset.UnifiedHandDataset"


def _validate_geometry_status(
    dataset_json_path: str,
    *,
    require_ready: bool,
    required_pipeline: str,
) -> Dict[str, str]:
    with open(dataset_json_path, "r") as handle:
        payload = json.load(handle)
    dataset_name = str(
        payload.get("dataset_name", Path(dataset_json_path).stem)
    ).lower()
    sampling_name = str(
        payload.get("sampling_name", dataset_name)
    ).strip().lower()
    if not sampling_name:
        raise ValueError(
            f"Dataset sampling_name must be non-empty: {dataset_json_path}"
        )
    status = str(
        payload.get("lingbotmap_geometry_status", "unmarked")
    ).strip().lower()
    pipeline = str(
        payload.get("lingbotmap_geometry_pipeline", "legacy_unverified")
    ).strip()
    record = {
        "dataset": dataset_name,
        "sampling_name": sampling_name,
        "status": status,
        "pipeline": pipeline,
        "config": dataset_json_path,
    }
    if require_ready:
        if status != "ready":
            raise RuntimeError(
                "Dataset is blocked from the centered virtual-camera route: "
                f"{dataset_name} status={status!r} pipeline={pipeline!r} "
                f"config={dataset_json_path}"
            )
        if required_pipeline and pipeline != required_pipeline:
            raise RuntimeError(
                "Dataset geometry pipeline mismatch: "
                f"{dataset_name} pipeline={pipeline!r}, "
                f"required={required_pipeline!r}"
            )
    return record


def allocate_weighted_virtual_lengths(
    dataset_names: List[str],
    sampling_weights: Dict[str, Any],
    nominal_length_per_dataset: int,
) -> Dict[str, int]:
    """Convert dataset-level weights into exact positive concat lengths.

    The resulting virtual lengths sum to
    ``len(dataset_names) * nominal_length_per_dataset``.  Dataset scene counts
    are intentionally not used here: the DOSS adapter samples scenes uniformly
    inside the selected dataset.
    """
    names = [str(name).lower() for name in dataset_names]
    if not names:
        raise ValueError("At least one dataset is required")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate dataset names are not allowed: {names}")
    nominal_length = int(nominal_length_per_dataset)
    if nominal_length <= 0:
        raise ValueError(
            "nominal_length_per_dataset must be positive, got "
            f"{nominal_length_per_dataset}"
        )

    missing = [name for name in names if name not in sampling_weights]
    if missing:
        raise ValueError(
            f"Missing sampling weights for selected datasets: {missing}"
        )
    weights = {name: float(sampling_weights[name]) for name in names}
    invalid = {
        name: value
        for name, value in weights.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if invalid:
        raise ValueError(f"Sampling weights must be finite and positive: {invalid}")

    total_length = len(names) * nominal_length
    weight_sum = sum(weights.values())
    exact_lengths = {
        name: total_length * weights[name] / weight_sum for name in names
    }
    lengths = {
        name: max(1, int(math.floor(exact_lengths[name]))) for name in names
    }
    delta = total_length - sum(lengths.values())
    add_order = sorted(
        names,
        key=lambda name: (
            exact_lengths[name] - math.floor(exact_lengths[name]),
            name,
        ),
        reverse=True,
    )
    for name in add_order[:max(0, delta)]:
        lengths[name] += 1
    while sum(lengths.values()) > total_length:
        donor = max(
            (name for name in names if lengths[name] > 1),
            key=lambda name: (lengths[name] - exact_lengths[name], lengths[name]),
        )
        lengths[donor] -= 1
    if sum(lengths.values()) != total_length or min(lengths.values()) <= 0:
        raise AssertionError((lengths, total_length))
    return lengths


def compute_filtered_frame_sampling_weights(
    dataset_names: List[str],
    manifest_root: str | Path,
    minimum_scene_frames: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Compute dataset weights after removing scenes that are too short.

    The Stage-2 sampling rule is sqrt(eligible frames), followed by the
    configured special-dataset and Stage-2-supervision multipliers.  Using
    manifest frame counts here keeps dataset-level allocation consistent with
    the scene filter applied by ``DossHandDatasetBase``.
    """
    root = Path(manifest_root)
    minimum = int(minimum_scene_frames)
    if minimum <= 0:
        raise ValueError(
            f"minimum_scene_frames must be positive, got {minimum_scene_frames}"
        )

    weights: Dict[str, float] = {}
    statistics: Dict[str, Dict[str, Any]] = {}
    for raw_name in dataset_names:
        name = str(raw_name).lower()
        manifest_path = root / name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Filtered-frame sampling requires manifest: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        scenes = manifest.get("scenes") or []
        frame_counts = []
        for scene in scenes:
            value = scene.get("frame_count") if isinstance(scene, dict) else None
            if value is None:
                raise ValueError(
                    f"Manifest scene is missing frame_count: {manifest_path}"
                )
            frame_counts.append(int(value))

        eligible_counts = [count for count in frame_counts if count >= minimum]
        eligible_frames = sum(eligible_counts)
        if eligible_frames <= 0:
            raise ValueError(
                f"Dataset {name!r} has no frames in scenes with at least "
                f"{minimum} frames"
            )
        special_multiplier = float(SPECIAL_DATASET_MULTIPLIERS.get(name, 1.0))
        supervision_multiplier = float(
            SUPERVISED_DATASET_MULTIPLIERS.get(name, 1.0)
        )
        weight = (
            math.sqrt(float(eligible_frames))
            * special_multiplier
            * supervision_multiplier
        )
        weights[name] = weight
        statistics[name] = {
            "total_scenes": len(frame_counts),
            "eligible_scenes": len(eligible_counts),
            "filtered_short_scenes": len(frame_counts) - len(eligible_counts),
            "total_frames": sum(frame_counts),
            "eligible_frames": eligible_frames,
            "special_multiplier": special_multiplier,
            "supervision_multiplier": supervision_multiplier,
            "effective_weight": weight,
        }
    return weights, statistics


def build_hand_dataloader(cfg: HandDataLoaderConfig):
    setup_data_imports(cfg)
    if str(cfg.annotation_mask_directory).strip() != "masks":
        raise ValueError(
            "The hand training route is pinned to annotation directory "
            "'masks'"
        )
    if str(cfg.annotation_detection_directory).strip() != "derived_from_masks":
        raise ValueError(
            "The hand training route is pinned to annotation directory "
            "'derived_from_masks'"
        )
    from infinihand.training.data.dynamic_dataloader import DynamicTorchDataset

    if int(cfg.image_size) != 518:
        raise ValueError(
            "The current DOSS full-image route is fixed at 378x518; "
            f"image_size must be 518, got {cfg.image_size}"
        )
    wilor_hw = (int(cfg.wilor_image_height), int(cfg.wilor_image_width))
    if wilor_hw != (256, 192):
        raise ValueError(
            "The current LingBot/WiLoR crop is fixed at 256x192; "
            f"got {wilor_hw}"
        )
    dataset_jsons = list(cfg.dataset_jsons) if cfg.dataset_jsons else [cfg.dataset_json]
    resolved_jsons = [
        resolve_dataset_manifest(path, cfg.ego3r_training_root)
        for path in dataset_jsons
    ]
    from .split_policy import validate_training_paths
    validate_training_paths(resolved_jsons, cfg.sampling_split_weights)
    if len(cfg.bbox_rescale_range) != 2:
        raise ValueError(
            "bbox_rescale_range must contain [min, max], "
            f"got {cfg.bbox_rescale_range}"
        )
    bbox_rescale_min, bbox_rescale_max = map(float, cfg.bbox_rescale_range)
    if bbox_rescale_min <= 0 or bbox_rescale_min > bbox_rescale_max:
        raise ValueError(
            "bbox_rescale_range must satisfy 0 < min <= max, "
            f"got [{bbox_rescale_min}, {bbox_rescale_max}]"
        )
    common = OmegaConf.create(
        {
            "training": True,
            "annotation_task": cfg.annotation_task,
            "random_full_scene_pool": True,
            "min_hand_frame_fraction": (1.0 if int(cfg.num_views) <= 4 else 0.7) if cfg.min_hand_frame_fraction is None else float(cfg.min_hand_frame_fraction),
            "inside_random": False,
            "fix_img_num": int(cfg.num_views),
            "fix_aspect_ratio": 1.0,
            "img_nums": [int(cfg.num_views), int(cfg.num_views)],
            "bbox_rescale_range": [bbox_rescale_min, bbox_rescale_max],
            "require_hand_det_files": bool(cfg.require_hand_det_files),
            "require_hand_mask_files": False,
            "annotation_mask_directory": "masks",
            "annotation_detection_directory": "derived_from_masks",
            "dataset_retry_attempts": int(cfg.dataset_retry_attempts),
            "file_cache_max_items": 0,
            "data_cache_mode": str(cfg.data_cache_mode).strip().lower(),
            "metadata_cache_dir": str(cfg.metadata_cache_dir),
            "preload_manifest_root": str(cfg.preload_manifest_root),
            "augs": {"aspects": [1.0, 1.0]},
        }
    )
    if common.data_cache_mode not in {"none", "metadata", "preload"}:
        raise ValueError(
            "data_cache_mode must be none, metadata, or preload; got "
            f"{cfg.data_cache_mode!r}"
        )
    if common.data_cache_mode in {"metadata", "preload"}:
        manifest_root = Path(common.preload_manifest_root)
        if not manifest_root.is_dir():
            raise FileNotFoundError(
                "metadata/preload cache mode requires preload_manifest_root; "
                f"not found: {manifest_root}"
            )

    geometry_records = [
        _validate_geometry_status(
            dataset_json,
            require_ready=bool(cfg.require_geometry_ready),
            required_pipeline=str(cfg.required_geometry_pipeline),
        )
        for dataset_json in resolved_jsons
    ]
    dataset_names = [record["sampling_name"] for record in geometry_records]
    if cfg.dataset_sampling_version not in cfg.sampling_weights:
        raise ValueError(
            "Unknown dataset_sampling_version "
            f"{cfg.dataset_sampling_version!r}; available="
            f"{sorted(cfg.sampling_weights)}"
        )
    configured_weights = cfg.sampling_weights[cfg.dataset_sampling_version]
    effective_weights: Dict[str, Any] = configured_weights
    sampling_weight_source = "configured"
    filtered_frame_statistics: Dict[str, Dict[str, Any]] = {}
    from .frame_sampling import SQRT_FRAME_SAMPLING_VERSION, manifest_sampling_plan
    if cfg.dataset_sampling_version == SQRT_FRAME_SAMPLING_VERSION:
        effective_weights, virtual_lengths, filtered_frame_statistics = manifest_sampling_plan(
            resolved_jsons, int(cfg.num_views), configured_weights, int(cfg.dataset_len_train),
            cfg.sampling_split_weights,
        )
        sampling_weight_source = "completed_manifest_sqrt_frames_with_split_budgets"
    elif cfg.dataset_sampling_version == FILTERED_FRAME_SAMPLING_VERSION:
        effective_weights, filtered_frame_statistics = (
            compute_filtered_frame_sampling_weights(
                dataset_names,
                manifest_root=common.preload_manifest_root,
                minimum_scene_frames=cfg.num_views,
            )
        )
        sampling_weight_source = "filtered_manifest_frames"
    if cfg.dataset_sampling_version != SQRT_FRAME_SAMPLING_VERSION:
        if cfg.sampling_split_weights:
            raise ValueError("Split budgets require ego_data_v2_sqrt_frames sampling")
        virtual_lengths = allocate_weighted_virtual_lengths(
            dataset_names,
            effective_weights,
            cfg.dataset_len_train,
        )
    print(
        "[dataset-read-contract] "
        + json.dumps(
            {
                "mask_directory": "masks",
                "bbox_source": "projected_masks",
                "hand_valid_source": "metadata.frames[].hands[].hand_valid",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "[dataset-sampling] "
        + json.dumps(
            {
                "version": cfg.dataset_sampling_version,
                "configured_weights": {
                    name: float(configured_weights[name]) for name in dataset_names
                },
                "effective_weights": {
                    name: float(effective_weights[name]) for name in dataset_names
                },
                "weight_source": sampling_weight_source,
                "split_weights": cfg.sampling_split_weights,
                "scene_sampling": "uniform_full_pool_with_replacement",
                "minimum_scene_frames": int(cfg.num_views),
                "filtered_frame_statistics": filtered_frame_statistics,
                "virtual_lengths": virtual_lengths,
                "total_virtual_length": sum(virtual_lengths.values()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    dataset_configs = []
    for dataset_json, geometry in zip(resolved_jsons, geometry_records):
        print(
            "[dataset-geometry] "
            f"dataset={geometry['dataset']} "
            f"sampling_name={geometry['sampling_name']} "
            f"status={geometry['status']} "
            f"pipeline={geometry['pipeline']}",
            flush=True,
        )
        target = _dataset_target(dataset_json)
        print(f"[build_hand_dataloader] {Path(dataset_json).name} -> {target}", flush=True)
        dataset_configs.append(
            {
                "_target_": target,
                "data_json": dataset_json,
                "len_train": virtual_lengths[geometry["sampling_name"]],
                "view_sampler_cfg": {
                    "name": "sequence",
                    "max_sampled_views": cfg.num_views,
                    "min_distance_between_views": 1,
                    "max_distance_between_views": cfg.max_gap,
                },
            }
        )
    dataset_config = OmegaConf.create(
        {
            "_target_": "infinihand.training.data.composed_dataset.ComposedDataset",
            "dataset_configs": dataset_configs,
        }
    )
    dynamic = DynamicTorchDataset(
        dataset=dataset_config,
        common_config=common,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        max_img_per_gpu=cfg.max_img_per_gpu,
        batch_size=cfg.batch_size,
        num_images_per_batch=cfg.num_views,
        seed=cfg.seed,
        persistent_workers=cfg.num_workers > 0,
    )
    return dynamic.get_loader(epoch=0)


class EmptyValidHandBatchError(RuntimeError):
    pass


def _next_batch(data_iter: Iterable, loader: Iterable):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(loader)
        return next(data_iter), data_iter


def _count_valid_hands(batch: Dict[str, Any]) -> int:
    hand_valid = batch.get("hand_valid")
    if not torch.is_tensor(hand_valid):
        return -1
    valid = hand_valid.bool()
    boxes = batch.get("hand_detections")
    if torch.is_tensor(boxes) and boxes.ndim >= valid.ndim + 1:
        box_valid = torch.isfinite(boxes).all(dim=-1)
        if box_valid.shape == valid.shape:
            valid = valid & box_valid
    return int(valid.sum().item())


def fetch_hand_batch_with_retry(
    *,
    data_iter: Iterable,
    loader: Iterable,
    step: int,
    rank: int,
    local_rank: int,
    max_attempts: int,
    require_valid_hands: bool = True,
) -> Tuple[Dict[str, Any], Iterable, int, float, List[Dict[str, Any]]]:
    attempts = 0
    fetch_ms = 0.0
    records: List[Dict[str, Any]] = []
    while True:
        attempts += 1
        started = time.time()
        try:
            batch, data_iter = _next_batch(data_iter, loader)
            elapsed_ms = (time.time() - started) * 1000.0
            fetch_ms += elapsed_ms
            if require_valid_hands and _count_valid_hands(batch) == 0:
                raise EmptyValidHandBatchError("M=0 valid hands in sampled batch")
            return batch, data_iter, attempts, fetch_ms, records
        except Exception as error:
            elapsed_ms = (time.time() - started) * 1000.0
            if not isinstance(error, EmptyValidHandBatchError):
                fetch_ms += elapsed_ms
            records.append(
                {
                    "step": step,
                    "rank": rank,
                    "local_rank": local_rank,
                    "host": socket.gethostname(),
                    "attempt": attempts,
                    "reason": "empty_valid_hands" if isinstance(error, EmptyValidHandBatchError) else "get_batch_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fetch_ms": elapsed_ms,
                }
            )
            if attempts >= max(1, int(max_attempts)):
                raise
            print(
                f"[warn] get batch failed step={step} rank={rank} "
                f"attempt={attempts}/{max_attempts}: {type(error).__name__}: {error}",
                flush=True,
            )
            if not isinstance(error, EmptyValidHandBatchError):
                data_iter = iter(loader)


__all__ = [
    "EmptyValidHandBatchError",
    "HandDataLoaderConfig",
    "build_hand_dataloader",
    "fetch_hand_batch_with_retry",
    "setup_data_imports",
]
