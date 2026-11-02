#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint_utils import (
    atomic_copy_file,
    atomic_torch_save,
    collect_trainable_module_state,
    compute_cosine_learning_rate,
    extract_model_state,
    infer_checkpoint_step,
    infer_finetune_step_offset,
    load_checkpoint_payload,
    restore_optimizer_state,
    validate_resume_schedule,
)
from infinihand.training.data.hand_dataloader import (
    HandDataLoaderConfig,
    build_hand_dataloader,
    fetch_hand_batch_with_retry,
)
from ..models.lingbotmap_hand_v6 import LingBotMapHand, LingBotMapHandConfig
from .losses import LingBotMapHandLoss, axis_angle_to_matrix
from .rank_step_trace import RankStepTracer, extract_batch_identity
from .sample_filter import RollingLossThreshold, slice_batch_tree, filtered_sample_backward
from .gradient_sync import sync_finite_sample_gradients


@dataclass
class TrainConfig:
    output_dir: str = "/path/to/outputs/lingbotmap_hand_arctic"
    ego3r_training_root: str = "/path/to/infinihand/infinihand/training"
    wilor_root: str = "/path/to/infinihand/infinihand/wilor"
    dataset_json: str = "/path/to/config/datasets/hand/ego_data_processed_dexycb.json"
    dataset_jsons: list[str] = field(default_factory=list)
    dataset_sampling_version: str = "dexycb_only"
    sampling_weights: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"dexycb_only": {"dexycb": 10000}}
    )
    sampling_split_weights: Dict[str, float] = field(default_factory=dict)
    dataset_len_train: int = 10000
    max_gap: int = 64
    dataset_retry_attempts: int = 128
    seed: int = 123
    batch_size: int = 1
    global_batch_size: int = 0
    num_views: int = 2
    image_size: int = 518
    num_workers: int = 2
    data_cache_mode: str = "none"
    metadata_cache_dir: str = ""
    preload_manifest_root: str = ""
    # Masks are projected MANO meshes; their file paths come from metadata.
    mask_source: str = "mesh"
    require_hand_det_files: bool = False
    require_geometry_ready: bool = False
    required_geometry_pipeline: str = ""
    get_batch_retry_attempts: int = 128
    train_steps: int = 5
    # Optional execution cap for smoke tests.  The LR scheduler still uses
    # train_steps/warmup_steps, so a short smoke exercises the exact prefix of
    # the eventual full schedule instead of compressing it.
    max_run_steps: int = 0
    warmup_steps: int = 10000
    loss_outlier_window: int = 1000
    loss_outlier_multiplier: float = 1.0
    loss_outlier_bootstrap_steps: int = 1000
    loss_outlier_bootstrap_threshold: float = 10.0
    loss_outlier_per_dataset: bool = False
    mano_random_valid_hand: bool = True
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 1000
    save_final_checkpoint: bool = True
    detailed_timing: bool = False
    rank_trace_every_step: bool = False
    rank_trace_flush_interval: int = 1
    finetune_checkpoint: str = ""
    require_finetune_checkpoint: bool = False
    finetune_step_offset: int = 0
    resume_checkpoint: str = ""
    resume_checkpoint_step: int = -1
    model_variant: str = "mutual_joint_fusion"
    lingbotmap_root: str = "/path/to/infinihand/infinihand/lingbot_map"
    lingbotmap_ckpt: str = ""
    wilor_cfg: str = "/path/to/infinihand/infinihand/wilor/pretrained_models/model_config.yaml"
    wilor_ckpt: str = "/path/to/weights/wilor_final.ckpt"
    load_wilor_checkpoint: bool = False
    image_height: int = 378
    image_width: int = 518
    wilor_crop_size: int = 256
    bbox_rescale: float = 1.5
    bbox_scale_factor: float = 0.0
    bbox_rescale_range: list[float] | None = field(default_factory=lambda: [1.5, 1.5])
    freeze_backbone: bool = True
    lingbot_tune_mode: str = "frozen"
    lingbot_tune_last_n_blocks: int = 4
    lingbot_tune_special_tokens: bool = False
    lingbot_gradient_checkpointing: bool = False
    lingbot_gradient_checkpoint_last_n_blocks: int = -1
    lingbot_lr: float = 2.0e-06
    lingbot_activation_offload: bool = False
    lingbot_enable_camera_head: bool = False
    camera_scale_head_enabled: bool = False
    camera_scale_head_hidden_dim: int = 512
    camera_scale_head_num_layers: int = 2
    camera_scale_head_num_heads: int = 8
    camera_scale_head_ffn_dim: int = 2048
    camera_scale_head_frames: int = 4
    camera_scale_log_min: float = -4.0
    camera_scale_log_max: float = 4.0
    hand_feature_provider: str = "wilor"
    hand_feature_chunk_size: int = 8
    freeze_hand_feature_provider: bool = False
    dinov2_input_height: int = 224
    dinov2_input_width: int = 168
    sapiens2_root: str = ""
    sapiens2_arch: str = "sapiens2_1b"
    sapiens2_ckpt: str = ""
    freeze_wilor_backbone: bool = False
    wilor_gradient_checkpointing: bool = False
    joint_heatmap_range_margin: float = 0.15
    mutual_fusion_num_heads: int = 8
    mutual_fusion_dropout: float = 0.0
    pnp_min_visible: int = 1
    hand_detection_dim: int = 256
    hand_detection_frames_chunk_size: int = 8
    hand_bbox_head_enabled: bool = False
    mask_only_threshold: float = 0.9
    use_soft_mask_boxes_for_mano: bool = False
    mano_gradient_to_mask_head: bool = False
    soft_mask_box_temperature: float = 1.0
    soft_mask_box_std_scale: float = 3.4641016151377544
    soft_mask_box_min_size: float = 0.02
    verify_mask_mano_gradient: bool = False
    train_detection_modules_only: bool = False
    train_probability_mask_heads_only: bool = False
    train_mano_modules_only: bool = False
    train_camera_head_only: bool = False
    train_joint_heads_and_camera: bool = False
    train_camera_scale_head_only: bool = False
    step_memory_enabled: bool = False
    step_memory_attention_mode: str = "gca"
    step_memory_scale_frames: int = 4
    step_memory_max_frames: int = 64
    step_memory_sliding_window: int = 64
    hand_probability_loss_weight: float = 0.0
    hand_bbox_smooth_l1_loss_weight: float = 5.0
    hand_bbox_giou_loss_weight: float = 2.0
    hand_mask_bce_loss_weight: float = 1.0
    hand_mask_dice_loss_weight: float = 1.0
    # ViDiHand Stage-2 names and weights in the WiLoR crop-camera frame.
    joints2d_direct_weight: float = 2.0
    reproj_loss_weight: float = 1.0
    go_loss_weight: float = 0.1
    hp_loss_weight: float = 0.1
    betas_loss_weight: float = 0.1
    trans_loss_weight: float = 3.0
    joints_loss_weight: float = 2.0
    velocity_smooth_weight: float = 0.0
    betas_consistency_weight: float = 0.0
    # Long-sequence Stage-2 calibration objective.
    stage1_loss_weight: float = 1.0
    world_joints_loss_weight: float = 0.0
    world_mano_loss_weight: float = 0.0
    abs_pose_loss_weight: float = 0.0
    rel_pose_loss_weight: float = 0.0
    pose_translation_loss_weight: float = 1.0
    abs_pose_translation_component_weight: float = 3.0 / 9.0
    abs_pose_quaternion_component_weight: float = 4.0 / 9.0
    abs_pose_fov_component_weight: float = 2.0 / 9.0
    rel_pose_rotation_component_weight: float = 1.0
    rel_pose_translation_component_weight: float = 1.0
    world_mano_global_orient_weight: float = 1.0
    world_mano_hand_pose_weight: float = 1.0
    world_mano_betas_weight: float = 1.0
    world_mano_translation_weight: float = 1.0
    stage2_huber_beta: float = 0.01
    compute_stage2_diagnostics: bool = False
    relative_pose_window: int = 16
    stage2_coordinate_protocol: str = (
        "opencv_c2w_dataset_world_metric_no_scale_alignment"
    )
    camera_joint_target_space: str = "image_camera"


DETECTION_ONLY_ROOTS = (
    "hand_detection_block",
    "hand_mask_head",
)
PROBABILITY_MASK_ONLY_ROOTS = (
    "hand_mask_head",
)
MANO_ONLY_ROOTS = (
    "lingbot_adapter",
    "hand_encoder",
    "feature_fusion",
    "mano_head",
    "joint_head",
    "mutual_fusion",
    "joint_offset_head",
)
CAMERA_HEAD_ONLY_PREFIX = "backbone.model.camera_head."
CAMERA_SCALE_HEAD_ONLY_PREFIX = "camera_scale_head."
POSE_PROTOCOL_DATASET_WORLD = (
    "opencv_c2w_dataset_world_metric_no_scale_alignment"
)
POSE_PROTOCOL_FIRST_CAMERA_SCALE = (
    "opencv_c2w_first_camera_metric_scale_v1"
)
POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE = (
    "opencv_c2w_first_camera_metric_no_scale_alignment"
)


def validate_trainable_scope_config(cfg: TrainConfig) -> None:
    if str(cfg.mask_source).strip().lower() != "mesh":
        raise ValueError(
            "this probability/mask-head experiment requires mask_source='mesh'"
        )
    if int(cfg.max_run_steps) < 0:
        raise ValueError("max_run_steps must be >= 0")
    enabled_scopes = {
        "detection": bool(cfg.train_detection_modules_only),
        "probability_mask_heads": bool(
            cfg.train_probability_mask_heads_only
        ),
        "mano": bool(cfg.train_mano_modules_only),
        "camera_head": bool(cfg.train_camera_head_only),
        "camera_scale_head": bool(cfg.train_camera_scale_head_only),
    }
    if sum(enabled_scopes.values()) > 1:
        raise ValueError(
            "train_detection_modules_only, "
            "train_probability_mask_heads_only, train_mano_modules_only, "
            "train_camera_head_only, and train_camera_scale_head_only are "
            f"mutually exclusive: {enabled_scopes}"
        )
    if cfg.train_camera_head_only or cfg.train_camera_scale_head_only:
        scope_name = (
            "train_camera_scale_head_only"
            if cfg.train_camera_scale_head_only
            else "train_camera_head_only"
        )
        allowed_pose_protocols = (
            {POSE_PROTOCOL_FIRST_CAMERA_SCALE}
            if cfg.train_camera_scale_head_only
            else {
                POSE_PROTOCOL_DATASET_WORLD,
                POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE,
            }
        )
        if cfg.stage2_coordinate_protocol not in allowed_pose_protocols:
            raise ValueError(
                f"{scope_name} requires stage2_coordinate_protocol in "
                f"{sorted(allowed_pose_protocols)!r}, got "
                f"{cfg.stage2_coordinate_protocol!r}"
            )
        if not cfg.lingbot_enable_camera_head:
            raise ValueError(
                f"{scope_name} requires lingbot_enable_camera_head=true"
            )
        if not cfg.freeze_backbone or str(cfg.lingbot_tune_mode).strip().lower() != "frozen":
            raise ValueError(
                f"{scope_name} requires the LingBot aggregator to be "
                "frozen: set freeze_backbone=true and lingbot_tune_mode='frozen'"
            )
        if not cfg.freeze_hand_feature_provider or not cfg.freeze_wilor_backbone:
            raise ValueError(
                f"{scope_name} requires the hand feature provider to "
                "be frozen"
            )
        if cfg.hand_bbox_head_enabled:
            raise ValueError(
                f"{scope_name} requires hand_bbox_head_enabled=false"
            )
        if cfg.use_soft_mask_boxes_for_mano or cfg.mano_gradient_to_mask_head:
            raise ValueError(
                f"{scope_name} requires GT boxes with no gradient to "
                "the mask branch"
            )
        non_pose_weights = {
            "hand_probability_loss_weight": cfg.hand_probability_loss_weight,
            "hand_bbox_smooth_l1_loss_weight": cfg.hand_bbox_smooth_l1_loss_weight,
            "hand_bbox_giou_loss_weight": cfg.hand_bbox_giou_loss_weight,
            "hand_mask_bce_loss_weight": cfg.hand_mask_bce_loss_weight,
            "hand_mask_dice_loss_weight": cfg.hand_mask_dice_loss_weight,
            "joints2d_direct_weight": cfg.joints2d_direct_weight,
            "reproj_loss_weight": cfg.reproj_loss_weight,
            "go_loss_weight": cfg.go_loss_weight,
            "hp_loss_weight": cfg.hp_loss_weight,
            "betas_loss_weight": cfg.betas_loss_weight,
            "trans_loss_weight": cfg.trans_loss_weight,
            "joints_loss_weight": cfg.joints_loss_weight,
            "velocity_smooth_weight": cfg.velocity_smooth_weight,
            "betas_consistency_weight": cfg.betas_consistency_weight,
            "stage1_loss_weight": cfg.stage1_loss_weight,
            "world_joints_loss_weight": cfg.world_joints_loss_weight,
            "world_mano_loss_weight": cfg.world_mano_loss_weight,
        }
        nonzero_non_pose = {
            name: value
            for name, value in non_pose_weights.items()
            if float(value) != 0.0
        }
        if nonzero_non_pose:
            raise ValueError(
                f"{scope_name} requires every non-pose objective "
                f"weight to be zero; nonzero={nonzero_non_pose}"
            )
        if float(cfg.abs_pose_loss_weight) <= 0.0 or float(cfg.rel_pose_loss_weight) <= 0.0:
            raise ValueError(
                f"{scope_name} requires positive absolute and relative "
                "pose loss weights"
            )
        if float(cfg.pose_translation_loss_weight) <= 0.0:
            raise ValueError(
                f"{scope_name} requires pose_translation_loss_weight>0"
            )
        if cfg.train_camera_head_only and cfg.camera_scale_head_enabled:
            raise ValueError(
                "train_camera_head_only requires camera_scale_head_enabled=false"
            )
        if cfg.train_camera_scale_head_only:
            if not cfg.camera_scale_head_enabled:
                raise ValueError(
                    "train_camera_scale_head_only requires "
                    "camera_scale_head_enabled=true"
                )
            if int(cfg.camera_scale_head_frames) != int(
                cfg.step_memory_scale_frames
            ):
                raise ValueError(
                    "camera_scale_head_frames must equal "
                    "step_memory_scale_frames"
                )
            if int(cfg.camera_scale_head_frames) > int(cfg.num_views):
                raise ValueError(
                    "camera_scale_head_frames cannot exceed num_views"
                )
        if cfg.train_camera_head_only and float(cfg.lr) != float(cfg.lingbot_lr):
            raise ValueError(
                "train_camera_head_only requires lingbot_lr == lr so the camera "
                "head uses the requested schedule"
            )
        return
    if cfg.train_probability_mask_heads_only:
        if cfg.hand_bbox_head_enabled:
            raise ValueError(
                "train_probability_mask_heads_only requires "
                "hand_bbox_head_enabled=false"
            )
        if cfg.use_soft_mask_boxes_for_mano or cfg.mano_gradient_to_mask_head:
            raise ValueError(
                "train_probability_mask_heads_only does not train MANO and "
                "requires use_soft_mask_boxes_for_mano=false and "
                "mano_gradient_to_mask_head=false"
            )
        if float(cfg.hand_probability_loss_weight) != 0.0:
            raise ValueError(
                "train_probability_mask_heads_only requires zero "
                "hand_probability_loss_weight (presence is mask-derived)"
            )
        if not any(
            float(weight) > 0.0
            for weight in (
                cfg.hand_mask_bce_loss_weight,
                cfg.hand_mask_dice_loss_weight,
            )
        ):
            raise ValueError(
                "train_probability_mask_heads_only requires a positive mask "
                "BCE or Dice loss weight"
            )
        disabled_weights = {
            "hand_bbox_smooth_l1_loss_weight": (
                cfg.hand_bbox_smooth_l1_loss_weight
            ),
            "hand_bbox_giou_loss_weight": cfg.hand_bbox_giou_loss_weight,
            "joints2d_direct_weight": cfg.joints2d_direct_weight,
            "reproj_loss_weight": cfg.reproj_loss_weight,
            "go_loss_weight": cfg.go_loss_weight,
            "hp_loss_weight": cfg.hp_loss_weight,
            "betas_loss_weight": cfg.betas_loss_weight,
            "trans_loss_weight": cfg.trans_loss_weight,
            "joints_loss_weight": cfg.joints_loss_weight,
            "velocity_smooth_weight": cfg.velocity_smooth_weight,
            "betas_consistency_weight": cfg.betas_consistency_weight,
            "world_joints_loss_weight": cfg.world_joints_loss_weight,
            "world_mano_loss_weight": cfg.world_mano_loss_weight,
            "abs_pose_loss_weight": cfg.abs_pose_loss_weight,
            "rel_pose_loss_weight": cfg.rel_pose_loss_weight,
            "pose_translation_loss_weight": (
                cfg.pose_translation_loss_weight
            ),
        }
        nonzero_disabled = {
            name: value
            for name, value in disabled_weights.items()
            if float(value) != 0.0
        }
        if nonzero_disabled:
            raise ValueError(
                "train_probability_mask_heads_only requires all bbox, MANO, "
                "world-hand, and camera-pose loss weights to be zero; "
                f"nonzero={nonzero_disabled}"
            )
        if float(cfg.stage1_loss_weight) <= 0.0:
            raise ValueError(
                "train_probability_mask_heads_only requires a positive "
                "stage1_loss_weight"
            )
        return

    if not cfg.train_mano_modules_only:
        return

    if str(cfg.model_variant).strip().lower() != "mutual_joint_fusion":
        raise ValueError(
            "train_mano_modules_only requires "
            "model_variant='mutual_joint_fusion'"
        )
    if str(cfg.hand_feature_provider).strip().lower() != "wilor":
        raise ValueError(
            "train_mano_modules_only requires hand_feature_provider='wilor'"
        )
    if cfg.freeze_hand_feature_provider or cfg.freeze_wilor_backbone:
        raise ValueError(
            "train_mano_modules_only requires the WiLoR hand encoder to be "
            "trainable: set freeze_hand_feature_provider=false and "
            "freeze_wilor_backbone=false"
        )
    if not cfg.freeze_backbone or str(cfg.lingbot_tune_mode).strip().lower() != "frozen":
        raise ValueError(
            "train_mano_modules_only requires the LingBot backbone to be "
            "frozen: set freeze_backbone=true and lingbot_tune_mode='frozen'"
        )
    if cfg.hand_bbox_head_enabled:
        raise ValueError(
            "train_mano_modules_only uses GT boxes and requires "
            "hand_bbox_head_enabled=false"
        )
    if cfg.use_soft_mask_boxes_for_mano or cfg.mano_gradient_to_mask_head:
        raise ValueError(
            "train_mano_modules_only uses GT boxes and requires both "
            "use_soft_mask_boxes_for_mano=false and "
            "mano_gradient_to_mask_head=false"
        )
    detection_weights = {
        "hand_probability_loss_weight": cfg.hand_probability_loss_weight,
        "hand_bbox_smooth_l1_loss_weight": cfg.hand_bbox_smooth_l1_loss_weight,
        "hand_bbox_giou_loss_weight": cfg.hand_bbox_giou_loss_weight,
        "hand_mask_bce_loss_weight": cfg.hand_mask_bce_loss_weight,
        "hand_mask_dice_loss_weight": cfg.hand_mask_dice_loss_weight,
    }
    nonzero_detection = {
        name: value
        for name, value in detection_weights.items()
        if float(value) != 0.0
    }
    if nonzero_detection:
        raise ValueError(
            "train_mano_modules_only requires every detection loss weight to "
            f"be zero; nonzero={nonzero_detection}"
        )
    camera_hand_weights = (
        cfg.joints2d_direct_weight,
        cfg.reproj_loss_weight,
        cfg.go_loss_weight,
        cfg.hp_loss_weight,
        cfg.betas_loss_weight,
        cfg.trans_loss_weight,
        cfg.joints_loss_weight,
    )
    if float(cfg.stage1_loss_weight) <= 0.0 or not any(
        float(weight) > 0.0 for weight in camera_hand_weights
    ):
        raise ValueError(
            "train_mano_modules_only requires a positive stage1_loss_weight "
            "and at least one positive camera-hand loss weight"
        )
    if str(cfg.camera_joint_target_space).strip().lower() not in {"image_camera", "wilor_crop_camera"}:
        raise ValueError(
            "train_mano_modules_only requires "
            "camera_joint_target_space='image_camera'"
        )


def configure_camera_head_only_scope(
    model: LingBotMapHand,
    *,
    rank: int,
) -> list[str]:
    """Freeze the full model and unlock exactly the native camera head."""
    model.requires_grad_(False)
    backbone_model = getattr(getattr(model, "backbone", None), "model", None)
    camera_head = getattr(backbone_model, "camera_head", None)
    if not isinstance(camera_head, torch.nn.Module):
        raise RuntimeError(
            "train_camera_head_only requires backbone.model.camera_head"
        )
    camera_head.requires_grad_(True)
    for module in model.children():
        module.eval()
    camera_head.train(True)

    expected_names = {
        CAMERA_HEAD_ONLY_PREFIX + name
        for name, _ in camera_head.named_parameters()
    }
    actual_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not expected_names or actual_names != expected_names:
        raise RuntimeError(
            "camera-head-only trainable parameters mismatch: "
            f"actual={sorted(actual_names)} expected={sorted(expected_names)}"
        )
    trainable_names = sorted(actual_names)
    if rank == 0:
        trainable_params = sum(
            parameter.numel()
            for parameter in camera_head.parameters()
            if parameter.requires_grad
        )
        print(
            "[trainable-scope] camera_head_only=true "
            f"prefix={CAMERA_HEAD_ONLY_PREFIX} "
            f"tensors={len(trainable_names)} params={trainable_params}",
            flush=True,
        )
    return trainable_names


def configure_camera_scale_head_only_scope(
    model: LingBotMapHand,
    *,
    rank: int,
) -> list[str]:
    """Freeze the full model and unlock exactly the metric scale head."""
    model.requires_grad_(False)
    scale_head = getattr(model, "camera_scale_head", None)
    if not isinstance(scale_head, torch.nn.Module):
        raise RuntimeError(
            "train_camera_scale_head_only requires model.camera_scale_head"
        )
    scale_head.requires_grad_(True)
    for module in model.children():
        module.eval()
    scale_head.train(True)
    expected_names = {
        CAMERA_SCALE_HEAD_ONLY_PREFIX + name
        for name, _ in scale_head.named_parameters()
    }
    actual_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not expected_names or actual_names != expected_names:
        raise RuntimeError(
            "camera-scale-head-only trainable parameters mismatch: "
            f"actual={sorted(actual_names)} expected={sorted(expected_names)}"
        )
    trainable_names = sorted(actual_names)
    if rank == 0:
        trainable_params = sum(
            parameter.numel()
            for parameter in scale_head.parameters()
            if parameter.requires_grad
        )
        print(
            "[trainable-scope] camera_scale_head_only=true "
            f"prefix={CAMERA_SCALE_HEAD_ONLY_PREFIX} "
            f"tensors={len(trainable_names)} params={trainable_params}",
            flush=True,
        )
    return trainable_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> TrainConfig:
    with open(path, "r") as handle:
        payload = yaml.safe_load(handle) or {}
    legacy_detection_weights = {
        "hand_id_loss_weight": "hand_probability_loss_weight",
        "hand_valid_loss_weight": "hand_probability_loss_weight",
        "hand_bbox_l1_loss_weight": "hand_bbox_smooth_l1_loss_weight",
    }
    for old_key, new_key in legacy_detection_weights.items():
        if old_key in payload and new_key not in payload:
            payload[new_key] = payload.pop(old_key)
    legacy_loss_weight_keys = {
        "keypoint_2d_direct_loss_weight": "joints2d_direct_weight",
        "keypoint_2d_reproj_loss_weight": "reproj_loss_weight",
        "keypoint_2d_l1_weight": "reproj_loss_weight",
        "mano_translation_loss_weight": "trans_loss_weight",
        "camera_translation_loss_weight": "trans_loss_weight",
        "mano_rot_loss_weight": "go_loss_weight",
        "mano_pose_loss_weight": "hp_loss_weight",
        "mano_shape_loss_weight": "betas_loss_weight",
        "keypoint_3d_l1_weight": "joints_loss_weight",
        "temporal_translation_acceleration_loss_weight": "velocity_smooth_weight",
        "temporal_shape_consistency_loss_weight": "betas_consistency_weight",
    }
    migrated = []
    for old_key, new_key in legacy_loss_weight_keys.items():
        if old_key not in payload:
            continue
        if new_key in payload:
            raise ValueError(
                f"config defines both legacy {old_key!r} and ViDiHand {new_key!r}"
            )
        payload[new_key] = payload.pop(old_key)
        migrated.append(f"{old_key}->{new_key}")
    if migrated:
        print("[config-migration] " + ", ".join(migrated), flush=True)
    for removed_key in (
        "hand_keypoint_3d_l1_weight",
        "normalized_3d_l1_weight",
        "vertices_3d_l1_weight",
    ):
        payload.pop(removed_key, None)
    cfg = TrainConfig(**payload)
    validate_trainable_scope_config(cfg)
    return cfg


def filter_disabled_bbox_head_state(
    checkpoint_state: Dict[str, torch.Tensor],
    *,
    bbox_head_enabled: bool,
) -> tuple[Dict[str, torch.Tensor], int]:
    """Drop legacy bbox-head tensors when the configured branch is absent."""
    if bbox_head_enabled:
        return checkpoint_state, 0
    prefix = "hand_bbox_head."
    filtered = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith(prefix)
    }
    return filtered, len(checkpoint_state) - len(filtered)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def batch_dataset_names(batch: Dict[str, Any], batch_size: int) -> list[str]:
    """Normalize the collated per-sample dataset labels used by DOSS."""
    # Prefer split-level sampling_name (e.g. arctic_eval) so train and eval
    # roots do not share one recent-loss window. Fall back to the historical
    # base dataset_name for compatibility with older/custom datasets.
    raw = batch.get("dataset_sampling_name", batch.get("dataset_name"))
    if isinstance(raw, str):
        return [raw.lower()] * int(batch_size)
    if isinstance(raw, (list, tuple)):
        values = [str(item).lower() for item in raw]
        if len(values) == int(batch_size):
            return values
        if len(values) == 1:
            return values * int(batch_size)
    raise ValueError(
        "per-dataset outlier filtering requires collated split dataset labels "
        f"with {batch_size} entries, got {type(raw).__name__}: {raw!r}"
    )


def build_loader(cfg: TrainConfig):
    from .data.training_profiles import training_task
    loader_cfg = HandDataLoaderConfig(
        ego3r_training_root=cfg.ego3r_training_root,
        dataset_json=cfg.dataset_json,
        dataset_jsons=cfg.dataset_jsons,
        dataset_sampling_version=cfg.dataset_sampling_version,
        sampling_weights=cfg.sampling_weights,
        sampling_split_weights=cfg.sampling_split_weights,
        dataset_len_train=cfg.dataset_len_train,
        max_gap=cfg.max_gap,
        seed=cfg.seed,
        image_size=cfg.image_size,
        wilor_image_height=cfg.wilor_crop_size,
        wilor_image_width=cfg.wilor_crop_size - 64,
        bbox_rescale_range=tuple(cfg.bbox_rescale_range or (cfg.bbox_rescale, cfg.bbox_rescale)),
        min_hand_frame_fraction=(
            0.0 if (cfg.train_detection_modules_only or cfg.train_probability_mask_heads_only
                    or cfg.train_camera_head_only or cfg.train_camera_scale_head_only)
            else (1.0 if cfg.num_views <= 4 else 0.7)
        ),
        num_views=cfg.num_views,
        annotation_task=training_task(cfg),
        batch_size=cfg.batch_size,
        max_img_per_gpu=cfg.num_views,
        num_workers=cfg.num_workers,
        data_cache_mode=cfg.data_cache_mode,
        metadata_cache_dir=cfg.metadata_cache_dir,
        preload_manifest_root=cfg.preload_manifest_root,
        require_hand_det_files=cfg.require_hand_det_files,
        require_geometry_ready=cfg.require_geometry_ready,
        required_geometry_pipeline=cfg.required_geometry_pipeline,
        dataset_retry_attempts=cfg.dataset_retry_attempts,
    )
    return build_hand_dataloader(loader_cfg)


def build_training_target(batch, cfg):
    from .data.training_profiles import build_target
    return build_target(batch, cfg)



def set_learning_rate(optimizer: torch.optim.Optimizer, step: int, cfg: TrainConfig) -> float:
    lr = compute_cosine_learning_rate(
        step=step,
        train_steps=cfg.train_steps,
        warmup_steps=cfg.warmup_steps,
        peak_lr=cfg.lr,
        min_lr=cfg.min_lr,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))
    return lr


def save_checkpoint(
    path: Path,
    model: LingBotMapHand,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    history,
    extra: Dict[str, Any] | None = None,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    if cfg.train_camera_head_only or cfg.train_camera_scale_head_only:
        scoped_prefix = (
            CAMERA_SCALE_HEAD_ONLY_PREFIX
            if cfg.train_camera_scale_head_only
            else CAMERA_HEAD_ONLY_PREFIX
        )
        scope_label = (
            "camera-scale-head"
            if cfg.train_camera_scale_head_only
            else "camera-head"
        )
        trainable_parameter_names = {
            name
            for name, parameter in raw_model.named_parameters()
            if parameter.requires_grad
        }
        unexpected = sorted(
            name
            for name in trainable_parameter_names
            if not name.startswith(scoped_prefix)
        )
        if unexpected:
            raise RuntimeError(
                f"{scope_label} checkpoint found unexpected trainable parameters: "
                + ", ".join(unexpected[:8])
            )
        trainable_state = {
            key: value.detach().cpu()
            for key, value in raw_model.state_dict().items()
            if key.startswith(scoped_prefix)
        }
        missing = sorted(trainable_parameter_names.difference(trainable_state))
        if not trainable_state or missing:
            raise RuntimeError(
                f"{scope_label} checkpoint is incomplete; missing="
                + ", ".join(missing[:8])
            )
        trainable_module_roots = (scoped_prefix.rstrip("."),)
    else:
        trainable_state, trainable_module_roots = collect_trainable_module_state(
            model
        )
    # Partial module checkpoints still need the geometry convention marker.
    # Without it, a new wrist-depth run is falsely rejected as an incompatible resume.
    trainable_state["_mano_translation_convention"] = raw_model._mano_translation_convention.detach().cpu().clone()
    head_prefixes = (
        "hand_detection_block.",
        "hand_probability_head.",
        "hand_bbox_head.",
        "hand_mask_head.",
        "lingbot_adapter.",
        "feature_fusion.",
        "mano_head.",
        "joint_head.",
        "mutual_fusion.",
        "joint_offset_head.",
    )
    head_state = (
        dict(trainable_state)
        if cfg.train_camera_head_only or cfg.train_camera_scale_head_only
        else {
            key: value
            for key, value in trainable_state.items()
            if key.startswith(head_prefixes)
        }
    )
    payload = {
        "checkpoint_format_version": (
            3 if cfg.model_variant == "mutual_joint_fusion" else 2
        ),
        "model_variant": cfg.model_variant,
        "trainable_state": trainable_state,
        "trainable_module_roots": trainable_module_roots,
        "trainable_parameter_names": tuple(sorted(trainable_parameter_names))
        if cfg.train_camera_head_only or cfg.train_camera_scale_head_only
        else tuple(
            sorted(
                name
                for name, parameter in raw_model.named_parameters()
                if parameter.requires_grad
            )
        ),
        "trainable_heads": head_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "history": history,
    }
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def _masked_float_mean(value, mask):
    mask = mask.bool()
    if not bool(mask.any().item()):
        return None
    return float(value[mask].detach().float().mean().item())


def detector_diagnostics(output, target):
    if "pred_hand_probability" not in output:
        return {"detector_executed": False}
    probability = output["pred_hand_probability"].detach()
    valid = target["hand_valid"].to(device=probability.device, dtype=torch.bool)
    pred_valid = probability >= 0.5
    total = valid.numel()
    pos = int(valid.sum().item())
    out = {
        "valid_pos": pos,
        "valid_neg": int(total - pos),
        "valid_ratio": round(pos / max(total, 1), 6),
        "valid_acc": round(float((pred_valid == valid).float().mean().item()), 6),
        "bbox_l1_valid_only": None,
        "probability_pos_mean": None,
        "probability_neg_mean": None,
    }
    bbox_l1 = None
    if "pred_hand_boxes" in output:
        pred_boxes = output["pred_hand_boxes"].detach()
        gt_boxes = target["hand_boxes"].to(
            device=pred_boxes.device,
            dtype=pred_boxes.dtype,
        )
        bbox_l1 = (pred_boxes - gt_boxes).abs().mean(dim=-1)
    for key, value in (
        (
            "bbox_l1_valid_only",
            _masked_float_mean(bbox_l1, valid)
            if bbox_l1 is not None
            else None,
        ),
        ("probability_pos_mean", _masked_float_mean(probability, valid)),
        ("probability_neg_mean", _masked_float_mean(probability, ~valid)),
    ):
        if value is not None:
            out[key] = round(value, 6)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_trainable_scope_config(cfg)
    stage2_geometry_enabled = cfg.compute_stage2_diagnostics or any(
        weight != 0.0
        for weight in (
            cfg.world_joints_loss_weight,
            cfg.world_mano_loss_weight,
            cfg.abs_pose_loss_weight,
            cfg.rel_pose_loss_weight,
        )
    )
    if stage2_geometry_enabled and not cfg.lingbot_enable_camera_head:
        raise ValueError(
            "non-zero Stage-2 world/pose loss requires "
            "lingbot_enable_camera_head=true"
        )
    if not cfg.hand_bbox_head_enabled and (
        cfg.hand_bbox_smooth_l1_loss_weight != 0.0
        or cfg.hand_bbox_giou_loss_weight != 0.0
    ):
        raise ValueError(
            "hand_bbox_head_enabled=false requires bbox loss weights to be zero"
        )
    if cfg.verify_mask_mano_gradient and not cfg.mano_gradient_to_mask_head:
        raise ValueError(
            "verify_mask_mano_gradient=true requires "
            "mano_gradient_to_mask_head=true"
        )
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    actual_global_batch_size = int(cfg.batch_size) * world_size
    if (
        int(cfg.global_batch_size) > 0
        and actual_global_batch_size != int(cfg.global_batch_size)
    ):
        raise ValueError(
            "global batch size mismatch: "
            f"per_rank={cfg.batch_size} world_size={world_size} "
            f"actual={actual_global_batch_size} "
            f"configured={cfg.global_batch_size}"
        )
    seed_everything(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    output_dir = Path(cfg.output_dir)
    resume_checkpoint_path = os.environ.get(
        "RESUME_CHECKPOINT", cfg.resume_checkpoint
    ).strip()
    finetune_checkpoint_path = os.environ.get(
        "FINETUNE_CHECKPOINT", cfg.finetune_checkpoint
    ).strip()
    from .initialization import validate_initialization, save_initial_state
    validate_initialization(cfg, finetune_checkpoint=finetune_checkpoint_path, resume_checkpoint=resume_checkpoint_path)
    if resume_checkpoint_path and not finetune_checkpoint_path:
        raise ValueError("Resume requires finetune_checkpoint: the complete immutable base weights.")
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {resume_checkpoint_path}"
            )
        if resume_path.parent.resolve() == output_dir.resolve():
            raise ValueError(
                "resume output_dir must be new and different from the source "
                f"checkpoint directory: {output_dir}"
            )
        existing_outputs = [
            path for path in (output_dir / "last.pt", output_dir / "best.pt")
            if path.exists()
        ]
        if existing_outputs:
            raise ValueError(
                "resume output_dir must not already contain checkpoints: "
                + ", ".join(str(path) for path in existing_outputs)
            )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "resolved_config.yaml", "w") as handle:
            yaml.safe_dump(asdict(cfg), handle, sort_keys=False)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29671")
            dist.init_process_group("gloo", rank=0, world_size=1)
    if rank == 0:
        print(f"[dist] rank={rank} local_rank={local_rank} world_size={world_size}", flush=True)
        print(
            "[loss-config] "
            + json.dumps(
                {
                    "stage1": cfg.stage1_loss_weight,
                    "world_joints": cfg.world_joints_loss_weight,
                    "world_mano": cfg.world_mano_loss_weight,
                    "abs_pose": cfg.abs_pose_loss_weight,
                    "rel_pose": cfg.rel_pose_loss_weight,
                    "pose_translation": cfg.pose_translation_loss_weight,
                    "world_mano_global_orient": (
                        cfg.world_mano_global_orient_weight
                    ),
                    "world_mano_hand_pose": cfg.world_mano_hand_pose_weight,
                    "world_mano_betas": cfg.world_mano_betas_weight,
                    "world_mano_translation": (
                        cfg.world_mano_translation_weight
                    ),
                    "stage2_huber_beta": cfg.stage2_huber_beta,
                    "compute_stage2_diagnostics": (
                        cfg.compute_stage2_diagnostics
                    ),
                    "relative_pose_window": cfg.relative_pose_window,
                    "loss_outlier_window": cfg.loss_outlier_window,
                    "loss_outlier_multiplier": cfg.loss_outlier_multiplier,
                    "loss_outlier_bootstrap_steps": (
                        cfg.loss_outlier_bootstrap_steps
                    ),
                    "loss_outlier_bootstrap_threshold": (
                        cfg.loss_outlier_bootstrap_threshold
                    ),
                    "loss_outlier_per_dataset": cfg.loss_outlier_per_dataset,
                    "hand_bbox_head_enabled": cfg.hand_bbox_head_enabled,
                    "use_soft_mask_boxes_for_mano": (
                        cfg.use_soft_mask_boxes_for_mano
                    ),
                    "mano_gradient_to_mask_head": (
                        cfg.mano_gradient_to_mask_head
                    ),
                    "train_detection_modules_only": (
                        cfg.train_detection_modules_only
                    ),
                    "train_probability_mask_heads_only": (
                        cfg.train_probability_mask_heads_only
                    ),
                    "train_mano_modules_only": cfg.train_mano_modules_only,
                    "train_camera_head_only": cfg.train_camera_head_only,
                    "train_camera_scale_head_only": (
                        cfg.train_camera_scale_head_only
                    ),
                    "camera_scale_head_enabled": (
                        cfg.camera_scale_head_enabled
                    ),
                    "coordinate_protocol": cfg.stage2_coordinate_protocol,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print("[stage] build_loader", flush=True)
    loader = build_loader(cfg)
    data_iter = iter(loader)
    if rank == 0:
        print("[stage] build_model", flush=True)
    model_cfg = LingBotMapHandConfig(
        lingbotmap_root=cfg.lingbotmap_root,
        lingbotmap_ckpt=cfg.lingbotmap_ckpt,
        wilor_root=cfg.wilor_root,
        wilor_cfg=cfg.wilor_cfg,
        wilor_ckpt=cfg.wilor_ckpt,
        load_wilor_checkpoint=cfg.load_wilor_checkpoint,
        mano_model_path=str(Path(cfg.wilor_root) / "mano_data"),
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        bbox_rescale=cfg.bbox_rescale,
        bbox_scale_factor=cfg.bbox_scale_factor,
        bbox_rescale_range=(
            tuple(cfg.bbox_rescale_range)
            if cfg.bbox_rescale_range is not None
            else None
        ),
        freeze_backbone=cfg.freeze_backbone,
        lingbot_tune_mode=cfg.lingbot_tune_mode,
        lingbot_tune_last_n_blocks=cfg.lingbot_tune_last_n_blocks,
        lingbot_tune_special_tokens=cfg.lingbot_tune_special_tokens,
        lingbot_gradient_checkpointing=cfg.lingbot_gradient_checkpointing,
        lingbot_gradient_checkpoint_last_n_blocks=(
            cfg.lingbot_gradient_checkpoint_last_n_blocks
        ),
        lingbot_activation_offload=cfg.lingbot_activation_offload,
        lingbot_enable_camera_head=cfg.lingbot_enable_camera_head,
        camera_scale_head_enabled=cfg.camera_scale_head_enabled,
        camera_scale_head_hidden_dim=cfg.camera_scale_head_hidden_dim,
        camera_scale_head_num_layers=cfg.camera_scale_head_num_layers,
        camera_scale_head_num_heads=cfg.camera_scale_head_num_heads,
        camera_scale_head_ffn_dim=cfg.camera_scale_head_ffn_dim,
        camera_scale_head_frames=cfg.camera_scale_head_frames,
        camera_scale_log_min=cfg.camera_scale_log_min,
        camera_scale_log_max=cfg.camera_scale_log_max,
        freeze_wilor_backbone=cfg.freeze_wilor_backbone,
        wilor_gradient_checkpointing=cfg.wilor_gradient_checkpointing,
        wilor_feature_chunk_size=cfg.hand_feature_chunk_size,
        hand_feature_provider=cfg.hand_feature_provider,
        hand_feature_chunk_size=cfg.hand_feature_chunk_size,
        freeze_hand_feature_provider=cfg.freeze_hand_feature_provider,
        dinov2_input_height=cfg.dinov2_input_height,
        dinov2_input_width=cfg.dinov2_input_width,
        sapiens2_root=cfg.sapiens2_root,
        sapiens2_arch=cfg.sapiens2_arch,
        sapiens2_ckpt=cfg.sapiens2_ckpt,
        decode_mano=True,
        use_gt_boxes_for_mano=not cfg.mano_gradient_to_mask_head,
        model_variant=cfg.model_variant,
        joint_heatmap_range_margin=cfg.joint_heatmap_range_margin,
        mutual_fusion_num_heads=cfg.mutual_fusion_num_heads,
        mutual_fusion_dropout=cfg.mutual_fusion_dropout,
        pnp_min_visible=cfg.pnp_min_visible,
        hand_detection_dim=cfg.hand_detection_dim,
        hand_detection_frames_chunk_size=cfg.hand_detection_frames_chunk_size,
        mask_only_threshold=cfg.mask_only_threshold,
        use_soft_mask_boxes_for_mano=cfg.mano_gradient_to_mask_head,
        mano_gradient_to_mask_head=cfg.mano_gradient_to_mask_head,
        soft_mask_box_temperature=cfg.soft_mask_box_temperature,
        soft_mask_box_std_scale=cfg.soft_mask_box_std_scale,
        soft_mask_box_min_size=cfg.soft_mask_box_min_size,
        step_memory_enabled=cfg.step_memory_enabled,
        step_memory_attention_mode=cfg.step_memory_attention_mode,
        step_memory_scale_frames=cfg.step_memory_scale_frames,
        step_memory_max_frames=cfg.step_memory_max_frames,
        step_memory_sliding_window=cfg.step_memory_sliding_window,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"[cuda] visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"name={torch.cuda.get_device_name(device)} "
        f"free_gb={free_bytes / (1024**3):.3f} total_gb={total_bytes / (1024**3):.3f}",
        flush=True,
    )
    model = LingBotMapHand(model_cfg)
    for module_name in (
        "backbone",
        "camera_scale_head",
        "hand_encoder",
        "hand_detection_block",
            "hand_bbox_head",
        "hand_mask_head",
        "lingbot_adapter",
        "feature_fusion",
        "mano_head",
        "joint_head",
        "mutual_fusion",
        "joint_offset_head",
        "mano_right",
        "mano_left",
    ):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        module.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        else:
            free_bytes = total_bytes = 0
        if rank == 0:
            print(
                f"[cuda] moved={module_name} allocated_gb={torch.cuda.memory_allocated() / (1024**3):.3f} "
                f"free_gb={free_bytes / (1024**3):.3f}",
                flush=True,
            )
    checkpoint_path = finetune_checkpoint_path
    finetune_source_step = 0
    finetune_step_offset = max(0, int(cfg.finetune_step_offset))
    resume_payload = None
    resume_schedule_step = 0
    if resume_checkpoint_path:
        # Saved training checkpoints contain only updated modules. Restore the frozen base first.
        base_payload = load_checkpoint_payload(finetune_checkpoint_path)
        model.load_state_dict(extract_model_state(base_payload), strict=True)
        del base_payload
        resume_payload = load_checkpoint_payload(resume_checkpoint_path)
        validate_resume_schedule(
            resume_payload,
            train_steps=cfg.train_steps,
            warmup_steps=cfg.warmup_steps,
            peak_lr=cfg.lr,
            min_lr=cfg.min_lr,
            weight_decay=cfg.weight_decay,
        )
        checkpoint_variant = (
            str(resume_payload["model_variant"]).strip().lower()
            if isinstance(resume_payload, dict) and "model_variant" in resume_payload
            else None
        )
        if (
            checkpoint_variant is not None
            and checkpoint_variant != str(cfg.model_variant).strip().lower()
        ):
            raise RuntimeError(
                "checkpoint model_variant mismatch: "
                f"checkpoint={checkpoint_variant!r}, config={cfg.model_variant!r}"
            )
        checkpoint_state = extract_model_state(resume_payload)
        checkpoint_state, dropped_bbox_tensors = filter_disabled_bbox_head_state(
            checkpoint_state,
            bbox_head_enabled=cfg.hand_bbox_head_enabled,
        )
        if rank == 0 and dropped_bbox_tensors:
            print(
                f"[checkpoint-filter] dropped_bbox_head_tensors={dropped_bbox_tensors}",
                flush=True,
            )
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected resume checkpoint keys: "
                + ", ".join(incompatible.unexpected_keys[:20])
            )
        resume_schedule_step = infer_checkpoint_step(
            resume_payload,
            configured_step=cfg.resume_checkpoint_step,
        )
        if resume_schedule_step >= int(cfg.train_steps):
            raise ValueError(
                f"checkpoint_step={resume_schedule_step} has already reached "
                f"train_steps={cfg.train_steps}"
            )
        finetune_step_offset = infer_finetune_step_offset(
            resume_payload, resume_schedule_step
        )
        if isinstance(resume_payload, dict):
            finetune_source_step = int(
                resume_payload.get("finetune_source_step", finetune_step_offset)
                or finetune_step_offset
            )
        if rank == 0:
            print(
                f"[resume-model] checkpoint={resume_checkpoint_path} "
                f"checkpoint_step={resume_schedule_step} "
                f"next_step={resume_schedule_step + 1} "
                f"loaded_keys={len(checkpoint_state)} "
                f"missing_model_keys={len(incompatible.missing_keys)}",
                flush=True,
            )
        del checkpoint_state
    elif checkpoint_path:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"finetune checkpoint does not exist: {checkpoint_path}")
        checkpoint_payload = load_checkpoint_payload(checkpoint_path)
        checkpoint_variant = str(
            checkpoint_payload.get("model_variant", cfg.model_variant)
        ).strip().lower()
        if checkpoint_variant != str(cfg.model_variant).strip().lower():
            raise RuntimeError(
                "checkpoint model_variant mismatch: "
                f"checkpoint={checkpoint_variant!r}, config={cfg.model_variant!r}. "
                "The checkpoint must match the configured model architecture."
            )
        checkpoint_state = extract_model_state(checkpoint_payload)
        checkpoint_state, dropped_bbox_tensors = filter_disabled_bbox_head_state(
            checkpoint_state,
            bbox_head_enabled=cfg.hand_bbox_head_enabled,
        )
        if rank == 0 and dropped_bbox_tensors:
            print(
                f"[checkpoint-filter] dropped_bbox_head_tensors={dropped_bbox_tensors}",
                flush=True,
            )
        if not any(key.startswith("hand_encoder.") for key in checkpoint_state):
            raise RuntimeError(
                f"finetune checkpoint is missing hand_encoder weights: {checkpoint_path}"
            )
        incompatible = model.load_state_dict(checkpoint_state, strict=True)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected finetune checkpoint keys: "
                + ", ".join(incompatible.unexpected_keys[:20])
            )
        finetune_source_step = int(checkpoint_payload.get("step", 0))
        if finetune_step_offset <= 0:
            finetune_step_offset = finetune_source_step
        if rank == 0:
            print(
                f"[finetune-init] checkpoint={checkpoint_path} "
                f"source_step={finetune_source_step} step_offset={finetune_step_offset} "
                f"loaded_keys={len(checkpoint_state)} missing_model_keys={len(incompatible.missing_keys)} "
                "optimizer=fresh",
                flush=True,
            )
        del checkpoint_payload, checkpoint_state
    elif finetune_step_offset:
        raise ValueError("finetune_step_offset requires finetune_checkpoint")
    if not finetune_checkpoint_path:
        # Save rank 0's initialization before DDP broadcasts it to other ranks.
        if rank == 0:
            save_initial_state(model, output_dir)
        cfg.finetune_checkpoint = str((output_dir / "initial_model.pt").resolve())
        if rank == 0:
            with open(output_dir / "resolved_config.yaml", "w") as handle:
                yaml.safe_dump(asdict(cfg), handle, sort_keys=False)
    if dist.get_world_size() > 1:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
    if cfg.train_joint_heads_and_camera:
        model.backbone.model.camera_head.requires_grad_(True)
    model.train()
    loss_fn = LingBotMapHandLoss(
        joint_heatmap_range_margin=cfg.joint_heatmap_range_margin,
        relative_pose_window=cfg.relative_pose_window,
        compute_stage2_diagnostics=cfg.compute_stage2_diagnostics,
        bbox_loss_enabled=cfg.hand_bbox_head_enabled,
        expose_graph_components=cfg.verify_mask_mano_gradient,
        camera_hand_loss_enabled=(
            not (
                cfg.train_detection_modules_only
                or cfg.train_probability_mask_heads_only
            )
            or cfg.mano_gradient_to_mask_head
        ),
        stage2_coordinate_protocol=cfg.stage2_coordinate_protocol,
        hand_probability=cfg.hand_probability_loss_weight,
        hand_bbox_smooth_l1=cfg.hand_bbox_smooth_l1_loss_weight,
        hand_bbox_giou=cfg.hand_bbox_giou_loss_weight,
        hand_mask_bce=cfg.hand_mask_bce_loss_weight,
        hand_mask_dice=cfg.hand_mask_dice_loss_weight,
        joints2d_direct_weight=cfg.joints2d_direct_weight,
        reproj_loss_weight=cfg.reproj_loss_weight,
        go_loss_weight=cfg.go_loss_weight,
        hp_loss_weight=cfg.hp_loss_weight,
        betas_loss_weight=cfg.betas_loss_weight,
        trans_loss_weight=cfg.trans_loss_weight,
        joints_loss_weight=cfg.joints_loss_weight,
        velocity_smooth_weight=cfg.velocity_smooth_weight,
        betas_consistency_weight=cfg.betas_consistency_weight,
        stage1_loss_weight=cfg.stage1_loss_weight,
        world_joints_loss_weight=cfg.world_joints_loss_weight,
        world_mano_loss_weight=cfg.world_mano_loss_weight,
        abs_pose_loss_weight=cfg.abs_pose_loss_weight,
        rel_pose_loss_weight=cfg.rel_pose_loss_weight,
        pose_translation_loss_weight=cfg.pose_translation_loss_weight,
        abs_pose_translation_component_weight=(
            cfg.abs_pose_translation_component_weight
        ),
        abs_pose_quaternion_component_weight=(
            cfg.abs_pose_quaternion_component_weight
        ),
        abs_pose_fov_component_weight=cfg.abs_pose_fov_component_weight,
        rel_pose_rotation_component_weight=(
            cfg.rel_pose_rotation_component_weight
        ),
        rel_pose_translation_component_weight=(
            cfg.rel_pose_translation_component_weight
        ),
        world_mano_global_orient_weight=(
            cfg.world_mano_global_orient_weight
        ),
        world_mano_hand_pose_weight=cfg.world_mano_hand_pose_weight,
        world_mano_betas_weight=cfg.world_mano_betas_weight,
        world_mano_translation_weight=cfg.world_mano_translation_weight,
        stage2_huber_beta=cfg.stage2_huber_beta,
    ).to(device)
    if cfg.train_probability_mask_heads_only:
        model.requires_grad_(False)
        trainable_modules = PROBABILITY_MASK_ONLY_ROOTS
        for module_name in trainable_modules:
            module = getattr(model, module_name)
            module.requires_grad_(True)
        for module_name, module in model.named_children():
            if module_name in trainable_modules:
                module.train(True)
            else:
                module.eval()
        trainable_roots = sorted(
            {
                name.split(".", 1)[0]
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
        if trainable_roots != sorted(trainable_modules):
            raise RuntimeError(
                "probability-mask-head-only trainable roots mismatch: "
                f"actual={trainable_roots} "
                f"expected={sorted(trainable_modules)}"
            )
        if rank == 0:
            print(
                "[trainable-scope] probability_mask_heads_only=true roots="
                + ",".join(trainable_roots),
                flush=True,
            )
    elif cfg.train_detection_modules_only:
        model.requires_grad_(False)
        detection_modules = DETECTION_ONLY_ROOTS
        for module_name in detection_modules:
            module = getattr(model, module_name)
            module.requires_grad_(True)
        for module_name, module in model.named_children():
            if module_name in detection_modules:
                module.train(True)
            else:
                module.eval()
        trainable_roots = sorted(
            {
                name.split(".", 1)[0]
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
        if trainable_roots != sorted(detection_modules):
            raise RuntimeError(
                "detection-only trainable roots mismatch: "
                f"actual={trainable_roots} expected={sorted(detection_modules)}"
            )
        if rank == 0:
            print(
                "[trainable-scope] detection_only=true roots="
                + ",".join(trainable_roots),
                flush=True,
            )
    elif cfg.train_mano_modules_only:
        model.requires_grad_(False)
        mano_modules = MANO_ONLY_ROOTS
        for module_name in mano_modules:
            module = getattr(model, module_name)
            module.requires_grad_(True)
        for module_name, module in model.named_children():
            if module_name in mano_modules:
                module.train(True)
            else:
                module.eval()
        trainable_roots = sorted(
            {
                name.split(".", 1)[0]
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
        if trainable_roots != sorted(mano_modules):
            raise RuntimeError(
                "MANO-only trainable roots mismatch: "
                f"actual={trainable_roots} expected={sorted(mano_modules)}"
            )
        if rank == 0:
            print(
                "[trainable-scope] mano_only=true roots="
                + ",".join(trainable_roots),
                flush=True,
            )
    elif cfg.train_camera_head_only:
        configure_camera_head_only_scope(model, rank=rank)
    elif cfg.train_camera_scale_head_only:
        configure_camera_scale_head_only_scope(model, rank=rank)
    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=True, broadcast_buffers=False)
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    lingbot_trainable = [
        parameter
        for name, parameter in named_trainable
        if "backbone.model." in name
    ]
    stage2_trainable = [
        parameter
        for name, parameter in named_trainable
        if "backbone.model." not in name
    ]
    camera_scale_trainable = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(
            (
                CAMERA_SCALE_HEAD_ONLY_PREFIX,
                "module." + CAMERA_SCALE_HEAD_ONLY_PREFIX,
            )
        )
    ]
    if cfg.train_camera_scale_head_only and (
        not camera_scale_trainable
        or {id(parameter) for parameter in camera_scale_trainable}
        != {id(parameter) for parameter in stage2_trainable}
    ):
        raise RuntimeError(
            "camera-scale-head-only gradient audit must track exactly the "
            "optimizer parameters after optional DDP wrapping"
        )
    optimizer_groups = []
    if cfg.train_camera_head_only:
        if stage2_trainable or not lingbot_trainable:
            raise RuntimeError(
                "camera-head-only optimizer must contain exactly the camera "
                "head and no Stage-2 parameters"
            )
        optimizer_groups.append(
            {
                "params": lingbot_trainable,
                "lr_scale": 1.0,
                "group_name": "camera_head",
            }
        )
    elif cfg.train_camera_scale_head_only:
        if lingbot_trainable or not stage2_trainable:
            raise RuntimeError(
                "camera-scale-head-only optimizer must contain exactly the "
                "metric scale head and no LingBot parameters"
            )
        optimizer_groups.append(
            {
                "params": stage2_trainable,
                "lr_scale": 1.0,
                "group_name": "camera_scale_head",
            }
        )
    elif stage2_trainable:
        optimizer_groups.append(
            {"params": stage2_trainable, "lr_scale": 1.0, "group_name": "stage2"}
        )
    if lingbot_trainable and not (
        cfg.train_camera_head_only or cfg.train_camera_scale_head_only
    ):
        if cfg.lr <= 0 or cfg.lingbot_lr <= 0:
            raise ValueError("lr and lingbot_lr must be positive")
        optimizer_groups.append(
            {
                "params": lingbot_trainable,
                "lr_scale": float(cfg.lingbot_lr) / float(cfg.lr),
                "group_name": "lingbot",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    optimizer_restored = restore_optimizer_state(optimizer, resume_payload)
    next_schedule_step = resume_schedule_step + 1
    resumed_lr = set_learning_rate(optimizer, next_schedule_step, cfg)
    if rank == 0:
        print(
            "[optimizer-groups] "
            + json.dumps(
                [
                    {
                        "group_name": group.get("group_name", "unnamed"),
                        "lr_scale": float(group.get("lr_scale", 1.0)),
                        "parameter_tensors": len(group["params"]),
                        "parameters": sum(
                            parameter.numel() for parameter in group["params"]
                        ),
                    }
                    for group in optimizer_groups
                ],
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            f"[stage] train trainable_tensors={len(trainable)} "
            f"trainable_params={sum(parameter.numel() for parameter in trainable)} "
            f"lingbot_tensors={len(lingbot_trainable)} "
            f"lingbot_params={sum(parameter.numel() for parameter in lingbot_trainable)}",
            flush=True,
        )
        if resume_payload is not None:
            print(
                f"[resume-state] checkpoint_step={resume_schedule_step} "
                f"next_step={next_schedule_step} "
                f"optimizer={'restored' if optimizer_restored else 'fresh_model_only'} "
                f"lr={resumed_lr:.10g} schedule_train_steps={cfg.train_steps} "
                f"warmup_steps={cfg.warmup_steps}",
                flush=True,
            )

    history = (
        list(resume_payload.get("history", []))
        if isinstance(resume_payload, dict)
        else []
    )
    best_window_loss = (
        float(
            resume_payload.get(
                "best_window_loss", resume_payload.get("best_loss", float("inf"))
            )
        )
        if isinstance(resume_payload, dict)
        else float("inf")
    )
    window_records = []
    if cfg.loss_outlier_per_dataset:
        raise ValueError("Sample filtering uses a shared rolling raw-loss threshold; per-dataset filtering is unsupported.")
    loss_filter = RollingLossThreshold(
        cfg.loss_outlier_window, cfg.loss_outlier_bootstrap_steps,
        cfg.loss_outlier_bootstrap_threshold, cfg.loss_outlier_multiplier,
        state=resume_payload.get("loss_filter_state") if isinstance(resume_payload, dict) else None,
    )
    restored_recent_source = "finite_raw_step_mean_v1"
    recent_observed_step_means = deque(maxlen=max(1, cfg.loss_outlier_window))
    recent_observed_step_means_by_dataset = {}
    world_size = dist.get_world_size()
    skipped_outliers = (
        int(resume_payload.get("rank0_total_skipped_outliers", 0))
        if isinstance(resume_payload, dict) and rank == 0
        else 0
    )
    window_skipped_outliers = 0
    total_outlier_rank_batches = (
        int(resume_payload.get("total_outlier_rank_batches", resume_payload.get("total_skipped_outliers", 0)))
        if isinstance(resume_payload, dict)
        else 0
    )
    window_outlier_rank_batches = 0
    total_steps_with_any_outlier = (
        int(resume_payload.get("total_steps_with_any_outlier", 0))
        if isinstance(resume_payload, dict)
        else 0
    )
    window_steps_with_any_outlier = 0
    total_fully_skipped_steps = (
        int(resume_payload.get("total_fully_skipped_steps", 0))
        if isinstance(resume_payload, dict)
        else 0
    )
    window_fully_skipped_steps = 0
    window_global_accepted_objective_sum = 0.0
    window_global_accepted_rank_batches = 0
    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    run_optimizer_updates = 0
    run_accepted_samples = run_rejected_samples = run_accepted_rank_batches = 0
    start_step = resume_schedule_step + 1
    run_end_step = int(cfg.train_steps)
    if int(cfg.max_run_steps) > 0:
        run_end_step = min(
            run_end_step,
            start_step + int(cfg.max_run_steps) - 1,
        )
    if run_end_step < start_step:
        raise ValueError(
            f"run has no steps: start_step={start_step} end_step={run_end_step}"
        )
    executed_steps = run_end_step - start_step + 1
    if rank == 0:
        print(
            f"[run-plan] start_step={start_step} end_step={run_end_step} "
            f"schedule_train_steps={cfg.train_steps} warmup_steps={cfg.warmup_steps} "
            f"max_run_steps={cfg.max_run_steps} "
            f"restored_history={len(history)} restored_best_loss={best_window_loss} "
            f"outlier_history={len(recent_observed_step_means)} "
            f"outlier_history_source={restored_recent_source}",
            flush=True,
        )
        if resume_checkpoint_path:
            prior_best_path = Path(resume_checkpoint_path).parent / "best.pt"
            if prior_best_path.is_file() and not best_checkpoint_path.exists():
                atomic_copy_file(prior_best_path, best_checkpoint_path)
                print(
                    f"[resume-best] source={prior_best_path} path={best_checkpoint_path}",
                    flush=True,
                )
    if resume_payload is not None:
        del resume_payload
    rank_step_tracer = (
        RankStepTracer(
            output_dir,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            flush_interval=cfg.rank_trace_flush_interval,
        )
        if cfg.rank_trace_every_step
        else None
    )
    if rank == 0 and rank_step_tracer is not None:
        print(
            f"[rank-trace] enabled=true directory={output_dir / 'rank_traces'} "
            f"flush_interval={cfg.rank_trace_flush_interval}",
            flush=True,
        )
    for step in range(start_step, run_end_step + 1):
        iteration_t0 = time.perf_counter()
        global_step = finetune_step_offset + step
        current_lr = set_learning_rate(optimizer, step, cfg)
        batch, data_iter, fetch_attempts, fetch_ms, slow_records = fetch_hand_batch_with_retry(
            data_iter=data_iter,
            loader=loader,
            step=global_step,
            rank=rank,
            local_rank=local_rank,
            max_attempts=cfg.get_batch_retry_attempts,
            require_valid_hands=not (
                cfg.train_camera_head_only
                or cfg.train_camera_scale_head_only
                or cfg.train_detection_modules_only
                or cfg.train_probability_mask_heads_only
            ),
        )
        if slow_records:
            for item in slow_records:
                print(f"[warn] get batch retry record: {json.dumps(item, ensure_ascii=False)}", flush=True)
        batch_identity = (
            extract_batch_identity(batch) if rank_step_tracer is not None else {}
        )
        pre_step_t0 = time.perf_counter()
        batch = move_tensors(batch, device)
        if batch["images"].dtype == torch.uint8:
            batch["images"] = batch["images"].float().div_(255.0)
        images = batch["images"]
        target, _ = build_training_target(batch, cfg)

        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            memory_before_forward = torch.cuda.memory_allocated(device)
            memory_before_forward_reserved = torch.cuda.memory_reserved(device)
        else:
            memory_before_forward = 0
            memory_before_forward_reserved = 0
        pre_step_ms = (time.perf_counter() - pre_step_t0) * 1000.0
        train_t0 = time.perf_counter()
        forward_t0 = train_t0

        def report_oom(oom_stage: str, **timing: float) -> None:
            raw_model = model.module if hasattr(model, "module") else model
            memory_stats = getattr(raw_model.backbone, "last_step_memory_stats", {})
            oom_record = {
                "status": "OOM",
                "oom_stage": oom_stage,
                "num_views": int(images.shape[1]),
                "step_memory_scale_frames": (
                    int(cfg.step_memory_scale_frames)
                    if cfg.step_memory_enabled
                    else 0
                ),
                "wilor_gradient_checkpointing": bool(
                    cfg.wilor_gradient_checkpointing
                ),
                "fetch_ms": round(fetch_ms, 2),
                "pre_step_ms": round(pre_step_ms, 2),
                "iteration_elapsed_ms": round(
                    (time.perf_counter() - iteration_t0) * 1000.0, 2
                ),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated(device) / (1024**3), 3
                ),
                "memory_reserved_gb": round(
                    torch.cuda.memory_reserved(device) / (1024**3), 3
                ),
                "peak_memory_allocated_gb": round(
                    torch.cuda.max_memory_allocated(device) / (1024**3), 3
                ),
                "peak_memory_reserved_gb": round(
                    torch.cuda.max_memory_reserved(device) / (1024**3), 3
                ),
                **{key: round(value, 2) for key, value in timing.items()},
                **{f"lingbot_{key}": value for key, value in memory_stats.items()},
            }
            print("[oom] " + json.dumps(oom_record, sort_keys=True), flush=True)

        original_batch_size = int(images.shape[0])
        full_batch = batch
        fixed_mano_selection = None
        mask_mano_grad_norm = mask_logits_mano_grad_norm = None
        threshold_value, outlier_threshold_mode = loss_filter.threshold(step)
        recent_mean = loss_filter.mean
        outlier_threshold = threshold_value if np.isfinite(threshold_value) else None
        outlier_thresholds_by_dataset = {}
        recent_means_by_dataset = {}
        dataset_names = batch_dataset_names(batch, original_batch_size)

        def evaluate_samples(selected_indices):
            nonlocal fixed_mano_selection, mask_mano_grad_norm, mask_logits_mano_grad_norm
            selected_batch = slice_batch_tree(full_batch, selected_indices, original_batch_size)
            selected_target, _ = build_training_target(selected_batch, cfg)
            selected_images = selected_batch["images"]
            hand_valid = selected_target["hand_valid"]
            if fixed_mano_selection is not None:
                hand_valid = hand_valid & fixed_mano_selection.index_select(0, selected_indices)
            from .data.training_profiles import training_task
            output = model(
                selected_images,
                heads=training_task(cfg),
                gt_hand_boxes=selected_target.get("hand_boxes"),
                gt_hand_valid=hand_valid,
                mano_left_basis_b=selected_batch.get("mano_left_basis_b"),
                mano_training_selection=bool(cfg.mano_random_valid_hand and (
                    cfg.train_mano_modules_only or cfg.train_joint_heads_and_camera)),
                use_gt_boxes=not cfg.mano_gradient_to_mask_head,
                intrinsics=selected_batch["intrinsics"].float(),
            )
            if fixed_mano_selection is None and selected_indices.numel() == original_batch_size:
                fixed_mano_selection = output.get("mano_selected_valid")
                if fixed_mano_selection is not None:
                    fixed_mano_selection = fixed_mano_selection.detach()
            losses = loss_fn(output, selected_target)
            objective = losses["objective"]
            camera_hand_objective_graph = losses.get("_camera_hand_objective_graph")
            if cfg.verify_mask_mano_gradient and step == start_step and selected_indices.numel() == original_batch_size:
                if camera_hand_objective_graph is None:
                    raise RuntimeError(
                        "mask-to-MANO gradient check requested but graph component "
                        "was not exposed"
                    )
                raw_model = model.module if hasattr(model, "module") else model
                mask_parameters = [
                    parameter
                    for parameter in raw_model.hand_mask_head.parameters()
                    if parameter.requires_grad
                ]
                parameter_grads = torch.autograd.grad(
                    camera_hand_objective_graph,
                    mask_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                parameter_terms = [
                    gradient.detach().float().norm()
                    for gradient in parameter_grads
                    if gradient is not None
                ]
                mask_mano_grad_norm = (
                    torch.linalg.vector_norm(torch.stack(parameter_terms))
                    if parameter_terms
                    else objective.new_zeros(())
                )
                (mask_logits_gradient,) = torch.autograd.grad(
                    camera_hand_objective_graph,
                    output["pred_hand_mask_logits"],
                    retain_graph=True,
                    allow_unused=True,
                )
                mask_logits_mano_grad_norm = (
                    mask_logits_gradient.detach().float().norm()
                    if mask_logits_gradient is not None
                    else objective.new_zeros(())
                )
                if (
                    not torch.isfinite(mask_mano_grad_norm)
                    or not torch.isfinite(mask_logits_mano_grad_norm)
                    or float(mask_mano_grad_norm) <= 0.0
                    or float(mask_logits_mano_grad_norm) <= 0.0
                ):
                    raise RuntimeError(
                        "MANO loss did not produce a finite non-zero gradient for "
                        "the mask head"
                    )
                if rank == 0:
                    print(
                        "[mask-mano-grad] "
                        f"mask_head_grad_norm={float(mask_mano_grad_norm):.9g} "
                        "mask_logits_grad_norm="
                        f"{float(mask_logits_mano_grad_norm):.9g} "
                        "status=nonzero",
                        flush=True,
                    )

            return output, selected_target, losses

        # DDP automatic reduction cannot safely traverse a discarded NaN graph.
        # Keep every forward/backward local, then synchronize only clean gradients.
        try:
            with model.no_sync() if isinstance(model, DDP) else nullcontext():
                filtered = filtered_sample_backward(
                    evaluate_samples, original_batch_size, device, threshold_value, trainable)
        except torch.OutOfMemoryError:
            report_oom("sample_forward_backward")
            raise
        output, target, losses = filtered.output, filtered.target, filtered.losses
        objective = losses["objective"]
        objective_value = float(objective)
        objective_per_sample = filtered.raw_losses
        local_accepted_samples = int(filtered.accepted_indices.numel())
        local_rejected_samples = original_batch_size - local_accepted_samples
        local_outlier = local_accepted_samples == 0
        forward_ms = filtered.forward_loss_seconds * 1000.0
        backward_ms = filtered.backward_seconds * 1000.0
        loss_ms = 0.0  # included in forward_ms, including isolated loss replay
        decision_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after_forward = torch.cuda.memory_allocated(device)
            memory_after_forward_reserved = torch.cuda.memory_reserved(device)
            forward_peak_memory = torch.cuda.max_memory_allocated(device)
            forward_peak_reserved_memory = torch.cuda.max_memory_reserved(device)
        else:
            memory_after_forward = memory_after_forward_reserved = 0
            forward_peak_memory = forward_peak_reserved_memory = 0
        finite_raw = objective_per_sample[torch.isfinite(objective_per_sample)]
        raw_sum = finite_raw.double().sum() if finite_raw.numel() else objective.new_zeros(())
        raw_max = float(finite_raw.max()) if finite_raw.numel() else 0.0
        local_ratio = raw_max / max(threshold_value, 1e-12) if np.isfinite(threshold_value) else 0.0
        # One collective supplies global sample counts and finite RAW history.
        outlier_sum_state = objective.new_tensor([
            float(local_rejected_samples > 0), float(local_accepted_samples > 0),
            filtered.objective_sum, float(local_accepted_samples),
            float(local_rejected_samples), float(raw_sum), float(finite_raw.numel()),
        ], dtype=torch.float64)
        outlier_max_state = objective.new_tensor([local_ratio, raw_max])
        if world_size > 1:
            dist.all_reduce(outlier_sum_state, op=dist.ReduceOp.SUM)
            dist.all_reduce(outlier_max_state, op=dist.ReduceOp.MAX)
        global_max_ratio, global_max_objective = outlier_max_state.tolist()
        global_outlier_ranks = int(outlier_sum_state[0])
        global_accepted_ranks = int(outlier_sum_state[1])
        global_accepted_objective_sum = float(outlier_sum_state[2])
        global_accepted_samples = int(outlier_sum_state[3])
        global_rejected_samples = int(outlier_sum_state[4])
        global_accepted_objective_mean = (global_accepted_objective_sum / global_accepted_samples
            if global_accepted_samples else None)
        loss_filter.observe(float(outlier_sum_state[5]), int(outlier_sum_state[6]))
        recent_observed_step_means.clear()
        recent_observed_step_means.extend(x for x in loss_filter.history if x is not None)
        stage2_availability_state = objective.new_tensor(
            [
                float(losses["loss_stage2_supervision_valid"].detach().item()),
                float(losses["loss_stage2_world_joints_available"].detach().item()),
                float(losses["loss_stage2_world_mano_available"].detach().item()),
                float(losses["loss_stage2_pose_available"].detach().item()),
            ]
        )
        if world_size > 1:
            dist.all_reduce(stage2_availability_state, op=dist.ReduceOp.SUM)
        global_stage2_supervision_valid_ranks = int(
            round(float(stage2_availability_state[0].item()))
        )
        global_stage2_world_joints_available_ranks = int(
            round(float(stage2_availability_state[1].item()))
        )
        global_stage2_world_mano_available_ranks = int(
            round(float(stage2_availability_state[2].item()))
        )
        global_stage2_pose_available_ranks = int(
            round(float(stage2_availability_state[3].item()))
        )

        if cfg.detailed_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        decision_ms = (time.perf_counter() - decision_t0) * 1000.0

        backward_scale = 1.0 / max(1, global_accepted_samples)
        sync_t0 = time.perf_counter()
        synchronized_count, gradients_finite = sync_finite_sample_gradients(
            trainable, local_accepted_samples)
        gradient_sync_ms = (time.perf_counter() - sync_t0) * 1000.0
        if synchronized_count != global_accepted_samples:
            raise RuntimeError("Accepted sample count disagrees across gradient synchronization")
        if not gradients_finite:
            # Never apply a nonfinite update. Per-sample local replay should
            # already have removed the offending sample; this is a final guard.
            optimizer.zero_grad(set_to_none=True)
        backward_t0 = time.perf_counter()
        try:
            lingbot_grad_terms = [
                parameter.grad.detach().float().norm()
                for parameter in lingbot_trainable
                if parameter.grad is not None
            ]
            lingbot_grad_norm = (
                torch.linalg.vector_norm(torch.stack(lingbot_grad_terms))
                if lingbot_grad_terms
                else objective.new_zeros(())
            )
            camera_head_grad_tensors = (
                sum(parameter.grad is not None for parameter in lingbot_trainable)
                if cfg.train_camera_head_only
                else 0
            )
            camera_head_unused_grad_tensors = (
                len(lingbot_trainable) - camera_head_grad_tensors
                if cfg.train_camera_head_only
                else 0
            )
            camera_scale_grad_terms = [
                parameter.grad.detach().float().norm()
                for parameter in camera_scale_trainable
                if parameter.grad is not None
            ]
            camera_scale_grad_norm = (
                torch.linalg.vector_norm(torch.stack(camera_scale_grad_terms))
                if camera_scale_grad_terms
                else objective.new_zeros(())
            )
            camera_scale_head_grad_tensors = (
                sum(
                    parameter.grad is not None
                    for parameter in camera_scale_trainable
                )
                if cfg.train_camera_scale_head_only
                else 0
            )
            camera_scale_head_unused_grad_tensors = (
                len(camera_scale_trainable) - camera_scale_head_grad_tensors
                if cfg.train_camera_scale_head_only
                else 0
            )
            if (
                cfg.train_camera_head_only
                and step == start_step
                and not local_outlier
                and (
                    not lingbot_grad_terms
                    or not torch.isfinite(lingbot_grad_norm)
                    or float(lingbot_grad_norm.detach().item()) <= 0.0
                )
            ):
                raise RuntimeError(
                    "camera pose objective did not produce a finite non-zero "
                    "gradient for the camera head"
                )
            if (
                cfg.train_camera_scale_head_only
                and step == start_step
                and not local_outlier
                and (
                    not camera_scale_grad_terms
                    or not torch.isfinite(camera_scale_grad_norm)
                    or float(camera_scale_grad_norm.detach().item()) <= 0.0
                )
            ):
                raise RuntimeError(
                    "camera pose objective did not produce a finite non-zero "
                    "gradient for the metric scale head"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm, error_if_nonfinite=True)
            if cfg.detailed_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            report_oom(
                "backward",
                forward_ms=forward_ms,
                loss_ms=loss_ms,
                decision_ms=decision_ms,
                backward_elapsed_ms=(time.perf_counter() - backward_t0) * 1000.0,
            )
            raise
        backward_ms += (time.perf_counter() - backward_t0) * 1000.0
        optimizer_t0 = time.perf_counter()
        if global_accepted_samples > 0 and gradients_finite:
            optimizer.step()
            run_optimizer_updates += 1
        else:
            optimizer.zero_grad(set_to_none=True)
        if local_rejected_samples:
            skipped_outliers += local_rejected_samples
            window_skipped_outliers += local_rejected_samples
        run_accepted_samples += global_accepted_samples
        run_rejected_samples += global_rejected_samples
        run_accepted_rank_batches += global_accepted_ranks
        total_outlier_rank_batches += global_outlier_ranks
        window_outlier_rank_batches += global_outlier_ranks
        window_global_accepted_rank_batches += global_accepted_samples
        window_global_accepted_objective_sum += global_accepted_objective_sum
        if global_outlier_ranks > 0:
            total_steps_with_any_outlier += 1
            window_steps_with_any_outlier += 1
        if global_accepted_ranks == 0:
            total_fully_skipped_steps += 1
            window_fully_skipped_steps += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optimizer_ms = (time.perf_counter() - optimizer_t0) * 1000.0
        step_ms = (time.perf_counter() - train_t0) * 1000.0
        iteration_ms = (time.perf_counter() - iteration_t0) * 1000.0
        raw_model = model.module if hasattr(model, "module") else model
        record: Dict[str, Any] = {
            "step": global_step,
            "finetune_step": step,
            "fetch_ms": round(fetch_ms, 2),
            "fetch_attempts": fetch_attempts,
            "pre_step_ms": round(pre_step_ms, 2),
            "forward_ms": round(forward_ms, 2),
            "loss_ms": round(loss_ms, 2),
            "decision_ms": round(decision_ms, 2),
            "backward_ms": round(backward_ms, 2),
            "gradient_sync_ms": round(gradient_sync_ms, 2),
            "optimizer_ms": round(optimizer_ms, 2),
            "step_ms": round(step_ms, 2),
            "iteration_ms": round(iteration_ms, 2),
            "detailed_timing": bool(cfg.detailed_timing),
            "wilor_gradient_checkpointing": bool(cfg.wilor_gradient_checkpointing),
            "hand_bbox_head_enabled": bool(cfg.hand_bbox_head_enabled),
            "mano_box_source": (
                "soft_mask_grad" if cfg.mano_gradient_to_mask_head else "gt"
            ),
            "train_detection_modules_only": bool(
                cfg.train_detection_modules_only
            ),
            "train_probability_mask_heads_only": bool(
                cfg.train_probability_mask_heads_only
            ),
            "train_mano_modules_only": bool(cfg.train_mano_modules_only),
            "train_camera_head_only": bool(cfg.train_camera_head_only),
            "train_camera_scale_head_only": bool(
                cfg.train_camera_scale_head_only
            ),
            "training_objective_scope": (
                "camera_pose"
                if cfg.train_camera_head_only
                else (
                    "camera_pose_scale"
                    if cfg.train_camera_scale_head_only
                    else (
                        "mano"
                        if cfg.train_mano_modules_only
                        else (
                            "probability_mask_heads"
                            if cfg.train_probability_mask_heads_only
                            else (
                                "detection"
                                if cfg.train_detection_modules_only
                                else "joint"
                            )
                        )
                    )
                )
            ),
            "mano_gradient_to_mask_head": bool(
                cfg.mano_gradient_to_mask_head
            ),
            "detection_objective_only": bool(
                (
                    cfg.train_detection_modules_only
                    or cfg.train_probability_mask_heads_only
                )
                and not cfg.mano_gradient_to_mask_head
            ),
            "mask_mano_grad_norm": (
                round(float(mask_mano_grad_norm), 9)
                if mask_mano_grad_norm is not None
                else None
            ),
            "mask_logits_mano_grad_norm": (
                round(float(mask_logits_mano_grad_norm), 9)
                if mask_logits_mano_grad_norm is not None
                else None
            ),
            "lingbot_tune_mode": str(cfg.lingbot_tune_mode),
            "lingbot_gradient_checkpointing": bool(
                cfg.lingbot_gradient_checkpointing
            ),
            "lingbot_gradient_checkpoint_last_n_blocks": int(
                raw_model.backbone.lingbot_gradient_checkpoint_effective_blocks
            ),
            "lingbot_lr": round(
                current_lr * float(cfg.lingbot_lr) / float(cfg.lr), 10
            ) if lingbot_trainable else 0.0,
            "lingbot_grad_norm": round(float(lingbot_grad_norm), 6),
            "camera_head_grad_norm": (
                round(float(lingbot_grad_norm), 6)
                if cfg.train_camera_head_only
                else 0.0
            ),
            "camera_head_grad_tensors": camera_head_grad_tensors,
            "camera_head_unused_grad_tensors": (
                camera_head_unused_grad_tensors
            ),
            "camera_scale_head_grad_norm": round(
                float(camera_scale_grad_norm), 6
            ),
            "camera_scale_head_grad_tensors": (
                camera_scale_head_grad_tensors
            ),
            "camera_scale_head_unused_grad_tensors": (
                camera_scale_head_unused_grad_tensors
            ),
            "grad_norm": round(float(grad_norm), 5),
            "valid_hands": int(target["hand_valid"].sum().item()),
            "num_views": int(images.shape[1]),
            "step_memory_scale_frames": (
                int(cfg.step_memory_scale_frames)
                if cfg.step_memory_enabled
                else 0
            ),
            "memory_before_forward_gb": round(memory_before_forward / (1024**3), 3),
            "memory_before_forward_reserved_gb": round(
                memory_before_forward_reserved / (1024**3), 3
            ),
            "memory_after_forward_gb": round(memory_after_forward / (1024**3), 3),
            "memory_after_forward_reserved_gb": round(
                memory_after_forward_reserved / (1024**3), 3
            ),
            "forward_peak_mem_gb": round(forward_peak_memory / (1024**3), 3),
            "forward_peak_reserved_gb": round(
                forward_peak_reserved_memory / (1024**3), 3
            ),
            "forward_peak_increment_gb": round(
                (forward_peak_memory - memory_before_forward) / (1024**3), 3
            ),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0,
            "peak_reserved_gb": round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 3
            ) if torch.cuda.is_available() else 0.0,
            "lr": current_lr,
            "camera_joint_target_space": cfg.camera_joint_target_space,
            "outlier_skipped": local_outlier,
            "outlier_scope": "sample_sequence",
            "accepted_samples": local_accepted_samples,
            "rejected_samples": local_rejected_samples,
            "accepted_sample_indices": filtered.accepted_indices.tolist(),
            "rejected_sample_indices": filtered.rejected_indices.tolist(),
            "gradient_rejected_sample_indices": filtered.gradient_rejected_indices,
            "sample_objectives": [float(x) if torch.isfinite(x) else None for x in objective_per_sample],
            "sample_replay_count": filtered.replay_count,
            "global_accepted_samples": global_accepted_samples,
            "global_rejected_samples": global_rejected_samples,
            "gradients_finite": gradients_finite,
            "optimizer_updated": bool(global_accepted_samples > 0 and gradients_finite),
            "mano_random_valid_hand": bool(cfg.mano_random_valid_hand),
            "mano_selected_hands": int(fixed_mano_selection.sum()) if fixed_mano_selection is not None else None,
            "timing_forward_includes_loss": True,
            "memory_sampled_after_backward": True,
            "diagnostics_scope": filtered.diagnostics_scope,
            "outlier_threshold": round(outlier_threshold, 6) if outlier_threshold is not None else None,
            "outlier_threshold_mode": outlier_threshold_mode,
            "loss_recent_mean": round(recent_mean, 6) if recent_mean is not None else None,
            "loss_outlier_per_dataset": bool(cfg.loss_outlier_per_dataset),
            "batch_dataset_names": dataset_names,
            "outlier_thresholds_by_dataset": {
                name: round(value, 6)
                for name, value in outlier_thresholds_by_dataset.items()
            },
            "loss_recent_means_by_dataset": {
                name: (
                    round(value, 6) if value is not None else None
                )
                for name, value in recent_means_by_dataset.items()
            },
            "loss_outlier_ratio": round(local_ratio, 4),
            "global_max_loss_outlier_ratio": round(global_max_ratio, 4),
            "global_max_objective": round(global_max_objective, 6),
            "global_outlier_ranks": global_outlier_ranks,
            "global_accepted_ranks": global_accepted_ranks,
            "accepted_rank_backward_scale": round(backward_scale, 6),
            "global_accepted_objective_mean": (
                round(global_accepted_objective_mean, 6)
                if global_accepted_objective_mean is not None
                else None
            ),
            "global_stage2_supervision_valid_ranks": (
                global_stage2_supervision_valid_ranks
            ),
            "global_stage2_fallback_ranks": (
                world_size - global_stage2_supervision_valid_ranks
            ),
            "global_stage2_world_joints_available_ranks": (
                global_stage2_world_joints_available_ranks
            ),
            "global_stage2_world_mano_available_ranks": (
                global_stage2_world_mano_available_ranks
            ),
            "global_stage2_pose_available_ranks": (
                global_stage2_pose_available_ranks
            ),
        }
        memory_stats = getattr(raw_model.backbone, "last_step_memory_stats", {})
        record.update(
            {
                f"lingbot_{key}": value
                for key, value in memory_stats.items()
            }
        )
        if "mano_crop_scale" in output:
            crop_scale = output["mano_crop_scale"].detach().float()
            record.update(
                {
                    "crop_scale_mean": round(float(crop_scale.mean()), 4),
                    "crop_scale_min": round(float(crop_scale.min()), 4),
                    "crop_scale_max": round(float(crop_scale.max()), 4),
                }
            )
        record.update(detector_diagnostics(output, target))
        record.update({key: round(float(value.detach()), 6) for key, value in losses.items()})
        if rank_step_tracer is not None:
            rank_step_tracer.write(
                record,
                batch_identity,
                torch_module=torch,
                device=device,
            )
        history.append(record)
        if not local_outlier:
            window_records.append(record)
        if rank == 0 and local_outlier:
            print(
                f"[skip-outlier] scope=rank0 step={global_step} finetune_step={step} "
                f"objective={objective_value:.6f} "
                f"threshold={threshold_value:.6f} mode={outlier_threshold_mode} "
                f"recent_mean={recent_mean if recent_mean is not None else float('nan'):.6f} "
                f"local_ratio={local_ratio:.3f} max_ratio={global_max_ratio:.3f} "
                f"global_max_objective={global_max_objective:.6f} "
                f"global_outlier_ranks={global_outlier_ranks}/{world_size} "
                f"rank0_skipped_total={skipped_outliers}",
                flush=True,
            )
        elif rank == 0 and (step == 1 or step % cfg.log_interval == 0):
            print("[train] " + json.dumps(record, sort_keys=True), flush=True)
        checkpoint_recent_by_rank = None
        checkpoint_recent_dataset_by_rank = None
        if step % cfg.checkpoint_interval == 0:
            checkpoint_recent_by_rank = [None for _ in range(world_size)]
            dist.all_gather_object(
                checkpoint_recent_by_rank, list(recent_observed_step_means)
            )
            if cfg.loss_outlier_per_dataset:
                checkpoint_recent_dataset_by_rank = [None for _ in range(world_size)]
                dist.all_gather_object(
                    checkpoint_recent_dataset_by_rank,
                    {
                        name: list(values)
                        for name, values in recent_observed_step_means_by_dataset.items()
                    },
                )
        if rank == 0 and step % cfg.checkpoint_interval == 0:
            window_avg_loss = (
                window_global_accepted_objective_sum
                / window_global_accepted_rank_batches
                if window_global_accepted_rank_batches > 0
                else float("inf")
            )
            improved = window_avg_loss < best_window_loss
            if improved:
                best_window_loss = window_avg_loss
            checkpoint_extra = {
                "loss_filter_state": loss_filter.state_dict(),
                "checkpoint_window_loss": window_avg_loss,
                "checkpoint_window_step": global_step,
                "checkpoint_window_finetune_step": step,
                "checkpoint_window_size": window_global_accepted_rank_batches,
                "checkpoint_window_skipped_outliers": window_outlier_rank_batches,
                "checkpoint_window_steps_with_any_outlier": window_steps_with_any_outlier,
                "checkpoint_window_fully_skipped_steps": window_fully_skipped_steps,
                "best_window_loss": best_window_loss,
                "total_skipped_outliers": total_outlier_rank_batches,
                "rank0_window_skipped_outliers": window_skipped_outliers,
                "rank0_total_skipped_outliers": skipped_outliers,
                "finetune_source_checkpoint": cfg.finetune_checkpoint,
                "finetune_source_step": finetune_source_step,
                "resume_source_checkpoint": resume_checkpoint_path,
                "optimizer_restored": optimizer_restored,
                "schedule_step": step,
                "checkpoint_step": step,
                "global_step": global_step,
                "recent_observed_step_means_by_rank": checkpoint_recent_by_rank,
                "recent_observed_step_means_by_dataset": (
                    checkpoint_recent_dataset_by_rank[0]
                    if checkpoint_recent_dataset_by_rank
                    else {
                        name: list(values)
                        for name, values in recent_observed_step_means_by_dataset.items()
                    }
                ),
                "recent_observed_step_means_by_dataset_by_rank": (
                    checkpoint_recent_dataset_by_rank
                    if checkpoint_recent_dataset_by_rank
                    else None
                ),
                "total_outlier_rank_batches": total_outlier_rank_batches,
                "total_steps_with_any_outlier": total_steps_with_any_outlier,
                "total_fully_skipped_steps": total_fully_skipped_steps,
            }
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                global_step,
                cfg,
                history,
                extra=checkpoint_extra,
            )
            print(
                f"[checkpoint-last] step={global_step} finetune_step={step} "
                f"window_avg_objective={window_avg_loss:.6f} "
                f"path={last_checkpoint_path}",
                flush=True,
            )
            if improved:
                atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
                print(
                    f"[checkpoint-best] step={global_step} finetune_step={step} "
                    f"window_avg_objective={window_avg_loss:.6f} "
                    f"best_window_loss={best_window_loss:.6f} "
                    f"accepted={window_global_accepted_rank_batches} "
                    f"skipped={window_outlier_rank_batches} "
                    f"steps_with_any_outlier={window_steps_with_any_outlier} "
                    f"fully_skipped_steps={window_fully_skipped_steps} "
                    f"path={best_checkpoint_path}",
                    flush=True,
                )
            else:
                print(
                    f"[checkpoint-skip] step={global_step} finetune_step={step} "
                    f"window_avg_objective={window_avg_loss:.6f} "
                    f"best_window_loss={best_window_loss:.6f} "
                    f"accepted={window_global_accepted_rank_batches} "
                    f"skipped={window_outlier_rank_batches} "
                    f"steps_with_any_outlier={window_steps_with_any_outlier} "
                    f"fully_skipped_steps={window_fully_skipped_steps}",
                    flush=True,
                )
            window_records = []
            window_skipped_outliers = 0
            window_outlier_rank_batches = 0
            window_steps_with_any_outlier = 0
            window_fully_skipped_steps = 0
            window_global_accepted_objective_sum = 0.0
            window_global_accepted_rank_batches = 0

    final_recent_by_rank = [None for _ in range(world_size)]
    dist.all_gather_object(
        final_recent_by_rank, list(recent_observed_step_means)
    )
    final_recent_dataset_by_rank = None
    if cfg.loss_outlier_per_dataset:
        final_recent_dataset_by_rank = [None for _ in range(world_size)]
        dist.all_gather_object(
            final_recent_dataset_by_rank,
            {
                name: list(values)
                for name, values in recent_observed_step_means_by_dataset.items()
            },
        )
    if rank == 0:
        if (
            cfg.save_final_checkpoint
            and run_end_step % cfg.checkpoint_interval != 0
        ):
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                finetune_step_offset + run_end_step,
                cfg,
                history,
                extra={
                    "loss_filter_state": loss_filter.state_dict(),
                    "best_window_loss": best_window_loss,
                    "finetune_source_checkpoint": cfg.finetune_checkpoint,
                    "finetune_source_step": finetune_source_step,
                    "resume_source_checkpoint": resume_checkpoint_path,
                    "optimizer_restored": optimizer_restored,
                    "finetune_steps": run_end_step,
                    "schedule_step": run_end_step,
                    "checkpoint_step": run_end_step,
                    "global_step": finetune_step_offset + run_end_step,
                    "recent_observed_step_means_by_rank": final_recent_by_rank,
                    "recent_observed_step_means_by_dataset": (
                        final_recent_dataset_by_rank[0]
                        if final_recent_dataset_by_rank
                        else {
                            name: list(values)
                            for name, values in recent_observed_step_means_by_dataset.items()
                        }
                    ),
                    "recent_observed_step_means_by_dataset_by_rank": (
                        final_recent_dataset_by_rank
                        if final_recent_dataset_by_rank
                        else None
                    ),
                    "total_outlier_rank_batches": total_outlier_rank_batches,
                    "total_steps_with_any_outlier": total_steps_with_any_outlier,
                    "total_fully_skipped_steps": total_fully_skipped_steps,
                    "rank0_total_skipped_outliers": skipped_outliers,
                },
            )
        possible_rank_batches = executed_steps * world_size
        with open(output_dir / "smoke_summary.json", "w") as handle:
            json.dump(
                {
                    "status": "ok",
                    "best_checkpoint": str(best_checkpoint_path),
                    "last_checkpoint": str(last_checkpoint_path),
                    "best_window_loss": best_window_loss,
                    "finetune_source_checkpoint": cfg.finetune_checkpoint,
                    "finetune_source_step": finetune_source_step,
                    "resume_source_checkpoint": resume_checkpoint_path,
                    "optimizer_restored": optimizer_restored,
                    "schedule_train_steps": cfg.train_steps,
                    "max_run_steps": cfg.max_run_steps,
                    "final_schedule_step": run_end_step,
                    "final_global_step": finetune_step_offset + run_end_step,
                    "accepted_steps": run_optimizer_updates,
                    "optimizer_updates": run_optimizer_updates,
                    "accepted_samples": run_accepted_samples,
                    "rejected_samples": run_rejected_samples,
                    "loss_filter_state": loss_filter.state_dict(),
                    "skipped_outliers": skipped_outliers,
                    "accepted_rank_batches": run_accepted_rank_batches,
                    "skipped_outlier_rank_batches": total_outlier_rank_batches,
                    "outlier_rank_batch_ratio": (
                        total_outlier_rank_batches / possible_rank_batches
                        if possible_rank_batches > 0
                        else 0.0
                    ),
                    "steps_with_any_outlier": total_steps_with_any_outlier,
                    "fully_skipped_steps": total_fully_skipped_steps,
                    "history": history,
                },
                handle,
                indent=2,
            )
        print(
            f"[done] best_checkpoint={best_checkpoint_path} last_checkpoint={last_checkpoint_path} "
            f"final_global_step={finetune_step_offset + run_end_step} "
            f"best_window_loss={best_window_loss:.6f} "
            f"rank0_accepted_steps={run_optimizer_updates} "
            f"rank0_skipped_outliers={skipped_outliers} "
            f"accepted_rank_batches={run_accepted_rank_batches} "
            f"skipped_rank_batches={total_outlier_rank_batches} "
            f"steps_with_any_outlier={total_steps_with_any_outlier} "
            f"fully_skipped_steps={total_fully_skipped_steps}",
            flush=True,
        )
    if rank_step_tracer is not None:
        rank_step_tracer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
