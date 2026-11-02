from __future__ import annotations

import copy
import os
import subprocess
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fcntl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mutual_hand_joint_fusion import (
    JointHeatmapDescriptorHead,
    JointOffsetHead,
    MutualHandJointFusion,
    decode_camera_translation_lsp,
    project_camera_joints_to_crop,
    uncanonicalize_crop_joints,
    uncanonicalize_crop_offsets,
)



@dataclass
class LingBotMapHandConfig:
    lingbotmap_root: str = "/path/to/infinihand/infinihand/lingbot_map"
    lingbotmap_ckpt: str = ""
    wilor_root: str = "/path/to/infinihand/infinihand/wilor"
    wilor_cfg: str = "/path/to/infinihand/infinihand/wilor/pretrained_models/model_config.yaml"
    wilor_ckpt: str = ""
    load_wilor_checkpoint: bool = True
    mano_model_path: str = "/path/to/weights/mano"
    image_height: int = 378
    image_width: int = 518
    feature_dim: int = 2048
    lingbot_pixel_channels: int = 12
    lingbot_pixel_patch_size: int = 14
    lingbot_mano_patch_size: int = 16
    mano_context_dim: int = 1280
    mano_token_dim: int = 1024
    head_dim: int = 1280
    hand_detection_dim: int = 256
    hand_detection_frames_chunk_size: int = 8
    # Disable the standalone box-regression branch when training uses GT boxes
    # for MANO crops and no box-regression objective is desired.
    mask_only_threshold: float = 0.95
    # Derive the MANO crop from mask logits with differentiable soft moments.
    # This route intentionally avoids threshold/nonzero/min/max so MANO losses
    # can propagate through grid_sample back into the mask head.
    use_soft_mask_boxes_for_mano: bool = False
    # Default off: MANO crops may still use the predicted soft-mask box, but
    # camera/MANO losses do not backpropagate through that box into the mask
    # branch unless this is explicitly enabled.
    mano_gradient_to_mask_head: bool = False
    soft_mask_box_temperature: float = 1.0
    soft_mask_box_std_scale: float = 3.4641016151377544
    soft_mask_box_min_size: float = 0.02
    # Step-local LingBotMap memory. The first scale frames are processed as a
    # bidirectional block; the remaining frames are causal one-frame updates.
    # KV state is always cleared at the beginning and end of every forward.
    step_memory_enabled: bool = False
    step_memory_attention_mode: str = "gca"
    step_memory_scale_frames: int = 4
    step_memory_max_frames: int = 64
    step_memory_sliding_window: int = 64
    wilor_crop_size: int = 256
    wilor_feature_chunk_size: int = 8
    bbox_rescale: float = 1.5
    bbox_scale_factor: float = 0.0
    bbox_rescale_range: Optional[Tuple[float, float]] = (1.5, 1.5)
    freeze_backbone: bool = True
    lingbot_tune_mode: str = "frozen"
    lingbot_tune_last_n_blocks: int = 4
    lingbot_tune_special_tokens: bool = False
    lingbot_gradient_checkpointing: bool = False
    # -1 preserves legacy all-stage checkpointing. Otherwise checkpoint the
    # last N alternating frame+GCA stages; zero requires checkpointing=False.
    lingbot_gradient_checkpoint_last_n_blocks: int = -1
    lingbot_activation_offload: bool = False
    # The hand-only Stage-1 path does not need LingBotMap's camera head.  The
    # Stage-2 world/pose objective enables it so camera tokens receive native
    # absolute/relative pose supervision while the head itself stays frozen.
    lingbot_enable_camera_head: bool = False
    # Optional metric-scale adapter trained on the causal aggregator's scale
    # token while the native camera head and all feature extractors stay frozen.
    camera_scale_head_enabled: bool = False
    camera_scale_head_hidden_dim: int = 512
    camera_scale_head_num_layers: int = 2
    camera_scale_head_num_heads: int = 8
    camera_scale_head_ffn_dim: int = 2048
    camera_scale_head_frames: int = 4
    camera_scale_log_min: float = -4.0
    camera_scale_log_max: float = 4.0
    freeze_wilor_backbone: bool = True
    wilor_gradient_checkpointing: bool = False
    hand_feature_provider: str = "wilor"
    hand_feature_chunk_size: int = 8
    freeze_hand_feature_provider: bool = True
    frozen_backbone_dtype: str = "bfloat16"
    decode_mano: bool = True
    use_gt_boxes_for_mano: bool = True
    model_variant: str = "mutual_joint_fusion"
    joint_heatmap_range_margin: float = 0.15
    mutual_fusion_num_heads: int = 8
    mutual_fusion_dropout: float = 0.0
    pnp_min_visible: int = 1


def materialize_checkpoint(checkpoint: str) -> str:
    """Stage an rclone checkpoint URI into a node-local cache exactly once."""
    if not checkpoint or ":" not in checkpoint or checkpoint.startswith("/"):
        return checkpoint

    cache_root = Path(
        os.environ.get(
            "LINGBOTMAP_CKPT_CACHE_DIR",
            f"/tmp/{os.environ.get('USER', 'user')}/lingbotmap_checkpoints",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / Path(checkpoint.split(":", 1)[1]).name
    lock_path = destination.with_suffix(destination.suffix + ".lock")

    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not destination.is_file() or destination.stat().st_size == 0:
            partial = destination.with_suffix(destination.suffix + f".partial.{os.getpid()}")
            print(f"[checkpoint] staging {checkpoint} -> {destination}", flush=True)
            try:
                subprocess.run(
                    ["rclone", "copyto", checkpoint, str(partial)],
                    check=True,
                )
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return str(destination)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotations to matrices."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(1e-4)
    return torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1).clamp(0.0, 1.0)


def expand_to_square_boxes(
    boxes: torch.Tensor,
    image_hw: Tuple[int, int],
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Expand normalized boxes to square boxes in pixel space, then renormalize."""
    image_h, image_w = image_hw
    center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
    pixel_wh = wh * boxes.new_tensor([image_w, image_h])
    scale_tensor = torch.as_tensor(scale, dtype=boxes.dtype, device=boxes.device)
    if scale_tensor.ndim == boxes.ndim - 1:
        scale_tensor = scale_tensor.unsqueeze(-1)
    side = pixel_wh.amax(dim=-1, keepdim=True) * scale_tensor
    half = 0.5 * side / boxes.new_tensor([image_w, image_h])
    return torch.cat((center - half, center + half), dim=-1)


def virtual_crop_camera_from_boxes(
    boxes_xyxy: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
    crop_hw: Tuple[int, int] = (256, 192),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-hand virtual camera used by the MANO crop target.

    ``boxes_xyxy`` are the normalized, already-expanded square crop boxes used
    by both hand feature branches.  The central 192 columns of each 256 square
    crop are consumed by the MANO head, so the corresponding perspective
    camera has a 256/box-side pixel scale and principal point at the centre of
    the 256x192 crop.  The returned rotation maps image-camera coordinates to
    crop-camera coordinates.
    """
    if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[-2:] != (2, 4):
        raise ValueError(
            "boxes_xyxy must have shape [B,N,2,4], got "
            f"{tuple(boxes_xyxy.shape)}"
        )
    if intrinsics.shape != (*boxes_xyxy.shape[:2], 3, 3):
        raise ValueError(
            "intrinsics must have shape [B,N,3,3], got "
            f"{tuple(intrinsics.shape)} for boxes {tuple(boxes_xyxy.shape)}"
        )
    image_h, image_w = map(float, image_hw)
    crop_h, crop_w = map(float, crop_hw)
    boxes = boxes_xyxy.float()
    camera_k = intrinsics.to(device=boxes.device, dtype=torch.float32)
    center_norm = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    center_px = center_norm * boxes.new_tensor((image_w, image_h))

    pixel_wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-6)
    pixel_wh = pixel_wh * boxes.new_tensor((image_w, image_h))
    side_px = pixel_wh.mean(dim=-1).clamp_min(1.0e-6)
    resize_scale = crop_h / side_px

    crop_k = camera_k[:, :, None].expand(-1, -1, 2, -1, -1).clone()
    crop_k[..., 0, :] *= resize_scale[..., None]
    crop_k[..., 1, :] *= resize_scale[..., None]
    crop_k[..., 0, 2] = 0.5 * (crop_w - 1.0)
    crop_k[..., 1, 2] = 0.5 * (crop_h - 1.0)
    crop_k[..., 2, :] = crop_k.new_tensor((0.0, 0.0, 1.0))

    homogeneous = torch.cat(
        (center_px, torch.ones_like(center_px[..., :1])), dim=-1
    )
    old_from_pixel = torch.linalg.inv(camera_k)[:, :, None]
    ray = torch.matmul(old_from_pixel, homogeneous[..., None]).squeeze(-1)
    z_axis = F.normalize(ray, dim=-1)
    down = z_axis.new_tensor((0.0, 1.0, 0.0)).expand_as(z_axis)
    x_axis = torch.cross(down, z_axis, dim=-1)
    degenerate = x_axis.norm(dim=-1, keepdim=True) < 1.0e-8
    alternate = torch.cross(
        z_axis.new_tensor((1.0, 0.0, 0.0)).expand_as(z_axis),
        z_axis,
        dim=-1,
    )
    x_axis = F.normalize(torch.where(degenerate, alternate, x_axis), dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)
    image_to_crop = torch.stack((x_axis, y_axis, z_axis), dim=-2)
    return crop_k, image_to_crop


def weak_perspective_to_camera_translation(
    pred_cam: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert crop-normalized weak-perspective camera to metric camera translation.

    ``pred_cam=[scale, tx, ty]`` follows this model's crop convention, where
    projected crop coordinates are ``scale * joints_xy + [tx, ty]`` and the
    crop spans ``[-0.5, 0.5]``. Boxes are normalized image-space xyxy boxes.
    The result is ``[Tx, Ty, Tz]`` in the same metric unit as MANO joints.
    """
    if pred_cam.shape[-1] != 3:
        raise ValueError(f"Expected pred_cam [...,3], got {pred_cam.shape}")
    if boxes_xyxy.shape != (*pred_cam.shape[:-1], 4):
        raise ValueError(
            f"boxes_xyxy must have shape {(*pred_cam.shape[:-1], 4)}, "
            f"got {boxes_xyxy.shape}"
        )
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [...,3,3], got {intrinsics.shape}")
    intrinsics_leading = intrinsics.shape[:-2]
    per_view_intrinsics = intrinsics_leading == pred_cam.shape[:-2]
    per_hand_intrinsics = intrinsics_leading == pred_cam.shape[:-1]
    if not (per_view_intrinsics or per_hand_intrinsics):
        raise ValueError(
            "intrinsics must match pred_cam batch/view or batch/view/hand "
            f"dimensions: {intrinsics_leading} vs "
            f"{pred_cam.shape[:-2]} or {pred_cam.shape[:-1]}"
        )

    image_h, image_w = image_hw
    boxes = boxes_xyxy.to(device=pred_cam.device, dtype=pred_cam.dtype)
    camera_k = intrinsics.to(device=pred_cam.device, dtype=pred_cam.dtype)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    box_width = ((x2 - x1) * float(image_w)).clamp_min(eps)
    box_height = ((y2 - y1) * float(image_h)).clamp_min(eps)
    box_center_x = 0.5 * (x1 + x2) * float(image_w)
    box_center_y = 0.5 * (y1 + y2) * float(image_h)

    fx = camera_k[..., 0, 0]
    fy = camera_k[..., 1, 1]
    cx = camera_k[..., 0, 2]
    cy = camera_k[..., 1, 2]
    if per_view_intrinsics:
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
    scale = pred_cam[..., 0].clamp_min(eps)

    # The crop coordinate range is [-0.5, 0.5], hence there is no factor of
    # two in the depth conversion (unlike HMR's [-1, 1] convention).
    depth_x = fx / (box_width * scale)
    depth_y = fy / (box_height * scale)
    tz = 0.5 * (depth_x + depth_y)
    tx = (
        pred_cam[..., 1] + (box_center_x - cx) / box_width
    ) / scale
    ty = (
        pred_cam[..., 2] + (box_center_y - cy) / box_height
    ) / scale
    return torch.stack((tx, ty, tz), dim=-1)


def sample_box_features(
    features: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Differentiably crop one normalized box per feature map with grid_sample."""
    if features.ndim != 4 or boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(f"Expected [M,C,H,W] features and [M,4] boxes, got {features.shape}, {boxes_xyxy.shape}")
    if features.shape[0] != boxes_xyxy.shape[0]:
        raise ValueError("features and boxes must have the same leading dimension")

    out_h, out_w = output_size
    dtype, device = features.dtype, features.device
    y = torch.linspace(0.0, 1.0, out_h, dtype=dtype, device=device)
    x = torch.linspace(0.0, 1.0, out_w, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    gx = x1[:, None, None] + xx * (x2 - x1)[:, None, None]
    gy = y1[:, None, None] + yy * (y2 - y1)[:, None, None]
    grid = torch.stack((gx * 2.0 - 1.0, gy * 2.0 - 1.0), dim=-1)
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def sample_perspective_features(features, homographies, image_hw, output_size):
    """Sample source features using image-pixel to destination-pixel H."""
    oh, ow = output_size
    ih, iw = image_hw
    with torch.autocast(device_type=features.device.type, enabled=False):
        h = homographies.float().reshape(-1, 3, 3)
        yy, xx = torch.meshgrid(
            torch.arange(oh, device=h.device, dtype=torch.float32),
            torch.arange(ow, device=h.device, dtype=torch.float32), indexing="ij")
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        source = torch.einsum("mij,hwj->mhwi", torch.linalg.inv(h), pixels)
        denominator = source[..., 2:]
        safe = torch.where(denominator.abs() > 1e-8, denominator, torch.full_like(denominator, 1e-8))
        uv = source[..., :2] / safe
        grid = uv / uv.new_tensor((iw-1, ih-1)) * 2 - 1
        result = F.grid_sample(features.float(), grid, align_corners=True,
                               mode="bilinear", padding_mode="zeros")
    return result.to(features.dtype)


def sample_one_valid_hand(valid: torch.Tensor) -> torch.Tensor:
    """Choose one valid right/left slot per frame, uniformly if both are valid."""
    if valid.shape[-1] != 2:
        raise ValueError("hand validity must end in the right/left slot dimension")
    valid = valid.bool()
    scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, -1.0)
    chosen = F.one_hot(scores.argmax(-1), num_classes=2).bool()
    return chosen & valid


def scatter_hand_rows(values: torch.Tensor, indices: Optional[torch.Tensor], total: int,
                      identity_rotation: bool = False) -> torch.Tensor:
    """Restore fixed hand slots without adding invalid hands to network batches."""
    if indices is None:
        return values
    shape = (total, *values.shape[1:])
    if identity_rotation:
        result = torch.eye(3, device=values.device, dtype=values.dtype).expand(shape).clone()
    else:
        result = values.new_zeros(shape)
    return result.index_copy(0, indices, values)


def sample_selected_hand_crops(source: torch.Tensor, boxes: torch.Tensor,
                               homographies: Optional[torch.Tensor], image_hw,
                               crop_size: int, indices: torch.Tensor,
                               canonicalize_left: bool = True) -> torch.Tensor:
    """Gather selected frames before sampling; odd flat indices are left hands."""
    channels = source.shape[2]
    if indices.numel() == 0:
        return source.new_empty((0, channels, crop_size, crop_size))
    flat_source = source.reshape(-1, channels, *source.shape[-2:])
    selected_source = flat_source.index_select(0, indices.div(2, rounding_mode="floor"))
    if homographies is None:
        crops = sample_box_features(selected_source, boxes.reshape(-1, 4).index_select(0, indices),
                                    (crop_size, crop_size))
    else:
        selected_h = homographies.reshape(-1, 3, 3).index_select(0, indices)
        crops = sample_perspective_features(selected_source, selected_h, image_hw,
                                            (crop_size, crop_size))
    if canonicalize_left:
        left = indices.remainder(2).bool()[:, None, None, None]
        crops = torch.where(left, crops.flip(-1), crops)
    return crops


def soft_mask_logits_to_boxes(
    mask_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    std_scale: float = 3.4641016151377544,
    min_size: float = 0.02,
) -> torch.Tensor:
    """Convert mask logits to normalized xyxy boxes without breaking autograd.

    A low-temperature sigmoid approximates the usual 0.5 mask threshold while
    remaining continuous.  For a filled rectangular mask, ``sqrt(12) * std``
    recovers its width/height, which is why sqrt(12) is the default scale.
    """
    if mask_logits.ndim != 5:
        raise ValueError(
            "soft_mask_logits_to_boxes expects [B,N,2,H,W], got "
            f"{tuple(mask_logits.shape)}"
        )
    if mask_logits.shape[2] != 2:
        raise ValueError(
            "soft_mask_logits_to_boxes expects exactly two hand slots, got "
            f"{mask_logits.shape[2]}"
        )
    if temperature <= 0.0 or std_scale <= 0.0:
        raise ValueError("temperature and std_scale must be positive")
    if not 0.0 < min_size <= 1.0:
        raise ValueError("min_size must be in (0, 1]")

    source_dtype = mask_logits.dtype
    logits = mask_logits.float()
    probability = torch.sigmoid(logits / float(temperature))
    height, width = probability.shape[-2:]
    x = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
    y = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
    mass = probability.sum(dim=(-1, -2)).clamp_min(1.0e-8)
    center_x = (probability * x.view(1, 1, 1, 1, width)).sum(
        dim=(-1, -2)
    ) / mass
    center_y = (probability * y.view(1, 1, 1, height, 1)).sum(
        dim=(-1, -2)
    ) / mass
    variance_x = (
        probability
        * (x.view(1, 1, 1, 1, width) - center_x[..., None, None]).square()
    ).sum(dim=(-1, -2)) / mass
    variance_y = (
        probability
        * (y.view(1, 1, 1, height, 1) - center_y[..., None, None]).square()
    ).sum(dim=(-1, -2)) / mass
    minimum_variance = (float(min_size) / float(std_scale)) ** 2
    box_width = float(std_scale) * variance_x.clamp_min(minimum_variance).sqrt()
    box_height = float(std_scale) * variance_y.clamp_min(minimum_variance).sqrt()
    boxes = torch.stack(
        (
            center_x - 0.5 * box_width,
            center_y - 0.5 * box_height,
            center_x + 0.5 * box_width,
            center_y + 0.5 * box_height,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    return boxes.to(dtype=source_dtype)


class LingBotMapBackbone(nn.Module):
    def __init__(self, cfg: LingBotMapHandConfig) -> None:
        super().__init__()
        from infinihand.lingbot_map.models.gct_stream_window import GCTStream

        self.patch_size = 14
        self.step_memory_enabled = bool(cfg.step_memory_enabled)
        self.step_memory_attention_mode = str(
            cfg.step_memory_attention_mode
        ).strip().lower()
        self.step_memory_scale_frames = int(cfg.step_memory_scale_frames)
        self.step_memory_max_frames = int(cfg.step_memory_max_frames)
        self.step_memory_sliding_window = int(cfg.step_memory_sliding_window)
        self.lingbot_tune_mode = str(cfg.lingbot_tune_mode).strip().lower()
        self.lingbot_tune_last_n_blocks = int(cfg.lingbot_tune_last_n_blocks)
        self.lingbot_tune_special_tokens = bool(cfg.lingbot_tune_special_tokens)
        self.lingbot_gradient_checkpointing = bool(
            cfg.lingbot_gradient_checkpointing
        )
        self.lingbot_gradient_checkpoint_last_n_blocks = int(
            cfg.lingbot_gradient_checkpoint_last_n_blocks
        )
        self.lingbot_activation_offload = bool(cfg.lingbot_activation_offload)
        self.lingbot_enable_camera_head = bool(cfg.lingbot_enable_camera_head)
        valid_tune_modes = {
            "frozen",
            "last_n_gca_attention",
            "last_n_gca_blocks",
        }
        if self.lingbot_tune_mode not in valid_tune_modes:
            raise ValueError(
                f"lingbot_tune_mode must be one of {sorted(valid_tune_modes)}, "
                f"got {cfg.lingbot_tune_mode!r}"
            )
        if self.lingbot_tune_last_n_blocks <= 0:
            raise ValueError("lingbot_tune_last_n_blocks must be positive")
        if self.lingbot_gradient_checkpointing and self.lingbot_tune_mode == "frozen":
            raise ValueError(
                "lingbot_gradient_checkpointing requires a trainable LingBotMap mode"
            )
        if self.lingbot_gradient_checkpoint_last_n_blocks < -1:
            raise ValueError(
                "lingbot_gradient_checkpoint_last_n_blocks must be -1 or non-negative"
            )
        if (
            not self.lingbot_gradient_checkpointing
            and self.lingbot_gradient_checkpoint_last_n_blocks not in (-1, 0)
        ):
            raise ValueError(
                "positive lingbot_gradient_checkpoint_last_n_blocks requires "
                "lingbot_gradient_checkpointing=true"
            )
        if (
            self.lingbot_gradient_checkpointing
            and self.lingbot_gradient_checkpoint_last_n_blocks == 0
        ):
            raise ValueError(
                "lingbot_gradient_checkpointing=true requires checkpoint blocks "
                "to be -1 or positive"
            )
        if self.step_memory_scale_frames <= 0:
            raise ValueError("step_memory_scale_frames must be positive")
        if self.step_memory_max_frames < self.step_memory_scale_frames:
            raise ValueError(
                "step_memory_max_frames must be >= step_memory_scale_frames"
            )
        if self.step_memory_sliding_window <= 0:
            raise ValueError("step_memory_sliding_window must be positive")
        if self.step_memory_enabled and self.step_memory_attention_mode != "gca":
            raise ValueError(
                "step-local memory currently supports attention_mode='gca' only"
            )
        expected_frozen = self.lingbot_tune_mode == "frozen"
        if bool(cfg.freeze_backbone) != expected_frozen:
            raise ValueError(
                "freeze_backbone must agree with lingbot_tune_mode: "
                f"freeze_backbone={cfg.freeze_backbone}, "
                f"lingbot_tune_mode={self.lingbot_tune_mode!r}"
            )
        self.model = GCTStream(
            img_size=max(cfg.image_height, cfg.image_width),
            patch_size=self.patch_size,
            embed_dim=1024,
            enable_camera=self.lingbot_enable_camera_head,
            enable_point=False,
            enable_local_point=False,
            # Materialize the pretrained depth DPT only long enough to copy
            # it into the trainable hand-detection branch.
            enable_depth=True,
            enable_track=False,
            enable_stream_inference=self.step_memory_enabled,
            # Lock the native LingBotMap GCA contract explicitly instead of
            # relying on third-party constructor defaults: full scale/anchor
            # frames, a dense recent window, and all special tokens retained
            # from evicted trajectory frames.
            attend_to_special_tokens=True,
            attend_to_scale_frames=True,
            num_frame_for_scale=self.step_memory_scale_frames,
            max_frame_num=self.step_memory_max_frames,
            kv_cache_sliding_window=self.step_memory_sliding_window,
            kv_cache_scale_frames=self.step_memory_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            kv_cache_camera_only=False,
            use_sdpa=True,
            use_gradient_checkpoint=self.lingbot_gradient_checkpointing,
        )
        if cfg.lingbotmap_ckpt:
            checkpoint_path = materialize_checkpoint(cfg.lingbotmap_ckpt)
            payload = torch.load(checkpoint_path, map_location="cpu")
            state = payload.get("model", payload.get("state_dict", payload))
            incompatible = self.model.load_state_dict(state, strict=False)
            print(
                f"[LingBotMapBackbone] loaded {cfg.lingbotmap_ckpt} via {checkpoint_path}; "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )
        self.freeze_backbone = bool(cfg.freeze_backbone)
        self._trainable_gca_modules: List[nn.Module] = []
        self._configure_lingbot_gradient_checkpointing()
        self._configure_lingbot_tuning()
        self.clean_kv_cache_each_forward = os.environ.get("LINGBOTMAP_CLEAN_KV_CACHE", "1") != "0"
        self.last_step_memory_stats: Dict[str, object] = {}
        if self.step_memory_enabled:
            self._assert_native_gca_configuration()

    def _configure_lingbot_gradient_checkpointing(self) -> None:
        aggregator = getattr(self.model, "aggregator", None)
        global_blocks = getattr(aggregator, "global_blocks", None)
        if not isinstance(global_blocks, nn.ModuleList):
            raise RuntimeError("LingBotMap aggregator has no global_blocks ModuleList")
        total_blocks = len(global_blocks)
        requested = self.lingbot_gradient_checkpoint_last_n_blocks
        if not self.lingbot_gradient_checkpointing:
            effective = 0
        elif requested == -1:
            effective = total_blocks
        else:
            effective = requested
        if effective > total_blocks:
            raise ValueError(
                f"requested checkpointing for last {effective} blocks, but model "
                f"has only {total_blocks}"
            )
        self.lingbot_gradient_checkpoint_effective_blocks = effective
        aggregator.gradient_checkpoint_last_n_blocks = effective
        aggregator.gradient_checkpoint_start_block = total_blocks - effective

    def _configure_lingbot_tuning(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()
        if self.lingbot_tune_mode == "frozen":
            self.lingbot_trainable_parameters = 0
            return

        aggregator = getattr(self.model, "aggregator", None)
        blocks = getattr(aggregator, "global_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError("LingBotMap aggregator has no global_blocks ModuleList")
        if self.lingbot_tune_last_n_blocks > len(blocks):
            raise ValueError(
                f"requested last {self.lingbot_tune_last_n_blocks} GCA blocks, "
                f"but model has only {len(blocks)}"
            )
        selected_blocks = list(blocks[-self.lingbot_tune_last_n_blocks :])
        if self.lingbot_tune_mode == "last_n_gca_attention":
            for block in selected_blocks:
                for module_name in ("norm1", "attn", "ls1"):
                    module = getattr(block, module_name, None)
                    if module is None:
                        raise RuntimeError(
                            f"GCA block is missing attention module {module_name!r}"
                        )
                    module.requires_grad_(True)
                    self._trainable_gca_modules.append(module)
        else:
            for block in selected_blocks:
                block.requires_grad_(True)
                self._trainable_gca_modules.append(block)

        if self.lingbot_tune_special_tokens:
            for token_name in ("camera_token", "register_token", "scale_token"):
                token = getattr(aggregator, token_name, None)
                if not isinstance(token, nn.Parameter):
                    raise RuntimeError(f"LingBotMap aggregator is missing {token_name}")
                token.requires_grad_(True)
        self.lingbot_trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(
            "[LingBotMapBackbone] tune_mode="
            f"{self.lingbot_tune_mode} last_n={self.lingbot_tune_last_n_blocks} "
            f"special_tokens={self.lingbot_tune_special_tokens} "
            f"checkpointing={self.lingbot_gradient_checkpointing} "
            f"checkpoint_last_n={self.lingbot_gradient_checkpoint_effective_blocks} "
            f"trainable_params={self.lingbot_trainable_parameters}",
            flush=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        if mode and not self.freeze_backbone:
            for module in self._trainable_gca_modules:
                module.train(True)
        return self

    def forward(self, images: torch.Tensor, *, run_camera: bool | None = None) -> "LingBotMapBackboneOutput":
        if images.ndim != 5:
            raise ValueError(f"LingBotMap expects [B,N,3,H,W], got {images.shape}")
        run_camera = self.lingbot_enable_camera_head if run_camera is None else bool(run_camera and self.lingbot_enable_camera_head)
        images = self._prepare_images(images)
        camera_head = getattr(self.model, "camera_head", None)
        camera_head_trainable = (
            run_camera
            and isinstance(camera_head, nn.Module)
            and any(parameter.requires_grad for parameter in camera_head.parameters())
        )
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        deferred_camera_outputs = None
        with context:
            activation_context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=True, device_type="cuda"
                )
                if self.lingbot_activation_offload
                and not self.freeze_backbone
                and images.is_cuda
                else nullcontext()
            )
            with activation_context:
                if self.step_memory_enabled:
                    outputs, patch_start_idx, pose_enc = self._forward_step_memory(
                        images, run_camera=run_camera
                    )
                else:
                    if self.clean_kv_cache_each_forward and hasattr(self.model, "clean_kv_cache"):
                        self.model.clean_kv_cache()
                    outputs, patch_start_idx = self.model._aggregate_features(images)
                    if camera_head_trainable:
                        # Keep the aggregator frozen and graph-free while
                        # allowing gradients only through the native camera
                        # head.  Detaching here also makes the scope explicit
                        # if a caller accidentally changes the outer context.
                        deferred_camera_outputs = [
                            output.detach() for output in outputs
                        ]
                        pose_enc = None
                    elif run_camera:
                        pose_output = self.model._predict_camera(
                            outputs,
                            causal_inference=False,
                            num_frame_for_scale=int(images.shape[1]),
                            num_frame_per_block=int(images.shape[1]),
                        )
                        pose_enc = pose_output.get("pose_enc")
                    else:
                        pose_enc = None
        if deferred_camera_outputs is not None:
            pose_output = self.model._predict_camera(
                deferred_camera_outputs,
                causal_inference=False,
                num_frame_for_scale=int(images.shape[1]),
                num_frame_per_block=int(images.shape[1]),
            )
            pose_enc = pose_output.get("pose_enc")
        if (
            not self.step_memory_enabled
            and self.clean_kv_cache_each_forward
            and hasattr(self.model, "clean_kv_cache")
        ):
            self.model.clean_kv_cache()
        return self._pack_backbone_output(images, outputs, patch_start_idx, pose_enc)

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        images = images.to(dtype=model_dtype)
        if images.detach().amax() > 1.5:
            images = images / 255.0
        return images.clamp(0.0, 1.0)

    def _pack_backbone_output(
        self,
        images: torch.Tensor,
        outputs: List[torch.Tensor],
        patch_start_idx: int,
        pose_enc: Optional[torch.Tensor],
    ) -> "LingBotMapBackboneOutput":
        tokens = outputs[-1][:, :, patch_start_idx:, :]
        b, n, p, c = tokens.shape
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        if p != h * w:
            raise RuntimeError(f"LingBotMap returned {p} patch tokens, expected {h}x{w}={h*w}")
        features = tokens.transpose(-1, -2).reshape(b, n, c, h, w)
        return LingBotMapBackboneOutput(
            patch_features=features,
            aggregated_tokens=outputs,
            patch_start_idx=int(patch_start_idx),
            pose_enc=pose_enc,
        )

    def begin_inference_stream(
        self,
        anchor_images: torch.Tensor,
    ) -> "LingBotMapBackboneOutput":
        """Start one persistent GCA stream from the initial anchor frames."""
        if not self.step_memory_enabled:
            raise RuntimeError("begin_inference_stream requires step_memory_enabled")
        if anchor_images.ndim != 5:
            raise ValueError(
                f"anchor_images must be [B,N,3,H,W], got {anchor_images.shape}"
            )
        if hasattr(self, "_inference_stream_state"):
            raise RuntimeError("an inference stream is already active")
        scale_frames = int(anchor_images.shape[1])
        if scale_frames != self.step_memory_scale_frames:
            raise ValueError(
                f"expected exactly {self.step_memory_scale_frames} anchors, got {scale_frames}"
            )
        images = self._prepare_images(anchor_images)
        self._clean_step_memory_cache()
        aggregator = getattr(self.model, "aggregator", None)
        if aggregator is not None:
            aggregator.gradient_checkpoint_frame_calls = 0
            aggregator.gradient_checkpoint_global_calls = 0
        outputs, patch_start_idx = self.model._aggregate_features(
            images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
        )
        pose_enc = None
        if self.lingbot_enable_camera_head:
            pose = self.model._predict_camera(
                outputs,
                causal_inference=True,
                num_frame_for_scale=scale_frames,
                sliding_window_size=self.step_memory_sliding_window,
                num_frame_per_block=scale_frames,
            )
            pose_enc = pose["pose_enc"]
        peak_cache_bytes = self._kv_cache_bytes() + self._camera_kv_cache_bytes()
        self._inference_stream_state = {
            "scale_frames": scale_frames,
            "processed_frames": scale_frames,
            "patch_start_idx": int(patch_start_idx),
            "tap_count": len(outputs),
            "peak_cache_bytes": peak_cache_bytes,
        }
        self._update_step_memory_stats(
            scale_frames=scale_frames,
            total_frames=scale_frames,
            peak_cache_bytes=peak_cache_bytes,
            cache_cleared=False,
            reset_stats=True,
        )
        return self._pack_backbone_output(images, outputs, patch_start_idx, pose_enc)

    def inference_stream_step(
        self,
        frame_image: torch.Tensor,
    ) -> "LingBotMapBackboneOutput":
        """Advance an active GCA stream by exactly one frame."""
        state = getattr(self, "_inference_stream_state", None)
        if state is None:
            raise RuntimeError("begin_inference_stream must be called first")
        if frame_image.ndim != 5 or int(frame_image.shape[1]) != 1:
            raise ValueError(
                f"frame_image must be [B,1,3,H,W], got {frame_image.shape}"
            )
        images = self._prepare_images(frame_image)
        outputs, patch_start_idx = self.model._aggregate_features(
            images,
            num_frame_for_scale=int(state["scale_frames"]),
            num_frame_per_block=1,
        )
        if int(patch_start_idx) != int(state["patch_start_idx"]):
            raise RuntimeError(
                "LingBotMap patch_start_idx changed within one stream: "
                f"{state['patch_start_idx']} -> {patch_start_idx}"
            )
        if len(outputs) != int(state["tap_count"]):
            raise RuntimeError(
                "LingBotMap output tap count changed within one stream: "
                f"{state['tap_count']} -> {len(outputs)}"
            )
        pose_enc = None
        if self.lingbot_enable_camera_head:
            pose = self.model._predict_camera(
                outputs,
                causal_inference=True,
                num_frame_for_scale=int(state["scale_frames"]),
                sliding_window_size=self.step_memory_sliding_window,
                num_frame_per_block=1,
            )
            pose_enc = pose["pose_enc"]
        state["processed_frames"] = int(state["processed_frames"]) + 1
        state["peak_cache_bytes"] = max(
            int(state["peak_cache_bytes"]),
            self._kv_cache_bytes() + self._camera_kv_cache_bytes(),
        )
        self._update_step_memory_stats(
            scale_frames=int(state["scale_frames"]),
            total_frames=int(state["processed_frames"]),
            peak_cache_bytes=int(state["peak_cache_bytes"]),
            cache_cleared=False,
        )
        return self._pack_backbone_output(images, outputs, patch_start_idx, pose_enc)

    def end_inference_stream(self) -> Dict[str, object]:
        """Close the active stream and return its audited cache statistics."""
        state = getattr(self, "_inference_stream_state", None)
        if state is None:
            raise RuntimeError("no inference stream is active")
        self._clean_step_memory_cache()
        self._update_step_memory_stats(
            scale_frames=int(state["scale_frames"]),
            total_frames=int(state["processed_frames"]),
            peak_cache_bytes=int(state["peak_cache_bytes"]),
            cache_cleared=True,
        )
        del self._inference_stream_state
        return dict(self.last_step_memory_stats)

    def _kv_cache_bytes(self) -> int:
        aggregator = getattr(self.model, "aggregator", None)
        cache = getattr(aggregator, "kv_cache", None)
        if not isinstance(cache, dict):
            return 0
        return sum(
            value.numel() * value.element_size()
            for value in cache.values()
            if torch.is_tensor(value)
        )

    def _camera_kv_cache_bytes(self) -> int:
        camera_head = getattr(self.model, "camera_head", None)
        cache = getattr(camera_head, "kv_cache", None)
        if not isinstance(cache, list):
            return 0
        return sum(
            value.numel() * value.element_size()
            for cache_dict in cache
            if isinstance(cache_dict, dict)
            for value in cache_dict.values()
            if torch.is_tensor(value)
        )

    def _assert_native_gca_configuration(self) -> None:
        aggregator = getattr(self.model, "aggregator", None)
        expected = {
            "kv_cache_scale_frames": self.step_memory_scale_frames,
            "kv_cache_sliding_window": self.step_memory_sliding_window,
            "kv_cache_cross_frame_special": True,
            "kv_cache_include_scale_frames": True,
            "kv_cache_camera_only": False,
        }
        actual = {
            key: getattr(aggregator, key, None)
            for key in expected
        }
        if actual != expected:
            raise RuntimeError(
                f"LingBotMap native GCA configuration mismatch: "
                f"expected={expected}, actual={actual}"
            )

    def _gca_cache_frame_counts(self) -> Tuple[int, int]:
        """Return dense-frame and evicted-trajectory-frame cache counts."""
        aggregator = getattr(self.model, "aggregator", None)
        cache = getattr(aggregator, "kv_cache", None)
        if not isinstance(cache, dict):
            return 0, 0
        dense_frames = 0
        trajectory_frames = 0
        for key, value in cache.items():
            if not (
                key.startswith("k_")
                and not key.endswith("_special")
                and torch.is_tensor(value)
                and value.ndim == 5
            ):
                continue
            dense_frames = max(dense_frames, int(value.shape[2]))
            special = cache.get(f"{key}_special")
            if torch.is_tensor(special) and special.ndim == 5:
                trajectory_frames = max(
                    trajectory_frames, int(special.shape[2])
                )
        return dense_frames, trajectory_frames

    def _clean_step_memory_cache(self) -> None:
        if self.lingbot_enable_camera_head and hasattr(
            self.model, "clean_kv_cache"
        ):
            self.model.clean_kv_cache()
            return
        aggregator = getattr(self.model, "aggregator", None)
        if aggregator is not None and hasattr(aggregator, "clean_kv_cache"):
            # The hand wrapper disables the camera head, so clean the native
            # aggregator directly instead of emitting a camera-cache warning.
            aggregator.clean_kv_cache()
            return
        if hasattr(self.model, "clean_kv_cache"):
            self.model.clean_kv_cache()
            return
        raise RuntimeError("LingBotMap model does not expose a KV-cache cleaner")

    def _update_step_memory_stats(
        self,
        *,
        scale_frames: int,
        total_frames: int,
        peak_cache_bytes: int,
        cache_cleared: bool,
        reset_stats: bool = False,
    ) -> None:
        dense_frames, trajectory_frames = self._gca_cache_frame_counts()
        previous = {} if reset_stats else self.last_step_memory_stats
        peak_dense_frames = max(
            int(previous.get("peak_dense_cache_frames", 0)), dense_frames
        )
        peak_trajectory_frames = max(
            int(previous.get("peak_trajectory_memory_frames", 0)),
            trajectory_frames,
        )
        self.last_step_memory_stats = {
            "enabled": True,
            "attention_mode": self.step_memory_attention_mode,
            "tune_mode": self.lingbot_tune_mode,
            "trainable_parameters": self.lingbot_trainable_parameters,
            "activation_offload": self.lingbot_activation_offload,
            "gradient_checkpointing": self.lingbot_gradient_checkpointing,
            "gradient_checkpoint_last_n_blocks": (
                self.lingbot_gradient_checkpoint_effective_blocks
            ),
            "gradient_checkpoint_frame_calls": int(
                getattr(
                    getattr(self.model, "aggregator", None),
                    "gradient_checkpoint_frame_calls",
                    0,
                )
            ),
            "gradient_checkpoint_global_calls": int(
                getattr(
                    getattr(self.model, "aggregator", None),
                    "gradient_checkpoint_global_calls",
                    0,
                )
            ),
            "tune_special_tokens": self.lingbot_tune_special_tokens,
            "scale_frames": scale_frames,
            "sliding_window_frames": self.step_memory_sliding_window,
            "dense_capacity_frames": (
                scale_frames + self.step_memory_sliding_window
            ),
            "total_frames": total_frames,
            "streamed_frames": total_frames - scale_frames,
            "peak_dense_cache_frames": peak_dense_frames,
            "peak_trajectory_memory_frames": peak_trajectory_frames,
            "gca_eviction_observed": peak_trajectory_frames > 0,
            "peak_kv_cache_bytes": peak_cache_bytes,
            "peak_kv_cache_gb": round(peak_cache_bytes / (1024**3), 4),
            "cache_cleared_at_step_end": cache_cleared,
        }

    def _forward_step_memory(
        self,
        images: torch.Tensor,
        *, run_camera: bool | None = None,
    ) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor]]:
        """Run one sequence with native LingBotMap KV memory, scoped to this step."""
        run_camera = self.lingbot_enable_camera_head if run_camera is None else bool(run_camera and self.lingbot_enable_camera_head)
        total_frames = int(images.shape[1])
        if total_frames > self.step_memory_max_frames:
            raise ValueError(
                f"step-local memory received {total_frames} frames, maximum is "
                f"{self.step_memory_max_frames}"
            )
        scale_frames = min(self.step_memory_scale_frames, total_frames)
        self._clean_step_memory_cache()
        aggregator = getattr(self.model, "aggregator", None)
        if aggregator is not None:
            aggregator.gradient_checkpoint_frame_calls = 0
            aggregator.gradient_checkpoint_global_calls = 0
        output_chunks: List[List[torch.Tensor]] = []
        pose_chunks: List[torch.Tensor] = []
        patch_start_idx: Optional[int] = None
        peak_cache_bytes = 0
        camera_head = getattr(self.model, "camera_head", None)
        camera_head_trainable = (
            run_camera
            and isinstance(camera_head, nn.Module)
            and any(
                parameter.requires_grad
                for parameter in camera_head.parameters()
            )
        )

        def predict_camera(
            chunk_outputs: List[torch.Tensor],
            *,
            num_frame_per_block: int,
        ) -> Dict[str, torch.Tensor]:
            # Keep the aggregator and its KV cache frozen/no-grad. Re-enter
            # autograd only for the native camera head; its causal KV cache
            # intentionally remains in the graph until this sequence's single
            # loss/backward. Detaching also makes the optimizer boundary
            # explicit if the outer context changes later.
            camera_inputs = (
                [output.detach() for output in chunk_outputs]
                if camera_head_trainable
                else chunk_outputs
            )
            camera_context = (
                torch.enable_grad()
                if camera_head_trainable
                else nullcontext()
            )
            with camera_context:
                return self.model._predict_camera(
                    camera_inputs,
                    causal_inference=True,
                    num_frame_for_scale=scale_frames,
                    sliding_window_size=self.step_memory_sliding_window,
                    num_frame_per_block=num_frame_per_block,
                )
        self._update_step_memory_stats(
            scale_frames=scale_frames,
            total_frames=total_frames,
            peak_cache_bytes=peak_cache_bytes,
            cache_cleared=False,
            reset_stats=True,
        )
        try:
            scale_outputs, scale_patch_start = self.model._aggregate_features(
                images[:, :scale_frames],
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
            )
            output_chunks.append(scale_outputs)
            if run_camera:
                scale_pose = predict_camera(
                    scale_outputs,
                    num_frame_per_block=scale_frames,
                )
                pose_chunks.append(scale_pose["pose_enc"])
            patch_start_idx = int(scale_patch_start)
            peak_cache_bytes = max(
                peak_cache_bytes,
                self._kv_cache_bytes() + self._camera_kv_cache_bytes(),
            )
            self._update_step_memory_stats(
                scale_frames=scale_frames,
                total_frames=total_frames,
                peak_cache_bytes=peak_cache_bytes,
                cache_cleared=False,
            )

            for frame_index in range(scale_frames, total_frames):
                frame_outputs, frame_patch_start = self.model._aggregate_features(
                    images[:, frame_index : frame_index + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                )
                if int(frame_patch_start) != patch_start_idx:
                    raise RuntimeError(
                        "LingBotMap patch_start_idx changed within one sequence: "
                        f"{patch_start_idx} -> {frame_patch_start}"
                    )
                if len(frame_outputs) != len(scale_outputs):
                    raise RuntimeError(
                        "LingBotMap output tap count changed within one sequence: "
                        f"{len(scale_outputs)} -> {len(frame_outputs)}"
                    )
                output_chunks.append(frame_outputs)
                if run_camera:
                    frame_pose = predict_camera(
                        frame_outputs,
                        num_frame_per_block=1,
                    )
                    pose_chunks.append(frame_pose["pose_enc"])
                peak_cache_bytes = max(
                    peak_cache_bytes,
                    self._kv_cache_bytes() + self._camera_kv_cache_bytes(),
                )
                self._update_step_memory_stats(
                    scale_frames=scale_frames,
                    total_frames=total_frames,
                    peak_cache_bytes=peak_cache_bytes,
                    cache_cleared=False,
                )

            outputs = [
                torch.cat([chunk[tap_index] for chunk in output_chunks], dim=1)
                for tap_index in range(len(scale_outputs))
            ]
            if pose_chunks:
                pose_context = (
                    torch.enable_grad()
                    if camera_head_trainable
                    else nullcontext()
                )
                with pose_context:
                    pose_enc = torch.cat(pose_chunks, dim=1)
            else:
                pose_enc = None
            return outputs, patch_start_idx, pose_enc
        finally:
            # This is the sequence boundary. KV state must never leak into the
            # next optimizer step, including on exceptions/OOM.
            self._clean_step_memory_cache()
            self._update_step_memory_stats(
                scale_frames=scale_frames,
                total_frames=total_frames,
                peak_cache_bytes=peak_cache_bytes,
                cache_cleared=True,
            )


@dataclass
class LingBotMapBackboneOutput:
    """Final LingBot patch map plus the intermediate DPT layer taps."""

    patch_features: torch.Tensor
    aggregated_tokens: List[torch.Tensor]
    patch_start_idx: int
    pose_enc: Optional[torch.Tensor] = None


@dataclass
class HandFeatureBatch:
    """Canonical fixed-slot hand features returned by a hand feature provider.

    Tensor contract:
      features: [B, N, 2, C, Hf, Wf]
      canonical_crops: [B, N, 2, 3, crop_size, crop_size]
      boxes: [B, N, 2, 4] normalized xyxy in the input image

    Slot 0 is the right hand. Slot 1 is the left hand, horizontally flipped
    into the canonical right-hand orientation when ``canonicalize_left`` is on.
    """

    features: torch.Tensor
    canonical_crops: torch.Tensor
    boxes: torch.Tensor


class HandFeatureProvider(nn.Module, ABC):
    """Stable full-image -> fixed-slot hand-feature interface.

    New feature backbones only need to implement ``encode_crops``. Cropping,
    fixed right/left slot ordering, optional left-hand canonicalization,
    chunking and output-shape validation stay shared here.
    """

    def __init__(
        self,
        *,
        crop_size: int,
        output_dim: int,
        output_hw: Tuple[int, int],
        chunk_size: int,
        canonicalize_left: bool = True,
    ) -> None:
        super().__init__()
        self.crop_size = int(crop_size)
        self.output_dim = int(output_dim)
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.chunk_size = max(1, int(chunk_size))
        self.canonicalize_left = bool(canonicalize_left)

    @abstractmethod
    def encode_crops(self, canonical_crops: torch.Tensor) -> torch.Tensor:
        """Encode [M,3,S,S] canonical crops as [M,C,Hf,Wf]."""

    def _canonical_hand_crops(self, images: torch.Tensor, boxes: torch.Tensor, crop_homographies=None) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"HandFeatureProvider expects images [B,N,3,H,W], got {images.shape}")
        if boxes.ndim != 4 or boxes.shape[2:] != (2, 4):
            raise ValueError(f"HandFeatureProvider expects boxes [B,N,2,4], got {boxes.shape}")
        if images.shape[:2] != boxes.shape[:2]:
            raise ValueError(
                f"images and boxes must share [B,N], got {images.shape[:2]} and {boxes.shape[:2]}"
            )
        if not torch.isfinite(boxes).all():
            raise ValueError("HandFeatureProvider boxes contain non-finite values")

        b, n = images.shape[:2]
        source = images.reshape(b * n, 3, *images.shape[-2:]).repeat_interleave(2, dim=0)
        if crop_homographies is None:
            crops = sample_box_features(source, boxes.reshape(-1, 4).to(images),
                                        (self.crop_size, self.crop_size))
        else:
            crops = sample_perspective_features(source, crop_homographies,
                        images.shape[-2:], (self.crop_size, self.crop_size))
        crops = crops.reshape(b * n, 2, 3, self.crop_size, self.crop_size)
        if self.canonicalize_left:
            crops = torch.stack((crops[:, 0], crops[:, 1].flip(-1)), dim=1)
        return crops.reshape(b, n, 2, 3, self.crop_size, self.crop_size)

    def forward(self, images: torch.Tensor, boxes: torch.Tensor, crop_homographies=None,
                hand_indices: Optional[torch.Tensor] = None) -> HandFeatureBatch:
        # Selection mode returns compact rows. The caller scatters only for losses/debug outputs.
        if hand_indices is not None:
            crops = sample_selected_hand_crops(images, boxes, crop_homographies, images.shape[-2:],
                                                self.crop_size, hand_indices, self.canonicalize_left)
            if hand_indices.numel() == 0:
                features = images.new_empty((0, self.output_dim, *self.output_hw))
            else:
                features = torch.cat([self.encode_crops(chunk) for chunk in crops.split(self.chunk_size)], 0)
            expected = (hand_indices.numel(), self.output_dim, *self.output_hw)
            if tuple(features.shape) != expected:
                raise RuntimeError(f"Hand crop features have shape {features.shape}, expected {expected}")
            return HandFeatureBatch(features=features, canonical_crops=crops, boxes=boxes)
        crops = self._canonical_hand_crops(images, boxes, crop_homographies)
        b, n = crops.shape[:2]
        flat_crops = crops.reshape(b * n * 2, 3, self.crop_size, self.crop_size)
        feature_chunks = [self.encode_crops(chunk) for chunk in flat_crops.split(self.chunk_size, dim=0)]
        flat_features = torch.cat(feature_chunks, dim=0)
        expected = (b * n * 2, self.output_dim, *self.output_hw)
        if tuple(flat_features.shape) != expected:
            raise RuntimeError(
                f"{type(self).__name__}.encode_crops returned {tuple(flat_features.shape)}, expected {expected}"
            )
        features = flat_features.reshape(b, n, 2, self.output_dim, *self.output_hw)
        return HandFeatureBatch(features=features, canonical_crops=crops, boxes=boxes)


class WiLoRHandFeatureProvider(HandFeatureProvider):
    """WiLoR crop ViT exposed through the shared hand-feature contract."""

    def __init__(self, cfg: LingBotMapHandConfig) -> None:
        crop_size = int(cfg.wilor_crop_size)
        output_hw = (crop_size // 16, (crop_size - 64) // 16)
        super().__init__(
            crop_size=crop_size,
            output_dim=int(cfg.mano_context_dim),
            output_hw=output_hw,
            chunk_size=int(cfg.wilor_feature_chunk_size),
            canonicalize_left=True,
        )
        from infinihand.wilor.configs import get_config
        from infinihand.wilor.models.backbones import create_backbone

        model_cfg = get_config(cfg.wilor_cfg, update_cachedir=True)
        model_cfg.defrost()
        if "vit" in model_cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in model_cfg.MODEL:
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.MANO.DATA_DIR = str(Path(cfg.mano_model_path))
        model_cfg.MANO.MODEL_PATH = str(Path(cfg.mano_model_path))
        model_cfg.MANO.MEAN_PARAMS = str(Path(cfg.mano_model_path) / "mano_mean_params.npz")
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

        self.backbone = create_backbone(model_cfg)
        if cfg.load_wilor_checkpoint:
            payload = torch.load(cfg.wilor_ckpt, map_location="cpu")
            state = payload.get("state_dict", payload)
            backbone_state = {
                key[len("backbone.") :]: value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
            incompatible = self.backbone.load_state_dict(backbone_state, strict=False)
            print(
                f"[WiLoRHandFeatureProvider] loaded {cfg.wilor_ckpt}; keys={len(backbone_state)} "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )
        else:
            print("[WiLoRHandFeatureProvider] checkpoint initialization disabled", flush=True)
        self.freeze_backbone = bool(cfg.freeze_wilor_backbone)
        self.gradient_checkpointing = bool(cfg.wilor_gradient_checkpointing)
        if self.gradient_checkpointing and self.freeze_backbone:
            raise ValueError(
                "wilor_gradient_checkpointing=true requires freeze_wilor_backbone=false"
            )
        if not hasattr(self.backbone, "use_checkpoint"):
            raise AttributeError(
                "WiLoR backbone does not expose the expected use_checkpoint flag"
            )
        self.backbone.use_checkpoint = self.gradient_checkpointing
        print(
            "[WiLoRHandFeatureProvider] gradient_checkpointing="
            f"{self.gradient_checkpointing}",
            flush=True,
        )
        self.register_buffer("image_mean", torch.tensor(model_cfg.MODEL.IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(model_cfg.MODEL.IMAGE_STD).view(1, 3, 1, 1), persistent=False)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode_crops(self, canonical_crops: torch.Tensor) -> torch.Tensor:
        backbone_dtype = next(self.backbone.parameters()).dtype
        crops = canonical_crops.to(dtype=backbone_dtype)
        if crops.detach().amax() > 1.5:
            crops = crops / 255.0
        if crops.shape[-2:] != (self.crop_size, self.crop_size):
            crops = F.interpolate(crops, (self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        crops = (crops.clamp(0.0, 1.0) - self.image_mean) / self.image_std
        crops = crops[..., 32:-32]
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            _, _, _, spatial_features = self.backbone(crops)
        return spatial_features


# Import compatibility for callers that used the previous class name. Its
# forward now follows the full-image HandFeatureProvider contract above.
WiLoRFeatureBackbone = WiLoRHandFeatureProvider


def _apply_dpt_uv_position_embedding(
    x: torch.Tensor,
    image_hw: Tuple[int, int],
    ratio: float = 0.1,
) -> torch.Tensor:
    """Apply LingBotMap DPT's exact UV positional embedding."""
    from infinihand.lingbot_map.heads.utils import create_uv_grid, position_grid_to_embed

    image_h, image_w = image_hw
    patch_h, patch_w = x.shape[-2:]
    pos = create_uv_grid(
        patch_w,
        patch_h,
        aspect_ratio=image_w / image_h,
        dtype=x.dtype,
        device=x.device,
    )
    pos = position_grid_to_embed(pos, x.shape[1]) * ratio
    return x + pos.permute(2, 0, 1).unsqueeze(0)


class HandDetectionBlock(nn.Module):
    """Depth-DPT feature preparation and coarse-to-fine RefineNet fusion.

    Every module in this block is deep-copied from LingBotMap's pretrained
    depth head.  The depth prediction layers themselves are deliberately not
    included: they become the separately and randomly initialized
    :class:`HandMaskHead` below.
    """

    def __init__(self, depth_head: nn.Module, frames_chunk_size: int = 8) -> None:
        super().__init__()
        if depth_head is None:
            raise ValueError(
                "HandDetectionBlock requires the pretrained LingBotMap depth_head"
            )
        self.patch_size = int(depth_head.patch_size)
        self.pos_embed = bool(depth_head.pos_embed)
        self.intermediate_layer_idx = tuple(depth_head.intermediate_layer_idx)
        self.frames_chunk_size = int(frames_chunk_size)
        if self.frames_chunk_size <= 0:
            raise ValueError("hand_detection_frames_chunk_size must be positive")

        # Stages 3 and 4 of the diagram: normalization/projection/scale
        # preparation followed by four-level DPT RefineNet fusion.
        self.norm = copy.deepcopy(depth_head.norm)
        self.projects = copy.deepcopy(depth_head.projects)
        self.resize_layers = copy.deepcopy(depth_head.resize_layers)
        self.layer_rn = nn.ModuleList(
            [
                copy.deepcopy(depth_head.scratch.layer1_rn),
                copy.deepcopy(depth_head.scratch.layer2_rn),
                copy.deepcopy(depth_head.scratch.layer3_rn),
                copy.deepcopy(depth_head.scratch.layer4_rn),
            ]
        )
        self.refinenet = nn.ModuleList(
            [
                copy.deepcopy(depth_head.scratch.refinenet1),
                copy.deepcopy(depth_head.scratch.refinenet2),
                copy.deepcopy(depth_head.scratch.refinenet3),
                copy.deepcopy(depth_head.scratch.refinenet4),
            ]
        )

        source_modules = [
            depth_head.norm,
            *depth_head.projects,
            *depth_head.resize_layers,
            depth_head.scratch.layer1_rn,
            depth_head.scratch.layer2_rn,
            depth_head.scratch.layer3_rn,
            depth_head.scratch.layer4_rn,
            depth_head.scratch.refinenet1,
            depth_head.scratch.refinenet2,
            depth_head.scratch.refinenet3,
            depth_head.scratch.refinenet4,
        ]
        copied_modules = [
            self.norm,
            *self.projects,
            *self.resize_layers,
            *self.layer_rn,
            *self.refinenet,
        ]
        max_abs_diff = 0.0
        for source_module, copied_module in zip(source_modules, copied_modules):
            source_state = source_module.state_dict()
            copied_state = copied_module.state_dict()
            if source_state.keys() != copied_state.keys():
                raise RuntimeError("depth-to-hand module state keys do not match")
            for key, source_tensor in source_state.items():
                difference = (
                    source_tensor.detach().float()
                    - copied_state[key].detach().float()
                ).abs().max()
                max_abs_diff = max(max_abs_diff, float(difference.item()))
        if max_abs_diff != 0.0:
            raise RuntimeError(
                f"depth-to-hand parameter copy is not exact: max_abs_diff={max_abs_diff}"
            )
        copied_tensors = len(self.state_dict())
        copied_parameters = sum(parameter.numel() for parameter in self.parameters())
        print(
            "[hand-detection-init] source=lingbot_depth_head "
            f"copied_tensors={copied_tensors} copied_parameters={copied_parameters} "
            f"max_abs_diff={max_abs_diff:.1f} "
            "stages=dpt_feature_preparation+refinenet_fusion",
            flush=True,
        )

    def _forward_chunk(
        self,
        aggregated_tokens: List[torch.Tensor],
        patch_start_idx: int,
        image_hw: Tuple[int, int],
        frame_start: int,
        frame_end: int,
    ) -> torch.Tensor:
        image_h, image_w = image_hw
        patch_h, patch_w = image_h // self.patch_size, image_w // self.patch_size
        prepared = []
        batch_size = aggregated_tokens[0].shape[0]
        frame_count = frame_end - frame_start
        block_dtype = self.norm.weight.dtype

        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens[layer_idx][
                :, frame_start:frame_end, patch_start_idx:
            ]
            if x.shape[2] != patch_h * patch_w:
                raise RuntimeError(
                    f"DPT layer {layer_idx} has {x.shape[2]} patches; "
                    f"expected {patch_h}x{patch_w}={patch_h * patch_w}"
                )
            x = x.reshape(batch_size * frame_count, -1, x.shape[-1])
            x = self.norm(x.to(dtype=block_dtype))
            x = x.permute(0, 2, 1).reshape(
                batch_size * frame_count, x.shape[-1], patch_h, patch_w
            )
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = _apply_dpt_uv_position_embedding(x, image_hw)
            prepared.append(self.resize_layers[dpt_idx](x))

        layer1, layer2, layer3, layer4 = [
            projection(feature)
            for projection, feature in zip(self.layer_rn, prepared)
        ]
        out = self.refinenet[3](layer4, size=layer3.shape[2:])
        out = self.refinenet[2](out, layer3, size=layer2.shape[2:])
        out = self.refinenet[1](out, layer2, size=layer1.shape[2:])
        out = self.refinenet[0](out, layer1)
        return out.view(batch_size, frame_count, *out.shape[1:])

    def forward(
        self,
        aggregated_tokens: List[torch.Tensor],
        patch_start_idx: int,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if len(aggregated_tokens) < len(self.intermediate_layer_idx):
            raise ValueError(
                "HandDetectionBlock requires four LingBot layer taps, got "
                f"{len(aggregated_tokens)}"
            )
        frame_count = aggregated_tokens[0].shape[1]
        chunks = [
            self._forward_chunk(
                aggregated_tokens,
                patch_start_idx,
                image_hw,
                start,
                min(start + self.frames_chunk_size, frame_count),
            )
            for start in range(0, frame_count, self.frames_chunk_size)
        ]
        return torch.cat(chunks, dim=1)


class HandMaskHead(nn.Module):
    """Randomly initialized copy of the dense Depth prediction head."""

    def __init__(self, feature_dim: int = 256, output_slots: int = 2) -> None:
        super().__init__()
        self.output_conv1 = nn.Conv2d(
            feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1
        )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_slots, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self, features: torch.Tensor, image_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                f"HandMaskHead expects [B,N,C,H,W], got {tuple(features.shape)}"
            )
        batch_size, frame_count, channels, height, width = features.shape
        x = features.reshape(batch_size * frame_count, channels, height, width)
        x = self.output_conv1(x)
        x = F.interpolate(x, size=image_hw, mode="bilinear", align_corners=True)
        x = _apply_dpt_uv_position_embedding(x, image_hw)
        x = self.output_conv2(x)
        return x.view(batch_size, frame_count, 2, *image_hw)


class LingBotFeatureAdapter(nn.Module):
    """Convert LingBotMap patch features into WiLoR-shaped hand context features."""

    def __init__(self, cfg: LingBotMapHandConfig) -> None:
        super().__init__()
        self.pixel_channels = int(cfg.lingbot_pixel_channels)
        self.pixel_patch_size = int(cfg.lingbot_pixel_patch_size)
        self.mano_patch_size = int(cfg.lingbot_mano_patch_size)
        self.crop_size = int(cfg.wilor_crop_size)
        self.crop_width = 192
        if self.crop_size % self.mano_patch_size != 0 or self.crop_width % self.mano_patch_size != 0:
            raise ValueError("WiLoR crop dimensions must be divisible by the MANO feature patch size")
        self.output_hw = (
            self.crop_size // self.mano_patch_size,
            self.crop_width // self.mano_patch_size,
        )
        lifted_dim = self.pixel_channels * self.pixel_patch_size * self.pixel_patch_size
        patch_dim = self.pixel_channels * self.mano_patch_size * self.mano_patch_size
        self.patch_to_pixels = nn.Conv2d(cfg.feature_dim, lifted_dim, 1)
        self.patch_to_context = nn.Conv2d(patch_dim, cfg.mano_context_dim, 1)

    def _unpatchify_to_pixel_image(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(f"Expected [B,N,C,H,W] LingBot features, got {features.shape}")
        b, n, c, h, w = features.shape
        x = self.patch_to_pixels(features.reshape(b * n, c, h, w))
        p = self.pixel_patch_size
        x = x.reshape(b * n, self.pixel_channels, p, p, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.reshape(b, n, self.pixel_channels, h * p, w * p)

    def _patchify_mano_crop(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 4:
            raise ValueError(f"Expected [M,C,256,192] LingBot hand crops, got {crops.shape}")
        m, c, h, w = crops.shape
        p = self.mano_patch_size
        gh, gw = h // p, w // p
        x = crops.reshape(m, c, gh, p, gw, p)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.reshape(m, c * p * p, gh, gw)

    def forward(self, features: torch.Tensor, boxes: torch.Tensor, crop_homographies=None, image_hw=None,
                hand_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n = features.shape[:2]
        if hand_indices is not None and hand_indices.numel() == 0:
            return (features.new_empty((0, self.patch_to_context.out_channels, *self.output_hw)),
                    features.new_empty((0, self.pixel_channels, self.crop_size, self.crop_size)))
        pixel_features = self._unpatchify_to_pixel_image(features)
        if hand_indices is not None:
            crops = sample_selected_hand_crops(pixel_features, boxes, crop_homographies, image_hw,
                                                self.crop_size, hand_indices)
            return self.patch_to_context(self._patchify_mano_crop(crops[..., 32:-32])), crops
        _, _, c, h, w = pixel_features.shape
        source = pixel_features.reshape(b * n, c, h, w).repeat_interleave(2, dim=0)
        if crop_homographies is None:
            crops = sample_box_features(source, boxes.reshape(-1, 4), (self.crop_size, self.crop_size))
        else:
            crops = sample_perspective_features(source, crop_homographies,
                        image_hw, (self.crop_size, self.crop_size))
        crops = crops.reshape(b * n, 2, c, self.crop_size, self.crop_size)
        crops = torch.stack((crops[:, 0], crops[:, 1].flip(-1)), dim=1)
        crops = crops.reshape(b * n * 2, c, self.crop_size, self.crop_size)
        crops_192 = crops[..., 32:-32]
        patchified = self._patchify_mano_crop(crops_192)
        context = self.patch_to_context(patchified)
        return context, crops


class HandFeatureFusion(nn.Module):
    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(2 * context_dim, context_dim, 1)

    def forward(self, lingbot_features: torch.Tensor, wilor_features: torch.Tensor) -> torch.Tensor:
        if lingbot_features.shape[-2:] != wilor_features.shape[-2:]:
            raise ValueError(
                f"LingBot and WiLoR contexts must share spatial shape, got "
                f"{lingbot_features.shape[-2:]} and {wilor_features.shape[-2:]}"
            )
        return self.fuse(torch.cat((lingbot_features, wilor_features), dim=1))


class _HaWoRFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HaWoRAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(b, n, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        ]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class _HaWoRCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        _, m, _ = context.shape
        q = self.to_q(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class _HaWoRTransformerDecoder(nn.Module):
    """Fallback matching HaWoR's one-query TransformerDecoder interface."""

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ) -> None:
        super().__init__()
        if norm != "layer" or norm_cond_dim != -1:
            raise ValueError("Fallback HaWoR decoder supports layer norm only")
        if emb_dropout_type not in {"drop", "normal"}:
            raise ValueError(f"Unsupported embedding dropout type: {emb_dropout_type}")
        if skip_token_embedding and token_dim != dim:
            raise ValueError(f"token_dim ({token_dim}) != dim ({dim})")
        self.to_token_embedding = nn.Identity() if skip_token_embedding else nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        context_dim = context_dim or dim
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        _HaWoRAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.LayerNorm(dim),
                        _HaWoRCrossAttention(
                            dim,
                            context_dim=context_dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        nn.LayerNorm(dim),
                        _HaWoRFeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        inp: torch.Tensor,
        *args,
        context: Optional[torch.Tensor] = None,
        context_list: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.dropout(self.to_token_embedding(inp))
        _, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        if context_list is None:
            if context is None:
                raise ValueError("HaWoR decoder requires context or context_list")
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(layers) ({len(context_list)} != {len(self.layers)})")
        for layer, layer_context in zip(self.layers, context_list):
            norm_sa, self_attn, norm_ca, cross_attn, norm_ff, ff = layer
            x = self_attn(norm_sa(x)) + x
            x = cross_attn(norm_ca(x), layer_context) + x
            x = ff(norm_ff(x)) + x
        return x


class HaWoRQueryMANOHead(nn.Module):
    """HaWoR-style one-query transformer MANO head."""

    def __init__(
        self,
        cfg: LingBotMapHandConfig,
        camera_mode: str = "weak_perspective",
    ) -> None:
        super().__init__()
        if camera_mode not in {"weak_perspective", "log_z"}:
            raise ValueError(f"Unsupported MANO camera_mode={camera_mode!r}")
        self.camera_mode = camera_mode
        TransformerDecoder = _HaWoRTransformerDecoder

        transformer_args = dict(
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=int(cfg.mano_context_dim),
            num_tokens=1,
            token_dim=1,
            dim=int(cfg.mano_token_dim),
        )
        self.transformer = TransformerDecoder(**transformer_args)
        dim = int(cfg.mano_token_dim)
        self.decpose = nn.Linear(dim, 16 * 6)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3 if camera_mode == "weak_perspective" else 1)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        if camera_mode == "weak_perspective":
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        else:
            nn.init.zeros_(self.deccam.weight)
            nn.init.constant_(self.deccam.bias, float(np.log(0.5)))

        mean_params = np.load(Path(cfg.mano_model_path) / "mano_mean_params.npz")
        self.register_buffer("init_hand_pose", torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0))
        self.register_buffer("init_betas", torch.from_numpy(mean_params["shape"].astype(np.float32)).unsqueeze(0))
        self.register_buffer("init_cam", torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0))

    def encode_hand_token(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.shape[0]
        context_tokens = context.flatten(2).transpose(1, 2).contiguous()
        query = context.new_zeros(batch_size, 1, 1)
        return self.transformer(query, context=context_tokens)

    def decode_hand_token(self, hand_token: torch.Tensor) -> Dict[str, torch.Tensor]:
        token_out = hand_token.squeeze(1)
        batch_size = token_out.shape[0]
        pose6d = self.decpose(token_out) + self.init_hand_pose.expand(batch_size, -1)
        betas = self.decshape(token_out) + self.init_betas.expand(batch_size, -1)
        rot6d = pose6d.reshape(-1, 16, 6)
        rotations = rotation_6d_to_matrix(rot6d)
        result = {
            "global_orient": rotations[:, :1],
            "hand_pose": rotations[:, 1:],
            "betas": betas,
        }
        if self.camera_mode == "weak_perspective":
            camera = self.deccam(token_out) + self.init_cam.expand(batch_size, -1)
            result["pred_cam"] = torch.cat(
                (F.softplus(camera[:, :1]) + 1e-4, camera[:, 1:]),
                dim=-1,
            )
        else:
            # deccam already carries the log-depth bias; init_cam belongs only
            # to the legacy weak-perspective parameterization.
            result["log_z"] = self.deccam(token_out)
        return result

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decode_hand_token(self.encode_hand_token(context))


class LingBotMapHand(nn.Module):
    """Shared hand and camera model for training and streaming inference."""

    def __init__(
        self,
        cfg: Optional[LingBotMapHandConfig] = None,
        hand_feature_provider: Optional[HandFeatureProvider] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LingBotMapHandConfig()
        self.model_variant = str(self.cfg.model_variant).strip().lower()
        if self.model_variant not in {"legacy", "mutual_joint_fusion"}:
            raise ValueError(
                f"model_variant must be legacy or mutual_joint_fusion, got "
                f"{self.cfg.model_variant!r}"
            )
        if self.model_variant == "mutual_joint_fusion" and not self.cfg.decode_mano:
            raise ValueError("mutual_joint_fusion requires decode_mano=True")
        self._bbox_rescale_range = self._validate_bbox_rescale_range()
        if self._bbox_rescale_range is None:
            print(
                "[LingBotMapHand] bbox_rescale_range=null; using legacy "
                f"bbox_rescale={self.cfg.bbox_rescale} "
                f"bbox_scale_factor={self.cfg.bbox_scale_factor}",
                flush=True,
            )
        else:
            scale_min, scale_max = self._bbox_rescale_range
            train_mode = "fixed" if scale_min == scale_max else "uniform"
            print(
                "[LingBotMapHand] "
                f"bbox_rescale_range=[{scale_min}, {scale_max}] "
                f"train_mode={train_mode} eval_scale={(scale_min + scale_max) * 0.5}",
                flush=True,
            )
        self.backbone = LingBotMapBackbone(self.cfg)
        self.cfg.camera_scale_head_enabled = False
        self.camera_scale_head = None  # Train camera translation directly in meters.
        if self.cfg.hand_detection_dim != 256:
            raise ValueError(
                "LingBot depth DPT fusion emits 256 channels; "
                f"hand_detection_dim must be 256, got {self.cfg.hand_detection_dim}"
            )
        depth_head = self.backbone.model.depth_head
        self.hand_detection_block = HandDetectionBlock(
            depth_head,
            frames_chunk_size=self.cfg.hand_detection_frames_chunk_size,
        )
        # The source belongs to the frozen backbone. The copied branch is the
        # trainable hand detector.
        self.hand_detection_block.requires_grad_(True)
        self.backbone.model.depth_head = None
        self.hand_mask_head = HandMaskHead(self.cfg.hand_detection_dim)
        self.lingbot_adapter = LingBotFeatureAdapter(self.cfg)
        self.hand_encoder = hand_feature_provider or self._build_hand_feature_provider()
        if not isinstance(self.hand_encoder, HandFeatureProvider):
            raise TypeError(
                f"hand_feature_provider must implement HandFeatureProvider, got {type(self.hand_encoder).__name__}"
            )
        if self.hand_encoder.output_dim != self.cfg.mano_context_dim:
            raise ValueError(
                f"Hand feature provider output_dim={self.hand_encoder.output_dim} must match "
                f"mano_context_dim={self.cfg.mano_context_dim}"
            )
        expected_hw = self.lingbot_adapter.output_hw
        if self.hand_encoder.output_hw != expected_hw:
            raise ValueError(
                f"Hand feature provider output_hw={self.hand_encoder.output_hw} must match {expected_hw}"
            )
        frozen_dtype = getattr(torch, self.cfg.frozen_backbone_dtype)
        # Match the paper's bfloat16 training path for partial GCA tuning.
        self.backbone.to(dtype=frozen_dtype)
        # Native LingBotMap explicitly converts camera-token features to FP32
        # and disables autocast in _predict_camera.  Keep the frozen camera
        # head in FP32 as well; casting it with the BF16 aggregator otherwise
        # makes its first LayerNorm reject the FP32 token input.
        camera_head = getattr(self.backbone.model, "camera_head", None)
        if camera_head is not None:
            camera_head.float()
        if self.cfg.freeze_backbone:
            print(f"[LingBotMapHand] frozen LingBotMap backbone dtype={frozen_dtype}", flush=True)
        else:
            print(
                "[LingBotMapHand] trainable LingBotMap "
                f"mode={self.cfg.lingbot_tune_mode} dtype={frozen_dtype}",
                flush=True,
            )
        self.feature_fusion = HandFeatureFusion(self.cfg.mano_context_dim)
        self.mano_head = HaWoRQueryMANOHead(
            self.cfg,
            camera_mode=(
                "log_z"
                if self.model_variant == "mutual_joint_fusion"
                else "weak_perspective"
            ),
        )
        self.joint_head = None
        self.mutual_fusion = None
        self.joint_offset_head = None
        if self.model_variant == "mutual_joint_fusion":
            self.joint_head = JointHeatmapDescriptorHead(
                input_dim=int(self.cfg.mano_context_dim),
                descriptor_dim=int(self.cfg.mano_token_dim),
                n_joints=21,
                range_margin=float(self.cfg.joint_heatmap_range_margin),
            )
            self.mutual_fusion = MutualHandJointFusion(
                dim=int(self.cfg.mano_token_dim),
                num_heads=int(self.cfg.mutual_fusion_num_heads),
                dropout=float(self.cfg.mutual_fusion_dropout),
            )
            self.joint_offset_head = JointOffsetHead(
                dim=int(self.cfg.mano_token_dim),
            )
        self.register_buffer("_mano_translation_convention", torch.tensor(1, dtype=torch.int64))
        self.mano_right = None
        self.mano_left = None
        if self.cfg.decode_mano:
            self._build_mano_layers()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Require the wrist-depth convention for training and inference weights."""
        version = state_dict.get("_mano_translation_convention")
        if version is None or int(version.item()) != 1:
            raise RuntimeError("Checkpoint must declare wrist-depth convention version 1.")
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            if assign:
                raise
            return super().load_state_dict(state_dict, strict=strict)

    def _validate_bbox_rescale_range(self) -> Optional[Tuple[float, float]]:
        value = self.cfg.bbox_rescale_range
        if value is None:
            if not np.isfinite(self.cfg.bbox_rescale) or self.cfg.bbox_rescale <= 0:
                raise ValueError(f"bbox_rescale must be finite and positive, got {self.cfg.bbox_rescale}")
            if not np.isfinite(self.cfg.bbox_scale_factor) or self.cfg.bbox_scale_factor < 0:
                raise ValueError(
                    "bbox_scale_factor must be finite and non-negative, "
                    f"got {self.cfg.bbox_scale_factor}"
                )
            return None
        if len(value) != 2:
            raise ValueError(f"bbox_rescale_range must contain [min, max], got {value}")
        scale_min, scale_max = (float(value[0]), float(value[1]))
        if not np.isfinite(scale_min) or not np.isfinite(scale_max):
            raise ValueError(f"bbox_rescale_range must be finite, got {value}")
        if scale_min <= 0 or scale_min > scale_max:
            raise ValueError(
                "bbox_rescale_range must satisfy 0 < min <= max, "
                f"got [{scale_min}, {scale_max}]"
            )
        return scale_min, scale_max

    def _sample_bbox_rescale(self, boxes: torch.Tensor) -> torch.Tensor:
        if self._bbox_rescale_range is None:
            crop_scale = boxes.new_full(boxes.shape[:-1], float(self.cfg.bbox_rescale))
            if self.training and self.cfg.bbox_scale_factor > 0:
                scale_jitter = torch.randn_like(crop_scale).clamp(-1.0, 1.0)
                scale_jitter = scale_jitter * float(self.cfg.bbox_scale_factor) + 1.0
                crop_scale = crop_scale * scale_jitter
            return crop_scale

        scale_min, scale_max = self._bbox_rescale_range
        if self.training and scale_min < scale_max:
            return boxes.new_empty(boxes.shape[:-1]).uniform_(scale_min, scale_max)
        return boxes.new_full(boxes.shape[:-1], (scale_min + scale_max) * 0.5)

    @property
    def hand_feature_provider(self) -> HandFeatureProvider:
        """Active crop-level hand feature provider."""
        return self.hand_encoder

    def _build_hand_feature_provider(self) -> HandFeatureProvider:
        name = str(self.cfg.hand_feature_provider).strip().lower()
        if name == "wilor":
            self.cfg.freeze_wilor_backbone = self.cfg.freeze_hand_feature_provider
            self.cfg.wilor_feature_chunk_size = self.cfg.hand_feature_chunk_size
            return WiLoRHandFeatureProvider(self.cfg)

        raise ValueError(
            f"Unsupported hand_feature_provider={self.cfg.hand_feature_provider!r}; "
            "the release model uses wilor"
        )

    def _build_mano_layers(self) -> None:
        from infinihand.wilor.models.mano_wrapper import MANO

        kwargs = dict(
            model_path=self.cfg.mano_model_path,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.mano_right = MANO(is_rhand=True, **kwargs)
        self.mano_left = MANO(is_rhand=False, **kwargs)

    def _decode_mano(
        self,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        mano_left_basis_b: Optional[torch.Tensor] = None,
        selected_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=global_orient.device.type, enabled=False):
            global_orient, hand_pose, betas = global_orient.float(), hand_pose.float(), betas.float()
            if self.mano_right is None or self.mano_left is None:
                raise RuntimeError("MANO decoding requested but MANO layers are not initialized")
            joints, vertices = [], []
            if selected_valid is not None:
                selected_valid = selected_valid.reshape(-1, 2).bool()
                for slot, layer in enumerate((self.mano_right, self.mano_left)):
                    ids = selected_valid[:, slot].nonzero(as_tuple=False).flatten()
                    side_joints = betas.new_zeros((betas.shape[0], 21, 3))
                    side_vertices = betas.new_zeros((betas.shape[0], 778, 3))
                    if ids.numel():
                        # A/B bases are evaluated only for samples that actually use that basis.
                        basis = torch.zeros_like(ids, dtype=torch.bool)
                        if slot == 1 and mano_left_basis_b is not None:
                            basis = mano_left_basis_b.to(betas.device).bool().reshape(-1).index_select(0, ids)
                        for use_b in (False, True):
                            subset = ids[basis == use_b]
                            if not subset.numel():
                                continue
                            kwargs = dict(global_orient=global_orient[subset, slot],
                                          hand_pose=hand_pose[subset, slot], betas=betas[subset, slot], pose2rot=False)
                            if use_b:
                                dirs = layer.shapedirs.clone()
                                dirs[:, 0, :] = -dirs[:, 0, :]
                                decoded = torch.func.functional_call(layer, {"shapedirs": dirs}, (), kwargs)
                            else:
                                decoded = layer(**kwargs)
                            wrist = decoded.joints[:, :1]
                            side_joints = side_joints.index_copy(0, subset, decoded.joints - wrist)
                            side_vertices = side_vertices.index_copy(0, subset, decoded.vertices - wrist)
                    joints.append(side_joints)
                    vertices.append(side_vertices)
                return torch.stack(joints, 1), torch.stack(vertices, 1)
            for slot, layer in enumerate((self.mano_right, self.mano_left)):
                out = layer(
                    global_orient=global_orient[:, slot],
                    hand_pose=hand_pose[:, slot],
                    betas=betas[:, slot],
                    pose2rot=False,
                )
                if slot == 1 and mano_left_basis_b is not None:
                    # Functional buffer override avoids mutating shared model/checkpoint assets.
                    use_b = mano_left_basis_b.to(device=betas.device, dtype=torch.bool).reshape(-1)
                    if use_b.numel() != betas.shape[0]:
                        raise ValueError("MANO basis flags must match flattened frames")
                    dirs = layer.shapedirs.clone()
                    dirs[:, 0, :] = -dirs[:, 0, :]
                    alt = torch.func.functional_call(layer, {"shapedirs": dirs}, (), dict(
                        global_orient=global_orient[:, slot], hand_pose=hand_pose[:, slot],
                        betas=betas[:, slot], pose2rot=False))
                    out.joints = torch.where(use_b[:, None, None], alt.joints, out.joints)
                    out.vertices = torch.where(use_b[:, None, None], alt.vertices, out.vertices)
                wrist = out.joints[:, :1, :]
                joints.append(out.joints - wrist)
                vertices.append(out.vertices - wrist)
            return torch.stack(joints, dim=1), torch.stack(vertices, dim=1)

    def _forward_mutual_joint_fusion(
        self,
        output: Dict[str, torch.Tensor],
        fused_features: torch.Tensor,
        batch_size: int,
        num_views: int,
        mano_crop_intrinsics: Optional[torch.Tensor],
        mano_crop_image_hw: Optional[Tuple[int, int]],
        mano_left_basis_b: Optional[torch.Tensor] = None,
        hand_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if (
            self.joint_head is None
            or self.mutual_fusion is None
            or self.joint_offset_head is None
        ):
            raise RuntimeError("mutual_joint_fusion modules are not initialized")
        if mano_crop_intrinsics is None:
            raise ValueError(
                "mutual_joint_fusion requires mano_crop_intrinsics for the "
                "256x192 virtual crop camera"
            )
        crop_hw = (
            tuple(int(value) for value in mano_crop_image_hw)
            if mano_crop_image_hw is not None
            else None
        )
        if crop_hw != (256, 192):
            raise ValueError(
                "mutual_joint_fusion only supports mano_crop_image_hw=(256, 192), "
                f"got {crop_hw}"
            )
        expected_intrinsics = (batch_size, num_views, 2, 3, 3)
        if tuple(mano_crop_intrinsics.shape) != expected_intrinsics:
            raise ValueError(
                f"mano_crop_intrinsics must be {expected_intrinsics}, got "
                f"{tuple(mano_crop_intrinsics.shape)}"
            )

        if fused_features.shape[0]:
            heatmap_logits, initial_canonical, joint_descriptors = self.joint_head(fused_features)
            hand_tokens = self.mano_head.encode_hand_token(fused_features)
            fused_hand_tokens, fused_joint_tokens = self.mutual_fusion(hand_tokens, joint_descriptors)
            mano = self.mano_head.decode_hand_token(fused_hand_tokens)
            offsets_canonical = self.joint_offset_head(fused_joint_tokens)
        else:
            # Empty frames do not enter any hand network. Finite placeholders preserve the API.
            heatmap_logits = fused_features.new_empty((0, 21, *fused_features.shape[-2:]))
            initial_canonical = fused_features.new_empty((0, 21, 2))
            offsets_canonical = fused_features.new_empty((0, 21, 2))
            mano = {"global_orient": fused_features.new_empty((0, 1, 3, 3)),
                    "hand_pose": fused_features.new_empty((0, 15, 3, 3)),
                    "betas": fused_features.new_empty((0, 10)), "log_z": fused_features.new_empty((0, 1))}
        total = batch_size * num_views * 2
        heatmap_logits = scatter_hand_rows(heatmap_logits, hand_indices, total)
        initial_canonical = scatter_hand_rows(initial_canonical, hand_indices, total)
        offsets_canonical = scatter_hand_rows(offsets_canonical, hand_indices, total)
        mano = {key: scatter_hand_rows(value, hand_indices, total,
                                      identity_rotation=key in ("global_orient", "hand_pose"))
                for key, value in mano.items()}
        refined_canonical = initial_canonical + offsets_canonical

        initial_canonical = initial_canonical.reshape(
            batch_size,
            num_views,
            2,
            21,
            2,
        )
        offsets_canonical = offsets_canonical.reshape(
            batch_size,
            num_views,
            2,
            21,
            2,
        )
        refined_canonical = refined_canonical.reshape(
            batch_size,
            num_views,
            2,
            21,
            2,
        )
        initial_crop = uncanonicalize_crop_joints(initial_canonical)
        refined_crop = uncanonicalize_crop_joints(refined_canonical)
        crop_offsets = uncanonicalize_crop_offsets(offsets_canonical)

        global_orient = mano["global_orient"].reshape(
            batch_size,
            num_views,
            2,
            1,
            3,
            3,
        )
        hand_pose = mano["hand_pose"].reshape(
            batch_size,
            num_views,
            2,
            15,
            3,
            3,
        )
        betas = mano["betas"].reshape(batch_size, num_views, 2, 10)
        log_z = mano["log_z"].reshape(batch_size, num_views, 2)

        flat_global = global_orient.reshape(batch_size * num_views, 2, 1, 3, 3)
        flat_pose = hand_pose.reshape(batch_size * num_views, 2, 15, 3, 3)
        flat_betas = betas.reshape(batch_size * num_views, 2, 10)
        joints, vertices = self._decode_mano(flat_global, flat_pose, flat_betas, mano_left_basis_b,
                                              output.get("mano_selected_valid"))
        oriented_joints = joints.reshape(batch_size, num_views, 2, 21, 3)
        oriented_vertices = vertices.reshape(batch_size, num_views, 2, 778, 3)

        margin = float(self.cfg.joint_heatmap_range_margin)
        lsp_valid = (
            (refined_crop[..., 0] >= -margin)
            & (refined_crop[..., 0] <= 1.0 + margin)
            & (refined_crop[..., 1] >= -margin)
            & (refined_crop[..., 1] <= 1.0 + margin)
        )
        intrinsics_fp32 = mano_crop_intrinsics.to(
            device=fused_features.device,
            dtype=torch.float32,
        )
        device_type = fused_features.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            camera_translation = decode_camera_translation_lsp(
                refined_crop.float(),
                oriented_joints.float(),
                log_z.float(),
                intrinsics_fp32,
                crop_hw,
                valid_mask=lsp_valid,
                min_visible=int(self.cfg.pnp_min_visible),
            )
            camera_joints = oriented_joints.float() + camera_translation[..., None, :]
            camera_vertices = (
                oriented_vertices.float() + camera_translation[..., None, :]
            )
            mano_reprojection = project_camera_joints_to_crop(
                camera_joints,
                intrinsics_fp32,
                crop_hw,
            )

        global_rotation = global_orient[..., 0, :, :]
        root = oriented_joints[..., :1, :]
        output["model_variant"] = self.model_variant
        output["heatmap_logits"] = heatmap_logits.reshape(
            batch_size,
            num_views,
            2,
            21,
            *heatmap_logits.shape[-2:],
        )
        output["pred_keypoints_2d_initial_crop"] = initial_crop - 0.5
        output["joint_offset_2d"] = crop_offsets
        output["pred_keypoints_2d"] = refined_crop - 0.5
        output["pred_keypoints_2d_mano_reproj"] = mano_reprojection - 0.5
        output["pred_keypoints_2d_crop_camera_projection"] = (
            mano_reprojection - 0.5
        )
        output["mano_crop_intrinsics"] = intrinsics_fp32
        output["lsp_valid_mask"] = lsp_valid
        output["log_z"] = log_z
        output["pred_mano_params"] = {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas,
            "transl": camera_translation,
        }
        output["mano_translation_convention"] = "wrist"
        output["depth_convention"] = "crop_camera_wrist_depth"
        output["pred_mano_transl"] = camera_translation
        output["pred_mano_transl_crop_camera"] = camera_translation
        output["pred_keypoints_3d_hand"] = torch.matmul(
            oriented_joints - root,
            global_rotation,
        )
        output["pred_vertices_hand"] = torch.matmul(
            oriented_vertices - root,
            global_rotation,
        )
        output["pred_keypoints_3d_oriented"] = oriented_joints
        output["pred_vertices_oriented"] = oriented_vertices
        output["pred_keypoints_3d"] = oriented_joints
        output["pred_vertices"] = oriented_vertices
        output["pred_keypoints_3d_camera"] = camera_joints
        output["pred_keypoints_3d_crop_camera"] = camera_joints
        output["pred_vertices_camera"] = camera_vertices
        output["pred_vertices_crop_camera"] = camera_vertices
        output["pred_wrist_transl_camera"] = camera_joints[..., 0, :]
        output["camera_output_space"] = "mano_crop_camera"

        # Deprecated approximation retained only for old diagnostics.  The
        # perspective outputs above are the training/evaluation authority.
        crop_height, crop_width = map(float, crop_hw)
        tz = camera_translation[..., 2].clamp_min(1.0e-6)
        scale_x = intrinsics_fp32[..., 0, 0] / (crop_width * tz)
        scale_y = intrinsics_fp32[..., 1, 1] / (crop_height * tz)
        translation_x = (
            intrinsics_fp32[..., 0, 0] * camera_translation[..., 0] / tz
            + intrinsics_fp32[..., 0, 2]
        ) / crop_width - 0.5
        translation_y = (
            intrinsics_fp32[..., 1, 1] * camera_translation[..., 1] / tz
            + intrinsics_fp32[..., 1, 2]
        ) / crop_height - 0.5
        output["pred_cam"] = torch.stack(
            (0.5 * (scale_x + scale_y), translation_x, translation_y),
            dim=-1,
        )
        output["pred_cam_is_diagnostic"] = True
        return output

    def forward(
        self,
        images: torch.Tensor,
        gt_hand_boxes: Optional[torch.Tensor] = None,
        gt_hand_valid: Optional[torch.Tensor] = None,
        use_gt_boxes: Optional[bool] = None,
        intrinsics: Optional[torch.Tensor] = None,
        mano_crop_intrinsics: Optional[torch.Tensor] = None,
        mano_crop_image_hw: Optional[Tuple[int, int]] = None,
        backbone_output: Optional[LingBotMapBackboneOutput] = None,
        mano_left_basis_b: Optional[torch.Tensor] = None,
        mano_training_selection: bool = False,
        heads: str = "all",
    ) -> Dict[str, torch.Tensor]:
        if heads not in {"all", "mask", "mano", "camera", "joint"}:
            raise ValueError(f"Unsupported head selection: {heads}")
        run_mask = heads in {"all", "mask", "joint"}
        run_mano = heads in {"all", "mano", "joint"}
        run_camera = heads in {"all", "camera", "joint"}
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = images.float()
            intrinsics = intrinsics.float() if intrinsics is not None else None
            original_hw = images.shape[-2:]
            target_hw = (self.cfg.image_height, self.cfg.image_width)
            if original_hw != target_hw:
                b, n = images.shape[:2]
                images = F.interpolate(
                    images.reshape(b * n, *images.shape[2:]).float(),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, n, 3, *target_hw)
                if intrinsics is not None:
                    resize = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
                    sx, sy = target_hw[1] / original_hw[1], target_hw[0] / original_hw[0]
                    resize[0,0], resize[1,1] = sx, sy
                    resize[0,2], resize[1,2] = (sx-1)/2, (sy-1)/2
                    intrinsics = resize @ intrinsics

            if backbone_output is None:
                backbone_output = self.backbone(images, run_camera=run_camera)
            output = {}
            if run_mask:
                detection_features = self.hand_detection_block(
                    backbone_output.aggregated_tokens,
                    backbone_output.patch_start_idx,
                    target_hw,
                )
                mask_logits = self.hand_mask_head(detection_features, target_hw)
                probability = torch.sigmoid(mask_logits)
                hand_probability = probability.amax(dim=(-1, -2))
                hand_probability_logits = torch.logit(hand_probability.clamp(1e-6, 1-1e-6))
                binary = probability > self.cfg.mask_only_threshold
                h, w = binary.shape[-2:]
                xs = torch.arange(w, device=binary.device).view(1,1,1,1,w)
                ys = torch.arange(h, device=binary.device).view(1,1,1,h,1)
                x1 = torch.where(binary, xs, w).amin(dim=(-1,-2))/w
                x2 = (torch.where(binary, xs, -1).amax(dim=(-1,-2))+1)/w
                y1 = torch.where(binary, ys, h).amin(dim=(-1,-2))/h
                y2 = (torch.where(binary, ys, -1).amax(dim=(-1,-2))+1)/h
                boxes = torch.stack((x1,y1,x2,y2),dim=-1)
                valid = binary.any(dim=(-1,-2))
                boxes = torch.where(valid[...,None], boxes, boxes.new_tensor((0,0,1,1)))
                output = {"pred_hand_boxes": boxes}
                output.update(
                    {
                        "pred_hand_probability": hand_probability,
                        "pred_hand_probability_logits": hand_probability_logits,
                        "pred_hand_mask_logits": mask_logits,
                        "pred_hand_masks": torch.sigmoid(mask_logits),
                    }
                )
            if run_camera and backbone_output.pose_enc is not None:
                output["pred_camera_pose_enc"] = backbone_output.pose_enc
            if run_camera and self.camera_scale_head is not None:
                if backbone_output.pose_enc is None:
                    raise RuntimeError(
                        "camera metric scale head requires native camera poses"
                    )
                output.update(
                    self.camera_scale_head(
                        backbone_output.aggregated_tokens,
                        backbone_output.patch_start_idx,
                    )
                )
            if not run_mano:
                return output
            features = backbone_output.patch_features.to(
                dtype=next(self.lingbot_adapter.parameters()).dtype
            )
            use_gt = self.cfg.use_gt_boxes_for_mano if use_gt_boxes is None else bool(use_gt_boxes)
            if self.cfg.mano_gradient_to_mask_head:
                if not run_mask:
                    raise ValueError("MANO predicted-box dependency requires the mask head")
                boxes = soft_mask_logits_to_boxes(
                    mask_logits,
                    temperature=self.cfg.soft_mask_box_temperature,
                    std_scale=self.cfg.soft_mask_box_std_scale,
                    min_size=self.cfg.soft_mask_box_min_size,
                )
                output["soft_mask_hand_boxes"] = boxes
            else:
                if not use_gt or gt_hand_boxes is None:
                    raise ValueError(
                        "mano_gradient_to_mask_head=false requires "
                        "use_gt_boxes=true and gt_hand_boxes for MANO crop construction"
                    )
                gt_boxes = gt_hand_boxes.to(
                    device=images.device,
                    dtype=images.dtype,
                )
                finite = torch.isfinite(gt_boxes).all(dim=-1)
                if gt_hand_valid is None:
                    valid = finite
                else:
                    valid = (
                        gt_hand_valid.to(
                            device=images.device,
                            dtype=torch.bool,
                        )
                        & finite
                    )
                default_boxes = gt_boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
                default_boxes = default_boxes.view(1, 1, 1, 4).expand_as(gt_boxes)
                boxes = torch.where(valid[..., None], gt_boxes, default_boxes)

            # Preserve hand slots; invalid crops use a finite fallback but carry no geometry loss.
            pixel_wh = (boxes[..., 2:] - boxes[..., :2]) * boxes.new_tensor((target_hw[1], target_hw[0]))
            overlap_wh = torch.minimum(boxes[..., 2:], boxes.new_ones(2)) - torch.maximum(boxes[..., :2], boxes.new_zeros(2))
            crop_valid = torch.isfinite(boxes).all(-1) & (pixel_wh >= 2).all(-1) & (overlap_wh > 0).all(-1)
            if self.cfg.mano_gradient_to_mask_head:
                crop_valid = crop_valid & (hand_probability > self.cfg.mask_only_threshold)
            hand_indices = None
            if self.training and mano_training_selection:
                if gt_hand_valid is None:
                    raise ValueError("Random valid-hand training requires gt_hand_valid")
                candidates = gt_hand_valid.to(device=boxes.device, dtype=torch.bool) & crop_valid
                selected = sample_one_valid_hand(candidates)
                hand_indices = selected.reshape(-1).nonzero(as_tuple=False).flatten()
                output["mano_selected_valid"] = selected
                crop_valid = crop_valid & selected
            output["geometry_crop_valid"] = crop_valid
            boxes = torch.where(crop_valid[..., None], boxes, boxes.new_tensor((0., 0., 1., 1.)))
            crop_scale = self._sample_bbox_rescale(boxes)
            boxes = expand_to_square_boxes(boxes, target_hw, crop_scale)
            if intrinsics is None:
                raise ValueError("Perspective hand crops require original image intrinsics")
            # Derive geometry from the actual boxes, including predicted boxes/augmentation.
            with torch.autocast(device_type=boxes.device.type, enabled=False):
                mano_crop_intrinsics, auto_crop_rotation = virtual_crop_camera_from_boxes(
                    boxes, intrinsics, target_hw, (256, 192))
                mano_crop_image_hw = (256, 192)
                crop_h = mano_crop_intrinsics @ auto_crop_rotation @ torch.linalg.inv(intrinsics.float())[:, :, None]
                # Encoders consume a 256-square image then discard columns 0:32 and 224:256.
                square_shift = torch.eye(3, device=crop_h.device, dtype=crop_h.dtype)
                square_shift[0, 2] = 32.0
                square_h = square_shift @ crop_h
            output["mano_crop_homographies"] = crop_h
            b, n, c, h, w = features.shape
            lingbot_context, lingbot_crops_256 = self.lingbot_adapter(
                features, boxes, square_h, target_hw, hand_indices=hand_indices)
            hand_feature_batch = self.hand_feature_provider(images, boxes, square_h, hand_indices=hand_indices)
            hand_features = hand_feature_batch.features.to(dtype=lingbot_context.dtype)
            flat_hand_features = hand_features.reshape(-1, self.hand_feature_provider.output_dim,
                                                      *self.hand_feature_provider.output_hw)
            if flat_hand_features.shape[0]:
                fused_features = self.feature_fusion(lingbot_context, flat_hand_features)
            else:
                fused_features = lingbot_context
            def restore(value):
                return scatter_hand_rows(value, hand_indices, b * n * 2)
            output["mano_boxes"] = boxes
            output["mano_crop_scale"] = crop_scale
            output["lingbotmap_feature_crops"] = restore(lingbot_crops_256).reshape(
                b, n, 2, self.cfg.lingbot_pixel_channels, self.cfg.wilor_crop_size, self.cfg.wilor_crop_size)
            output["lingbotmap_mano_features"] = restore(lingbot_context).reshape(
                b, n, 2, self.cfg.mano_context_dim, *self.lingbot_adapter.output_hw)
            output["hand_feature_crops"] = restore(hand_feature_batch.canonical_crops.reshape(
                -1, 3, self.cfg.wilor_crop_size, self.cfg.wilor_crop_size)).reshape(
                b, n, 2, 3, self.cfg.wilor_crop_size, self.cfg.wilor_crop_size)
            output["hand_features"] = restore(flat_hand_features).reshape(
                b, n, 2, self.hand_feature_provider.output_dim, *self.hand_feature_provider.output_hw)
            output["fused_mano_features"] = restore(fused_features).reshape(
                b, n, 2, self.cfg.mano_context_dim, *self.lingbot_adapter.output_hw)
            if self.model_variant == "mutual_joint_fusion":
                output = self._forward_mutual_joint_fusion(
                    output,
                    fused_features,
                    b,
                    n,
                    mano_crop_intrinsics,
                    mano_crop_image_hw,
                    mano_left_basis_b,
                    hand_indices=hand_indices,
                )
                if auto_crop_rotation is not None:
                    with torch.autocast(device_type=auto_crop_rotation.device.type, enabled=False):
                        crop_to_image = auto_crop_rotation.transpose(-1, -2)
                        output["mano_virtual_rotations"] = auto_crop_rotation
                        output["pred_keypoints_3d_camera"] = torch.matmul(
                            output["pred_keypoints_3d_crop_camera"],
                            auto_crop_rotation,
                        )
                        output["pred_vertices_camera"] = torch.matmul(
                            output["pred_vertices_crop_camera"],
                            auto_crop_rotation,
                        )
                        crop_global = output["pred_mano_params"]["global_orient"].float()
                        output["pred_global_orient_crop_camera"] = crop_global
                        output["pred_global_orient_camera"] = torch.matmul(
                            crop_to_image[..., None, :, :], crop_global
                        )
                        output["pred_mano_transl_camera"] = torch.matmul(
                            crop_to_image,
                            output["pred_mano_transl_crop_camera"].float()[..., None],
                        ).squeeze(-1)
                        output["pred_wrist_transl_crop_camera"] = output["pred_wrist_transl_camera"]
                        output["pred_wrist_transl_camera"] = output["pred_keypoints_3d_camera"][..., 0, :]
                        output["camera_output_space"] = "image_camera"
                return output

            output["model_variant"] = self.model_variant
            if fused_features.shape[0]:
                mano = self.mano_head(fused_features)
            else:
                mano = {"global_orient": fused_features.new_empty((0, 1, 3, 3)),
                        "hand_pose": fused_features.new_empty((0, 15, 3, 3)),
                        "betas": fused_features.new_empty((0, 10)), "pred_cam": fused_features.new_empty((0, 3))}
            mano = {key: scatter_hand_rows(value, hand_indices, b * n * 2,
                                          identity_rotation=key in ("global_orient", "hand_pose"))
                    for key, value in mano.items()}
            output["pred_mano_params"] = {
                "global_orient": mano["global_orient"].reshape(b, n, 2, 1, 3, 3),
                "hand_pose": mano["hand_pose"].reshape(b, n, 2, 15, 3, 3),
                "betas": mano["betas"].reshape(b, n, 2, 10),
            }
            output["pred_cam"] = mano["pred_cam"].reshape(b, n, 2, 3)
            if self.cfg.decode_mano:
                flat_global = mano["global_orient"].reshape(b * n, 2, 1, 3, 3)
                flat_pose = mano["hand_pose"].reshape(b * n, 2, 15, 3, 3)
                flat_betas = mano["betas"].reshape(b * n, 2, 10)
                joints, vertices = self._decode_mano(flat_global, flat_pose, flat_betas, mano_left_basis_b,
                                                      output.get("mano_selected_valid"))
                oriented_joints = joints.reshape(b, n, 2, 21, 3)
                oriented_vertices = vertices.reshape(b, n, 2, 778, 3)
                global_rotation = output["pred_mano_params"]["global_orient"][..., 0, :, :]
                root = oriented_joints[..., :1, :]

                # Explicit coordinate-space outputs:
                # - hand: wrist-rooted and inverse-global-rotated MANO coordinates
                # - oriented: global rotation applied, but no camera translation
                # The legacy aliases remain oriented for visualization/checkpoint
                # compatibility. Camera-space losses use the explicit camera keys.
                output["pred_keypoints_3d_hand"] = torch.matmul(
                    oriented_joints - root, global_rotation
                )
                output["pred_vertices_hand"] = torch.matmul(
                    oriented_vertices - root, global_rotation
                )
                output["pred_keypoints_3d_oriented"] = oriented_joints
                output["pred_vertices_oriented"] = oriented_vertices
                output["pred_keypoints_3d"] = oriented_joints
                output["pred_vertices"] = oriented_vertices

                cam = mano["pred_cam"].reshape(b, n, 2, 3)
                pred_xy = cam[..., None, :1] * oriented_joints[..., :2] + cam[..., None, 1:3]
                output["pred_keypoints_2d"] = pred_xy
                if mano_crop_intrinsics is not None:
                    crop_hw = (
                        tuple(int(value) for value in mano_crop_image_hw)
                        if mano_crop_image_hw is not None
                        else (self.cfg.wilor_crop_size, self.cfg.wilor_crop_size - 64)
                    )
                    if len(crop_hw) != 2 or min(crop_hw) <= 0:
                        raise ValueError(
                            "mano_crop_image_hw must contain positive (height, width), "
                            f"got {crop_hw}"
                        )
                    crop_boxes = cam.new_tensor((0.0, 0.0, 1.0, 1.0)).expand(
                        *cam.shape[:-1], 4
                    )
                    camera_translation = weak_perspective_to_camera_translation(
                        cam,
                        crop_boxes,
                        mano_crop_intrinsics,
                        crop_hw,
                    )
                    camera_joints = oriented_joints + camera_translation[..., None, :]
                    camera_vertices = oriented_vertices + camera_translation[..., None, :]
                    output["pred_mano_transl"] = camera_translation
                    output["pred_mano_transl_crop_camera"] = camera_translation
                    output["pred_mano_params"]["transl"] = camera_translation
                    output["pred_keypoints_3d_camera"] = camera_joints
                    output["pred_keypoints_3d_crop_camera"] = camera_joints
                    output["pred_vertices_camera"] = camera_vertices
                    output["pred_vertices_crop_camera"] = camera_vertices
                    output["camera_output_space"] = "mano_crop_camera"
                with torch.autocast(device_type=auto_crop_rotation.device.type, enabled=False):
                    output["mano_virtual_rotations"] = auto_crop_rotation
                    output["mano_crop_intrinsics"] = mano_crop_intrinsics
                    output["pred_global_orient_crop_camera"] = output["pred_mano_params"]["global_orient"].float()
                    output["pred_global_orient_camera"] = auto_crop_rotation.transpose(-1,-2)[...,None,:,:] @ output["pred_global_orient_crop_camera"]
                    output["pred_keypoints_3d_camera"] = output["pred_keypoints_3d_crop_camera"] @ auto_crop_rotation
                    output["pred_vertices_camera"] = output["pred_vertices_crop_camera"] @ auto_crop_rotation
                    output["pred_mano_transl_camera"] = (auto_crop_rotation.transpose(-1,-2) @ camera_translation.float()[...,None]).squeeze(-1)
                    output["pred_wrist_transl_camera"] = output["pred_keypoints_3d_camera"][...,0,:]
                    output["camera_output_space"] = "image_camera"
            return output


def build_lingbotmap_hand(**kwargs) -> LingBotMapHand:
    return LingBotMapHand(LingBotMapHandConfig(**kwargs))
