from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


N_JOINTS_2D = 21


def soft_argmax_2d_norm(
    heatmap: torch.Tensor,
    range_margin: float = 0.15,
) -> torch.Tensor:
    """Soft-argmax heatmaps into crop-normalized coordinates.

    The center of spatial cell ``(row, col)`` maps into
    ``[-range_margin, 1 + range_margin]``.  The leading dimensions are
    preserved and the returned tensor ends in ``[..., 2]`` as ``(u, v)``.
    """
    if heatmap.ndim < 2:
        raise ValueError(f"heatmap must have at least 2 dimensions, got {heatmap.shape}")
    *leading, height, width = heatmap.shape
    flat = heatmap.reshape(-1, height * width)
    weights = F.softmax(flat, dim=-1)
    scale = 1.0 + 2.0 * float(range_margin)
    offset = float(range_margin)
    u = (
        (torch.arange(width, device=heatmap.device, dtype=heatmap.dtype) + 0.5)
        / float(width)
        * scale
        - offset
    )
    v = (
        (torch.arange(height, device=heatmap.device, dtype=heatmap.dtype) + 0.5)
        / float(height)
        * scale
        - offset
    )
    grid_u = u.unsqueeze(0).expand(height, -1).reshape(-1)
    grid_v = v.unsqueeze(1).expand(-1, width).reshape(-1)
    pred_u = (weights * grid_u).sum(dim=-1)
    pred_v = (weights * grid_v).sum(dim=-1)
    return torch.stack((pred_u, pred_v), dim=-1).reshape(*leading, 2)


class JointHeatmapDescriptorHead(nn.Module):
    """Predict 2D joint heatmaps and pool one descriptor per joint."""

    def __init__(
        self,
        input_dim: int,
        descriptor_dim: int,
        n_joints: int = N_JOINTS_2D,
        range_margin: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_joints = int(n_joints)
        self.range_margin = float(range_margin)
        self.heatmap_head = nn.Conv2d(input_dim, self.n_joints, kernel_size=1)
        self.descriptor_proj = nn.Conv2d(input_dim, descriptor_dim, kernel_size=1)
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.zeros_(self.heatmap_head.bias)

    def forward(
        self,
        crop_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if crop_features.ndim != 4:
            raise ValueError(
                f"crop_features must be [M,C,H,W], got {crop_features.shape}"
            )
        logits = self.heatmap_head(crop_features)
        joints_2d = soft_argmax_2d_norm(
            logits,
            range_margin=self.range_margin,
        )
        weights = F.softmax(logits.flatten(2), dim=-1)
        descriptor_map = self.descriptor_proj(crop_features)
        descriptor_tokens = descriptor_map.flatten(2).transpose(1, 2)
        descriptors = torch.einsum(
            "mjh,mhd->mjd",
            weights,
            descriptor_tokens,
        )
        return logits, joints_2d, descriptors


class MutualHandJointFusion(nn.Module):
    """One exactly identity-initialized bidirectional cross-attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hand_norm = nn.LayerNorm(dim)
        self.joint_norm = nn.LayerNorm(dim)
        self.hand_from_joint = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.joint_from_hand = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        for attention in (self.hand_from_joint, self.joint_from_hand):
            nn.init.zeros_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)

    def forward(
        self,
        hand_tokens: torch.Tensor,
        joint_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hand_tokens.ndim != 3 or joint_tokens.ndim != 3:
            raise ValueError(
                "hand_tokens and joint_tokens must be [M,L,D], got "
                f"{hand_tokens.shape} and {joint_tokens.shape}"
            )
        if hand_tokens.shape[0] != joint_tokens.shape[0]:
            raise ValueError("hand and joint token batches must match")
        hand_norm = self.hand_norm(hand_tokens)
        joint_norm = self.joint_norm(joint_tokens)
        hand_delta, _ = self.hand_from_joint(
            hand_norm,
            joint_norm,
            joint_norm,
            need_weights=False,
        )
        joint_delta, _ = self.joint_from_hand(
            joint_norm,
            hand_norm,
            hand_norm,
            need_weights=False,
        )
        return hand_tokens + hand_delta, joint_tokens + joint_delta


class JointOffsetHead(nn.Module):
    """Predict a residual 2D offset from fused joint descriptors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(joint_tokens)


def uncanonicalize_crop_joints(
    joints_2d: torch.Tensor,
) -> torch.Tensor:
    """Undo the model's horizontal canonicalization for slot 1 (left hand).

    Input and output use normalized crop coordinates.  Expected shape is
    ``[..., 2 slots, J, 2]`` with slot 0=right and slot 1=left.
    """
    if joints_2d.ndim < 4 or joints_2d.shape[-3] != 2 or joints_2d.shape[-1] != 2:
        raise ValueError(
            "joints_2d must end in [2 slots,J,2], got "
            f"{joints_2d.shape}"
        )
    right = joints_2d[..., 0, :, :]
    left = joints_2d[..., 1, :, :].clone()
    left[..., 0] = (191.0 / 192.0) - left[..., 0]
    return torch.stack((right, left), dim=-3)


def uncanonicalize_crop_offsets(
    offsets_2d: torch.Tensor,
) -> torch.Tensor:
    """Undo canonicalization for offsets (left-hand x changes sign)."""
    if offsets_2d.ndim < 4 or offsets_2d.shape[-3] != 2 or offsets_2d.shape[-1] != 2:
        raise ValueError(
            "offsets_2d must end in [2 slots,J,2], got "
            f"{offsets_2d.shape}"
        )
    right = offsets_2d[..., 0, :, :]
    left = offsets_2d[..., 1, :, :].clone()
    left[..., 0] = -left[..., 0]
    return torch.stack((right, left), dim=-3)


def decode_camera_translation_lsp(
    joints_2d_norm: torch.Tensor,
    joints_3d_canonical: torch.Tensor,
    log_z: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
    valid_mask: Optional[torch.Tensor] = None,
    min_visible: int = 1,
) -> torch.Tensor:
    """Recover MANO-origin ``(tx, ty, tz)`` by perspective least squares.

    ``tz`` is ``exp(log_z)``.  Given fixed depth, each visible joint contributes
    a linear pixel-space equation for ``tx`` and ``ty``.  The solve stays fully
    differentiable.
    """
    if joints_2d_norm.shape[:-1] != joints_3d_canonical.shape[:-1]:
        raise ValueError(
            "2D and 3D joints must share leading dimensions, got "
            f"{joints_2d_norm.shape} and {joints_3d_canonical.shape}"
        )
    if joints_2d_norm.shape[-1] != 2 or joints_3d_canonical.shape[-1] != 3:
        raise ValueError("Expected joints ending in 2D and 3D coordinates")
    if intrinsics.shape != (*joints_2d_norm.shape[:-2], 3, 3):
        raise ValueError(
            f"intrinsics must be {(*joints_2d_norm.shape[:-2], 3, 3)}, "
            f"got {intrinsics.shape}"
        )
    if log_z.shape == (*joints_2d_norm.shape[:-2], 1):
        log_z = log_z.squeeze(-1)
    if log_z.shape != joints_2d_norm.shape[:-2]:
        raise ValueError(
            f"log_z must be {joints_2d_norm.shape[:-2]}, got {log_z.shape}"
        )

    image_height, image_width = map(float, image_hw)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    tz = log_z.exp()

    u_px = joints_2d_norm[..., 0] * image_width
    v_px = joints_2d_norm[..., 1] * image_height
    canonical_x = joints_3d_canonical[..., 0]
    canonical_y = joints_3d_canonical[..., 1]
    canonical_z = joints_3d_canonical[..., 2]
    raw_depth = canonical_z + tz.unsqueeze(-1)
    depth = raw_depth.clamp_min(0.05)

    coefficient_u = fx.unsqueeze(-1) / depth
    reduced_u = (
        u_px
        - cx.unsqueeze(-1)
        - fx.unsqueeze(-1) * canonical_x / depth
    )
    coefficient_v = fy.unsqueeze(-1) / depth
    reduced_v = (
        v_px
        - cy.unsqueeze(-1)
        - fy.unsqueeze(-1) * canonical_y / depth
    )

    mask = raw_depth > 0.05
    if valid_mask is not None:
        if valid_mask.shape != mask.shape:
            raise ValueError(
                f"valid_mask must be {mask.shape}, got {valid_mask.shape}"
            )
        mask = mask & valid_mask.bool()
    weights = mask.to(dtype=joints_3d_canonical.dtype)
    eps = 1.0e-8
    tx = (
        (coefficient_u * reduced_u * weights).sum(dim=-1)
        / (coefficient_u.square() * weights).sum(dim=-1).clamp_min(eps)
    )
    ty = (
        (coefficient_v * reduced_v * weights).sum(dim=-1)
        / (coefficient_v.square() * weights).sum(dim=-1).clamp_min(eps)
    )
    if min_visible > 0:
        invalid = weights.sum(dim=-1) < float(min_visible)
        tx = torch.where(invalid, torch.zeros_like(tx), tx)
        ty = torch.where(invalid, torch.zeros_like(ty), ty)
    return torch.stack((tx, ty, tz), dim=-1)


def project_camera_joints_to_crop(
    joints_3d_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    """Project camera-space joints into crop-normalized ``[0,1]`` coordinates."""
    if intrinsics.shape != (*joints_3d_camera.shape[:-2], 3, 3):
        raise ValueError(
            "intrinsics and joints leading dimensions do not match: "
            f"{intrinsics.shape} vs {joints_3d_camera.shape}"
        )
    image_height, image_width = map(float, image_hw)
    depth = joints_3d_camera[..., 2].clamp_min(1.0e-6)
    u = (
        joints_3d_camera[..., 0]
        / depth
        * intrinsics[..., 0, 0].unsqueeze(-1)
        + intrinsics[..., 0, 2].unsqueeze(-1)
    ) / image_width
    v = (
        joints_3d_camera[..., 1]
        / depth
        * intrinsics[..., 1, 1].unsqueeze(-1)
        + intrinsics[..., 1, 2].unsqueeze(-1)
    ) / image_height
    return torch.stack((u, v), dim=-1)
