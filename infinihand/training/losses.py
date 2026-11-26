from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


POSE_PROTOCOL_DATASET_WORLD = (
    "opencv_c2w_dataset_world_metric_no_scale_alignment"
)
POSE_PROTOCOL_FIRST_CAMERA_SCALE = (
    "opencv_c2w_first_camera_metric_scale_v1"
)
POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE = (
    "opencv_c2w_first_camera_metric_no_scale_alignment"
)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues conversion for tensors shaped [..., 3]."""
    orig_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[0], 3, 3)
    angle = angle.reshape(-1, 1, 1)
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    return (eye + sin * skew + (1.0 - cos) * (skew @ skew)).reshape(*orig_shape, 3, 3)


@dataclass
class LingBotMapHandLossWeights:
    hand_probability: float = 1.0
    hand_bbox_smooth_l1: float = 5.0
    hand_bbox_giou: float = 2.0
    hand_mask_bce: float = 1.0
    hand_mask_dice: float = 1.0
    # ViDiHand Stage-2 direct L1 on the final heatmap+offset joint prediction.
    joints2d_direct_weight: float = 2.0
    # MANO/camera perspective reprojection L1.
    reproj_loss_weight: float = 1.0
    go_loss_weight: float = 0.1
    hp_loss_weight: float = 0.1
    betas_loss_weight: float = 0.1
    trans_loss_weight: float = 3.0
    # Absolute crop-camera-space MANO21 loss.
    joints_loss_weight: float = 2.0
    velocity_smooth_weight: float = 0.0
    betas_consistency_weight: float = 0.0
    # Long-sequence Stage-2 objective.  Stage-1 is the complete pre-existing
    # hand objective above; each Stage-2 macro loss is logged both raw and
    # weighted so the initial smoke can calibrate lambdas without ambiguity.
    stage1_loss_weight: float = 1.0
    world_joints_loss_weight: float = 0.0
    world_mano_loss_weight: float = 0.0
    abs_pose_loss_weight: float = 0.0
    rel_pose_loss_weight: float = 0.0
    pose_translation_loss_weight: float = 1.0
    # Explicit component coefficients for the five pose terms.  Defaults
    # preserve the historical nine-coordinate absolute-pose mean and the
    # original relative rotation + translation objective.
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


def _slice_loss_batch(value, sample_index: int, batch_size: int, *, key: str = ""):
    """Slice batch tensors, preserving global image sizes and scalar metadata.

    Loss targets store image_hw/crop_image_hw as global [H,W] tensors rather
    than [B,2]. In particular, they must not be sliced when batch_size is two.
    """
    if isinstance(value, dict):
        return {
            name: _slice_loss_batch(item, sample_index, batch_size, key=name)
            for name, item in value.items()
        }
    if isinstance(value, torch.Tensor):
        if key in {"image_hw", "crop_image_hw"} and value.ndim == 1:
            return value
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[sample_index : sample_index + 1]
        return value
    if isinstance(value, tuple):
        return tuple(_slice_loss_batch(item, sample_index, batch_size, key=key) for item in value)
    if isinstance(value, list):
        return [_slice_loss_batch(item, sample_index, batch_size, key=key) for item in value]
    return value


def _weight_loss(value: torch.Tensor, weight: float) -> torch.Tensor:
    """A disabled term must not contaminate the objective through 0 * NaN."""
    return value * float(weight) if float(weight) != 0.0 else value.new_zeros(())


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(device=value.device, dtype=torch.bool)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    expanded = valid.expand_as(value)
    safe_value = torch.where(expanded, value, torch.zeros_like(value))
    return safe_value.sum() / expanded.sum().clamp_min(1).to(value.dtype)


def _masked_mean_per_sample(
    value: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean over all non-batch axes, retaining one value per sample."""
    valid = valid.to(device=value.device, dtype=torch.bool)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    expanded = valid.expand_as(value)
    safe_value = torch.where(expanded, value, torch.zeros_like(value))
    reduce_dims = tuple(range(1, value.ndim))
    numerator = safe_value.sum(dim=reduce_dims)
    denominator = expanded.sum(dim=reduce_dims).clamp_min(1).to(value.dtype)
    return numerator / denominator


def _geodesic_per_element(
    pred_rotation: torch.Tensor,
    gt_rotation: torch.Tensor,
) -> torch.Tensor:
    """ViDiHand SO(3) geodesic distance, returned in radians."""
    relative = pred_rotation.transpose(-1, -2) @ gt_rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine)


def _invert_se3(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"SE(3) tensor must end in [4,4], got {transform.shape}")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse_rotation = rotation.transpose(-1, -2)
    inverse[..., :3, :3] = inverse_rotation
    inverse[..., :3, 3] = -torch.matmul(
        inverse_rotation, translation.unsqueeze(-1)
    ).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


def _pose_encoding_to_w2c(pose_encoding: torch.Tensor) -> torch.Tensor:
    """Decode LingBotMap's [T, quat_xyzw, fov] camera encoding to W2C."""
    if pose_encoding.ndim != 3 or pose_encoding.shape[-1] != 9:
        raise ValueError(
            "pred_camera_pose_enc must be [B,T,9], got "
            f"{tuple(pose_encoding.shape)}"
        )
    translation = pose_encoding[..., :3]
    quaternion = pose_encoding[..., 3:7]
    i, j, k, r = quaternion.unbind(dim=-1)
    two_s = 2.0 / quaternion.square().sum(dim=-1).clamp_min(1.0e-8)
    rotation = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    w2c = pose_encoding.new_zeros(*pose_encoding.shape[:-1], 4, 4)
    w2c[..., :3, :3] = rotation
    w2c[..., :3, 3] = translation
    w2c[..., 3, 3] = 1.0
    return w2c


def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply transforms [...,4,4] to points with matching leading axes."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def _matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to scalar-last XYZW quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in [3,3], got {matrix.shape}")
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(*batch_shape, 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            1.0e-8,
        )
    )
    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    floor = q_abs.new_tensor(0.1)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp_min(floor))
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion_rijk = candidates.gather(-2, gather_index).squeeze(-2)
    quaternion_xyzw = quaternion_rijk[..., (1, 2, 3, 0)]
    return torch.where(
        quaternion_xyzw[..., 3:4] < 0.0,
        -quaternion_xyzw,
        quaternion_xyzw,
    )


def _masked_smooth_l1_mean(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    return _masked_mean(
        F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=float(beta),
        ),
        valid,
    )


def compute_pose_losses(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    relative_pose_window: int,
    pose_translation_weight: float,
    huber_beta: float,
    abs_pose_translation_component_weight: float = 3.0 / 9.0,
    abs_pose_quaternion_component_weight: float = 4.0 / 9.0,
    abs_pose_fov_component_weight: float = 2.0 / 9.0,
    rel_pose_rotation_component_weight: float = 1.0,
    rel_pose_translation_component_weight: float = 1.0,
    coordinate_protocol: str = POSE_PROTOCOL_DATASET_WORLD,
) -> Dict[str, torch.Tensor]:
    """Metric absolute/relative pose losses under an explicit C2W protocol."""
    coordinate_protocol = str(coordinate_protocol).strip().lower()
    supported_protocols = {
        POSE_PROTOCOL_DATASET_WORLD,
        POSE_PROTOCOL_FIRST_CAMERA_SCALE,
        POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE,
    }
    if coordinate_protocol not in supported_protocols:
        raise ValueError(
            "unsupported pose coordinate protocol: "
            f"{coordinate_protocol!r}; expected one of "
            f"{sorted(supported_protocols)}"
        )
    required_output = ("pred_camera_pose_enc",)
    required_target = (
        "camera_poses",
        "intrinsics",
        "image_hw",
        "camera_trajectory_valid",
    )
    missing = [key for key in required_output if key not in output]
    missing += [key for key in required_target if key not in target]
    if missing:
        raise KeyError("Pose loss requires: " + ", ".join(missing))

    pred_pose_encoding = output["pred_camera_pose_enc"].float()
    pred_c2w = _invert_se3(_pose_encoding_to_w2c(pred_pose_encoding))
    # Absolute metric camera supervision: no learned or fitted scale.
    metric_scale = pred_pose_encoding.new_ones(pred_pose_encoding.shape[0])
    gt_c2w = target["camera_poses"].to(
        device=pred_pose_encoding.device, dtype=pred_pose_encoding.dtype
    )
    camera_valid = target["camera_trajectory_valid"].to(
        device=pred_pose_encoding.device, dtype=torch.bool
    )
    camera_valid = (
        camera_valid
        & torch.isfinite(gt_c2w).all(dim=(-1, -2))
    )

    if coordinate_protocol in {
        POSE_PROTOCOL_FIRST_CAMERA_SCALE,
        POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE,
    }:
        # Remove only the arbitrary SE(3) world gauge. Scale remains metric:
        # canonical first-camera translation is compared directly with metric GT.
        pred_first_w2c = _invert_se3(pred_c2w[:, 0])
        gt_first_w2c = _invert_se3(gt_c2w[:, 0])
        pred_c2w = torch.matmul(pred_first_w2c[:, None], pred_c2w)
        gt_c2w = torch.matmul(gt_first_w2c[:, None], gt_c2w)
        first_frame_valid = camera_valid[:, :1]
        camera_valid = camera_valid & first_frame_valid

    scaled_translation = (
        pred_c2w[..., :3, 3] * metric_scale[:, None, None]
    ).unsqueeze(-1)
    pred_c2w = torch.cat(
        (
            torch.cat((pred_c2w[..., :3, :3], scaled_translation), dim=-1),
            pred_c2w[..., 3:4, :],
        ),
        dim=-2,
    )

    pred_quaternion = _matrix_to_quaternion_xyzw(pred_c2w[..., :3, :3])
    gt_quaternion = _matrix_to_quaternion_xyzw(gt_c2w[..., :3, :3])
    quaternion_dot = (pred_quaternion * gt_quaternion).sum(
        dim=-1, keepdim=True
    )
    gt_quaternion = torch.where(
        quaternion_dot < 0.0, -gt_quaternion, gt_quaternion
    )
    intrinsics = target["intrinsics"].to(
        device=pred_pose_encoding.device, dtype=pred_pose_encoding.dtype
    )
    image_hw = target["image_hw"].to(
        device=pred_pose_encoding.device, dtype=pred_pose_encoding.dtype
    ).reshape(-1)
    image_height, image_width = image_hw[0], image_hw[1]
    gt_fov = torch.stack(
        (
            2.0 * torch.atan((image_height * 0.5) / intrinsics[..., 1, 1]),
            2.0 * torch.atan((image_width * 0.5) / intrinsics[..., 0, 0]),
        ),
        dim=-1,
    )
    pred_pose_world_encoding = torch.cat(
        (
            pred_c2w[..., :3, 3],
            pred_quaternion,
            pred_pose_encoding[..., 7:9],
        ),
        dim=-1,
    )
    gt_pose_world_encoding = torch.cat(
        (gt_c2w[..., :3, 3], gt_quaternion, gt_fov),
        dim=-1,
    )
    camera_abs_valid = (
        camera_valid
        & torch.isfinite(gt_pose_world_encoding).all(dim=-1)
    )
    abs_pose_translation = _masked_smooth_l1_mean(
        pred_pose_world_encoding[..., :3],
        gt_pose_world_encoding[..., :3],
        camera_abs_valid,
        beta=huber_beta,
    )
    abs_pose_quaternion = _masked_smooth_l1_mean(
        pred_pose_world_encoding[..., 3:7],
        gt_pose_world_encoding[..., 3:7],
        camera_abs_valid,
        beta=huber_beta,
    )
    abs_pose_fov = _masked_smooth_l1_mean(
        pred_pose_world_encoding[..., 7:9],
        gt_pose_world_encoding[..., 7:9],
        camera_abs_valid,
        beta=huber_beta,
    )
    # Keep each component explicit so experiments can report raw and weighted
    # values without hiding the objective behind a hard-coded mean.
    abs_pose = (
        float(abs_pose_translation_component_weight) * abs_pose_translation
        + float(abs_pose_quaternion_component_weight) * abs_pose_quaternion
        + float(abs_pose_fov_component_weight) * abs_pose_fov
    )

    frame_count = int(pred_c2w.shape[1])
    pair_window = min(max(int(relative_pose_window), 2), frame_count)
    window_count = max(frame_count - pair_window + 1, 0)
    if frame_count >= 2 and window_count > 0:
        pair_template = torch.ones(
            pair_window,
            pair_window,
            dtype=torch.bool,
            device=pred_pose_encoding.device,
        )
        pair_template.fill_diagonal_(False)
        pair_i, pair_j = pair_template.nonzero(as_tuple=True)
        window_offsets = torch.arange(
            window_count, device=pred_pose_encoding.device
        )[:, None]
        pair_i_all = window_offsets + pair_i[None]
        pair_j_all = window_offsets + pair_j[None]
        pred_i = pred_c2w[:, pair_i_all]
        pred_j = pred_c2w[:, pair_j_all]
        gt_i = gt_c2w[:, pair_i_all]
        gt_j = gt_c2w[:, pair_j_all]
        pred_relative = torch.matmul(_invert_se3(pred_j), pred_i)
        gt_relative = torch.matmul(_invert_se3(gt_j), gt_i)
        pair_valid = (
            camera_valid[:, pair_i_all] & camera_valid[:, pair_j_all]
        )
        rel_rotation = _masked_mean(
            _geodesic_per_element(
                pred_relative[..., :3, :3], gt_relative[..., :3, :3]
            ),
            pair_valid,
        )
        rel_translation = _masked_mean(
            (pred_relative[..., :3, 3] - gt_relative[..., :3, 3])
            .abs()
            .sum(dim=-1),
            pair_valid,
        )
        relative_pose_pairs = pair_valid.sum().to(pred_pose_encoding.dtype)
    else:
        rel_rotation = pred_pose_encoding.sum() * 0.0
        rel_translation = pred_pose_encoding.sum() * 0.0
        relative_pose_pairs = pred_pose_encoding.new_zeros(())
    rel_pose = (
        float(rel_pose_rotation_component_weight) * rel_rotation
        + float(rel_pose_translation_component_weight)
        * float(pose_translation_weight)
        * rel_translation
    )
    return {
        "abs_pose": abs_pose,
        "abs_pose_translation": abs_pose_translation,
        "abs_pose_quaternion": abs_pose_quaternion,
        "abs_pose_fov": abs_pose_fov,
        "rel_pose": rel_pose,
        "rel_pose_rotation": rel_rotation,
        "rel_pose_translation": rel_translation,
        "relative_pose_pairs": relative_pose_pairs,
        "relative_pose_windows": pred_pose_encoding.new_tensor(window_count),
        "camera_metric_scale_mean": metric_scale.mean(),
        "camera_metric_scale_min": metric_scale.min(),
        "camera_metric_scale_max": metric_scale.max(),
        "pose_first_camera_protocol": pred_pose_encoding.new_tensor(
            float(
                coordinate_protocol
                in {
                    POSE_PROTOCOL_FIRST_CAMERA_SCALE,
                    POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE,
                }
            )
        ),
    }


def compute_stage2_world_pose_losses(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    relative_pose_window: int,
    pose_translation_weight: float,
    huber_beta: float,
    world_mano_global_orient_weight: float,
    world_mano_hand_pose_weight: float,
    world_mano_betas_weight: float,
    world_mano_translation_weight: float,
    coordinate_protocol: str = POSE_PROTOCOL_DATASET_WORLD,
    compute_world_mano: bool = True,
) -> Dict[str, torch.Tensor]:
    """Four metric Stage-2 losses in each sample's OpenCV world frame.

    The native camera pose encoding is decoded as W2C and converted once to
    C2W.  Ground-truth poses enter this boundary as OpenCV C2W.  No first-frame
    rebasing, scene-scale normalization, or prediction-to-target alignment is
    applied.
    """
    required_output = ("pred_camera_pose_enc", "pred_keypoints_3d_camera")
    required_target = (
        "camera_poses",
        "intrinsics",
        "image_hw",
        "world_joints",
        "world_joints_valid",
        "world_geometry_valid",
        "camera_trajectory_valid",
    )
    if compute_world_mano:
        required_target += ("world_mano", "world_mano_valid")
    missing = [key for key in required_output if key not in output]
    missing += [key for key in required_target if key not in target]
    if missing:
        raise KeyError("Stage-2 world/pose loss requires: " + ", ".join(missing))

    pred_pose_encoding = output["pred_camera_pose_enc"].float()
    pred_crop_joints = output.get("pred_keypoints_3d_crop_camera", output["pred_keypoints_3d_camera"]).float()
    gt_c2w = target["camera_poses"].to(
        device=pred_pose_encoding.device, dtype=pred_pose_encoding.dtype
    )
    pred_w2c = _pose_encoding_to_w2c(pred_pose_encoding)
    pred_c2w = _invert_se3(pred_w2c)
    gt_w2c = _invert_se3(gt_c2w)

    crop_to_full_camera = None
    if "mano_virtual_rotations" in output:
        crop_to_full_camera = torch.eye(4, device=pred_crop_joints.device, dtype=pred_crop_joints.dtype).expand(*pred_crop_joints.shape[:3], 4, 4).clone()
        crop_to_full_camera[..., :3, :3] = output["mano_virtual_rotations"].float().transpose(-1, -2)
    elif "wilor_camera_poses" in target:
        crop_c2w = target["wilor_camera_poses"].to(pred_crop_joints)
        if pred_crop_joints.shape[:3] != crop_c2w.shape[:3]:
            raise ValueError("Crop joints and crop camera poses must share B,T,hand dimensions")
        crop_to_full_camera = gt_w2c.unsqueeze(2) @ crop_c2w
    if output.get("camera_output_space") == "image_camera":
        pred_full_camera_joints = output["pred_keypoints_3d_camera"].float()
    elif crop_to_full_camera is not None:
        pred_full_camera_joints = _transform_points(crop_to_full_camera, pred_crop_joints)
    else:
        raise KeyError("World joints require image-camera predictions or an explicit crop rotation")
    first_camera = coordinate_protocol in {
        POSE_PROTOCOL_FIRST_CAMERA_SCALE, POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE}
    gt_gauge = _invert_se3(gt_c2w[:, 0])
    if first_camera:
        pred_c2w = _invert_se3(pred_c2w[:, :1]) @ pred_c2w
    pred_world_joints = _transform_points(
        pred_c2w.unsqueeze(2), pred_full_camera_joints
    )
    gt_world_joints = target["world_joints"].to(
        device=pred_world_joints.device, dtype=pred_world_joints.dtype
    )

    if first_camera:
        gt_world_joints = _transform_points(gt_gauge[:, None, None], gt_world_joints)
    world_geometry_valid = target["world_geometry_valid"].to(
        device=pred_world_joints.device, dtype=torch.bool
    )
    if first_camera:
        world_geometry_valid = world_geometry_valid & target["camera_trajectory_valid"][:, :1].bool()
    world_joint_valid = target["world_joints_valid"].to(
        device=pred_world_joints.device, dtype=torch.bool
    )
    world_joint_valid = (
        world_joint_valid
        & world_geometry_valid[..., None, None]
        & torch.isfinite(gt_world_joints).all(dim=-1)
    )
    world_mano_valid = target.get("world_mano_valid", world_joint_valid.any(-1)).to(
        device=pred_world_joints.device, dtype=torch.bool
    )
    world_mano_valid = (
        world_mano_valid & world_geometry_valid[..., None]
    )
    for validity_key in ("geometry_crop_valid", "mano_selected_valid"):
        if validity_key in output:
            crop_valid = output[validity_key].bool()
            world_joint_valid = world_joint_valid & crop_valid[..., None]
            world_mano_valid = world_mano_valid & crop_valid
    world_joints = _masked_smooth_l1_mean(
        pred_world_joints,
        gt_world_joints,
        world_joint_valid,
        beta=huber_beta,
    )
    world_mano = world_mano_global_orient = world_mano_hand_pose = world_mano_betas = world_mano_translation = pred_world_joints.new_zeros(())
    if compute_world_mano:
        if crop_to_full_camera is None:
            raise KeyError("World MANO diagnostics require a crop rotation")
        gt_world_mano = target["world_mano"].to(
            device=pred_world_joints.device, dtype=pred_world_joints.dtype
        )
        gt_world_mano_rotations = axis_angle_to_matrix(
            gt_world_mano[..., :48].reshape(
                *gt_world_mano.shape[:-1], 16, 3
            )
        )
        pred_mano = output.get("pred_mano_params")
        if not isinstance(pred_mano, dict):
            raise KeyError("Stage-2 world MANO loss requires pred_mano_params")
        missing_mano = [
            key
            for key in ("global_orient", "hand_pose", "betas", "transl")
            if key not in pred_mano
        ]
        if missing_mano:
            raise KeyError(
                "Stage-2 world MANO loss requires pred_mano_params fields: "
                + ", ".join(missing_mano)
            )
        pred_global_crop = pred_mano["global_orient"][..., 0, :, :].float()
        pred_global_world = torch.matmul(
            torch.matmul(
                pred_c2w.unsqueeze(2)[..., :3, :3],
                crop_to_full_camera[..., :3, :3],
            ),
            pred_global_crop,
        )
        gt_global_world = gt_world_mano_rotations[..., 0, :, :]
        if first_camera:
            gt_global_world = gt_gauge[:, None, None, :3, :3] @ gt_global_world
        world_mano_valid = (
            world_mano_valid
            & torch.isfinite(gt_world_mano).all(dim=-1)
        )
        world_mano_global_orient = _masked_mean(
            _geodesic_per_element(pred_global_world, gt_global_world),
            world_mano_valid,
        )
        world_mano_hand_pose = _masked_mean(
            _geodesic_per_element(
                pred_mano["hand_pose"].float(),
                gt_world_mano_rotations[..., 1:, :, :],
            ).mean(dim=-1),
            world_mano_valid,
        )
        world_mano_betas = _masked_mean(
            F.smooth_l1_loss(
                pred_mano["betas"].float(),
                gt_world_mano[..., 48:58],
                reduction="none",
                beta=float(huber_beta),
            ),
            world_mano_valid,
        )
        gt_world_translation = gt_world_mano[..., 58:61]
        if first_camera:
            gt_world_translation = _transform_points(gt_gauge[:, None, None], gt_world_translation.unsqueeze(-2)).squeeze(-2)
        pred_full_translation = _transform_points(
            crop_to_full_camera,
            pred_crop_joints[..., :1, :],
        ).squeeze(-2)
        pred_world_translation = _transform_points(
            pred_c2w.unsqueeze(2),
            pred_full_translation.unsqueeze(-2),
        ).squeeze(-2)
        world_mano_translation = _masked_smooth_l1_mean(
            pred_world_translation,
            gt_world_translation,
            world_mano_valid,
            beta=huber_beta,
        )
        world_mano = (
            float(world_mano_global_orient_weight) * world_mano_global_orient
            + float(world_mano_hand_pose_weight) * world_mano_hand_pose
            + float(world_mano_betas_weight) * world_mano_betas
            + float(world_mano_translation_weight) * world_mano_translation
        )

    pose_losses = compute_pose_losses(
        output,
        target,
        relative_pose_window=relative_pose_window,
        pose_translation_weight=pose_translation_weight,
        huber_beta=huber_beta,
        coordinate_protocol=coordinate_protocol,
    )
    return {
        "world_joints": world_joints,
        "world_mano": world_mano,
        "world_mano_global_orient": world_mano_global_orient,
        "world_mano_hand_pose": world_mano_hand_pose,
        "world_mano_betas": world_mano_betas,
        "world_mano_translation": world_mano_translation,
        **pose_losses,
    }


def temporal_time_seconds(target):
    """Use recorded time for a whole clip, otherwise its 10 FPS frame indices."""
    fallback = target["frame_ids"].double() * 0.1
    times = target.get("timestamps_seconds")
    available = target.get("timestamps_valid")
    if times is None or available is None:
        return fallback
    usable = available.bool().all(dim=1) & torch.isfinite(times).all(dim=1)
    usable = usable & (times[:, 1:] > times[:, :-1]).all(dim=1)
    return torch.where(usable[:, None], times.double(), fallback)


def compute_temporal_smoothness_loss(
    pred_crop_translation: torch.Tensor,
    crop_virtual_rotations: torch.Tensor,
    frame_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """ViDiHand acceleration penalty; frame_ids here contains time in seconds.

    InfinityHand tensors are [batch, frames, hand_slots, ...]. Crop-camera
    translation is mapped back with R_virtual^T before differencing. Velocities
    are divided by elapsed seconds, so uneven temporal sampling does
    not turn crop/frame spacing into apparent hand acceleration.
    """
    if pred_crop_translation.ndim != 4 or pred_crop_translation.shape[-1] != 3:
        raise ValueError(
            "pred_crop_translation must be [B,T,2,3], got "
            f"{tuple(pred_crop_translation.shape)}"
        )
    expected_rotation_shape = pred_crop_translation.shape[:-1] + (3, 3)
    if tuple(crop_virtual_rotations.shape) != expected_rotation_shape:
        raise ValueError(
            "crop_virtual_rotations must match translations as [B,T,2,3,3], "
            f"expected {expected_rotation_shape}, got "
            f"{tuple(crop_virtual_rotations.shape)}"
        )
    if tuple(frame_ids.shape) != tuple(pred_crop_translation.shape[:2]):
        raise ValueError(
            "frame_ids must be [B,T], expected "
            f"{tuple(pred_crop_translation.shape[:2])}, got {tuple(frame_ids.shape)}"
        )
    if pred_crop_translation.shape[1] < 3:
        return torch.nan_to_num(pred_crop_translation).sum() * 0.0

    rotations = crop_virtual_rotations.to(
        device=pred_crop_translation.device,
        dtype=pred_crop_translation.dtype,
    )
    # Unselected slots are placeholders rather than observations. Remove
    # them before arithmetic, including NaN placeholders, without suppressing
    # a non-finite prediction on a genuinely selected/supervised slot.
    safe_translation = torch.where(
        valid[..., None], pred_crop_translation, torch.zeros_like(pred_crop_translation)
    )
    safe_rotations = torch.where(
        valid[..., None, None], rotations,
        torch.eye(3, device=rotations.device, dtype=rotations.dtype),
    )
    source_translation = torch.matmul(
        safe_rotations.transpose(-1, -2), safe_translation.unsqueeze(-1)
    ).squeeze(-1)

    ids = frame_ids.to(device=pred_crop_translation.device)
    previous_gap = ids[:, 1:-1] - ids[:, :-2]
    next_gap = ids[:, 2:] - ids[:, 1:-1]
    positive_gaps = (previous_gap > 0) & (next_gap > 0)
    safe_previous_gap = previous_gap.clamp_min(1e-6).to(pred_crop_translation.dtype)
    safe_next_gap = next_gap.clamp_min(1e-6).to(pred_crop_translation.dtype)
    previous_velocity = (
        source_translation[:, 1:-1] - source_translation[:, :-2]
    ) / safe_previous_gap[..., None, None]
    next_velocity = (
        source_translation[:, 2:] - source_translation[:, 1:-1]
    ) / safe_next_gap[..., None, None]
    # Nonuniform central acceleration, normalized to the 0.1-second
    # reference step. At uniform 10 FPS this equals ViDiHand's second difference.
    velocity_change = (next_velocity - previous_velocity) * (
        2.0 * 0.1 ** 2 / (safe_previous_gap + safe_next_gap)
    )[..., None, None]

    # Do not compact valid observations in time: a missing middle slot
    # breaks the triple, so random left/right selection cannot create an
    # acceleration across a missing observation or across hand identities.
    valid_triple = (
        valid[:, 2:]
        & valid[:, 1:-1]
        & valid[:, :-2]
        & positive_gaps[..., None]
    )
    return _masked_mean(velocity_change.abs().sum(dim=-1), valid_triple)


def compute_betas_consistency_loss(
    pred_betas: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """ViDiHand per-slot betas consistency, adapted to batched clips."""
    if pred_betas.ndim != 4 or pred_betas.shape[-1] != 10:
        raise ValueError(
            f"pred_betas must be [B,T,2,10], got {tuple(pred_betas.shape)}"
        )
    valid = valid.to(device=pred_betas.device, dtype=torch.bool)
    count = valid.sum(dim=1).clamp_min(1).unsqueeze(-1).to(pred_betas.dtype)
    selected_betas = torch.where(
        valid.unsqueeze(-1), pred_betas, torch.zeros_like(pred_betas)
    )
    temporal_mean = (selected_betas.sum(dim=1) / count).detach()
    squared_error = (selected_betas - temporal_mean.unsqueeze(1)).pow(2).mean(dim=-1)
    return _masked_mean(squared_error, valid)


def aligned_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter = (rb - lt).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - inter
    iou = inter / union.clamp_min(1e-7)
    enc_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enc_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enc_area = (enc_rb - enc_lt).clamp_min(0).prod(dim=-1)
    return iou - (enc_area - union) / enc_area.clamp_min(1e-7)


def canonicalize_hand_targets(
    hand_ids: torch.Tensor,
    hand_bboxes: torch.Tensor,
    image_hw: Tuple[int, int],
    hand_mano_params: Optional[torch.Tensor] = None,
    hand_mano_valid: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Map arbitrary K hands to slot 0=right and slot 1=left."""
    ids = hand_ids.squeeze(-1).long()
    b, n, _ = ids.shape
    boxes = hand_bboxes.new_zeros((b, n, 2, 4))
    valid = torch.zeros((b, n, 2), dtype=torch.bool, device=ids.device)
    params = None
    params_valid = None
    if hand_mano_params is not None:
        params = hand_mano_params.new_zeros((b, n, 2, hand_mano_params.shape[-1]))
        params_valid = torch.zeros_like(valid)

    for slot in (0, 1):
        matches = ids == slot
        has_match = matches.any(dim=-1)
        index = matches.float().argmax(dim=-1)
        gather_box = index[..., None, None].expand(b, n, 1, 4)
        boxes[:, :, slot] = hand_bboxes.gather(2, gather_box).squeeze(2)
        valid[:, :, slot] = has_match & torch.isfinite(boxes[:, :, slot]).all(dim=-1)
        if params is not None:
            gather_param = index[..., None, None].expand(b, n, 1, hand_mano_params.shape[-1])
            params[:, :, slot] = hand_mano_params.gather(2, gather_param).squeeze(2)
            if hand_mano_valid is None:
                this_valid = torch.isfinite(params[:, :, slot]).all(dim=-1)
            else:
                mano_valid = hand_mano_valid.squeeze(-1) if hand_mano_valid.ndim == ids.ndim + 1 else hand_mano_valid
                this_valid = mano_valid.gather(2, index[..., None]).squeeze(-1).bool()
            params_valid[:, :, slot] = has_match & this_valid

    height, width = image_hw
    scale = boxes.new_tensor([width, height, width, height])
    boxes = (boxes / scale).clamp(0.0, 1.0)
    result = {"hand_valid": valid, "hand_boxes": boxes}
    if params is not None:
        result["mano_params"] = params
        result["mano_valid"] = params_valid
    return result


def _loss_reference(output):
    for key in ("pred_hand_probability", "pred_camera_pose_enc", "pred_keypoints_3d_camera"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    raise KeyError("No active head output is available for the loss")


class LingBotMapHandLoss(nn.Module):
    def __init__(
        self,
        joint_heatmap_range_margin: float = 0.15,
        relative_pose_window: int = 16,
        compute_stage2_diagnostics: bool = False,
        bbox_loss_enabled: bool = True,
        expose_graph_components: bool = False,
        camera_hand_loss_enabled: bool = True,
        stage2_coordinate_protocol: str = POSE_PROTOCOL_DATASET_WORLD,
        **weights: float,
    ) -> None:
        super().__init__()
        self.weights = LingBotMapHandLossWeights(**weights)
        self.joint_heatmap_range_margin = float(joint_heatmap_range_margin)
        self.relative_pose_window = int(relative_pose_window)
        self.compute_stage2_diagnostics = bool(compute_stage2_diagnostics)
        self.bbox_loss_enabled = bool(bbox_loss_enabled)
        self.expose_graph_components = bool(expose_graph_components)
        self.camera_hand_loss_enabled = bool(camera_hand_loss_enabled)
        self.stage2_coordinate_protocol = str(
            stage2_coordinate_protocol
        ).strip().lower()
        if self.stage2_coordinate_protocol not in {
            POSE_PROTOCOL_DATASET_WORLD,
            POSE_PROTOCOL_FIRST_CAMERA_SCALE,
            POSE_PROTOCOL_FIRST_CAMERA_NO_SCALE,
        }:
            raise ValueError(
                "unsupported stage2_coordinate_protocol: "
                f"{self.stage2_coordinate_protocol!r}"
            )
        if self.joint_heatmap_range_margin < 0.0:
            raise ValueError("joint_heatmap_range_margin must be non-negative")
        if self.relative_pose_window < 2:
            raise ValueError("relative_pose_window must be at least 2")
        if not self.bbox_loss_enabled and (
            self.weights.hand_bbox_smooth_l1 != 0.0
            or self.weights.hand_bbox_giou != 0.0
        ):
            raise ValueError(
                "bbox_loss_enabled=false requires both bbox loss weights to be zero"
            )

    def forward(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return a complete differentiable objective for each batch sample.

        Every sample first normalizes each loss by its own valid supervision.
        The public scalar is the equal-weight sample mean. The trainer can
        reject individual samples using objective_per_sample without changing
        the accepted samples' objectives. No NaN/Inf is hidden for active terms.
        """
        batch_size = int(_loss_reference(output).shape[0])
        if batch_size < 1:
            raise ValueError("loss requires a non-empty batch")
        if batch_size == 1:
            return self._forward_single(output, target)
        sample_losses = []
        projected = {}
        projection_keys = (
            "pred_keypoints_2d_crop_camera_projection",
            "pred_keypoints_2d_image_camera_projection",
        )
        for sample_index in range(batch_size):
            sample_output = _slice_loss_batch(output, sample_index, batch_size)
            sample_target = _slice_loss_batch(target, sample_index, batch_size)
            sample_losses.append(self._forward_single(sample_output, sample_target))
            for key in projection_keys:
                if key in sample_output:
                    projected.setdefault(key, []).append(sample_output[key])
        for key, values in projected.items():
            if len(values) == batch_size:
                output[key] = torch.cat(values, dim=0)
        losses = {
            key: torch.stack([sample[key] for sample in sample_losses]).mean(dim=0)
            for key in sample_losses[0]
            if key != "objective_per_sample"
        }
        losses["objective_per_sample"] = torch.stack(
            [sample["objective"] for sample in sample_losses]
        )
        losses["objective"] = losses["objective_per_sample"].mean()
        return losses

    def _forward_single(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reference = _loss_reference(output)
        probability = output.get("pred_hand_probability")
        if probability is None:
            if self.weights.hand_probability != 0:
                raise KeyError("Active hand presence loss requires mask-head output")
            probability = reference.new_zeros(target["hand_valid"].shape)
        device, dtype = probability.device, probability.dtype
        detection_valid = target["hand_valid"].to(device=device, dtype=torch.bool)
        valid = detection_valid
        for validity_key in ("geometry_crop_valid", "mano_selected_valid"):
            if validity_key in output:
                valid = valid & output[validity_key].to(device=device, dtype=torch.bool)

        # Keep the public/evaluation contract on ``pred_hand_probability``.
        # During training, use the paired logits when available so a sigmoid
        # value saturated at exactly 0/1 does not lose its gradient after
        # probability clamping.  The fallback preserves compatibility with
        # older callers that only provide the probability tensor.
        if self.weights.hand_probability == 0 and "pred_hand_probability" not in output:
            l_vis = probability.new_zeros(())
            l_vis_per_sample = probability.new_zeros(probability.shape[0])
        else:
            probability_logits = output.get("pred_hand_probability_logits")
            if probability_logits is not None:
                if tuple(probability_logits.shape) != tuple(probability.shape):
                    raise ValueError(
                        "pred_hand_probability_logits must match "
                        "pred_hand_probability: "
                        f"logits={tuple(probability_logits.shape)} "
                        f"probability={tuple(probability.shape)}"
                    )
                l_vis_per = F.binary_cross_entropy_with_logits(
                    probability_logits,
                    detection_valid.to(dtype),
                    reduction="none",
                )
            else:
                probability_safe = probability.clamp(1e-7, 1.0 - 1e-7)
                l_vis_per = -(
                    detection_valid.to(dtype) * torch.log(probability_safe)
                    + (~detection_valid).to(dtype) * torch.log(1.0 - probability_safe)
                )
            l_vis_per = l_vis_per.clamp(max=5.0)
            l_vis = l_vis_per.mean()
            l_vis_per_sample = _masked_mean_per_sample(l_vis_per, valid)
        if self.bbox_loss_enabled:
            if "pred_hand_boxes" not in output:
                raise KeyError(
                    "bbox loss is enabled but output lacks 'pred_hand_boxes'"
                )
            gt_boxes = target["hand_boxes"].to(device=device, dtype=dtype)
            pred_boxes = output["pred_hand_boxes"]
            bbox_smooth_l1 = _masked_mean(
                F.smooth_l1_loss(
                    pred_boxes,
                    gt_boxes,
                    reduction="none",
                    beta=0.1,
                ),
                valid,
            )
            bbox_giou = _masked_mean(
                1.0 - aligned_giou(pred_boxes, gt_boxes),
                valid,
            )
        else:
            bbox_smooth_l1 = probability.new_zeros(())
            bbox_giou = probability.new_zeros(())
        mask_logits = output.get("pred_hand_mask_logits")
        if mask_logits is None or "hand_masks" not in target:
            if self.weights.hand_mask_bce != 0.0 or self.weights.hand_mask_dice != 0.0:
                raise KeyError("hand mask loss requires target['hand_masks']")
            mask_bce = mask_dice = probability.new_zeros(())
            mask_bce_per_sample = probability.new_zeros(
                probability.shape[0]
            )
            mask_dice_per_sample = probability.new_zeros(
                probability.shape[0]
            )
        else:
            gt_masks = target["hand_masks"].to(
                device=device,
                dtype=mask_logits.dtype,
            )
            if tuple(gt_masks.shape) != tuple(mask_logits.shape):
                raise ValueError(
                    "hand_masks must match pred_hand_mask_logits: "
                    f"target={tuple(gt_masks.shape)} pred={tuple(mask_logits.shape)}"
                )
            mask_valid = target.get(
                "hand_mask_valid",
                torch.ones_like(valid, dtype=torch.bool),
            ).to(device=device, dtype=torch.bool)
            mask_bce_per_slot = F.binary_cross_entropy_with_logits(
                mask_logits,
                gt_masks.clamp(0.0, 1.0),
                reduction="none",
            ).mean(dim=(-1, -2))
            mask_bce = _masked_mean(mask_bce_per_slot, mask_valid)
            mask_bce_per_sample = _masked_mean_per_sample(
                mask_bce_per_slot,
                mask_valid,
            )
            mask_probability = torch.sigmoid(mask_logits)
            intersection = (mask_probability * gt_masks).sum(dim=(-1, -2))
            denominator = (
                mask_probability.sum(dim=(-1, -2))
                + gt_masks.sum(dim=(-1, -2))
            )
            mask_dice_per_slot = 1.0 - (2.0 * intersection + 1.0) / (
                denominator + 1.0
            )
            mask_dice = _masked_mean(mask_dice_per_slot, mask_valid)
            mask_dice_per_sample = _masked_mean_per_sample(
                mask_dice_per_slot,
                mask_valid,
            )
        detection_objective_per_sample = (
            self.weights.hand_probability * l_vis_per_sample
            + self.weights.hand_mask_bce * mask_bce_per_sample
            + self.weights.hand_mask_dice * mask_dice_per_sample
        )
        objective = (
            _weight_loss(l_vis, self.weights.hand_probability)
            + _weight_loss(bbox_smooth_l1, self.weights.hand_bbox_smooth_l1)
            + _weight_loss(bbox_giou, self.weights.hand_bbox_giou)
            + _weight_loss(mask_bce, self.weights.hand_mask_bce)
            + _weight_loss(mask_dice, self.weights.hand_mask_dice)
        )
        detection_component_weighted = objective
        losses: Dict[str, torch.Tensor] = {
            "L_vis": l_vis.detach(),
            "hand_bbox_smooth_l1_loss": bbox_smooth_l1.detach(),
            "hand_bbox_giou_loss": bbox_giou.detach(),
            "hand_mask_bce_loss": mask_bce.detach(),
            "hand_mask_dice_loss": mask_dice.detach(),
        }

        mano_valid = target.get("mano_valid", valid).to(device=device, dtype=torch.bool) & valid
        if "mano_params" in target:
            gt = target["mano_params"].to(device=device, dtype=dtype)
            if gt.shape[-1] != 61:
                raise ValueError(
                    "ViDiHand-style crop-camera loss requires MANO61 targets "
                    "[global_orient(3), hand_pose(45), betas(10), translation(3)], "
                    f"got final dimension {gt.shape[-1]}"
                )
            gt_rot = axis_angle_to_matrix(gt[..., :48].reshape(*gt.shape[:-1], 16, 3))
            pred = output["pred_mano_params"]
            pred_global_orient = output.get(
                "pred_global_orient_camera", pred["global_orient"]
            )
            rot_loss = _masked_mean(
                _geodesic_per_element(
                    pred_global_orient[..., 0, :, :],
                    gt_rot[..., 0, :, :],
                ),
                mano_valid,
            )
            pose_loss = _masked_mean(
                _geodesic_per_element(
                    pred["hand_pose"], gt_rot[..., 1:, :, :]
                ).mean(dim=-1),
                mano_valid,
            )
            shape_loss = _masked_mean(
                (pred["betas"] - gt[..., 48:58]).pow(2).mean(dim=-1),
                mano_valid,
            )
            if "transl" not in pred:
                if self.weights.trans_loss_weight != 0.0:
                    raise KeyError(
                        "crop-camera translation loss requires "
                        "output['pred_mano_params']['transl']"
                    )
                translation_loss = probability.new_zeros(())
                pred_translation = None
            else:
                pred_translation = output.get(
                    "pred_mano_transl_camera", pred["transl"]
                )
                gt_translation = gt[..., 58:61].to(device=device, dtype=dtype)
                translation_loss = _masked_mean(
                    (pred_translation - gt_translation).pow(2).mean(dim=-1),
                    mano_valid,
                )
        else:
            zero = probability.new_zeros(())
            rot_loss = pose_loss = shape_loss = translation_loss = zero
            pred_translation = None
        go_weighted = _weight_loss(rot_loss, self.weights.go_loss_weight)
        hp_weighted = _weight_loss(pose_loss, self.weights.hp_loss_weight)
        betas_weighted = _weight_loss(shape_loss, self.weights.betas_loss_weight)
        trans_weighted = _weight_loss(translation_loss, self.weights.trans_loss_weight)
        objective = (
            objective
            + go_weighted
            + hp_weighted
            + betas_weighted
            + trans_weighted
        )
        losses.update(
            go=rot_loss.detach(),
            go_weighted=go_weighted.detach(),
            hp=pose_loss.detach(),
            hp_weighted=hp_weighted.detach(),
            betas=shape_loss.detach(),
            betas_weighted=betas_weighted.detach(),
            trans=translation_loss.detach(),
            trans_weighted=trans_weighted.detach(),
        )

        # ViDiHand Stage-2 supervision for the final joint prediction:
        #   P_final = P_init + joint_offset_2d.
        # Keep this separate from reproj below, which is the existing
        # MANO/camera perspective reprojection term.  There is intentionally no
        # auxiliary loss on pred_keypoints_2d_initial_crop.
        direct_2d_loss = probability.new_zeros(())
        if (
            output.get("model_variant") == "mutual_joint_fusion"
            and self.weights.joints2d_direct_weight != 0.0
        ):
            if "pred_keypoints_2d" not in output:
                raise KeyError(
                    "mutual_joint_fusion direct-2D loss requires "
                    "output['pred_keypoints_2d']"
                )
            if "keypoints_2d_crop" in target:
                gt2d_direct = target["keypoints_2d_crop"].to(
                    device=device,
                    dtype=dtype,
                )
            elif all(
                key in output
                for key in ("mano_virtual_rotations", "mano_crop_intrinsics")
            ) and "keypoints_3d" in target:
                gt_camera = target["keypoints_3d"].to(
                    device=device, dtype=torch.float32
                )
                image_to_crop = output["mano_virtual_rotations"].float()
                gt_crop = torch.matmul(
                    image_to_crop[..., None, :, :], gt_camera[..., None]
                ).squeeze(-1)
                crop_k = output["mano_crop_intrinsics"].float()
                projected = torch.matmul(
                    crop_k[..., None, :, :], gt_crop[..., None]
                ).squeeze(-1)
                crop_pixels = projected[..., :2] / projected[..., 2:].clamp_min(
                    1.0e-6
                )
                crop_hw = crop_pixels.new_tensor((192.0, 256.0))
                crop_valid = target.get(
                    "keypoints_3d_joint_valid",
                    torch.isfinite(gt_camera).all(dim=-1),
                ).to(device=device, dtype=torch.bool)
                crop_valid = crop_valid & (gt_crop[..., 2] > 1.0e-6)
                gt2d_direct = torch.cat(
                    (
                        crop_pixels / crop_hw - 0.5,
                        crop_valid.to(dtype=torch.float32)[..., None],
                    ),
                    dim=-1,
                ).to(dtype=dtype)
            else:
                raise KeyError(
                    "mutual_joint_fusion direct-2D loss requires either "
                    "target['keypoints_2d_crop'] or dynamic virtual-camera "
                    "targets"
                )
            pred2d_direct = output["pred_keypoints_2d"]
            if pred2d_direct.shape != gt2d_direct[..., :2].shape:
                raise ValueError(
                    "refined 2D prediction and target shapes must match, got "
                    f"{tuple(pred2d_direct.shape)} and "
                    f"{tuple(gt2d_direct[..., :2].shape)}"
                )

            direct_valid = gt2d_direct[..., 2] > 0
            direct_valid = direct_valid & valid[..., None]
            finite_gt = torch.isfinite(gt2d_direct[..., :2]).all(dim=-1)
            direct_valid = direct_valid & finite_gt

            # Stage-2 per-hand OOS gating, adapted from [0,1] image coordinates
            # to this model's centered [-0.5,0.5] crop-coordinate interface.
            gt_u = gt2d_direct[..., 0]
            gt_v = gt2d_direct[..., 1]
            on_screen = (
                (gt_u >= -0.5)
                & (gt_u <= 0.5)
                & (gt_v >= -0.5)
                & (gt_v <= 0.5)
                & direct_valid
            )
            hand_in = on_screen.any(dim=-1, keepdim=True)
            margin = self.joint_heatmap_range_margin
            in_margin = (
                (gt_u >= -0.5 - margin)
                & (gt_u <= 0.5 + margin)
                & (gt_v >= -0.5 - margin)
                & (gt_v <= 0.5 + margin)
            )
            direct_valid = direct_valid & hand_in & in_margin

            direct_error = (
                pred2d_direct - gt2d_direct[..., :2]
            ).abs().sum(dim=-1)
            direct_2d_loss = _masked_mean(direct_error, direct_valid)

        direct_2d_weighted = (
            _weight_loss(direct_2d_loss, self.weights.joints2d_direct_weight)
        )
        objective = objective + direct_2d_weighted
        losses["joints2d_direct"] = direct_2d_loss.detach()
        losses["joints2d_direct_weighted"] = direct_2d_weighted.detach()

        if "keypoints_2d_crop" in target and "pred_keypoints_2d" in output:
            gt2d_crop = target["keypoints_2d_crop"].to(device=device, dtype=dtype)
            pred2d_crop = output["pred_keypoints_2d"]
            if (
                "crop_intrinsics" in target
                and "crop_image_hw" in target
                and "pred_keypoints_3d_camera" in output
            ):
                pred_camera = output["pred_keypoints_3d_camera"]
                crop_intrinsics = target["crop_intrinsics"].to(
                    device=device, dtype=dtype
                )
                crop_hw = target["crop_image_hw"].to(
                    device=device, dtype=dtype
                )
                crop_height, crop_width = crop_hw.unbind()
                pred_z = pred_camera[..., 2].clamp_min(1e-6)
                pred_u = (
                    pred_camera[..., 0]
                    / pred_z
                    * crop_intrinsics[..., 0, 0][..., None]
                    + crop_intrinsics[..., 0, 2][..., None]
                )
                pred_v = (
                    pred_camera[..., 1]
                    / pred_z
                    * crop_intrinsics[..., 1, 1][..., None]
                    + crop_intrinsics[..., 1, 2][..., None]
                )
                pred2d_crop = torch.stack(
                    (
                        pred_u / crop_width - 0.5,
                        pred_v / crop_height - 0.5,
                    ),
                    dim=-1,
                )
                output["pred_keypoints_2d_crop_camera_projection"] = (
                    pred2d_crop
                )
            kp2d_valid = gt2d_crop[..., 2] > 0
            kp2d_valid = kp2d_valid & valid[..., None]
            kp2d_valid = kp2d_valid & torch.isfinite(gt2d_crop[..., :2]).all(dim=-1)
            loss_2d = _masked_mean(
                (pred2d_crop - gt2d_crop[..., :2]).abs().sum(dim=-1),
                kp2d_valid,
            )
        elif all(
            key in target for key in ("keypoints_2d_image", "intrinsics", "image_hw")
        ) and "pred_keypoints_3d_camera" in output:
            gt2d = target["keypoints_2d_image"].to(device=device, dtype=dtype)
            pred_camera = output["pred_keypoints_3d_camera"]
            image_intrinsics = target["intrinsics"].to(
                device=device, dtype=dtype
            )
            image_hw = target["image_hw"].to(device=device, dtype=dtype)
            image_height, image_width = image_hw.unbind()
            pred_z = pred_camera[..., 2].clamp_min(1.0e-6)
            pred_u = (
                pred_camera[..., 0]
                / pred_z
                * image_intrinsics[..., 0, 0][:, :, None, None]
                + image_intrinsics[..., 0, 2][:, :, None, None]
            ) / image_width
            pred_v = (
                pred_camera[..., 1]
                / pred_z
                * image_intrinsics[..., 1, 1][:, :, None, None]
                + image_intrinsics[..., 1, 2][:, :, None, None]
            ) / image_height
            pred2d_image = torch.stack((pred_u, pred_v), dim=-1)
            kp2d_valid = gt2d[..., 2] > 0
            kp2d_valid = kp2d_valid & valid[..., None]
            kp2d_valid = kp2d_valid & torch.isfinite(gt2d[..., :2]).all(dim=-1)
            loss_2d = _masked_mean(
                (pred2d_image - gt2d[..., :2]).abs().sum(dim=-1),
                kp2d_valid,
            )
            output["pred_keypoints_2d_image_camera_projection"] = pred2d_image
        elif "keypoints_2d_image" in target and "pred_keypoints_2d" in output and "mano_boxes" in output:
            gt2d = target["keypoints_2d_image"].to(device=device, dtype=dtype)
            boxes = output["mano_boxes"].to(device=device, dtype=dtype)
            xy1 = boxes[..., None, :2]
            wh = (boxes[..., None, 2:] - boxes[..., None, :2]).clamp_min(1e-6)
            gt2d_crop = (gt2d[..., :2] - xy1) / wh - 0.5
            kp2d_valid = gt2d[..., 2] > 0
            kp2d_valid = kp2d_valid & valid[..., None]
            kp2d_valid = kp2d_valid & torch.isfinite(gt2d_crop).all(dim=-1)
            loss_2d = _masked_mean(
                (output["pred_keypoints_2d"] - gt2d_crop).abs().sum(dim=-1),
                kp2d_valid,
            )
        else:
            loss_2d = probability.new_zeros(())
        reprojection_2d_weighted = _weight_loss(loss_2d, self.weights.reproj_loss_weight)
        objective = objective + reprojection_2d_weighted
        losses["reproj"] = loss_2d.detach()
        losses["reproj_weighted"] = reprojection_2d_weighted.detach()

        if "keypoints_3d" in target:
            gt_joints = target["keypoints_3d"].to(device=device, dtype=dtype)
            joint_point_valid = target.get(
                "keypoints_3d_joint_valid",
                target.get("keypoints_3d_valid", mano_valid),
            ).to(device=device, dtype=torch.bool)
            # A sampled/valid crop is required even when the loader supplies
            # an explicit per-joint validity mask for both hands.
            hand_joint_valid = valid
            while hand_joint_valid.ndim < joint_point_valid.ndim:
                hand_joint_valid = hand_joint_valid.unsqueeze(-1)
            joint_point_valid = joint_point_valid & hand_joint_valid
            if "pred_keypoints_3d_camera" in output:
                camera_joint_loss = _masked_mean(
                    (output["pred_keypoints_3d_camera"] - gt_joints).abs(),
                    joint_point_valid,
                )
            else:
                if self.weights.joints_loss_weight != 0.0:
                    raise KeyError(
                        "camera-space 3D loss requires pred_keypoints_3d_camera; "
                        "pass batch intrinsics into LingBotMapHand.forward"
                    )
                camera_joint_loss = probability.new_zeros(())
        else:
            camera_joint_loss = probability.new_zeros(())
        joints_weighted = _weight_loss(camera_joint_loss, self.weights.joints_loss_weight)
        objective = objective + joints_weighted

        if self.weights.velocity_smooth_weight != 0.0:
            if pred_translation is None:
                raise KeyError(
                    "temporal translation loss requires crop-camera translation"
                )
            missing_temporal = [
                key
                for key in (("frame_ids",) if output.get("camera_output_space") == "image_camera" else ("crop_virtual_rotations", "frame_ids"))
                if key not in target
            ]
            if missing_temporal:
                raise KeyError(
                    "source-camera temporal translation loss requires target "
                    "fields: " + ", ".join(missing_temporal)
                )
            temporal_translation_loss = compute_temporal_smoothness_loss(
                pred_translation,
                torch.eye(3, device=pred_translation.device, dtype=pred_translation.dtype).expand(*pred_translation.shape[:-1],3,3) if output.get("camera_output_space") == "image_camera" else target["crop_virtual_rotations"],
                temporal_time_seconds(target),
                mano_valid,
            )
        else:
            temporal_translation_loss = probability.new_zeros(())
        if self.weights.betas_consistency_weight != 0.0:
            if "pred_mano_params" not in output:
                raise KeyError(
                    "temporal shape loss requires output['pred_mano_params']"
                )
            temporal_shape_loss = compute_betas_consistency_loss(
                output["pred_mano_params"]["betas"], mano_valid
            )
        else:
            temporal_shape_loss = probability.new_zeros(())
        velocity_smooth_weighted = (
            _weight_loss(temporal_translation_loss, self.weights.velocity_smooth_weight)
        )
        betas_consistency_weighted = (
            _weight_loss(temporal_shape_loss, self.weights.betas_consistency_weight)
        )
        objective = (
            objective
            + velocity_smooth_weighted
            + betas_consistency_weighted
        )
        losses["joints"] = camera_joint_loss.detach()
        losses["joints_weighted"] = joints_weighted.detach()
        losses["velocity_smooth"] = temporal_translation_loss.detach()
        losses["velocity_smooth_weighted"] = velocity_smooth_weighted.detach()
        losses["betas_consistency"] = temporal_shape_loss.detach()
        losses["betas_consistency_weighted"] = (
            betas_consistency_weighted.detach()
        )
        # Preserve the complete pre-existing objective as L_stage1.  Stage-2
        # geometry is optional so all historical configs remain bitwise on the
        # old route when their new lambda values are zero.
        full_stage1_raw = objective
        configured_camera_hand_component = (
            go_weighted + hp_weighted + betas_weighted + trans_weighted
            + direct_2d_weighted + reprojection_2d_weighted + joints_weighted
            + velocity_smooth_weighted + betas_consistency_weighted
        )
        camera_hand_component_weighted = (
            configured_camera_hand_component
            if self.camera_hand_loss_enabled
            else configured_camera_hand_component.new_zeros(())
        )
        stage1_raw = detection_component_weighted + camera_hand_component_weighted
        world_stage2_enabled = self.compute_stage2_diagnostics or any(
            weight != 0.0
            for weight in (
                self.weights.world_joints_loss_weight,
                self.weights.world_mano_loss_weight,
            )
        )
        pose_stage2_enabled = self.compute_stage2_diagnostics or any(
            weight != 0.0
            for weight in (
                self.weights.abs_pose_loss_weight,
                self.weights.rel_pose_loss_weight,
            )
        )
        stage2_enabled = world_stage2_enabled or pose_stage2_enabled
        stage2_supervision = target.get("stage2_supervision_valid")
        if stage2_supervision is None:
            stage2_supervision_fraction = 1.0
            stage2_supervision_valid = True
        else:
            supervision = stage2_supervision.to(
                device=stage1_raw.device, dtype=torch.bool
            )
            stage2_supervision_fraction = float(
                supervision.float().mean().item()
            )
            stage2_supervision_valid = bool(supervision.any().item())
        stage1_fallback_multiplier = (
            2.0 if stage2_enabled and not stage2_supervision_valid else 1.0
        )
        stage1_effective_weight = (
            self.weights.stage1_loss_weight * stage1_fallback_multiplier
        )
        stage1_weighted = _weight_loss(stage1_raw, stage1_effective_weight)
        zero = stage1_raw.new_zeros(())
        stage2 = {
            "world_joints": zero,
            "world_mano": zero,
            "world_mano_global_orient": zero,
            "world_mano_hand_pose": zero,
            "world_mano_betas": zero,
            "world_mano_translation": zero,
            "abs_pose": zero,
            "abs_pose_translation": zero,
            "abs_pose_quaternion": zero,
            "abs_pose_fov": zero,
            "rel_pose": zero,
            "rel_pose_rotation": zero,
            "rel_pose_translation": zero,
            "relative_pose_pairs": zero,
            "relative_pose_windows": zero,
            "camera_metric_scale_mean": zero + 1.0,
            "camera_metric_scale_min": zero + 1.0,
            "camera_metric_scale_max": zero + 1.0,
            "pose_first_camera_protocol": zero,
        }
        if world_stage2_enabled:
            stage2 = compute_stage2_world_pose_losses(
                output,
                target,
                compute_world_mano=(self.compute_stage2_diagnostics or self.weights.world_mano_loss_weight != 0),
                relative_pose_window=self.relative_pose_window,
                pose_translation_weight=self.weights.pose_translation_loss_weight,
                huber_beta=self.weights.stage2_huber_beta,
                coordinate_protocol=self.stage2_coordinate_protocol,
                world_mano_global_orient_weight=(
                    self.weights.world_mano_global_orient_weight
                ),
                world_mano_hand_pose_weight=(
                    self.weights.world_mano_hand_pose_weight
                ),
                world_mano_betas_weight=self.weights.world_mano_betas_weight,
                world_mano_translation_weight=(
                    self.weights.world_mano_translation_weight
                ),
            )
        elif pose_stage2_enabled:
            stage2.update(
                compute_pose_losses(
                    output,
                    target,
                    relative_pose_window=self.relative_pose_window,
                    pose_translation_weight=(
                        self.weights.pose_translation_loss_weight
                    ),
                    abs_pose_translation_component_weight=(
                        self.weights.abs_pose_translation_component_weight
                    ),
                    abs_pose_quaternion_component_weight=(
                        self.weights.abs_pose_quaternion_component_weight
                    ),
                    abs_pose_fov_component_weight=(
                        self.weights.abs_pose_fov_component_weight
                    ),
                    rel_pose_rotation_component_weight=(
                        self.weights.rel_pose_rotation_component_weight
                    ),
                    rel_pose_translation_component_weight=(
                        self.weights.rel_pose_translation_component_weight
                    ),
                    huber_beta=self.weights.stage2_huber_beta,
                    coordinate_protocol=self.stage2_coordinate_protocol,
                )
            )
        world_joints_weighted = (
            _weight_loss(stage2["world_joints"], self.weights.world_joints_loss_weight)
        )
        world_mano_weighted = (
            _weight_loss(stage2["world_mano"], self.weights.world_mano_loss_weight)
        )
        abs_pose_weighted = (
            _weight_loss(stage2["abs_pose"], self.weights.abs_pose_loss_weight)
        )
        rel_pose_weighted = (
            _weight_loss(stage2["rel_pose"], self.weights.rel_pose_loss_weight)
        )
        stage2_weighted = (
            world_joints_weighted
            + world_mano_weighted
            + abs_pose_weighted
            + rel_pose_weighted
        )
        detection_group_weighted = (
            _weight_loss(detection_component_weighted, stage1_effective_weight)
        )
        camera_hand_group_weighted = (
            _weight_loss(camera_hand_component_weighted, stage1_effective_weight)
        )
        world_hand_group_weighted = (
            world_joints_weighted + world_mano_weighted
        )
        pose_group_weighted = abs_pose_weighted + rel_pose_weighted
        grouped_objective = (
            detection_group_weighted
            + camera_hand_group_weighted
            + world_hand_group_weighted
            + pose_group_weighted
        )
        objective = stage1_weighted + stage2_weighted
        scalar = stage1_raw.new_tensor
        def availability_mean(key: str) -> torch.Tensor:
            value = target.get(key)
            if value is None:
                return scalar(1.0)
            return value.to(device=stage1_raw.device, dtype=stage1_raw.dtype).mean()

        losses.update(
            loss_stage1_raw=stage1_raw.detach(),
            loss_stage1_base_weight=scalar(self.weights.stage1_loss_weight),
            loss_stage1_fallback_multiplier=scalar(
                stage1_fallback_multiplier
            ),
            loss_stage1_weight=scalar(stage1_effective_weight),
            loss_stage1_weighted=stage1_weighted.detach(),
            loss_stage2_supervision_valid=scalar(
                stage2_supervision_fraction
            ),
            loss_stage2_world_joints_available=availability_mean(
                "stage2_world_joints_available"
            ).detach(),
            loss_stage2_world_mano_available=availability_mean(
                "stage2_world_mano_available"
            ).detach(),
            loss_stage2_pose_available=availability_mean(
                "stage2_pose_available"
            ).detach(),
            loss_world_joints_raw=stage2["world_joints"].detach(),
            loss_world_joints_weight=scalar(
                self.weights.world_joints_loss_weight
            ),
            loss_world_joints_weighted=world_joints_weighted.detach(),
            loss_world_mano_raw=stage2["world_mano"].detach(),
            loss_world_mano_weight=scalar(self.weights.world_mano_loss_weight),
            loss_world_mano_weighted=world_mano_weighted.detach(),
            loss_world_mano_global_orient_raw=stage2[
                "world_mano_global_orient"
            ].detach(),
            loss_world_mano_hand_pose_raw=stage2[
                "world_mano_hand_pose"
            ].detach(),
            loss_world_mano_betas_raw=stage2[
                "world_mano_betas"
            ].detach(),
            loss_world_mano_translation_raw=stage2[
                "world_mano_translation"
            ].detach(),
            loss_abs_pose_raw=stage2["abs_pose"].detach(),
            loss_abs_pose_translation_raw=stage2[
                "abs_pose_translation"
            ].detach(),
            loss_abs_pose_quaternion_raw=stage2[
                "abs_pose_quaternion"
            ].detach(),
            loss_abs_pose_fov_raw=stage2["abs_pose_fov"].detach(),
            loss_abs_pose_weight=scalar(self.weights.abs_pose_loss_weight),
            loss_abs_pose_translation_weight=scalar(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_translation_component_weight
            ),
            loss_abs_pose_translation_weighted=(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_translation_component_weight
                * stage2["abs_pose_translation"]
            ).detach(),
            loss_abs_pose_quaternion_weight=scalar(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_quaternion_component_weight
            ),
            loss_abs_pose_quaternion_weighted=(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_quaternion_component_weight
                * stage2["abs_pose_quaternion"]
            ).detach(),
            loss_abs_pose_fov_weight=scalar(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_fov_component_weight
            ),
            loss_abs_pose_fov_weighted=(
                self.weights.abs_pose_loss_weight
                * self.weights.abs_pose_fov_component_weight
                * stage2["abs_pose_fov"]
            ).detach(),
            loss_abs_pose_weighted=abs_pose_weighted.detach(),
            loss_rel_pose_raw=stage2["rel_pose"].detach(),
            loss_rel_pose_weight=scalar(self.weights.rel_pose_loss_weight),
            loss_rel_pose_weighted=rel_pose_weighted.detach(),
            loss_rel_pose_rotation_raw=stage2["rel_pose_rotation"].detach(),
            loss_rel_pose_translation_raw=stage2[
                "rel_pose_translation"
            ].detach(),
            loss_rel_pose_rotation_weight=scalar(
                self.weights.rel_pose_loss_weight
                * self.weights.rel_pose_rotation_component_weight
            ),
            loss_rel_pose_rotation_weighted=(
                self.weights.rel_pose_loss_weight
                * self.weights.rel_pose_rotation_component_weight
                * stage2["rel_pose_rotation"]
            ).detach(),
            loss_rel_pose_translation_weight=scalar(
                self.weights.rel_pose_loss_weight
                * self.weights.rel_pose_translation_component_weight
                * self.weights.pose_translation_loss_weight
            ),
            loss_rel_pose_translation_weighted=(
                self.weights.rel_pose_loss_weight
                * self.weights.rel_pose_translation_component_weight
                * self.weights.pose_translation_loss_weight
                * stage2["rel_pose_translation"]
            ).detach(),
            loss_pose_translation_weight=scalar(
                self.weights.pose_translation_loss_weight
            ),
            loss_relative_pose_pairs=stage2["relative_pose_pairs"].detach(),
            loss_relative_pose_windows=stage2[
                "relative_pose_windows"
            ].detach(),
            camera_metric_scale_mean=stage2[
                "camera_metric_scale_mean"
            ].detach(),
            camera_metric_scale_min=stage2[
                "camera_metric_scale_min"
            ].detach(),
            camera_metric_scale_max=stage2[
                "camera_metric_scale_max"
            ].detach(),
            pose_first_camera_protocol=stage2[
                "pose_first_camera_protocol"
            ].detach(),
            loss_stage2_weighted=stage2_weighted.detach(),
            loss_detection_weighted=detection_group_weighted.detach(),
            loss_camera_hand_weighted=camera_hand_group_weighted.detach(),
            loss_camera_hand_configured=(
                stage1_effective_weight * configured_camera_hand_component
            ).detach(),
            loss_camera_hand_enabled=scalar(
                float(self.camera_hand_loss_enabled)
            ),
            loss_world_hand_weighted=world_hand_group_weighted.detach(),
            loss_pose_weighted=pose_group_weighted.detach(),
            loss_grouped_objective=grouped_objective.detach(),
        )
        # Hierarchical aliases make raw/weight/weighted calibration records
        # explicit without breaking the existing flat JSON consumers.
        losses.update(
            {
                "raw/stage1": stage1_raw.detach(),
                "raw/world_joints": stage2["world_joints"].detach(),
                "raw/world_mano_global_orient": stage2[
                    "world_mano_global_orient"
                ].detach(),
                "raw/world_mano_hand_pose": stage2[
                    "world_mano_hand_pose"
                ].detach(),
                "raw/world_mano_betas": stage2["world_mano_betas"].detach(),
                "raw/world_mano_translation": stage2[
                    "world_mano_translation"
                ].detach(),
                "raw/world_mano": stage2["world_mano"].detach(),
                "raw/abs_pose": stage2["abs_pose"].detach(),
                "raw/abs_pose_translation": stage2[
                    "abs_pose_translation"
                ].detach(),
                "raw/abs_pose_quaternion": stage2[
                    "abs_pose_quaternion"
                ].detach(),
                "raw/abs_pose_fov": stage2["abs_pose_fov"].detach(),
                "raw/rel_pose_rotation": stage2["rel_pose_rotation"].detach(),
                "raw/rel_pose_translation": stage2[
                    "rel_pose_translation"
                ].detach(),
                "raw/rel_pose": stage2["rel_pose"].detach(),
                "weight/stage1_base": scalar(
                    self.weights.stage1_loss_weight
                ),
                "weight/stage1_fallback_multiplier": scalar(
                    stage1_fallback_multiplier
                ),
                "weight/stage1": scalar(stage1_effective_weight),
                "weight/world_joints": scalar(
                    self.weights.world_joints_loss_weight
                ),
                "weight/world_mano": scalar(self.weights.world_mano_loss_weight),
                "weight/world_mano_global_orient": scalar(
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_global_orient_weight
                ),
                "weight/world_mano_hand_pose": scalar(
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_hand_pose_weight
                ),
                "weight/world_mano_betas": scalar(
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_betas_weight
                ),
                "weight/world_mano_translation": scalar(
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_translation_weight
                ),
                "weight/abs_pose": scalar(self.weights.abs_pose_loss_weight),
                "weight/rel_pose": scalar(self.weights.rel_pose_loss_weight),
                "weight/abs_pose_translation": scalar(
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_translation_component_weight
                ),
                "weight/abs_pose_quaternion": scalar(
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_quaternion_component_weight
                ),
                "weight/abs_pose_fov": scalar(
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_fov_component_weight
                ),
                "weight/rel_pose_rotation": scalar(
                    self.weights.rel_pose_loss_weight
                    * self.weights.rel_pose_rotation_component_weight
                ),
                "weight/rel_pose_translation": scalar(
                    self.weights.rel_pose_loss_weight
                    * self.weights.rel_pose_translation_component_weight
                    * self.weights.pose_translation_loss_weight
                ),
                "weighted/stage1": stage1_weighted.detach(),
                "weighted/world_joints": world_joints_weighted.detach(),
                "weighted/world_mano": world_mano_weighted.detach(),
                "weighted/world_mano_global_orient": (
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_global_orient_weight
                    * stage2["world_mano_global_orient"]
                ).detach(),
                "weighted/world_mano_hand_pose": (
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_hand_pose_weight
                    * stage2["world_mano_hand_pose"]
                ).detach(),
                "weighted/world_mano_betas": (
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_betas_weight
                    * stage2["world_mano_betas"]
                ).detach(),
                "weighted/world_mano_translation": (
                    self.weights.world_mano_loss_weight
                    * self.weights.world_mano_translation_weight
                    * stage2["world_mano_translation"]
                ).detach(),
                "weighted/abs_pose": abs_pose_weighted.detach(),
                "weighted/abs_pose_translation": (
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_translation_component_weight
                    * stage2["abs_pose_translation"]
                ).detach(),
                "weighted/abs_pose_quaternion": (
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_quaternion_component_weight
                    * stage2["abs_pose_quaternion"]
                ).detach(),
                "weighted/abs_pose_fov": (
                    self.weights.abs_pose_loss_weight
                    * self.weights.abs_pose_fov_component_weight
                    * stage2["abs_pose_fov"]
                ).detach(),
                "weighted/rel_pose": rel_pose_weighted.detach(),
                "weighted/rel_pose_rotation": (
                    self.weights.rel_pose_loss_weight
                    * self.weights.rel_pose_rotation_component_weight
                    * stage2["rel_pose_rotation"]
                ).detach(),
                "weighted/rel_pose_translation": (
                    self.weights.rel_pose_loss_weight
                    * self.weights.rel_pose_translation_component_weight
                    * self.weights.pose_translation_loss_weight
                    * stage2["rel_pose_translation"]
                ).detach(),
                "weighted/stage2": stage2_weighted.detach(),
                "weighted/detection": detection_group_weighted.detach(),
                "weighted/camera_hand": camera_hand_group_weighted.detach(),
                "weighted/world_hand": world_hand_group_weighted.detach(),
                "weighted/pose": pose_group_weighted.detach(),
                "weighted/grouped_objective": grouped_objective.detach(),
            }
        )
        # A camera annotation declared valid but containing NaN/Inf is
        # corrupt supervision, not an absent label. Preserve that fact for
        # per-sample rejection. SE(3) inversion above is analytic (R^T,-R^Tt),
        # so no linalg exception handler or broad exception swallowing is
        # needed. Shape/configuration errors still raise normally.
        active_camera_supervision = any(weight != 0.0 for weight in (
            self.weights.world_joints_loss_weight, self.weights.world_mano_loss_weight,
            self.weights.abs_pose_loss_weight, self.weights.rel_pose_loss_weight,
        ))
        if active_camera_supervision and "camera_trajectory_valid" in target:
            declared_camera_valid = target["camera_trajectory_valid"].bool()
            bad_camera = declared_camera_valid & (
                ~torch.isfinite(target["camera_poses"]).all(dim=(-1, -2))
                | ~torch.isfinite(target["intrinsics"]).all(dim=(-1, -2))
            )
            objective = torch.where(
                bad_camera.any().to(device=objective.device),
                objective.new_full((), float("nan")), objective
            )
        losses["objective"] = objective
        # Includes every active loss term and retains its autograd graph.
        losses["objective_per_sample"] = objective.reshape(1)
        if self.expose_graph_components:
            # Private graph-bearing diagnostic. The trainer removes this key
            # before scalar logging/checkpoint history.
            losses["_camera_hand_objective_graph"] = (
                camera_hand_group_weighted
            )
        return losses
