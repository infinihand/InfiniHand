"""Centered virtual-camera full-image transform for LingBotMap.

This module is deliberately independent from the DOSS reader.  It transforms
the complete canonical DOSS field dictionary from source image coordinates to
the final LingBotMap virtual camera in one step:

    source HxW -> mask-balanced crop selection -> virtual-camera warp 378x518

The output principal point is the image centre.  Therefore camera-frame MANO,
camera joints, and world-to-camera extrinsics rotate into the virtual camera.
World-frame MANO/joints and point-validity flags remain unchanged.  The DOSS
loader recomputes ``hand_valid`` after this transform with the same visibility
rule as evaluation, so the label corresponds to the final model image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


TARGET_HEIGHT = 378
TARGET_WIDTH = 518
EVALUATION_HAND_MIN_DEPTH_M = 0.01
SPATIAL_FIELDS = {
    'images',
    'intrinsics',
    'hand_masks',
    'hand_detections',
}
UNCHANGED_GEOMETRY_FIELDS = {
    'mano_valid',
    'hand_valid',
}
VIRTUAL_CAMERA_FIELDS = {
    'camera_mano',
    'camera_joints',
}
FULL_REQUIRED_FIELDS = (
    SPATIAL_FIELDS
    | {'hand_valid'}
    | {'mano_valid'}
    | VIRTUAL_CAMERA_FIELDS
)


def evaluation_aligned_hand_valid(
    camera_joints,
    intrinsics,
    joint_valid,
    image_hw,
):
    """Return the evaluation visibility label for ``[T,2,21]`` hands.

    A hand is valid when at least one valid, finite camera-space GT joint has
    ``z > 0.01 m`` and projects inside the final image.  This is the GT
    visibility rule used by the FAcc/Recall/F1 paper evaluator.
    """
    joints = np.asarray(camera_joints, dtype=np.float32)
    cameras = np.asarray(intrinsics, dtype=np.float32)
    valid = np.asarray(joint_valid, dtype=bool)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(
            'camera_joints must have shape [T,2,21,3], got '
            f'{joints.shape}'
        )
    frame_count = joints.shape[0]
    if cameras.shape != (frame_count, 3, 3):
        raise ValueError(
            'intrinsics must have shape [T,3,3], got '
            f'{cameras.shape}'
        )
    if valid.shape != (frame_count, 2, 21):
        raise ValueError(
            'joint_valid must have shape [T,2,21], got '
            f'{valid.shape}'
        )
    image_h, image_w = map(int, image_hw)
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f'image_hw must be positive, got {image_hw}')

    z = joints[..., 2]
    safe_z = np.where(np.abs(z) > 1.0e-8, z, np.nan)
    u = (
        cameras[:, None, None, 0, 0] * joints[..., 0] / safe_z
        + cameras[:, None, None, 0, 2]
    )
    v = (
        cameras[:, None, None, 1, 1] * joints[..., 1] / safe_z
        + cameras[:, None, None, 1, 2]
    )
    onscreen = (
        valid
        & np.isfinite(joints).all(axis=-1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (z > EVALUATION_HAND_MIN_DEPTH_M)
        & (u >= 0.0)
        & (u < float(image_w))
        & (v >= 0.0)
        & (v < float(image_h))
    )
    return onscreen.any(axis=-1)


@dataclass(frozen=True)
class ResizeCropRecord:
    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    target_hw: tuple[int, int]
    scale_xy: tuple[float, float]
    crop_xy: tuple[int, int]
    crop_axis: str
    valid_mask_centers_xy: tuple[tuple[float, float], ...]
    crop_center_source_uv: tuple[float, float]
    virtual_rotation_new_from_old: tuple[tuple[float, float, float], ...]
    homography_source_to_crop: tuple[tuple[float, float, float], ...]


def _tight_mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def _mask_centers(masks):
    centers = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs):
            centers.append((float(xs.mean()), float(ys.mean())))
    return centers


def _resize_shape(source_hw, target_hw):
    source_h, source_w = map(int, source_hw)
    target_h, target_w = map(int, target_hw)
    rh = float(source_h) / float(target_h)
    rw = float(source_w) / float(target_w)
    if rh < rw:
        resized_h = target_h
        resized_w = max(target_w, int(round(float(source_w) / rh)))
        crop_axis = 'horizontal'
    elif rh > rw:
        resized_h = max(target_h, int(round(float(source_h) / rw)))
        resized_w = target_w
        crop_axis = 'vertical'
    else:
        resized_h, resized_w = target_h, target_w
        crop_axis = 'none'
    return (resized_h, resized_w), crop_axis


def _balanced_crop_origin(resized_hw, target_hw, centers, crop_axis):
    resized_h, resized_w = resized_hw
    target_h, target_w = target_hw
    if centers:
        center_x = float(np.mean([center[0] for center in centers]))
        center_y = float(np.mean([center[1] for center in centers]))
    else:
        center_x = 0.5 * float(resized_w)
        center_y = 0.5 * float(resized_h)
    if crop_axis == 'horizontal':
        x0 = int(round(center_x - 0.5 * float(target_w)))
        x0 = int(np.clip(x0, 0, resized_w - target_w))
        y0 = 0
    elif crop_axis == 'vertical':
        x0 = 0
        y0 = int(round(center_y - 0.5 * float(target_h)))
        y0 = int(np.clip(y0, 0, resized_h - target_h))
    else:
        x0 = y0 = 0
    return x0, y0


def _virtual_camera_rotation(intrinsic, crop_center_uv):
    pixel = np.asarray(
        [crop_center_uv[0], crop_center_uv[1], 1.0],
        dtype=np.float64,
    )
    ray = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64)) @ pixel
    z_axis_in_old = ray / np.linalg.norm(ray)
    down_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis_in_old = np.cross(down_reference, z_axis_in_old)
    if np.linalg.norm(x_axis_in_old) < 1.0e-8:
        x_axis_in_old = np.cross(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            z_axis_in_old,
        )
    x_axis_in_old /= np.linalg.norm(x_axis_in_old)
    y_axis_in_old = np.cross(z_axis_in_old, x_axis_in_old)
    y_axis_in_old /= np.linalg.norm(y_axis_in_old)
    return np.stack(
        (x_axis_in_old, y_axis_in_old, z_axis_in_old),
        axis=0,
    ).astype(np.float32)


def _transform_points(points, rotation):
    return np.einsum(
        'ij,...j->...i',
        np.asarray(rotation, dtype=np.float32),
        np.asarray(points, dtype=np.float32),
    ).astype(np.float32)


def _rotate_mano61(params, rotation, valid):
    output = np.asarray(params, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    for slot in range(output.shape[0]):
        if not valid[slot]:
            continue
        global_rotation, _ = cv2.Rodrigues(
            output[slot, :3].astype(np.float64)
        )
        new_global_rotation = (
            np.asarray(rotation, dtype=np.float64) @ global_rotation
        )
        new_axis_angle, _ = cv2.Rodrigues(new_global_rotation)
        output[slot, :3] = new_axis_angle.reshape(3).astype(np.float32)
        output[slot, 58:61] = (
            np.asarray(rotation, dtype=np.float32) @ output[slot, 58:61]
        )
    return output


def _transform_fallback_box(box, homography, target_hw):
    target_h, target_w = target_hw
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
        return np.zeros(4, dtype=np.float32)
    corners = np.asarray(
        [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        dtype=np.float32,
    )
    transformed = cv2.perspectiveTransform(
        corners[None],
        np.asarray(homography, dtype=np.float32),
    )[0]
    result = np.asarray(
        [
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
        ],
        dtype=np.float32,
    )
    result[0::2] = np.clip(result[0::2], 0.0, float(target_w))
    result[1::2] = np.clip(result[1::2], 0.0, float(target_h))
    if result[2] <= result[0] or result[3] <= result[1]:
        return np.zeros(4, dtype=np.float32)
    return result


def resize_crop_hand_fields(
    fields,
    target_hw=(TARGET_HEIGHT, TARGET_WIDTH),
):
    """Transform a time-major DOSS hand dictionary to final model pixels.

    Args:
        fields: Mapping containing all full-geometry fields. ``images`` has
            shape ``[T,H,W,3]`` and hand fields use slot order [right, left].
            ``extrinsics/camera_poses`` and ``world_mano/world_joints`` are
            optional. This lets datasets without a global camera trajectory
            keep a camera-only public batch instead of fabricating identity
            world geometry.
        target_hw: Final LingBotMap height and width.
    Returns:
        ``(transformed_fields, records)``.  Geometry fields are copied without
        modification where appropriate; records describe each virtual camera.
    """
    required_fields = FULL_REQUIRED_FIELDS
    missing = sorted(required_fields - set(fields))
    if missing:
        raise KeyError(f'DOSS resize/crop is missing fields: {missing}')
    target_h, target_w = map(int, target_hw)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f'Invalid target_hw={target_hw}')

    images = np.asarray(fields['images'])
    intrinsics = np.asarray(fields['intrinsics'], dtype=np.float32)
    masks = np.asarray(fields['hand_masks'])
    detections = np.asarray(fields['hand_detections'], dtype=np.float32)
    hand_valid = np.asarray(fields['hand_valid'], dtype=bool)
    frame_count = images.shape[0]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f'images must be [T,H,W,3], got {images.shape}')
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError(f'intrinsics must be [T,3,3], got {intrinsics.shape}')
    if masks.shape[:2] != (frame_count, 2) or masks.ndim != 4:
        raise ValueError(f'hand_masks must be [T,2,H,W], got {masks.shape}')
    if detections.shape != (frame_count, 2, 4):
        raise ValueError(
            f'hand_detections must be [T,2,4], got {detections.shape}'
        )
    if hand_valid.shape != (frame_count, 2):
        raise ValueError(f'hand_valid must be [T,2], got {hand_valid.shape}')

    output = {
        key: np.asarray(value).copy()
        for key, value in fields.items()
        if key not in SPATIAL_FIELDS
    }
    output_images = []
    output_intrinsics = []
    output_masks = []
    output_detections = []
    records = []

    for frame_index in range(frame_count):
        image = images[frame_index]
        source_h, source_w = image.shape[:2]
        if masks[frame_index].shape[-2:] != (source_h, source_w):
            raise ValueError(
                f'Frame {frame_index} image/mask size mismatch: '
                f'{(source_h, source_w)} vs {masks[frame_index].shape[-2:]}'
            )
        resized_hw, crop_axis = _resize_shape(
            (source_h, source_w), (target_h, target_w)
        )
        resized_h, resized_w = resized_hw
        scale_x = float(resized_w) / float(source_w)
        scale_y = float(resized_h) / float(source_h)
        resized_masks = np.stack(
            [
                cv2.resize(
                    (mask > 0).astype(np.uint8),
                    (resized_w, resized_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks[frame_index]
            ]
        )
        centers = _mask_centers(resized_masks)
        crop_x, crop_y = _balanced_crop_origin(
            resized_hw,
            (target_h, target_w),
            centers,
            crop_axis,
        )
        output_center = np.asarray(
            [
                0.5 * float(target_w - 1),
                0.5 * float(target_h - 1),
            ],
            dtype=np.float32,
        )
        crop_center_source = np.asarray(
            [
                (float(crop_x) + float(output_center[0])) / scale_x,
                (float(crop_y) + float(output_center[1])) / scale_y,
            ],
            dtype=np.float32,
        )
        old_intrinsic = intrinsics[frame_index].copy()
        virtual_rotation = _virtual_camera_rotation(
            old_intrinsic,
            crop_center_source,
        )
        intrinsic = old_intrinsic.copy()
        intrinsic[0, :] *= np.float32(scale_x)
        intrinsic[1, :] *= np.float32(scale_y)
        intrinsic[0, 2] = output_center[0]
        intrinsic[1, 2] = output_center[1]
        homography = (
            intrinsic.astype(np.float64)
            @ virtual_rotation.astype(np.float64)
            @ np.linalg.inv(old_intrinsic.astype(np.float64))
        )
        image_crop = cv2.warpPerspective(
            image,
            homography,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask_crop = np.stack(
            [
                cv2.warpPerspective(
                    (mask > 0).astype(np.uint8),
                    homography,
                    (target_w, target_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                for mask in masks[frame_index]
            ]
        )
        if image_crop.shape[:2] != (target_h, target_w):
            raise AssertionError(
                f'Bad crop shape at frame {frame_index}: {image_crop.shape}'
            )

        frame_detections = np.zeros((2, 4), dtype=np.float32)
        for slot in range(2):
            tight_box = _tight_mask_box(mask_crop[slot])
            if tight_box is not None:
                frame_detections[slot] = tight_box
            else:
                frame_detections[slot] = _transform_fallback_box(
                    detections[frame_index, slot],
                    homography,
                    (target_h, target_w),
                )

        output_images.append(image_crop.astype(images.dtype, copy=False))
        output_intrinsics.append(intrinsic.astype(np.float32))
        output_masks.append(mask_crop.astype(np.float32))
        output_detections.append(frame_detections)
        output['camera_joints'][frame_index] = _transform_points(
            fields['camera_joints'][frame_index],
            virtual_rotation,
        )
        output['camera_mano'][frame_index] = _rotate_mano61(
            fields['camera_mano'][frame_index],
            virtual_rotation,
            fields['mano_valid'][frame_index],
        )
        virtual_transform = np.eye(4, dtype=np.float32)
        virtual_transform[:3, :3] = virtual_rotation
        if 'extrinsics' in fields:
            output['extrinsics'][frame_index] = (
                virtual_transform @ fields['extrinsics'][frame_index]
            ).astype(np.float32)
        if 'camera_poses' in fields and 'extrinsics' in fields:
            output['camera_poses'][frame_index] = np.linalg.inv(
                output['extrinsics'][frame_index]
            ).astype(np.float32)
        if 'camera_mano_joints' in fields:
            output['camera_mano_joints'][frame_index] = _transform_points(
                fields['camera_mano_joints'][frame_index],
                virtual_rotation,
            )
        records.append(
            asdict(
                ResizeCropRecord(
                    source_hw=(source_h, source_w),
                    resized_hw=resized_hw,
                    target_hw=(target_h, target_w),
                    scale_xy=(scale_x, scale_y),
                    crop_xy=(crop_x, crop_y),
                    crop_axis=crop_axis,
                    valid_mask_centers_xy=tuple(centers),
                    crop_center_source_uv=tuple(
                        map(float, crop_center_source)
                    ),
                    virtual_rotation_new_from_old=tuple(
                        tuple(map(float, row)) for row in virtual_rotation
                    ),
                    homography_source_to_crop=tuple(
                        tuple(map(float, row)) for row in homography
                    ),
                )
            )
        )

    output['images'] = np.stack(output_images)
    output['intrinsics'] = np.stack(output_intrinsics).astype(np.float32)
    output['hand_masks'] = np.stack(output_masks).astype(np.float32)
    output['hand_detections'] = np.stack(output_detections).astype(np.float32)
    output['hand_valid'] = hand_valid.copy()
    return output, records
