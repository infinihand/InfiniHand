"""Perspective hand crops with centered intrinsics and rotated camera labels.

RGB, masks and 2D labels share H = K_crop Q inv(K_image). MANO global
orientation, translation and 3D joints rotate by Q; world geometry is unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np


TARGET_HEIGHT = 256
TARGET_WIDTH = 192
DEFAULT_BBOX_RESCALE = 1.5


def _tight_mask_box(mask):
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def _valid_box(box):
    box = np.asarray(box, dtype=np.float32)
    return (
        box.shape == (4,)
        and np.isfinite(box).all()
        and box[2] > box[0]
        and box[3] > box[1]
    )


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
        "ij,...j->...i",
        np.asarray(rotation, dtype=np.float32),
        np.asarray(points, dtype=np.float32),
    ).astype(np.float32)


def _rotate_mano61(params, rotation, valid):
    output = np.asarray(params, dtype=np.float32).copy()
    if not bool(valid):
        return output
    global_rotation, _ = cv2.Rodrigues(output[:3].astype(np.float64))
    new_global_rotation = np.asarray(rotation, dtype=np.float64) @ global_rotation
    new_axis_angle, _ = cv2.Rodrigues(new_global_rotation)
    output[:3] = new_axis_angle.reshape(3).astype(np.float32)
    output[58:61] = np.asarray(rotation, dtype=np.float32) @ output[58:61]
    return output


def prepare_wilor_virtual_crops(
    fields,
    *,
    target_hw=(TARGET_HEIGHT, TARGET_WIDTH),
    bbox_rescale=DEFAULT_BBOX_RESCALE,
):
    """Create model-aligned affine crop images and synchronized labels."""
    required = {
        "images",
        "intrinsics",
        "camera_mano",
        "camera_joints",
        "mano_valid",
        "hand_valid",
        "hand_masks",
        "hand_detections",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise KeyError(f"DOSS WiLoR crop is missing fields: {missing}")

    target_h, target_w = map(int, target_hw)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target_hw={target_hw}")
    bbox_rescale = float(bbox_rescale)
    if not np.isfinite(bbox_rescale) or bbox_rescale <= 0:
        raise ValueError(f"bbox_rescale must be finite and positive, got {bbox_rescale}")

    images = np.asarray(fields["images"])
    intrinsics = np.asarray(fields["intrinsics"], dtype=np.float32)
    camera_mano = np.asarray(fields["camera_mano"], dtype=np.float32)
    camera_joints = np.asarray(fields["camera_joints"], dtype=np.float32)
    mano_valid = np.asarray(fields["mano_valid"], dtype=bool)
    hand_valid = np.asarray(fields["hand_valid"], dtype=bool)
    masks = np.asarray(fields["hand_masks"])
    detections = np.asarray(fields["hand_detections"], dtype=np.float32)

    frame_count = images.shape[0]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must be [T,H,W,3], got {images.shape}")
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError(f"intrinsics must be [T,3,3], got {intrinsics.shape}")
    if camera_joints.shape != (frame_count, 2, 21, 3):
        raise ValueError(f"camera_joints must be [T,2,21,3], got {camera_joints.shape}")
    if camera_mano.shape != (frame_count, 2, 61):
        raise ValueError(f"camera_mano must be [T,2,61], got {camera_mano.shape}")
    if "extrinsics" in fields:
        extrinsics = np.asarray(fields["extrinsics"], dtype=np.float32)
        if extrinsics.shape != (frame_count, 4, 4):
            raise ValueError(f"extrinsics must be [T,4,4], got {extrinsics.shape}")
    else:
        extrinsics = None
    if "world_joints" in fields:
        world_joints = np.asarray(fields["world_joints"], dtype=np.float32)
        if world_joints.shape != (frame_count, 2, 21, 3):
            raise ValueError(f"world_joints must be [T,2,21,3], got {world_joints.shape}")
    else:
        world_joints = None
    if "world_mano" in fields:
        world_mano = np.asarray(fields["world_mano"], dtype=np.float32)
        if world_mano.shape != (frame_count, 2, 61):
            raise ValueError(f"world_mano must be [T,2,61], got {world_mano.shape}")
    else:
        world_mano = None

    result = {
        "wilor_images": [],
        "wilor_intrinsics": [],
        "wilor_camera_mano": [],
        "wilor_camera_joints": [],
        "wilor_hand_masks": [],
        "wilor_hand_detections": [],
        "wilor_virtual_rotations": [],
        "wilor_crop_homographies": [],
        "wilor_crop_boxes_source": [],
        "wilor_crop_valid": [],
    }
    if extrinsics is not None:
        result["wilor_extrinsics"] = []
        result["wilor_camera_poses"] = []
    if world_mano is not None:
        result["wilor_world_mano"] = world_mano.copy()
    if world_joints is not None:
        result["wilor_world_joints"] = world_joints.copy()
    if "camera_mano_joints" in fields:
        result["wilor_camera_mano_joints"] = []
    if "world_mano_joints" in fields:
        result["wilor_world_mano_joints"] = np.asarray(
            fields["world_mano_joints"],
            dtype=np.float32,
        ).copy()

    output_center = np.asarray(
        [0.5 * float(target_w - 1), 0.5 * float(target_h - 1)],
        dtype=np.float32,
    )

    for frame_index in range(frame_count):
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        frame_camera_poses = []
        frame_camera_mano = []
        frame_camera_joints = []
        frame_masks = []
        frame_detections = []
        frame_rotations = []
        frame_homographies = []
        frame_boxes = []
        frame_valid = []
        frame_camera_mano_joints = []
        image_h, image_w = images[frame_index].shape[:2]

        for slot in range(2):
            box = _tight_mask_box(masks[frame_index, slot])
            if box is None and _valid_box(detections[frame_index, slot]):
                box = detections[frame_index, slot].copy()
            crop_valid = bool(hand_valid[frame_index, slot] and box is not None)
            if box is None:
                box = np.asarray(
                    [0.0, 0.0, float(image_w), float(image_h)],
                    dtype=np.float32,
                )

            center = 0.5 * (box[:2] + box[2:])
            box_width = max(float(box[2] - box[0]), image_w * 1.0e-4)
            box_height = max(float(box[3] - box[1]), image_h * 1.0e-4)
            import torch
            from infinihand.models.hand_model import (
                expand_to_square_boxes, virtual_crop_camera_from_boxes,
            )
            old_intrinsic = intrinsics[frame_index]
            box_tensor = torch.from_numpy(box / np.asarray([image_w, image_h, image_w, image_h])).float()[None,None,None].expand(1,1,2,4)
            expanded = expand_to_square_boxes(box_tensor, (image_h, image_w), bbox_rescale)
            kt, qt = virtual_crop_camera_from_boxes(expanded,
                torch.from_numpy(old_intrinsic)[None,None], (image_h,image_w), (target_h,target_w))
            crop_intrinsic = kt[0,0,slot].numpy()
            virtual_rotation = qt[0,0,slot].numpy()
            homography = crop_intrinsic @ virtual_rotation @ np.linalg.inv(old_intrinsic)
            source_crop_box = box.copy()  # source bbox; H defines the actual perspective footprint

            crop_image = cv2.warpPerspective(
                images[frame_index],
                homography,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            crop_mask = cv2.warpPerspective(
                (masks[frame_index, slot] > 0).astype(np.uint8),
                homography,
                (target_w, target_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            crop_detection = _tight_mask_box(crop_mask)
            if crop_detection is None:
                crop_detection = np.zeros(4, dtype=np.float32)

            virtual_transform = np.eye(4, dtype=np.float32)
            virtual_transform[:3, :3] = virtual_rotation
            frame_images.append(crop_image.astype(images.dtype, copy=False))
            frame_intrinsics.append(crop_intrinsic)
            if extrinsics is not None:
                crop_extrinsic = virtual_transform @ extrinsics[frame_index]
                frame_extrinsics.append(crop_extrinsic.astype(np.float32))
                frame_camera_poses.append(
                    np.linalg.inv(crop_extrinsic).astype(np.float32)
                )
            frame_camera_mano.append(
                _rotate_mano61(
                    camera_mano[frame_index, slot],
                    virtual_rotation,
                    mano_valid[frame_index, slot],
                )
            )
            frame_camera_joints.append(
                _transform_points(
                    camera_joints[frame_index, slot],
                    virtual_rotation,
                )
            )
            if "camera_mano_joints" in fields:
                frame_camera_mano_joints.append(
                    _transform_points(
                        fields["camera_mano_joints"][frame_index, slot],
                        virtual_rotation,
                    )
                )
            frame_masks.append(crop_mask.astype(np.float32))
            frame_detections.append(crop_detection)
            frame_rotations.append(virtual_rotation)
            frame_homographies.append(homography)
            frame_boxes.append(source_crop_box)
            frame_valid.append(crop_valid)

        result["wilor_images"].append(np.stack(frame_images))
        result["wilor_intrinsics"].append(np.stack(frame_intrinsics))
        if extrinsics is not None:
            result["wilor_extrinsics"].append(np.stack(frame_extrinsics))
            result["wilor_camera_poses"].append(np.stack(frame_camera_poses))
        result["wilor_camera_mano"].append(np.stack(frame_camera_mano))
        result["wilor_camera_joints"].append(np.stack(frame_camera_joints))
        result["wilor_hand_masks"].append(np.stack(frame_masks))
        result["wilor_hand_detections"].append(np.stack(frame_detections))
        result["wilor_virtual_rotations"].append(np.stack(frame_rotations))
        result["wilor_crop_homographies"].append(np.stack(frame_homographies))
        result["wilor_crop_boxes_source"].append(np.stack(frame_boxes))
        result["wilor_crop_valid"].append(np.asarray(frame_valid, dtype=bool))
        if "camera_mano_joints" in fields:
            result["wilor_camera_mano_joints"].append(
                np.stack(frame_camera_mano_joints)
            )

    for key in (
        "wilor_images",
        "wilor_intrinsics",
        "wilor_extrinsics",
        "wilor_camera_poses",
        "wilor_camera_mano",
        "wilor_camera_joints",
        "wilor_hand_masks",
        "wilor_hand_detections",
        "wilor_virtual_rotations",
        "wilor_crop_homographies",
        "wilor_crop_boxes_source",
        "wilor_crop_valid",
        "wilor_camera_mano_joints",
    ):
        if key in result:
            result[key] = np.stack(result[key])
    return result
