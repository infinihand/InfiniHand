#!/usr/bin/env python3
"""Streaming hand and camera inference from a single checkpoint.

The camera-space boundary includes the persistent LingBotMap/GCA backbone,
camera head, mask/detection heads, WiLoR hand feature provider, feature fusion,
and MANO decoding.  The global-space boundary adds only the conversion from
predicted camera coordinates through the Stage-2 predicted camera trajectory.

The production contract is fixed to four anchor frames plus sixteen recent
dense frames. CUDA stream 1 runs LingBotMap/GCA and the mask/detection heads
ahead of CUDA stream 2. CUDA stream 2 consumes the cached LingBot features and
predicted boxes for MANO; it never calls the LingBotMap backbone.

All image decoding/resizing and host-to-device copies are intentionally kept
outside the measured interval. Model loading and output serialization are
reported separately. This module is inference-only and is not imported by the
training entrypoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import cv2
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch


from .models.hand_model import LingBotMapBackboneOutput, LingBotMapHand, LingBotMapHandConfig, expand_to_square_boxes, virtual_crop_camera_from_boxes
from .models.geometry import _invert_se3, _pose_encoding_to_w2c

ANCHOR_FRAMES = 4
SLIDING_WINDOW_FRAMES = 16
DENSE_CAPACITY_FRAMES = ANCHOR_FRAMES + SLIDING_WINDOW_FRAMES
LOOKAHEAD_PACKETS = 4


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def build_model(checkpoint, device, config_path):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(config_path).read_text())
    for key in ('lingbotmap_root', 'wilor_root', 'wilor_cfg', 'mano_model_path'):
        config[key] = str((root / config[key]).resolve())
    config.update(lingbotmap_ckpt='', wilor_ckpt='', load_wilor_checkpoint=False,
                  step_memory_enabled=True, step_memory_scale_frames=4,
                  step_memory_sliding_window=16, step_memory_max_frames=8192)
    model = LingBotMapHand(LingBotMapHandConfig(**config))
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Expected a complete exported tensor state_dict')
    from .weights import load_inference_weights
    load_inference_weights(model, state)
    model.to(device).eval()
    return model, time.perf_counter() - started


@dataclass
class DetectionPacket:
    order: int
    images: torch.Tensor
    intrinsics: torch.Tensor
    output: dict[str, Any]
    features: torch.Tensor
    gt_hand_boxes: torch.Tensor | None
    gt_hand_valid: torch.Tensor | None
    ready: torch.cuda.Event


def detection_forward(
    model: LingBotMapHand,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    backbone_output: LingBotMapBackboneOutput,
    order: int,
    gt_hand_boxes: torch.Tensor | None = None,
    gt_hand_valid: torch.Tensor | None = None,
    gt_hand_masks: torch.Tensor | None = None,
    bbox_source: str = "mask",
    mask_threshold: float = 0.5,
    mask_blend_alpha: float = 0.0,
) -> DetectionPacket:
    with torch.autocast(device_type=images.device.type, enabled=False):
        images = images.float()
        intrinsics = intrinsics.float()
        if bbox_source == "gt":
            if gt_hand_boxes is None or gt_hand_valid is None:
                raise ValueError(
                    "bbox_source='gt' requires gt_hand_boxes and gt_hand_valid"
                )
            valid = gt_hand_valid.to(device=images.device, dtype=torch.bool)
            boxes = gt_hand_boxes.to(device=images.device, dtype=images.dtype)
            fallback = boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
            boxes = torch.where(valid[..., None], boxes, fallback)
            valid_float = valid.float()
            output = {
                "pred_hand_boxes": boxes,
                "pred_hand_probability": valid_float,
                "pred_hand_probability_logits": torch.where(
                    valid, valid_float.new_tensor(20.0), valid_float.new_tensor(-20.0)
                ),
                "pred_bbox_mask_probability": valid_float[..., None, None],
                "pred_mask_tight_boxes": gt_hand_boxes.to(
                    device=images.device, dtype=images.dtype
                ),
                "pred_mask_valid": valid,
                "mano_bbox_source": "gt",
            }
            if backbone_output.pose_enc is None:
                raise RuntimeError("Stage2 camera head did not return pose_enc")
            output["pred_camera_pose_enc"] = backbone_output.pose_enc
            features = backbone_output.patch_features.to(
                dtype=next(model.lingbot_adapter.parameters()).dtype
            )
            return DetectionPacket(
                order=order,
                images=images,
                intrinsics=intrinsics,
                output=output,
                features=features,
                gt_hand_boxes=gt_hand_boxes,
                gt_hand_valid=gt_hand_valid,
                ready=torch.cuda.Event(),
            )

        target_hw = (model.cfg.image_height, model.cfg.image_width)
        detection_features = model.hand_detection_block(
            backbone_output.aggregated_tokens,
            backbone_output.patch_start_idx,
            target_hw,
        )
        mask_logits = model.hand_mask_head(detection_features, target_hw)
        mask_probability = torch.sigmoid(mask_logits)
        hand_probability = mask_probability.amax(dim=(-1, -2))
        hand_probability_logits = torch.logit(hand_probability.clamp(1e-6, 1-1e-6))
        if gt_hand_masks is None:
            if mask_blend_alpha != 0.0:
                raise ValueError("mask_blend_alpha requires gt_hand_masks")
            bbox_mask_probability = mask_probability
        else:
            if gt_hand_masks.shape != mask_probability.shape:
                raise ValueError(
                    "gt_hand_masks must match predicted mask shape, got "
                    f"gt={tuple(gt_hand_masks.shape)} pred={tuple(mask_probability.shape)}"
                )
            if not 0.0 <= mask_blend_alpha <= 1.0:
                raise ValueError(
                    f"mask_blend_alpha must be in [0,1], got {mask_blend_alpha}"
                )
            gt_binary = (gt_hand_masks > 0.5).to(dtype=mask_probability.dtype)
            bbox_mask_probability = (
                (1.0 - mask_blend_alpha) * mask_probability
                + mask_blend_alpha * gt_binary
            )
        # The evaluation protocol defines visibility_score > threshold.  Keep the
        # same strict comparison for mixed mask probabilities, especially at
        # alpha=0.5 where an equality comparison would leak saturated predictions.
        binary_mask = bbox_mask_probability > mask_threshold
        mask_boxes, mask_valid = tight_mask_boxes(binary_mask)
        if bbox_source == "mask":
            # Empty masks are gated out by ``mask_valid``. A full-image fallback
            # keeps the MANO branch numerically defined without inventing a box.
            fallback = mask_boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
            mano_input_boxes = torch.where(
                mask_valid[..., None], mask_boxes, fallback
            )
            output = {"pred_hand_boxes": mano_input_boxes}
        else:
            raise ValueError(f"unsupported bbox_source={bbox_source!r}")
        output.update(
            {
                "pred_hand_probability": hand_probability,
                "pred_hand_probability_logits": hand_probability_logits,
                "pred_hand_mask_logits": mask_logits,
                "pred_hand_masks": mask_probability,
                "pred_bbox_mask_probability": bbox_mask_probability,
                "pred_mask_tight_boxes": mask_boxes,
                "pred_mask_valid": mask_valid,
                "mano_bbox_source": bbox_source,
            }
        )
        if backbone_output.pose_enc is None:
            raise RuntimeError("Stage2 camera head did not return pose_enc")
        output["pred_camera_pose_enc"] = backbone_output.pose_enc
        features = backbone_output.patch_features.to(
            dtype=next(model.lingbot_adapter.parameters()).dtype
        )
        return DetectionPacket(
            order=order,
            images=images,
            intrinsics=intrinsics,
            output=output,
            features=features,
            gt_hand_boxes=gt_hand_boxes,
            gt_hand_valid=gt_hand_valid,
            ready=torch.cuda.Event(),
        )


def tight_mask_boxes(binary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized tight xyxy boxes and non-empty flags for [B,N,2,H,W]."""
    if binary.ndim != 5 or binary.shape[2] != 2:
        raise ValueError(
            f"mask tensor must be [B,N,2,H,W], got {tuple(binary.shape)}"
        )
    batch, views, slots, height, width = binary.shape
    flat = binary.reshape(batch * views, slots, height, width)
    rows = flat.any(dim=-1)
    cols = flat.any(dim=-2)
    nonempty = rows.any(dim=-1)
    y = torch.arange(height, device=binary.device).view(1, 1, height)
    x = torch.arange(width, device=binary.device).view(1, 1, width)
    y1 = torch.where(rows, y, height).amin(dim=-1)
    y2 = torch.where(rows, y + 1, 0).amax(dim=-1)
    x1 = torch.where(cols, x, width).amin(dim=-1)
    x2 = torch.where(cols, x + 1, 0).amax(dim=-1)
    boxes = torch.stack(
        (
            x1.float() / float(width),
            y1.float() / float(height),
            x2.float() / float(width),
            y2.float() / float(height),
        ),
        dim=-1,
    )
    boxes = torch.where(nonempty[..., None], boxes, torch.zeros_like(boxes))
    return (
        boxes.reshape(batch, views, slots, 4),
        nonempty.reshape(batch, views, slots),
    )


def mano_forward(
    model: LingBotMapHand,
    packet: DetectionPacket,
) -> dict[str, Any]:
    with torch.autocast(device_type=packet.images.device.type, enabled=False):
        packet.images = packet.images.float()
        packet.features = packet.features.float()
        packet.intrinsics = packet.intrinsics.float()
        target_hw = (model.cfg.image_height, model.cfg.image_width)
        boxes = packet.output["pred_hand_boxes"]
        if packet.gt_hand_boxes is not None:
            valid = (
                torch.isfinite(packet.gt_hand_boxes).all(dim=-1)
                if packet.gt_hand_valid is None
                else packet.gt_hand_valid.to(dtype=torch.bool)
            )
            boxes = torch.where(
                valid[..., None],
                packet.gt_hand_boxes.to(device=boxes.device, dtype=boxes.dtype),
                boxes,
            )
        crop_scale = model._sample_bbox_rescale(boxes)
        boxes = expand_to_square_boxes(boxes, target_hw, crop_scale)
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            mano_crop_intrinsics, auto_crop_rotation = virtual_crop_camera_from_boxes(
                boxes,
                packet.intrinsics,
                target_hw,
                (256, 192),
            )

            crop_h = mano_crop_intrinsics @ auto_crop_rotation @ torch.linalg.inv(packet.intrinsics.float())[:, :, None]
            shift = torch.eye(3, device=crop_h.device, dtype=crop_h.dtype)
            shift[0,2] = 32.0
            square_h = shift @ crop_h
        batch_size, num_views = packet.features.shape[:2]
        lingbot_context, lingbot_crops_256 = model.lingbot_adapter(
            packet.features, boxes, square_h, target_hw
        )
        hand_feature_batch = model.hand_feature_provider(packet.images, boxes, square_h)
        hand_features = hand_feature_batch.features.to(dtype=lingbot_context.dtype)
        flat_hand_features = hand_features.reshape(
            batch_size * num_views * 2,
            model.hand_feature_provider.output_dim,
            *model.hand_feature_provider.output_hw,
        )
        fused_features = model.feature_fusion(lingbot_context, flat_hand_features)
        output = packet.output
        output["mano_crop_homographies"] = crop_h
        output["mano_boxes"] = boxes
        output["mano_crop_scale"] = crop_scale
        output["lingbotmap_feature_crops"] = lingbot_crops_256.reshape(
            batch_size,
            num_views,
            2,
            model.cfg.lingbot_pixel_channels,
            model.cfg.wilor_crop_size,
            model.cfg.wilor_crop_size,
        )
        output["lingbotmap_mano_features"] = lingbot_context.reshape(
            batch_size,
            num_views,
            2,
            model.cfg.mano_context_dim,
            *model.lingbot_adapter.output_hw,
        )
        output["hand_feature_crops"] = hand_feature_batch.canonical_crops
        output["hand_features"] = hand_features
        output["fused_mano_features"] = fused_features.reshape(
            batch_size,
            num_views,
            2,
            model.cfg.mano_context_dim,
            *model.lingbot_adapter.output_hw,
        )
        output = model._forward_mutual_joint_fusion(
            output,
            fused_features,
            batch_size,
            num_views,
            mano_crop_intrinsics,
            (256, 192),
        )

        # Preserve the exact full-forward virtual-camera to image-camera contract.
        with torch.autocast(device_type=auto_crop_rotation.device.type, enabled=False):
            crop_to_image = auto_crop_rotation.transpose(-1, -2)
            output["mano_virtual_rotations"] = auto_crop_rotation
            output["pred_keypoints_3d_camera"] = torch.matmul(
                output["pred_keypoints_3d_crop_camera"], auto_crop_rotation
            )
            output["pred_vertices_camera"] = torch.matmul(
                output["pred_vertices_crop_camera"], auto_crop_rotation
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
            output["pred_wrist_transl_camera"] = output["pred_keypoints_3d_camera"][...,0,:]
            output["camera_output_space"] = "image_camera"
        return output


def output_subset(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    if output.get("camera_output_space") != "image_camera":
        raise RuntimeError(
            f"unexpected camera_output_space={output.get('camera_output_space')!r}"
        )
    return {
        "camera_joints": output["pred_keypoints_3d_camera"][0].detach().clone(),
        "camera_vertices": output["pred_vertices_camera"][0].detach().clone(),
        "camera_global_orient": output[
            "pred_global_orient_camera"
        ][0, :, :, 0].detach().clone(),
        "presence": output["pred_hand_probability"][0].detach().clone(),
        "presence_mask": output["pred_mask_valid"][0].float().detach().clone(),
        "mask_score": output["pred_bbox_mask_probability"][0]
        .amax(dim=(-1, -2))
        .detach()
        .clone(),
        # Retain dense probabilities only for the temporary mask-visualization
        # runner. Production evaluation intentionally omits this large tensor.
        "mask_tight_boxes": output["pred_mask_tight_boxes"][0]
        .detach()
        .clone(),
        "mano_input_boxes": output["mano_boxes"][0].detach().clone(),
        "pose_encoding": output["pred_camera_pose_enc"][0].detach().clone(),
        "hand_pose": output["pred_mano_params"]["hand_pose"][0].detach().clone(),
        "betas": output["pred_mano_params"]["betas"][0].detach().clone(),
        "camera_translation": output["pred_mano_transl_camera"][0].detach().clone(),
    }


def record_stream_tree(value: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(value):
        value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_stream_tree(item, stream)


def concatenate_outputs(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in chunks[0]
    }


@torch.inference_mode()
def run_dual_stream(
    model: LingBotMapHand,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_hand_boxes: torch.Tensor | None = None,
    gt_hand_valid: torch.Tensor | None = None,
    gt_hand_masks: torch.Tensor | None = None,
    bbox_source: str = "mask",
    mask_threshold: float = 0.5,
    mask_blend_alpha: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
    """Run the only supported production schedule: 4-16, two CUDA streams."""
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(f"images must have shape [1,T,3,H,W], got {images.shape}")
    if intrinsics.shape[:2] != images.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"intrinsics must match [1,T,3,3], got {intrinsics.shape}"
        )
    if int(images.shape[1]) < ANCHOR_FRAMES:
        raise RuntimeError("Stage2 inference requires at least four frames")
    if gt_hand_boxes is not None and gt_hand_boxes.shape != (*images.shape[:2], 2, 4):
        raise ValueError(
            "gt_hand_boxes must have shape [1,T,2,4], got "
            f"{gt_hand_boxes.shape}"
        )
    if gt_hand_valid is not None and gt_hand_valid.shape != (*images.shape[:2], 2):
        raise ValueError(
            "gt_hand_valid must have shape [1,T,2], got "
            f"{gt_hand_valid.shape}"
        )
    expected_mask_shape = (*images.shape[:2], 2, images.shape[-2], images.shape[-1])
    if gt_hand_masks is not None and gt_hand_masks.shape != expected_mask_shape:
        raise ValueError(
            "gt_hand_masks must have shape [1,T,2,H,W], got "
            f"{gt_hand_masks.shape}; expected {expected_mask_shape}"
        )
    if gt_hand_masks is None and mask_blend_alpha != 0.0:
        raise ValueError("mask_blend_alpha requires gt_hand_masks")
    if not 0.0 <= mask_blend_alpha <= 1.0:
        raise ValueError(f"mask_blend_alpha must be in [0,1], got {mask_blend_alpha}")
    device = images.device
    backbone = model.backbone
    detection_stream = torch.cuda.Stream(device=device)
    mano_stream = torch.cuda.Stream(device=device)
    packet_count = int(images.shape[1]) - ANCHOR_FRAMES + 1
    packets: queue.Queue[DetectionPacket | None] = queue.Queue(
        maxsize=LOOKAHEAD_PACKETS
    )
    prefilled = threading.Event()
    abort = threading.Event()
    errors: list[BaseException] = []
    detection_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    mano_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    chunks: list[dict[str, torch.Tensor]] = []
    stream_stats_holder: list[dict[str, Any]] = []

    def producer() -> None:
        try:
            with torch.inference_mode(), torch.cuda.stream(detection_stream):
                for order in range(packet_count):
                    if abort.is_set():
                        return
                    if order == 0:
                        packet_images = images[:, :ANCHOR_FRAMES]
                        packet_intrinsics = intrinsics[:, :ANCHOR_FRAMES]
                        packet_gt_boxes = (
                            gt_hand_boxes[:, :ANCHOR_FRAMES]
                            if gt_hand_boxes is not None
                            else None
                        )
                        packet_gt_valid = (
                            gt_hand_valid[:, :ANCHOR_FRAMES]
                            if gt_hand_valid is not None
                            else None
                        )
                        packet_gt_masks = (
                            gt_hand_masks[:, :ANCHOR_FRAMES]
                            if gt_hand_masks is not None
                            else None
                        )
                    else:
                        index = ANCHOR_FRAMES + order - 1
                        packet_images = images[:, index : index + 1]
                        packet_intrinsics = intrinsics[:, index : index + 1]
                        packet_gt_boxes = (
                            gt_hand_boxes[:, index : index + 1]
                            if gt_hand_boxes is not None
                            else None
                        )
                        packet_gt_valid = (
                            gt_hand_valid[:, index : index + 1]
                            if gt_hand_valid is not None
                            else None
                        )
                        packet_gt_masks = (
                            gt_hand_masks[:, index : index + 1]
                            if gt_hand_masks is not None
                            else None
                        )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(detection_stream)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if order == 0:
                            backbone_output = backbone.begin_inference_stream(
                                packet_images
                            )
                        else:
                            backbone_output = backbone.inference_stream_step(
                                packet_images
                            )
                        packet = detection_forward(
                            model=model,
                            images=packet_images,
                            intrinsics=packet_intrinsics,
                            backbone_output=backbone_output,
                            order=order,
                            gt_hand_boxes=packet_gt_boxes,
                            gt_hand_valid=packet_gt_valid,
                            gt_hand_masks=packet_gt_masks,
                            bbox_source=bbox_source,
                            mask_threshold=mask_threshold,
                            mask_blend_alpha=mask_blend_alpha,
                        )
                    end.record(detection_stream)
                    packet.ready.record(detection_stream)
                    detection_events.append((start, end))
                    while not abort.is_set():
                        try:
                            packets.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if abort.is_set():
                        return
                    if order + 1 == min(LOOKAHEAD_PACKETS, packet_count):
                        prefilled.set()
                    del backbone_output
                stream_stats_holder.append(backbone.end_inference_stream())
                while not abort.is_set():
                    try:
                        packets.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:
            errors.append(error)
            abort.set()
            prefilled.set()
            try:
                packets.put_nowait(None)
            except queue.Full:
                pass
        finally:
            if abort.is_set() and hasattr(backbone, "_inference_stream_state"):
                try:
                    backbone.end_inference_stream()
                except BaseException:
                    pass

    def consumer() -> None:
        try:
            prefilled.wait()
            with torch.inference_mode(), torch.cuda.stream(mano_stream):
                while True:
                    if abort.is_set() and packets.empty():
                        break
                    try:
                        packet = packets.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if packet is None:
                        break
                    mano_stream.wait_event(packet.ready)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(mano_stream)
                    with torch.autocast(device_type="cuda", enabled=False):
                        output = mano_forward(model, packet)
                    chunks.append(output_subset(output))
                    end.record(mano_stream)
                    mano_events.append((start, end))
                    record_stream_tree(packet.output, mano_stream)
                    packet.features.record_stream(mano_stream)
                    packet.images.record_stream(mano_stream)
                    packet.intrinsics.record_stream(mano_stream)
                    del output, packet
        except BaseException as error:
            errors.append(error)
            abort.set()
            prefilled.set()

    synchronize(device)
    started = time.perf_counter()
    producer_thread = threading.Thread(target=producer, name="stage2-detection")
    consumer_thread = threading.Thread(target=consumer, name="stage2-mano")
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    synchronize(device)
    wall_seconds = time.perf_counter() - started
    if errors:
        raise errors[0]
    if len(chunks) != packet_count:
        raise RuntimeError(
            f"parallel pipeline returned {len(chunks)} packets, expected {packet_count}"
        )
    if len(stream_stats_holder) != 1:
        raise RuntimeError("parallel stream did not return exactly one cache receipt")
    timing = {
        "camera_estimation_seconds": wall_seconds,
        "detection_gpu_stage_sum_seconds": sum(
            start.elapsed_time(end) for start, end in detection_events
        )
        / 1000.0,
        "mano_gpu_stage_sum_seconds": sum(
            start.elapsed_time(end) for start, end in mano_events
        )
        / 1000.0,
    }
    return concatenate_outputs(chunks), stream_stats_holder[0], timing


@torch.inference_mode()
def camera_to_predicted_global(
    output: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float]:
    device = output["camera_joints"].device
    synchronize(device)
    started = time.perf_counter()
    pose_encoding = output["pose_encoding"].unsqueeze(0).float()
    pred_w2c = _pose_encoding_to_w2c(pose_encoding)
    pred_c2w = _invert_se3(pred_w2c)
    # Express every camera in the first predicted anchor-camera coordinate.
    c2anchor = torch.matmul(pred_w2c[:, :1], pred_c2w)[0]
    rotation = c2anchor[:, :3, :3]
    translation = c2anchor[:, :3, 3]
    world_joints = torch.matmul(
        output["camera_joints"].float(), rotation[:, None].transpose(-1, -2)
    ) + translation[:, None, None, :]
    world_vertices = torch.matmul(
        output["camera_vertices"].float(), rotation[:, None].transpose(-1, -2)
    ) + translation[:, None, None, :]
    world_orient = torch.matmul(
        rotation[:, None], output["camera_global_orient"].float()
    )
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "pred_camera_to_anchor": c2anchor,
        "world_joints": world_joints,
        "world_vertices": world_vertices,
        "world_global_orient": world_orient,
    }, seconds


def validate_contract(
    stats: dict[str, Any],
    frame_count: int,
) -> None:
    expected = {
        "enabled": True,
        "scale_frames": ANCHOR_FRAMES,
        "sliding_window_frames": SLIDING_WINDOW_FRAMES,
        "total_frames": frame_count,
        "streamed_frames": frame_count - ANCHOR_FRAMES,
        "cache_cleared_at_step_end": True,
    }
    mismatch = {
        key: (stats.get(key), value)
        for key, value in expected.items()
        if stats.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Stage2 stream contract mismatch: {mismatch}")
    if int(stats["peak_dense_cache_frames"]) > DENSE_CAPACITY_FRAMES:
        raise RuntimeError(
            f"dense cache exceeded {DENSE_CAPACITY_FRAMES} frames: {stats}"
        )


ROOT = Path(__file__).resolve().parents[1]

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Complete weights exported from the current training runs')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/infinihand_v1.json')
    p.add_argument('--intrinsics', type=Path, required=True, help='JSON: fx,fy,cx,cy,image_width,image_height, in original video pixels')
    p.add_argument('--mask-source', choices=['base', 'sam3-tracking'], default='base')
    p.add_argument('--mask-threshold', type=float, default=0.9)
    p.add_argument('--sam3-python', default=sys.executable)
    p.add_argument('--sam3-source', type=Path, default=Path('/path/to/infinihand/third_party/sam3'))
    p.add_argument('--sam3-checkpoint', type=Path, default=Path('/path/to/weights/sam3.pt'))
    p.add_argument('--detector', type=Path, default=Path('/path/to/weights/detector.pt'))
    p.add_argument('--ba-python', default=sys.executable)
    p.add_argument('--droid-weights', type=Path, default=Path('/path/to/weights/droid.pth'))
    p.add_argument('--no-ba', action='store_true')
    p.add_argument('--ba-stride', type=int, default=10)
    p.add_argument('--ba-depth-init', choices=['none', 'sparse'], default='none')
    p.add_argument('--ba-no-reject-outliers', action='store_true', help='Disable update rejection for ablations')
    p.add_argument('--max-frames', type=int, default=8192)
    return p

def main():
    a = parser().parse_args()
    if not 0 < a.mask_threshold < 1:
        raise ValueError('mask threshold must be in (0,1)')
    if not 4 <= a.max_frames <= 8192:
        raise ValueError('max-frames must be between 4 and 8192')
    required = [a.video, a.intrinsics, a.checkpoint, a.config]
    if a.ba_stride < 1:
        raise ValueError('Require ba-stride >= 1')
    if a.mask_source == 'sam3-tracking':
        required += [a.sam3_checkpoint, a.detector, a.sam3_source]
    if not a.no_ba:
        required += [a.droid_weights]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    a.output.mkdir(parents=True, exist_ok=True)
    D = a.output / 'frames'
    D.mkdir(exist_ok=True)
    if list(D.glob('*.png')):
        raise ValueError('Use a fresh output directory to avoid stale frame/mask mixing')
    Kd = json.loads(a.intrinsics.read_text())
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError('Cannot open video or invalid FPS')
    n = 0
    while n < a.max_frames:
        (ok, im) = cap.read()
        if not ok:
            break
        (h, w) = im.shape[:2]
        if (w, h) != (Kd['image_width'], Kd['image_height']):
            raise ValueError('Video dimensions differ from intrinsics metadata; explicitly calibrate/rescale before inference')
        cv2.imwrite(str(D / f'{n:06d}.png'), im)
        n += 1
    cap.release()
    if n < 4:
        raise ValueError('At least four frames required')
    if n > 8192:
        raise ValueError('Maximum supported temporal index is 8192')
    boxes = np.zeros((n, 2, 4), np.float32)
    valid = np.zeros((n, 2), bool)
    if a.mask_source == 'sam3-tracking':
        import yaml
        cfg = yaml.safe_load((ROOT / 'third_party/sam3_tracking/configs/default.yaml').read_text())
        cfg.update(output_fps=fps, write_visualization=False, wandb_mode='disabled')
        cf = a.output / 'sam3_config.yaml'
        cf.write_text(yaml.safe_dump(cfg))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'third_party/sam3_tracking/src')
        subprocess.run([a.sam3_python, '-m', 'sam3_tracking', 'frames', '--frames-dir', str(D.resolve()), '--output-dir', str((a.output / 'sam3').resolve()), '--config', str(cf.resolve()), '--detector-weight', str(a.detector.resolve()), '--sam3-checkpoint', str(a.sam3_checkpoint.resolve()), '--sam3-src', str(a.sam3_source.resolve())], env=env, check=True)
        records = [json.loads(line) for line in (a.output / 'sam3/masks.jsonl').read_text().splitlines()]
        assert len(records) == n * 2
        seen = set()
        for row in records:
            i = int(row['frame_idx'])
            s = {'right': 0, 'left': 1}[row['side_name']]
            assert 0 <= i < n and (i, s) not in seen
            seen.add((i, s))
            if row['mask_box_xyxy'] is not None and row['mask_area'] > 0:
                boxes[i, s] = np.array(row['mask_box_xyxy']) / [w, h, w, h]
                valid[i, s] = True
    (model, load_seconds) = build_model(a.checkpoint, torch.device('cuda:0'), a.config)
    images = np.stack([cv2.resize(cv2.imread(str(D / f'{i:06d}.png')), (518, 378))[:, :, ::-1] for i in range(n)])
    images = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).cuda()
    K = np.array([[Kd['fx'], 0, Kd['cx']], [0, Kd['fy'], Kd['cy']], [0, 0, 1]], np.float32)
    K[0] *= 518 / w
    K[1] *= 378 / h
    intr = torch.from_numpy(K).view(1, 1, 3, 3).expand(1, n, 3, 3).cuda()
    opts = dict(bbox_source='gt' if a.mask_source == 'sam3-tracking' else 'mask', mask_threshold=a.mask_threshold)
    if a.mask_source == 'sam3-tracking':
        opts.update(gt_hand_boxes=torch.from_numpy(boxes)[None].cuda(), gt_hand_valid=torch.from_numpy(valid)[None].cuda())
    with torch.inference_mode():
        (out, stats, timing) = run_dual_stream(model, images, intr, **opts)
        (world, _) = camera_to_predicted_global(out)
    validate_contract(stats, n)
    arr = {k: v.float().cpu().numpy() for (k, v) in out.items()}
    poses = world['pred_camera_to_anchor'].float().cpu().numpy()
    if a.mask_source == 'base':
        boxes = arr['mask_tight_boxes']
        valid = arr['presence_mask'] > 0.5
    for (key, value) in arr.items():
        if not np.isfinite(value).all():
            raise ValueError('Non-finite model output: ' + key)
    np.savez_compressed(a.output / 'camera_space.npz', **arr, valid=valid)
    np.savez_compressed(a.output / 'ba_inputs.npz', poses=poses, boxes=boxes)
    del out, world, model, images, intr
    torch.cuda.empty_cache()
    if not a.no_ba:
        subprocess.run([a.ba_python, '-m', 'infinihand.sparse_ba_backend', '--output', str(a.output.resolve()), '--intrinsics', str(a.intrinsics.resolve()), '--weights', str(a.droid_weights.resolve()), '--stride', str(a.ba_stride), '--depth-init', a.ba_depth_init] + (['--no-reject-outliers'] if a.ba_no_reject_outliers else []), cwd=ROOT, check=True)
        poses = np.load(a.output / 'trajectory.npz')['poses']
    if not np.isfinite(poses).all():
        raise ValueError('Non-finite trajectory')
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    np.savez_compressed(a.output / 'world_space.npz', camera_to_world=poses, world_joints=np.einsum('tij,thnj->thni', rot, arr['camera_joints']) + trans[:, None, None, :], world_vertices=np.einsum('tij,thnj->thni', rot, arr['camera_vertices']) + trans[:, None, None, :], world_global_orient=rot[:, None] @ arr['camera_global_orient'], world_translation=np.einsum('tij,thj->thi', rot, arr['camera_translation']) + trans[:, None, :], hand_pose=arr['hand_pose'], betas=arr['betas'], valid=valid)
    receipt = dict(ba_depth_init=None if a.no_ba else a.ba_depth_init, ba_reject_outliers=not (a.no_ba or a.ba_no_reject_outliers), ba_mode=None if a.no_ba else 'fixed', status='complete', frames=n, fps=fps, hand_slots=['right', 'left'], mask_source=a.mask_source, mask_threshold=a.mask_threshold, ba_stride=None if a.no_ba else a.ba_stride, coordinate_frame='first predicted camera', checkpoint=str(a.checkpoint), model_load_seconds=load_seconds, timing=timing, cache=stats)
    (a.output / 'SUCCESS.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
if __name__ == '__main__':
    main()
