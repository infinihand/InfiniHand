"""Composition and tensor conversion for the current DOSS hand pipeline."""

from __future__ import annotations

import bisect
import random

from hydra.utils import instantiate
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


FLOAT_FIELDS = (
    "camera_mano",
    "camera_joints",
    "world_mano",
    "world_joints",
    "extrinsics",
    "camera_poses",
    "intrinsics",
    "hand_masks",
    "hand_detections",
    "camera_mano_joints",
    "world_mano_joints",
    "hand_points_confidence",
    "wilor_camera_mano",
    "wilor_camera_joints",
    "wilor_world_mano",
    "wilor_world_joints",
    "wilor_extrinsics",
    "wilor_camera_poses",
    "wilor_intrinsics",
    "wilor_hand_masks",
    "wilor_hand_detections",
    "wilor_virtual_rotations",
    "wilor_crop_homographies",
    "wilor_crop_boxes_source",
    "wilor_camera_mano_joints",
    "wilor_world_mano_joints",
)

BOOL_FIELDS = (
    "mano_valid",
    "hand_valid",
    "hand_points_valid",
    "wilor_crop_valid",
    "world_geometry_valid",
    "camera_trajectory_valid",
)


OPTIONAL_WORLD_FIELDS = {
    "world_mano": (2, 61),
    "world_joints": (2, 21, 3),
    "world_mano_joints": (2, 21, 3),
    "wilor_world_mano": (2, 61),
    "wilor_world_joints": (2, 21, 3),
    "wilor_world_mano_joints": (2, 21, 3),
}

OPTIONAL_CAMERA_TRAJECTORY_FIELDS = {
    "extrinsics": (4, 4),
    "camera_poses": (4, 4),
    "wilor_extrinsics": (2, 4, 4),
    "wilor_camera_poses": (2, 4, 4),
}


class ComposedDataset(Dataset):
    """Combine DOSS adapters and emit the fixed LingBot batch."""

    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        del kwargs
        datasets = [
            instantiate(dataset_cfg, common_conf=common_config)
            for dataset_cfg in dataset_configs
        ]
        self.base_dataset = TupleConcatDataset(datasets, common_config)
        self.fixed_num_images = int(common_config.fix_img_num)
        self.fixed_aspect_ratio = float(common_config.fix_aspect_ratio)
        self.total_samples = len(self.base_dataset)

    def set_epoch(self, epoch: int):
        for dataset in self.base_dataset.datasets:
            dataset.set_epoch(epoch)

    def __len__(self):
        return self.total_samples

    @staticmethod
    def _stack(batch, key, dtype):
        if key not in batch:
            return None
        return torch.from_numpy(np.stack(batch[key]).astype(dtype))

    @staticmethod
    def _normalize_optional_fields(batch):
        """Make optional geometry collatable across heterogeneous datasets.

        World-space geometry and camera trajectories are not available for
        every source dataset.  PyTorch's default collate requires every sample
        in a batch to expose exactly the same mapping keys, so absent values
        are represented by shape-correct placeholders plus explicit
        per-frame availability masks.
        """
        frame_count = int(np.asarray(batch["ids"]).shape[0])

        has_world_geometry = all(key in batch for key in OPTIONAL_WORLD_FIELDS)
        if any(key in batch for key in OPTIONAL_WORLD_FIELDS) and not has_world_geometry:
            present = sorted(key for key in OPTIONAL_WORLD_FIELDS if key in batch)
            missing = sorted(key for key in OPTIONAL_WORLD_FIELDS if key not in batch)
            raise ValueError(
                "Incomplete optional world geometry: "
                f"present={present}, missing={missing}"
            )
        availability = np.asarray(batch.get("world_geometry_valid", np.full((frame_count,), has_world_geometry)), dtype=np.bool_)
        if availability.shape != (frame_count,):
            raise ValueError("world_geometry_valid must have one flag per frame")
        if availability.any() and not has_world_geometry:
            raise ValueError("world_geometry_valid is true but geometry fields are absent")
        batch["world_geometry_valid"] = availability & has_world_geometry
        if not has_world_geometry:
            for key, trailing_shape in OPTIONAL_WORLD_FIELDS.items():
                batch[key] = np.zeros(
                    (frame_count, *trailing_shape), dtype=np.float32
                )

        has_camera_trajectory = all(
            key in batch for key in OPTIONAL_CAMERA_TRAJECTORY_FIELDS
        )
        if (
            any(key in batch for key in OPTIONAL_CAMERA_TRAJECTORY_FIELDS)
            and not has_camera_trajectory
        ):
            present = sorted(
                key for key in OPTIONAL_CAMERA_TRAJECTORY_FIELDS if key in batch
            )
            missing = sorted(
                key for key in OPTIONAL_CAMERA_TRAJECTORY_FIELDS if key not in batch
            )
            raise ValueError(
                "Incomplete optional camera trajectory: "
                f"present={present}, missing={missing}"
            )
        availability = np.asarray(batch.get("camera_trajectory_valid", np.full((frame_count,), has_camera_trajectory)), dtype=np.bool_)
        if availability.shape != (frame_count,):
            raise ValueError("camera_trajectory_valid must have one flag per frame")
        if availability.any() and not has_camera_trajectory:
            raise ValueError("camera_trajectory_valid is true but geometry fields are absent")
        batch["camera_trajectory_valid"] = availability & has_camera_trajectory
        if not has_camera_trajectory:
            identity = np.eye(4, dtype=np.float32)
            for key, trailing_shape in OPTIONAL_CAMERA_TRAJECTORY_FIELDS.items():
                repeat_shape = (frame_count, *trailing_shape[:-2], 1, 1)
                batch[key] = np.tile(identity, repeat_shape)

        return batch

    def __getitem__(self, index_tuple):
        if self.fixed_num_images > 0:
            sequence_index = (
                index_tuple[0] if isinstance(index_tuple, tuple) else index_tuple
            )
            index_tuple = (
                sequence_index,
                self.fixed_num_images,
                self.fixed_aspect_ratio,
            )
        batch = self.base_dataset[index_tuple]
        if batch.get("annotation_task", "full") != "full":
            from .training_profiles import tensorize_training_sample
            return tensorize_training_sample(batch)
        batch = self._normalize_optional_fields(batch)
        sample = {
            "seq_name": batch["seq_name"],
            "dataset_name": batch["dataset_name"],
            # Split-level label used by per-dataset outlier statistics. Keep
            # dataset_name unchanged for existing consumers and metadata QA.
            "dataset_sampling_name": batch.get(
                "sampling_name", batch["dataset_name"]
            ),
            "ids": torch.from_numpy(np.asarray(batch["ids"], dtype=np.int64)),
            "frame_num": torch.tensor(int(batch["frame_num"]), dtype=torch.long),
        }

        for key, dtype in (("source_frame_ids", np.int64), ("timestamps_seconds", np.float64), ("timestamps_valid", np.bool_), ("hand_mask_supervision_valid", np.bool_), ("mano_left_basis_b", np.bool_)):
            sample[key] = torch.from_numpy(np.asarray(batch[key], dtype=dtype))

        images = self._stack(batch, "images", np.float32)
        sample["images"] = images.permute(0, 3, 1, 2).contiguous().div(255.0)

        wilor_images = self._stack(batch, "wilor_images", np.float32)
        if wilor_images is None or wilor_images.ndim != 5 or wilor_images.shape[-1] != 3:
            shape = None if wilor_images is None else tuple(wilor_images.shape)
            raise ValueError(f"wilor_images must be [N,2,H,W,3], got {shape}")
        sample["wilor_images"] = (
            wilor_images.permute(0, 1, 4, 2, 3).contiguous().div(255.0)
        )

        for key in FLOAT_FIELDS:
            tensor = self._stack(batch, key, np.float32)
            if tensor is not None:
                sample[key] = tensor
        for key in BOOL_FIELDS:
            tensor = self._stack(batch, key, np.bool_)
            if tensor is not None:
                sample[key] = tensor.bool()
        original_sizes = self._stack(batch, "original_sizes", np.int64)
        if original_sizes is not None:
            sample["original_sizes"] = original_sizes.long()
        return sample


class TupleConcatDataset(ConcatDataset):
    def __init__(self, datasets, common_config):
        super().__init__(datasets)
        self.inside_random = bool(common_config.inside_random)

    def _resolve_index(self, index):
        index = int(index)
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index += len(self)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        sample_index = index if dataset_index == 0 else (
            index - self.cumulative_sizes[dataset_index - 1]
        )
        return dataset_index, sample_index

    def get_scene_key(self, index):
        dataset_index, sample_index = self._resolve_index(index)
        dataset = self.datasets[dataset_index]
        scene_name = dataset.get_scene_name_for_index(sample_index)
        if getattr(dataset, "random_full_scene_pool", False):
            raise ValueError("Unique-scene batching requires resolved scene IDs; full-pool random sampling does not expose them before loading")
        return f"{dataset_index}:{scene_name or sample_index}"

    def __getitem__(self, index):
        index_tuple = index if isinstance(index, tuple) else None
        raw_index = index_tuple[0] if index_tuple is not None else index
        if self.inside_random:
            raw_index = random.randint(0, self.cumulative_sizes[-1] - 1)
        dataset_index, sample_index = self._resolve_index(raw_index)
        if index_tuple is None:
            return self.datasets[dataset_index][sample_index]
        return self.datasets[dataset_index][
            (sample_index, index_tuple[1], index_tuple[2])
        ]
