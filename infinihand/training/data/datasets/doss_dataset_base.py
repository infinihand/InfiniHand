"""Minimal dataset base for the current InfinityHand DOSS pipeline.

Only self-contained ``doss:`` scenes are supported.  Raw videos, online MANO
fitting, Ego4D processed archives, pseudo labels, and legacy Petrel/S3 routes
intentionally do not exist in this base.
"""

from __future__ import annotations

import io
import json
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from infinihand.training.data.datasets.petrel_io import (
    DossIO,
    is_remote_path,
    join_remote,
    normalize_remote_path,
)


MANO21_NAMES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)


def _cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DossDatasetBase(Dataset):
    """Shared sampling and I/O for canonical DOSS hand datasets."""

    DATASET_NAME = None
    SINGLE_IMAGE_MULTIVIEW = False

    def __init__(
        self,
        common_conf=None,
        common_config=None,
        view_sampler_cfg=None,
        data_json: str = "",
        len_train: int = 10000,
    ):
        super().__init__()
        common = common_conf if common_conf is not None else common_config
        if common is None:
            raise ValueError(f"{type(self).__name__} requires common_conf/common_config")
        if not self.DATASET_NAME:
            with open(data_json) as handle:
                self.DATASET_NAME = str(json.load(handle)["dataset_name"]).lower()

        self.data_json = str(data_json)
        self.len_train = int(len_train)
        self.view_sampler_cfg = view_sampler_cfg
        self.training = bool(_cfg_value(common, "training", False))
        self.random_full_scene_pool = self.training and bool(
            _cfg_value(common, "random_full_scene_pool", True)
        )
        self.annotation_task = str(_cfg_value(common, "annotation_task", "full"))
        from ..training_profiles import TASKS
        if self.annotation_task not in TASKS:
            raise ValueError(f"Unknown annotation_task: {self.annotation_task}")
        # Applied only to training sampling; evaluation keeps requested frames.
        self.min_hand_frame_fraction = float(_cfg_value(common, "min_hand_frame_fraction", 1.0 if int(self._view_sampler_value("max_sampled_views", 4)) <= 4 else 0.7)) if self.training else 0.0
        if not 0 <= self.min_hand_frame_fraction <= 1:
            raise ValueError("min_hand_frame_fraction must be in [0, 1]")
        self.hand_sampling_attempts = int(_cfg_value(common, "hand_sampling_attempts", 1024))
        self.dataset_retry_attempts = int(
            _cfg_value(common, "dataset_retry_attempts", 128)
        )
        self.require_hand_det_files = bool(
            _cfg_value(common, "require_hand_det_files", False)
        )
        self.require_hand_mask_files = bool(
            _cfg_value(common, "require_hand_mask_files", False)
        )
        # Annotation storage is canonicalised to the mesh-derived assets.
        self.annotation_mask_directory = str(
            _cfg_value(common, "annotation_mask_directory", "masks")
        ).strip()
        self.annotation_detection_directory = str(
            _cfg_value(
                common,
                "annotation_detection_directory",
                "derived_from_masks",
            )
        ).strip()
        if self.annotation_mask_directory != "masks":
            raise ValueError(
                "DOSS hand annotations must be read from masks, got "
                f"{self.annotation_mask_directory!r}"
            )
        if self.annotation_detection_directory != "derived_from_masks":
            raise ValueError(
                "DOSS hand annotations must be read from derived_from_masks, got "
                f"{self.annotation_detection_directory!r}"
            )
        self.file_cache_max_items = int(
            _cfg_value(common, "file_cache_max_items", 0)
        )
        self.data_cache_mode = str(
            _cfg_value(common, "data_cache_mode", "none")
        ).strip().lower()
        metadata_cache_dir = str(_cfg_value(common, "metadata_cache_dir", "")).strip()
        from .metadata_cache import MetadataCache
        self._metadata_disk_cache = MetadataCache(metadata_cache_dir) if metadata_cache_dir else None
        self.preload_manifest_root = str(
            _cfg_value(common, "preload_manifest_root", "")
        ).strip()
        if self.data_cache_mode not in {"none", "metadata", "preload"}:
            raise ValueError(
                "data_cache_mode must be none, metadata, or preload; got "
                f"{self.data_cache_mode!r}"
            )
        self.bbox_rescale_range = tuple(
            map(float, _cfg_value(common, "bbox_rescale_range", (1.5, 1.5)))
        )

        with open(self.data_json, "r", encoding="utf-8") as handle:
            self.cfg = json.load(handle)
        if self.training:
            from ..split_policy import validate_training_manifest
            validate_training_manifest(self.cfg, str(data_json))
        self.detection_only_annotations = bool(
            self.cfg.get("detection_only_annotations", False)
        )
        self.camera_pose_only_annotations = bool(
            self.cfg.get("camera_pose_only_annotations", False)
        )
        if self.detection_only_annotations and self.camera_pose_only_annotations:
            raise ValueError(
                "detection_only_annotations and camera_pose_only_annotations "
                "are mutually exclusive"
            )
        configured_name = str(
            self.cfg.get("dataset_name", Path(self.data_json).stem)
        ).lower()
        if configured_name != self.DATASET_NAME:
            raise ValueError(
                f"{type(self).__name__} expects dataset_name={self.DATASET_NAME!r}, "
                f"got {configured_name!r}"
            )
        if self.cfg.get("data_format") != "unified_preprocessed":
            raise ValueError(
                f"{configured_name} must use data_format='unified_preprocessed'"
            )

        self.dataset_name = configured_name
        # Keep the split-level sampling label (for example
        # ``egodex_train`` vs ``egodex_eval``) alongside the base dataset
        # name. The trainer uses this label to maintain independent recent
        # loss windows and outlier thresholds for every configured split.
        self.sampling_name = str(
            self.cfg.get("sampling_name", self.dataset_name)
        ).strip().lower()
        if not self.sampling_name:
            raise ValueError(
                f"{configured_name} must use a non-empty sampling_name"
            )
        self.raw_s3_root = normalize_remote_path(
            self.cfg.get("raw_s3_root", "")
        ).rstrip("/")
        if not (":" in self.raw_s3_root and not self.raw_s3_root.startswith("s3://")):
            raise ValueError(
                f"{configured_name} must use the current doss: route, got "
                f"{self.raw_s3_root!r}"
            )
        if int(self.cfg.get("hand_num_points", 21)) != 21:
            raise ValueError(f"{configured_name} must expose canonical MANO21")
        if str(self.cfg.get("hand_keypoint_layout", "mano21")).lower() not in {
            "mano21",
            "mano_21",
        }:
            raise ValueError(f"{configured_name} must use hand_keypoint_layout='mano21'")

        self.hand_num_points = 21
        self.hand_point_names = list(MANO21_NAMES)
        self.hand_target_type = "mano21"
        self.min_scene_frames = int(self.cfg.get("min_scene_frames", 1))
        self._file_cache = OrderedDict()
        self._petrel_io = DossIO(
            self.cfg.get("rclone_config"),
            remote=self.raw_s3_root.split(":", 1)[0],
        )

        configured_scenes = self.cfg.get("scenes")
        manifest_path = str(
            self.cfg.get("preload_manifest_path", "")
        ).strip()
        if not manifest_path and self.preload_manifest_root:
            manifest_path = str(
                Path(self.preload_manifest_root)
                / self.dataset_name
                / "manifest.json"
            )
        if self.data_cache_mode in {"metadata", "preload"}:
            if not manifest_path or not Path(manifest_path).is_file():
                raise FileNotFoundError(
                    f"{self.dataset_name} preload manifest is missing: "
                    f"{manifest_path or '<unset>'}"
                )
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if str(manifest.get("dataset_name", "")).lower() != self.dataset_name:
                raise ValueError(
                    f"manifest dataset mismatch for {self.dataset_name}: "
                    f"{manifest.get('dataset_name')!r}"
                )
            manifest_root = normalize_remote_path(
                manifest.get("raw_s3_root", "")
            ).rstrip("/")
            if manifest_root != self.raw_s3_root:
                raise ValueError(
                    f"manifest root mismatch for {self.dataset_name}: "
                    f"{manifest_root!r} != {self.raw_s3_root!r}"
                )
            configured_scenes = manifest.get("scenes") or []
            print(
                f"[{self.dataset_name}] loaded preload manifest "
                f"{manifest_path} scenes={len(configured_scenes)}",
                flush=True,
            )
        scenes = (
            configured_scenes
            if configured_scenes is not None
            else self._discover_doss_scenes(self.cfg)
        )
        self.scene_ids = {}
        self.scenes = {}
        minimum_scene_frames = self.get_min_required_scene_frames()
        scene_length_sources = {"manifest": 0, "metadata": 0, "unknown": 0}
        filtered_short_scenes = 0
        for scene in scenes or []:
            frame_count, frame_count_source = self._preload_scene_frame_count(scene)
            scene_length_sources[frame_count_source] += 1
            if frame_count is not None and frame_count < minimum_scene_frames:
                filtered_short_scenes += 1
                continue
            payload = self._normalize_doss_scene(scene)
            if payload is None:
                continue
            if frame_count is not None:
                payload["frame_count"] = frame_count
            index = len(self.scene_ids)
            self.scene_ids[index] = payload["scene_name"]
            self.scenes[payload["scene_name"]] = payload
        self.num_scenes = len(self.scene_ids)
        print(
            f"[{self.dataset_name}] scene-length prefilter required>="
            f"{minimum_scene_frames} kept={self.num_scenes} "
            f"filtered_short={filtered_short_scenes} "
            f"sources={scene_length_sources}",
            flush=True,
        )
        if self.num_scenes == 0:
            raise RuntimeError(
                f"No {self.dataset_name} scenes have at least "
                f"{minimum_scene_frames} frames"
            )
        if self.data_cache_mode == "preload":
            for preload_index, scene_name in enumerate(
                self.scene_ids.values(), start=1
            ):
                self._ensure_scene_loaded(self.scenes[scene_name])
                if (
                    preload_index == 1
                    or preload_index % 1000 == 0
                    or preload_index == self.num_scenes
                ):
                    print(
                        f"[{self.dataset_name}] metadata preload "
                        f"{preload_index}/{self.num_scenes}",
                        flush=True,
                    )
            print(
                f"[{self.dataset_name}] preloaded {self.num_scenes} metadata.json files",
                flush=True,
            )
        print(
            f"{self.dataset_name} loaded {self.num_scenes} DOSS scenes from "
            f"{self.raw_s3_root}"
        )

    def _discover_doss_scenes(self, cfg):
        raise NotImplementedError

    def _normalize_doss_scene(self, scene):
        if isinstance(scene, str):
            prefix = scene if is_remote_path(scene) else join_remote(
                self.raw_s3_root, scene
            )
            scene_name = str(scene).rstrip("/").rsplit("/", 1)[-1]
        else:
            scene_name = scene.get("scene_name") or scene.get("name") or scene.get("id")
            prefix = scene.get("prefix") or scene.get("path") or scene.get("scene_root")
            if prefix is None and scene_name:
                prefix = join_remote(self.raw_s3_root, scene_name)
        if not scene_name or not prefix:
            return None
        payload = {
            "scene_name": str(scene_name),
            "prefix": normalize_remote_path(prefix).rstrip("/"),
            "loaded": False,
        }
        if isinstance(scene, dict) and scene.get("metadata_path"):
            payload["metadata_path"] = str(scene["metadata_path"])
        if isinstance(scene, dict) and scene.get("metadata_sha256"):
            payload["metadata_sha256"] = str(scene["metadata_sha256"])
        return payload

    def _preload_scene_frame_count(self, scene):
        """Return a scene length before it is admitted to the sampler.

        New preload manifests carry ``frame_count`` directly.  For legacy
        manifests, read the already-local cached ``metadata.json`` once so a
        short scene is rejected before metadata preloading and retry-based
        sampling.  Non-manifest discovery keeps its previous lazy behavior.
        """
        if not isinstance(scene, dict):
            return None, "unknown"
        declared = scene.get("frame_count")
        if declared is not None:
            frame_count = int(declared)
            if frame_count <= 0:
                raise ValueError(
                    f"{self.dataset_name}/{scene.get('scene_name')} has invalid "
                    f"manifest frame_count={frame_count}"
                )
            return frame_count, "manifest"
        metadata_path = scene.get("metadata_path")
        if not metadata_path:
            return None, "unknown"
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        frames = metadata.get("frames") or []
        frame_count = int(metadata.get("frame_count", len(frames)))
        if frame_count <= 0 or frame_count != len(frames):
            raise ValueError(
                f"{self.dataset_name}/{scene.get('scene_name')} has invalid "
                f"cached metadata frame_count={frame_count}, frames={len(frames)}"
            )
        return frame_count, "metadata"

    def _read_bytes_cached(self, path):
        key = str(path)
        if self.file_cache_max_items > 0 and key in self._file_cache:
            raw = self._file_cache.pop(key)
            self._file_cache[key] = raw
            return raw
        raw = self._petrel_io.read_bytes(key)
        if self.file_cache_max_items > 0:
            self._file_cache[key] = raw
            while len(self._file_cache) > self.file_cache_max_items:
                self._file_cache.popitem(last=False)
        return raw

    def _read_json_remote(self, path):
        return json.loads(self._read_bytes_cached(path).decode("utf-8"))

    def _read_npz_remote(self, path, keys=None):
        with np.load(io.BytesIO(self._read_bytes_cached(path)), allow_pickle=False) as data:
            selected = data.files if keys is None else [key for key in keys if key in data.files]
            return {key: data[key] for key in selected}

    def _read_image_remote(self, path):
        raw = np.frombuffer(self._read_bytes_cached(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode DOSS image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _view_sampler_value(self, key, default=None):
        return _cfg_value(self.view_sampler_cfg, key, default)

    def get_min_required_scene_frames(self):
        if self.SINGLE_IMAGE_MULTIVIEW:
            return 1
        requested = int(self._view_sampler_value("max_sampled_views", 1))
        return max(self.min_scene_frames, requested)

    def _sample_ids(self, scene_name, frame_count, num_views, ids, explicit_scene):
        repeat_single = self.SINGLE_IMAGE_MULTIVIEW and int(num_views) > 1
        if ids is not None and explicit_scene:
            sampled = np.asarray(ids, dtype=np.int64)
        elif repeat_single:
            sampled = np.asarray([random.randrange(frame_count)], dtype=np.int64)
        else:
            count = int(num_views)
            if count <= 0:
                raise ValueError(f"num_views must be positive, got {count}")
            if frame_count < count:
                raise ValueError(
                    f"{self.dataset_name}/{scene_name} has {frame_count} frames, "
                    f"but {count} views were requested"
                )
            if count == 1:
                sampled = np.asarray([random.randrange(frame_count)], dtype=np.int64)
            else:
                min_distance = int(
                    self._view_sampler_value("min_distance_between_views", 1)
                )
                max_distance = int(
                    self._view_sampler_value("max_distance_between_views", 64)
                )
                min_gap = max(count - 1, min(count * 3, min_distance))
                max_gap = min(max(count * 5, count**2), max_distance, frame_count - 1)
                if max_gap < count - 1:
                    raise ValueError(f"max_distance_between_views={max_distance} cannot contain {count} distinct frames")
                min_gap = min(min_gap, max_gap)
                gap = random.randint(min_gap, max_gap)
                start = random.randint(0, frame_count - 1 - gap)
                middle = sorted(
                    random.sample(range(start + 1, start + gap), count - 2)
                )
                sampled = np.asarray(
                    [start, *middle, start + gap], dtype=np.int64
                )
        if sampled.ndim != 1 or len(sampled) != (1 if repeat_single else int(num_views)):
            raise ValueError("Explicit frame IDs must match num_views")
        if np.any(sampled < 0) or np.any(sampled >= frame_count):
            raise ValueError("Frame IDs outside the scene; clipping would silently change samples")
        if repeat_single:
            sampled = np.repeat(sampled[:1], int(num_views))
        return sampled

    def _sample_valid_ids(self, scene, num_views, ids, explicit_scene):
        """Reject unsuitable windows without filtering or renumbering the timeline."""
        frames = scene["metadata"]["frames"]
        validity = np.asarray([any(f["hands"][side]["hand_valid"] for side in ("left", "right")) for f in frames], dtype=bool)
        required = int(np.ceil(self.min_hand_frame_fraction * int(num_views) - 1e-9))
        if required > int(validity.sum()):
            raise ValueError(f"{scene['scene_name']}: only {validity.sum()} hand-valid frames; require {required}/{num_views}")
        attempts = 1 if ids is not None and explicit_scene else max(1, self.hand_sampling_attempts)
        for _ in range(attempts):
            sampled = self._sample_ids(scene["scene_name"], scene["frame_count"], num_views, ids, explicit_scene)
            if int(validity[sampled].sum()) >= required:
                return sampled
        raise ValueError(f"{scene['scene_name']}: no sampled window meets {required}/{num_views} hand-valid frames after {attempts} attempts")

    def _sample_has_required_files(self, output):
        if self.require_hand_det_files:
            present = np.asarray(output.get("_hand_det_file_present", False), dtype=bool)
            if present.ndim < 2 or not present.any(axis=1).all():
                return False, "hand_detections"
        if self.require_hand_mask_files:
            present = np.asarray(output.get("_hand_mask_file_present", False), dtype=bool)
            if present.ndim < 2 or not present.any(axis=1).all():
                return False, "hand_masks"
        return True, None

    @staticmethod
    def _strip_private_fields(output):
        return {key: value for key, value in output.items() if not key.startswith("_")}

    def get_scene_name_for_index(self, index):
        if self.num_scenes == 0:
            return None
        if getattr(self, "random_full_scene_pool", False):
            return None
        return self.scene_ids[int(index) % self.num_scenes]

    def _scene_candidate_indices(self, index, attempts):
        if getattr(self, "random_full_scene_pool", False):
            # Worker seeds handle reproducibility/rank independence. Virtual
            # length controls dataset frequency, never the scene pool size.
            selected = []
            while len(selected) < attempts:
                candidate = int(np.random.randint(self.num_scenes))
                if candidate not in selected:
                    selected.append(candidate)
            return selected
        start = int(index or 0) % self.num_scenes
        return [(start + offset) % self.num_scenes for offset in range(attempts)]

    def set_epoch(self, epoch: int):
        self.current_epoch = max(int(epoch), 0)

    def __len__(self):
        return self.len_train

    def __getitem__(self, index_tuple):
        sequence_index, num_views, aspect_ratio = index_tuple
        return self.get_data(
            seq_index=sequence_index,
            img_per_seq=num_views,
            aspect_ratio=aspect_ratio,
        )

    def get_data(
        self,
        seq_index=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
        img_per_seq=None,
    ):
        del aspect_ratio  # The current geometry path has fixed 378x518 output.
        if self.num_scenes == 0:
            raise RuntimeError(
                f"No {self.dataset_name} DOSS scenes found under {self.raw_s3_root}"
            )
        explicit_scene = seq_name is not None
        if img_per_seq is None:
            img_per_seq = int(self._view_sampler_value("max_sampled_views", 1))

        max_attempts = 1 if explicit_scene else min(
            self.num_scenes, max(1, self.dataset_retry_attempts)
        )
        candidates = [] if explicit_scene else self._scene_candidate_indices(seq_index, max_attempts)
        last_error = None
        for attempt in range(max_attempts):
            current_name = seq_name if explicit_scene else self.scene_ids[
                candidates[attempt]
            ]
            try:
                scene = self._ensure_scene_loaded(self.scenes[current_name])
                frame_count = int(scene["frame_count"])
                minimum = self.get_min_required_scene_frames()
                if frame_count < minimum:
                    raise ValueError(
                        f"{self.dataset_name}/{current_name} has {frame_count} "
                        f"frames, need {minimum}"
                    )
                sampled_ids = self._sample_valid_ids(scene, img_per_seq, ids, explicit_scene)
                output = self._get_doss_data(scene, sampled_ids)
                sample_ok, missing = self._sample_has_required_files(output)
                if not sample_ok:
                    raise ValueError(
                        f"{self.dataset_name}/{current_name} is missing required {missing}"
                    )
                return self._strip_private_fields(output)
            except Exception as exc:
                last_error = (current_name, exc)
                if explicit_scene:
                    raise
        scene, error = last_error
        raise RuntimeError(
            f"{self.dataset_name} failed to read a valid DOSS sample after "
            f"{max_attempts} attempts; last scene={scene}: "
            f"{type(error).__name__}: {error}"
        ) from error
