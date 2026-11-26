"""Unified loader for prepared ego_data_v2 scenes.

The public hand-slot order remains the training convention used by
InfinityHand: slot 0 = right, slot 1 = left. The processed files store left
then right, so every side-first payload is permuted exactly once here.

Dataset names affect sampling and reporting only. Metadata controls per-frame
validity; manifests can disable optional camera/world supervision.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from infinihand.training.data.datasets.wilor_virtual_crop import (
    prepare_wilor_virtual_crops,
)
from infinihand.training.data.datasets.doss_dataset_base import DossDatasetBase
from infinihand.training.data.datasets.petrel_io import (
    is_remote_path,
    join_remote,
    normalize_list_item,
    normalize_remote_path,
)


SOURCE_SIDES = ('left', 'right')
CANONICAL_SIDE_INDICES = np.asarray([1, 0], dtype=np.int64)
MANO21_TIP_JOINT_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
MANO21_TIP_VERTEX_INDICES = np.asarray([744, 320, 443, 554, 671], dtype=np.int64)


class UnifiedHandDataset(DossDatasetBase):
    """Canonical MANO21/MANO61 DOSS loader and LingBot input transform."""

    DATASET_NAME = None
    SINGLE_IMAGE_MULTIVIEW = False
    SCHEMA_VERSION = None
    SCHEMA_VERSIONS = None
    SCHEMA_FAMILY = None
    HAS_CAMERA_TRAJECTORY = True
    HAS_WORLD_GEOMETRY = True
    CANONICALIZE_MANO21_FINGERTIPS = False

    def _camera_trajectory_enabled(self):
        return (
            not self.detection_only_annotations
            and getattr(self, "annotation_task", "full") in {"full", "camera", "joint"}
            and self.HAS_CAMERA_TRAJECTORY
            and bool(
                self.cfg.get(
                    'camera_trajectory_available',
                    self.HAS_CAMERA_TRAJECTORY,
                )
            )
        )

    def _world_geometry_enabled(self):
        return (
            not self.detection_only_annotations
            and not self.camera_pose_only_annotations
            and getattr(self, "annotation_task", "full") in {"full", "joint"}
            and self.HAS_WORLD_GEOMETRY
            and bool(
                self.cfg.get(
                    'world_geometry_available',
                    self.HAS_WORLD_GEOMETRY,
                )
            )
        )

    def _hand_geometry_enabled(self):
        return not (
            self.detection_only_annotations
            or self.camera_pose_only_annotations
        )

    def _discover_doss_scenes(self, cfg):
        """Discover one-level self-contained scenes with a single DOSS listing.

        The production conversion only publishes complete scene directories.
        Avoiding one HEAD/metadata read per scene keeps worker construction
        cheap while scene metadata is loaded on first use.
        """
        max_scenes = int(cfg.get('discover_max_scenes', cfg.get('max_scenes', 100000)))
        scenes = []
        root = normalize_remote_path(self.raw_s3_root).rstrip('/')
        try:
            for item in self._petrel_io.list(root + '/'):
                child = normalize_list_item(root, item)
                if not str(item).endswith('/'):
                    continue
                child = child.rstrip('/')
                scene_name = child.rsplit('/', 1)[-1]
                if not scene_name or scene_name.startswith('_'):
                    continue
                scenes.append({'name': scene_name, 'prefix': child})
                if len(scenes) >= max_scenes:
                    break
        except Exception as exc:
            print(
                f'[{self.dataset_name}] warning: failed to discover '
                f'DOSS scenes from {root}: {exc}'
            )
        print(
            f'[{self.dataset_name}] discovered {len(scenes)} '
            f'self-contained DOSS scenes from {root}'
        )
        return scenes

    @staticmethod
    def _frame_intrinsic(frame):
        intrinsic = np.asarray(frame['intrinsics'], dtype=np.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(f'Invalid intrinsic shape: {intrinsic.shape}')
        return intrinsic

    @staticmethod
    def _frame_extrinsic(frame):
        extrinsic = np.asarray(frame['world_to_camera'], dtype=np.float32)
        if extrinsic.shape != (4, 4):
            raise ValueError(f'Invalid world_to_camera shape: {extrinsic.shape}')
        return extrinsic

    def _ensure_scene_loaded(self, scene):
        if scene.get('loaded'):
            return scene
        # The persistent scene registry contains descriptors only. All decoded
        # metadata and camera arrays belong to this sample-local copy.
        scene = dict(scene)
        metadata_path = scene.get('metadata_path')
        source = metadata_path or join_remote(scene['prefix'], 'metadata.json')
        def fetch_bytes():
            return (Path(metadata_path).read_bytes() if metadata_path
                    else self._read_bytes_cached(source))
        cache = getattr(self, '_metadata_disk_cache', None)
        if cache is not None:
            metadata = cache.read_json(source, fetch_bytes, scene.get('metadata_sha256'))
        else:
            metadata = json.loads(fetch_bytes())
        if self.detection_only_annotations or self.camera_pose_only_annotations:
            raise ValueError("Unified scenes use their complete compact annotations; select losses in the training config")
        from .compact import validate_metadata
        validate_metadata(metadata, scene['scene_name'])
        from ..mano_basis_policy import resolve_basis
        scene['mano_left_basis'] = resolve_basis(
            self.dataset_name, metadata, getattr(self, 'annotation_task', 'full'))
        for frame in metadata['frames']:
            for hand in frame['hands'].values():
                hand['in_image'] = bool(hand['hand_valid'])
        frames = metadata.get('frames') or []
        frame_count = int(metadata.get('frame_count', len(frames)))
        if frame_count <= 0 or frame_count != len(frames):
            raise ValueError(
                f'{self.dataset_name} scene {scene["scene_name"]} has invalid frame_count='
                f'{frame_count}, frames={len(frames)}'
            )
        if tuple(metadata.get('side_order', ())) != SOURCE_SIDES:
            raise ValueError(
                f'{self.dataset_name} scene {scene["scene_name"]} has side_order='
                f'{metadata.get("side_order")}, expected {SOURCE_SIDES}'
            )
        if all('intrinsics' in frame for frame in frames):
            intrinsics = np.stack(
                [self._frame_intrinsic(frame) for frame in frames]
            ).astype(np.float32)
        else:
            raise ValueError(
                f'{self.dataset_name} scene {scene["scene_name"]} has no '
                'per-frame intrinsics'
            )
        scene_update = {
            'metadata': metadata,
            'frame_count': frame_count,
            'intrinsics': intrinsics,
            'loaded': True,
        }
        if self._camera_trajectory_enabled():
            extrinsics = np.stack(
                [self._frame_extrinsic(frame) if frame['camera_valid'] else np.eye(4, dtype=np.float32) for frame in frames]
            ).astype(np.float32)
            camera_poses = np.stack(
                [
                    np.asarray(
                        frame['camera_to_world']
                        if frame['camera_valid'] and 'camera_to_world' in frame
                        else np.linalg.inv(extrinsic),
                        dtype=np.float32,
                    )
                    for frame, extrinsic in zip(frames, extrinsics)
                ]
            ).astype(np.float32)
            scene_update.update(
                {
                    'camera_poses': camera_poses,
                    'extrinsics': extrinsics,
                }
            )
        else:
            # The shared view sampler requires a pose tensor. These identities
            # are internal temporal-sampling placeholders only and are never
            # exposed in the public batch.
            scene_update['camera_poses'] = np.repeat(
                np.eye(4, dtype=np.float32)[None],
                frame_count,
                axis=0,
            )
            scene_update['camera_poses_are_sampling_placeholders'] = True
        scene.update(scene_update)
        return scene

    @staticmethod
    def _require_shape(value, shape, name):
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f'{name} has shape={array.shape}, expected={shape}')
        return array

    @staticmethod
    def _frame_path(frame, key, side=None):
        value = frame['paths'][key]
        return value[side] if side is not None else value

    @staticmethod
    def _source_to_canonical(value):
        array = np.asarray(value)
        if array.shape[0] != 2:
            raise ValueError(f'Expected side-first payload, got {array.shape}')
        return array[CANONICAL_SIDE_INDICES]

    def _absolute_frame_path(self, scene, frame, key, side=None):
        path = self._frame_path(frame, key, side=side)
        return path if is_remote_path(path) else join_remote(scene['prefix'], path)

    def _absolute_image_path(self, scene, frame):
        if 'paths' in frame and 'image' in frame['paths']:
            path = frame['paths']['image']
        else:
            path = frame['path']
        return path if is_remote_path(path) else join_remote(scene['prefix'], path)

    def _absolute_annotation_path(
        self,
        scene,
        frame,
        kind,
        side,
    ):
        if kind not in {'masks', 'detections'}:
            raise ValueError(f'Unsupported annotation kind: {kind!r}')
        if side not in SOURCE_SIDES:
            raise ValueError(f'Unsupported hand side: {side!r}')
        target_kind = (
            self.annotation_mask_directory
            if kind == 'masks'
            else self.annotation_detection_directory
        )
        if 'paths' in frame and kind in frame['paths']:
            path = frame['paths'][kind][side]
            parts = str(path).split('/')
            if target_kind not in parts:
                for index in range(len(parts) - 1, -1, -1):
                    if parts[index] == kind:
                        parts[index] = target_kind
                        break
                else:
                    raise ValueError(
                        f'Could not find {kind!r} or {target_kind!r} in annotation path {path!r}'
                    )
                path = '/'.join(parts)
        else:
            local_index = int(
                frame['local_index']
                if 'local_index' in frame
                else frame['frame_id']
            )
            suffix = '.png' if kind == 'masks' else '.npz'
            path = f'{target_kind}/{side}/{local_index:06d}{suffix}'
        return path if is_remote_path(path) else join_remote(scene['prefix'], path)

    def _read_hand_annotations(self, scene, frame, image_hw):
        from .compact import annotations
        return annotations(self, scene, frame, image_hw)

    def _preprocess_source_fields(self, arrays):
        """Dataset-specific transform after DOSS read and before model crops.

        The default route preserves the source camera exactly.  A dataset may
        override this hook when its stored sensor orientation must be
        canonicalized, provided that image-space fields, camera-space fields,
        intrinsics, and camera trajectory are transformed together.
        """
        return arrays

    def _get_doss_data(self, scene, ids):
        if getattr(self, "annotation_task", "full") != "full":
            from ..training_profiles import load_training_sample
            return load_training_sample(self, scene, ids)
        target_hw = (
            int(self.cfg.get('lingbotmap_image_height', 378)),
            int(self.cfg.get('lingbotmap_image_width', 518)),
        )
        frame_records = scene['metadata']['frames']

        output_lists = {
            'images': [],
            'camera_mano': [],
            'camera_joints': [],
            'intrinsics': [],
            'mano_valid': [],
            'hand_valid': [],
            'hand_masks': [],
            'hand_detections': [],
            'camera_mano_joints': [],
            'hand_points_valid': [],
            'hand_points_confidence': [],
            'original_sizes': [],
            '_hand_det_file_present': [],
            '_hand_mask_file_present': [],
        }
        if self._world_geometry_enabled():
            output_lists.update(
                {
                    'world_mano': [],
                    'world_joints': [],
                    'world_mano_joints': [],
                }
            )
        if self._camera_trajectory_enabled():
            output_lists.update(
                {
                    'extrinsics': [],
                    'camera_poses': [],
                }
            )

        for frame_index in np.asarray(ids, dtype=np.int64):
            frame = frame_records[int(frame_index)]
            image = self._read_image_remote(
                self._absolute_image_path(scene, frame)
            )
            original_h, original_w = image.shape[:2]
            if (original_h, original_w) != tuple(frame['image_size_hw']):
                raise ValueError("RGB dimensions disagree with metadata image_size_hw")
            intrinsic = self._frame_intrinsic(frame)

            if not self._hand_geometry_enabled():
                camera_joints = np.zeros((2, 21, 3), dtype=np.float32)
                camera_mano = np.zeros((2, 61), dtype=np.float32)
                camera_mano_joints = np.zeros((2, 21, 3), dtype=np.float32)
                joint_valid = np.zeros((2, 21), dtype=bool)
                confidence = np.zeros((2, 21), dtype=np.float32)
                hand_valid = np.zeros(2, dtype=bool)
                mano_valid = np.zeros(2, dtype=bool)
            else:
                from .compact import geometry
                camera_mano_payload, camera_joints_payload, compact_world_mano, compact_world_joints = geometry(self, scene, frame)
                camera_joints = self._source_to_canonical(
                    self._require_shape(
                        camera_joints_payload['joints'], (2, 21, 3), 'camera_joints'
                    )
                ).astype(np.float32)
                camera_mano = self._source_to_canonical(
                    self._require_shape(
                        camera_mano_payload['params_61'], (2, 61), 'camera_mano.params_61'
                    )
                ).astype(np.float32)
                camera_mano_joints = self._source_to_canonical(
                    self._require_shape(
                        camera_mano_payload['joints'], (2, 21, 3), 'camera_mano.joints'
                    )
                ).astype(np.float32)
                camera_joint_valid = self._source_to_canonical(
                    self._require_shape(
                        camera_joints_payload['joints_valid'], (2, 21), 'camera_joints_valid'
                    )
                ).astype(bool)
                joint_valid = camera_joint_valid.copy()
                confidence = self._source_to_canonical(
                    np.asarray(
                        camera_joints_payload.get(
                            'joints_confidence', camera_joint_valid.astype(np.float32)
                        ),
                        dtype=np.float32,
                    )
                )

                source_hand_valid = np.asarray(
                    [
                        bool(frame['hands'][side].get('in_image', False))
                        for side in SOURCE_SIDES
                    ],
                    dtype=bool,
                )
                hand_valid = self._source_to_canonical(source_hand_valid)
                hand_valid &= joint_valid.any(axis=-1)
                source_mano_valid = np.asarray(
                    camera_mano_payload['mano_valid'],
                    dtype=bool,
                )
                mano_valid = self._source_to_canonical(source_mano_valid)

            if self._world_geometry_enabled():
                if compact_world_mano is None:
                    compact_world_mano = {'params_61': np.zeros((2,61),np.float32), 'joints': np.zeros((2,21,3),np.float32), 'mano_valid': np.zeros(2,bool)}
                    compact_world_joints = {'joints': np.zeros((2,21,3),np.float32), 'joints_valid': np.zeros((2,21),bool)}
                world_mano_payload, world_joints_payload = compact_world_mano, compact_world_joints
                world_joints = self._source_to_canonical(
                    self._require_shape(
                        world_joints_payload['joints'],
                        (2, 21, 3),
                        'world_joints',
                    )
                ).astype(np.float32)
                world_mano = self._source_to_canonical(
                    self._require_shape(
                        world_mano_payload['params_61'],
                        (2, 61),
                        'world_mano.params_61',
                    )
                ).astype(np.float32)
                world_mano_joints = self._source_to_canonical(
                    self._require_shape(
                        world_mano_payload['joints'],
                        (2, 21, 3),
                        'world_mano.joints',
                    )
                ).astype(np.float32)
                world_joint_valid = self._source_to_canonical(
                    self._require_shape(
                        world_joints_payload['joints_valid'],
                        (2, 21),
                        'world_joints_valid',
                    )
                ).astype(bool)
                # Camera-space hand supervision does not depend on camera-pose availability.

            if self.camera_pose_only_annotations:
                # Camera-head supervision needs only RGB, intrinsics and the
                # camera trajectory.  Evaluation exports are not required to
                # carry MANO, side order, masks or detections; use shape-stable
                # zero placeholders so the shared collate/model interface stays
                # unchanged without touching any hand-annotation object.
                hand_masks = np.zeros(
                    (2, original_h, original_w), dtype=np.float32
                )
                hand_detections = np.zeros((2, 4), dtype=np.float32)
                hand_mask_present = np.zeros(2, dtype=bool)
                hand_detection_present = np.zeros(2, dtype=bool)
                annotation_valid = np.zeros(2, dtype=bool)
            else:
                (
                    hand_masks,
                    hand_detections,
                    hand_mask_present,
                    hand_detection_present,
                    annotation_valid,
                ) = self._read_hand_annotations(
                    scene, frame, (original_h, original_w)
                )
            if not self._hand_geometry_enabled():
                hand_valid = annotation_valid.astype(bool)

            output_lists['images'].append(image)
            output_lists['camera_mano'].append(camera_mano)
            output_lists['camera_joints'].append(camera_joints)
            output_lists['intrinsics'].append(intrinsic)
            output_lists['mano_valid'].append(mano_valid)
            output_lists['hand_valid'].append(hand_valid)
            output_lists['hand_masks'].append(hand_masks)
            output_lists['hand_detections'].append(hand_detections)
            output_lists['camera_mano_joints'].append(camera_mano_joints)
            output_lists['hand_points_valid'].append(joint_valid)
            output_lists['hand_points_confidence'].append(confidence)
            output_lists['original_sizes'].append(
                np.asarray((original_h, original_w), dtype=np.int32)
            )
            output_lists['_hand_det_file_present'].append(
                hand_detection_present
            )
            output_lists['_hand_mask_file_present'].append(
                hand_mask_present
            )
            if self._world_geometry_enabled():
                output_lists['world_mano'].append(world_mano)
                output_lists['world_joints'].append(world_joints)
                output_lists['world_mano_joints'].append(world_mano_joints)
            if self._camera_trajectory_enabled():
                extrinsic = self._frame_extrinsic(frame) if frame['camera_valid'] else np.eye(4, dtype=np.float32)
                camera_to_world = (
                    frame['camera_to_world']
                    if frame['camera_valid'] and 'camera_to_world' in frame
                    else np.linalg.inv(extrinsic)
                )
                camera_pose = np.asarray(
                    camera_to_world,
                    dtype=np.float32,
                )
                self._require_shape(camera_pose, (4, 4), 'camera_to_world')
                output_lists['extrinsics'].append(extrinsic)
                output_lists['camera_poses'].append(camera_pose)

        arrays = {
            key: np.stack(values)
            for key, values in output_lists.items()
        }
        arrays = self._preprocess_source_fields(arrays)
        crop_field_names = (
            'images',
            'camera_mano',
            'camera_joints',
            'world_mano',
            'world_joints',
            'extrinsics',
            'intrinsics',
            'mano_valid',
            'hand_valid',
            'hand_masks',
            'hand_detections',
            'camera_poses',
            'camera_mano_joints',
            'world_mano_joints',
        )
        crop_fields = {
            key: arrays[key]
            for key in crop_field_names
            if key in arrays
        }
        if tuple(arrays["images"].shape[1:3]) != tuple(target_hw):
            raise ValueError("Prepared images must match model resolution; no implicit warp is allowed")
        cropped_fields, crop_records = crop_fields, []
        scene['_last_lingbotmap_resize_crop_records'] = crop_records
        bbox_rescale_range = self.cfg.get(
            'bbox_rescale_range', self.bbox_rescale_range
        )
        if len(bbox_rescale_range) != 2:
            raise ValueError(
                'bbox_rescale_range must contain [min, max], got '
                f'{bbox_rescale_range}'
            )
        bbox_rescale_min, bbox_rescale_max = map(
            float,
            bbox_rescale_range,
        )
        if bbox_rescale_min <= 0 or bbox_rescale_min > bbox_rescale_max:
            raise ValueError(
                'bbox_rescale_range must satisfy 0 < min <= max, got '
                f'[{bbox_rescale_min}, {bbox_rescale_max}]'
            )
        bbox_rescale = (
            float(np.random.uniform(bbox_rescale_min, bbox_rescale_max))
            if self.training
            and bbox_rescale_min < bbox_rescale_max
            else 0.5 * (bbox_rescale_min + bbox_rescale_max)
        )
        wilor_crop_fields = prepare_wilor_virtual_crops(
            cropped_fields,
            target_hw=(
                int(self.cfg.get('wilor_image_height', 256)),
                int(self.cfg.get('wilor_image_width', 192)),
            ),
            bbox_rescale=bbox_rescale,
        )
        arrays.update(wilor_crop_fields)

        camera_joints = arrays['camera_joints'].astype(np.float32)
        camera_mano = arrays['camera_mano'].astype(np.float32)
        mano_valid = arrays['mano_valid'].astype(bool)
        hand_valid = arrays['hand_valid'].astype(bool)
        public = {
            'seq_name': scene['scene_name'],
            'ids': np.asarray(ids, dtype=np.int64),
            'frame_num': int(scene['frame_count']),
            'dataset_name': self.dataset_name,
            'sampling_name': self.sampling_name,
            'images': arrays['images'].astype(np.uint8),
            'camera_mano': camera_mano,
            'camera_joints': camera_joints,
            'intrinsics': arrays['intrinsics'].astype(np.float32),
            'mano_valid': mano_valid,
            'hand_valid': hand_valid,
            'hand_masks': arrays['hand_masks'].astype(np.float32),
            'hand_detections': arrays['hand_detections'].astype(np.float32),
            'wilor_images': arrays['wilor_images'].astype(np.uint8),
            'wilor_camera_mano': arrays['wilor_camera_mano'].astype(np.float32),
            'wilor_camera_joints': arrays['wilor_camera_joints'].astype(np.float32),
            'wilor_intrinsics': arrays['wilor_intrinsics'].astype(np.float32),
            'wilor_hand_masks': arrays['wilor_hand_masks'].astype(np.float32),
            'wilor_hand_detections': arrays['wilor_hand_detections'].astype(np.float32),
            'wilor_virtual_rotations': arrays['wilor_virtual_rotations'].astype(np.float32),
            'wilor_crop_homographies': arrays['wilor_crop_homographies'].astype(np.float32),
            'wilor_crop_boxes_source': arrays['wilor_crop_boxes_source'].astype(np.float32),
            'wilor_crop_valid': arrays['wilor_crop_valid'].astype(bool),
            'camera_mano_joints': arrays['camera_mano_joints'].astype(np.float32),
            'wilor_camera_mano_joints': arrays[
                'wilor_camera_mano_joints'
            ].astype(np.float32),
            'original_sizes': arrays['original_sizes'].astype(np.int32),
            'hand_points_valid': arrays['hand_points_valid'].astype(bool),
            'hand_points_confidence': arrays['hand_points_confidence'].astype(
                np.float32
            ),
            '_hand_det_file_present': arrays['_hand_det_file_present'].astype(bool),
            '_hand_mask_file_present': arrays['_hand_mask_file_present'].astype(bool),
        }
        if self._world_geometry_enabled():
            world_joints = arrays['world_joints'].astype(np.float32)
            public.update(
                {
                    'world_mano': arrays['world_mano'].astype(np.float32),
                    'world_joints': world_joints,
                    'world_mano_joints': arrays['world_mano_joints'].astype(
                        np.float32
                    ),
                    'wilor_world_mano': arrays['wilor_world_mano'].astype(
                        np.float32
                    ),
                    'wilor_world_joints': arrays['wilor_world_joints'].astype(
                        np.float32
                    ),
                    'wilor_world_mano_joints': arrays[
                        'wilor_world_mano_joints'
                    ].astype(np.float32),
                }
            )
        if self._camera_trajectory_enabled():
            public.update(
                {
                    'extrinsics': arrays['extrinsics'].astype(np.float32),
                    'camera_poses': arrays['camera_poses'].astype(np.float32),
                    'wilor_extrinsics': arrays['wilor_extrinsics'].astype(
                        np.float32
                    ),
                    'wilor_camera_poses': arrays['wilor_camera_poses'].astype(
                        np.float32
                    ),
                }
            )
        camera_valid = np.array([bool(frame_records[int(i)]['camera_valid']) for i in ids], dtype=bool)
        selected_frames = [frame_records[int(i)] for i in ids]
        public['source_frame_ids'] = np.asarray([f['frame_id'] for f in selected_frames], dtype=np.int64)
        # MANO geometry validity is distinct from projected-mask supervision.
        # Missing annotation/failed fitting is unknown, never an automatic negative.
        mask_valid = []
        for i in ids:
            frame = scene['metadata']['frames'][int(i)]
            slots = []
            for side in ('right', 'left'):
                hand = frame.get('hands', {}).get(side, {})
                reason = hand.get('invalid_reason')
                slots.append(bool(hand.get('mask_supervision_valid',
                    hand.get('hand_valid', False) or reason in
                    ('insufficient_in_image_projection', 'no_projected_area', 'confirmed_absent'))))
            mask_valid.append(slots)
        public['hand_mask_supervision_valid'] = np.asarray(mask_valid, dtype=bool)
        basis = scene['mano_left_basis']
        public['mano_left_basis_b'] = np.full(len(ids), basis == 'B', dtype=bool)
        timestamps, timestamp_valid = [], []
        for frame in selected_frames:
            value = None
            for key, scale in (("timestamp_seconds", 1.0), ("timestamp_s", 1.0), ("timestamp_ns", 1e-9), ("timestamp_us", 1e-6)):
                if frame.get(key) is not None:
                    value = float(frame[key]) * scale
                    break
            if value is None:
                source = frame.get("source_paths", {})
                if source.get("timestamp_ns") is not None:
                    value = float(source["timestamp_ns"]) * 1e-9
                elif source.get("source_fps") and float(source["source_fps"]) > 0:
                    value = float(frame["frame_id"]) / float(source["source_fps"])
            valid_time = value is not None and np.isfinite(value)
            timestamps.append(value if valid_time else 0.0)
            timestamp_valid.append(valid_time)
        public['timestamps_seconds'] = np.asarray(timestamps, dtype=np.float64)
        public['timestamps_valid'] = np.asarray(timestamp_valid, dtype=bool)
        public['camera_trajectory_valid'] = camera_valid & self._camera_trajectory_enabled()
        public['world_geometry_valid'] = camera_valid & self._world_geometry_enabled()
        return public
