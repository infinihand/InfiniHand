"""Task I/O, exact compact transfer, and selective real-model forward contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from infinihand.training.data.training_profiles import (
    build_target, load_training_sample, tensorize_training_sample,
)
from infinihand.models.lingbotmap_hand_v6 import LingBotMapBackbone
from test_hand_selection import tiny_model_and_inputs
from test_loss_per_sample import fixture, criterion


class Source:
    cfg = {'lingbotmap_image_height': 8, 'lingbotmap_image_width': 8}
    dataset_name = sampling_name = 'arctic'

    def __init__(self, task):
        self.annotation_task = task
        self.reads = []
        self.p = np.zeros((2, 61), np.float32)
        self.p[:, 58:61] = [[.1, .2, 1.], [.3, .4, 2.]]
        self.j = np.repeat(self.p[:, None, 58:61], 21, axis=1)

    def _absolute_image_path(self, scene, frame): return 'rgb'
    def _absolute_frame_path(self, scene, frame, kind, side=None): return (kind, side)
    def _frame_intrinsic(self, frame): return np.asarray(frame['intrinsics'], np.float32)
    def _camera_trajectory_enabled(self): return True
    def _world_geometry_enabled(self): return True
    def _source_to_canonical(self, x): return x[[1, 0]]

    def _read_npz_remote(self, path):
        self.reads.append(path)
        return {'params_61': self.p} if path[0] == 'camera_mano' else {'joints': self.j}

    def _read_image_remote(self, path):
        self.reads.append(path)
        if path == 'rgb': return np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
        mask = np.zeros((8, 8), np.uint8)
        mask[2:5, 1:4] = 255
        return mask


def scene():
    C = np.eye(4); C[:3, 3] = [.5, -.1, .2]
    frame = dict(image_size_hw=[8, 8], intrinsics=[[10, 0, 4], [0, 10, 4], [0, 0, 1]],
                 camera_valid=True, camera_to_world=C, frame_id=123,
                 timestamp_seconds=12.3,
                 hands={s: dict(hand_valid=True, bbox_xyxy=[1, 2, 4, 5], mask_supervision_valid=True)
                        for s in ('left', 'right')})
    return dict(scene_name='fixture', frame_count=1, metadata=dict(frames=[frame], mano_left_basis='A', mano_basis_policy='dataset_source_v1_20261015'))


class RecordingBackbone(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.camera = nn.Parameter(torch.tensor(.1))
        self.flags = []

    def forward(self, images, *, run_camera=None):
        self.flags.append(run_camera)
        return SimpleNamespace(aggregated_tokens=self.source.aggregated_tokens,
                               patch_start_idx=0, patch_features=self.source.patch_features,
                               pose_enc=self.source.pose_enc * self.camera if run_camera else None)


class TrainingProfileTests(unittest.TestCase):
    def test_backbone_camera_skip_and_cache_boundary_in_both_memory_modes(self):
        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.camera_head = nn.Linear(3, 9)
                self.camera_calls = 0
                self.clean_calls = 0
            def _aggregate_features(self, x, **kw):
                return [x.mean((-1, -2))], 0
            def _predict_camera(self, tokens, **kw):
                self.camera_calls += 1
                return {'pose_enc': self.camera_head(tokens[0])}
            def clean_kv_cache(self):
                self.clean_calls += 1
        for memory in (False, True):
            for enabled in (False, True):
                with self.subTest(memory=memory, enabled=enabled):
                    backbone = LingBotMapBackbone.__new__(LingBotMapBackbone)
                    nn.Module.__init__(backbone)
                    backbone.model = Core()
                    backbone.lingbot_enable_camera_head = True
                    backbone.freeze_backbone = True
                    backbone.lingbot_activation_offload = False
                    backbone.step_memory_enabled = memory
                    backbone.step_memory_max_frames = 6
                    backbone.step_memory_scale_frames = 2
                    backbone.step_memory_sliding_window = 2
                    backbone.clean_kv_cache_each_forward = True
                    backbone._prepare_images = lambda x: x
                    backbone._pack_backbone_output = lambda images, outputs, start, pose: SimpleNamespace(pose_enc=pose)
                    backbone._clean_step_memory_cache = backbone.model.clean_kv_cache
                    backbone._update_step_memory_stats = lambda **kw: None
                    backbone._kv_cache_bytes = lambda: 0
                    backbone._camera_kv_cache_bytes = lambda: 0
                    output = backbone(torch.rand(1, 6, 3, 4, 4), run_camera=enabled)
                    self.assertEqual(backbone.model.camera_calls, (5 if memory else 1) if enabled else 0)
                    self.assertEqual(backbone.model.clean_calls, 2)
                    if enabled:
                        output.pose_enc.square().mean().backward()
                        self.assertGreater(backbone.model.camera_head.weight.grad.abs().sum(), 0.)
                    else:
                        self.assertIsNone(output.pose_enc)
                        self.assertIsNone(backbone.model.camera_head.weight.grad)

    def test_only_required_storage_objects_are_read(self):
        expected = {'mask': ['rgb', ('masks', 'left'), ('masks', 'right')],
                    'mano': ['rgb', ('camera_mano', None), ('camera_joints', None)],
                    'camera': ['rgb'],
                    'joint': ['rgb', ('masks', 'left'), ('masks', 'right'),
                              ('camera_mano', None), ('camera_joints', None)]}
        for mode in expected:
            with self.subTest(mode=mode):
                source = Source(mode)
                sample = tensorize_training_sample(load_training_sample(source, scene(), [0]))
                self.assertEqual(source.reads, expected[mode])
                self.assertEqual(sample['images'].dtype, torch.uint8)
                self.assertFalse(any(k.startswith(('wilor_', 'world_')) for k in sample if k != 'world_geometry_valid'))
                self.assertEqual(sample['source_frame_ids'].tolist(), [123])
                self.assertAlmostEqual(sample['timestamps_seconds'].item(), 12.3)
                if 'hand_masks' in sample:
                    self.assertEqual(sample['hand_masks'].dtype, torch.bool)
                if 'camera_mano' in sample:
                    np.testing.assert_allclose(sample['camera_mano'][0, 0, 58:61], [.3, .4, 2.])

    def test_uint8_normalization_and_binary_masks_are_exact(self):
        rgb = np.arange(256, dtype=np.uint8)
        old = torch.from_numpy(rgb.astype(np.float32) / 255.)
        new = torch.from_numpy(rgb).float().div_(255.)
        self.assertTrue(torch.equal(old, new))
        raw = np.array([0, 1, 255], np.uint8)
        self.assertTrue(torch.equal(torch.from_numpy((raw > 0).astype(np.float32)),
                                    torch.from_numpy(raw > 0).float()))

    def test_metadata_boxes_equal_projected_mask_boxes(self):
        a = load_training_sample(Source('mano'), scene(), [0])
        b = load_training_sample(Source('joint'), scene(), [0])
        np.testing.assert_array_equal(a['hand_detections'], b['hand_detections'])

    def test_targets_use_camera_labels_and_derive_world_once(self):
        raw = tensorize_training_sample(load_training_sample(Source('joint'), scene(), [0]))
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in raw.items()}
        batch['images'] = batch['images'].float().div_(255.)
        cfg = SimpleNamespace(compute_stage2_diagnostics=False, world_joints_loss_weight=1.,
                              world_mano_loss_weight=0., abs_pose_loss_weight=1., rel_pose_loss_weight=.2)
        target, crop_hw = build_target(batch, cfg)
        self.assertIsNone(crop_hw)
        self.assertNotIn('world_mano', target)
        self.assertNotIn('keypoints_2d_crop', target)
        expected = target['keypoints_3d'] + torch.tensor([.5, -.1, .2])
        torch.testing.assert_close(target['world_joints'], expected)
        torch.testing.assert_close(target['mano_params'], batch['camera_mano'])

    def test_selective_forward_skips_other_heads_and_their_gradients(self):
        for mode in ('mask', 'mano', 'camera', 'joint'):
            with self.subTest(mode=mode):
                model, args = tiny_model_and_inputs()
                model.backbone = RecordingBackbone(args.pop('backbone_output'))
                model.train()
                with patch.object(model.hand_mask_head, 'forward', wraps=model.hand_mask_head.forward) as mask_forward:
                    output = model(**args, heads=mode, mano_training_selection=True)
                expected_mask = mode in ('mask', 'joint')
                expected_mano = mode in ('mano', 'joint')
                expected_camera = mode in ('camera', 'joint')
                self.assertEqual(mask_forward.call_count, int(expected_mask))
                self.assertEqual(model.hand_encoder.calls, [2] if expected_mano else [])
                self.assertEqual(model.mano_head.calls, [2] if expected_mano else [])
                self.assertEqual(model.backbone.flags, [expected_camera])
                self.assertEqual('pred_hand_masks' in output, expected_mask)
                self.assertEqual('pred_keypoints_3d_camera' in output, expected_mano)
                self.assertEqual('pred_camera_pose_enc' in output, expected_camera)
                terms = []
                if expected_mask: terms.append(output['pred_hand_mask_logits'].square().mean())
                if expected_mano: terms.append(output['pred_mano_transl_camera'].square().mean())
                if expected_camera: terms.append(output['pred_camera_pose_enc'].square().mean())
                sum(terms).backward()
                for parameter, expected in ((model.hand_mask_head.proj.weight, expected_mask),
                                            (model.mano_head.proj.weight, expected_mano),
                                            (model.backbone.camera, expected_camera)):
                    self.assertEqual(parameter.grad is not None, expected)
                    if expected:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum(), 0.)

    def test_losses_are_unchanged_when_disabled_predictions_are_absent(self):
        for mode in ('mask', 'mano', 'camera'):
            with self.subTest(mode=mode):
                output, target = fixture()
                c = criterion(world_joints_loss_weight=0., world_mano_loss_weight=0.,
                              abs_pose_loss_weight=1. if mode == 'camera' else 0.,
                              rel_pose_loss_weight=.2 if mode == 'camera' else 0.,
                              camera_hand_loss_enabled=mode == 'mano',
                              hand_mask_bce=1. if mode == 'mask' else 0.,
                              hand_mask_dice=1. if mode == 'mask' else 0.,
                              velocity_smooth_weight=0., betas_consistency_weight=0.)
                expected = c(output, target)['objective_per_sample']
                if mode == 'camera':
                    selected = {'pred_camera_pose_enc': output['pred_camera_pose_enc']}
                    target = {k: v for k, v in target.items() if k in (
                        'camera_poses', 'camera_trajectory_valid', 'intrinsics', 'image_hw',
                        'frame_ids', 'timestamps_seconds', 'timestamps_valid', 'hand_valid')}
                elif mode == 'mask':
                    selected = {k: v for k, v in output.items() if k.startswith('pred_hand_')}
                    target = {k: v for k, v in target.items() if k in ('hand_masks', 'hand_mask_valid', 'hand_valid')}
                else:
                    selected = {k: v for k, v in output.items() if not k.startswith(('pred_hand_', 'pred_camera_'))}
                result = c(selected, target)
                torch.testing.assert_close(result['objective_per_sample'], expected)
                result['objective'].backward()


if __name__ == '__main__':
    unittest.main()
