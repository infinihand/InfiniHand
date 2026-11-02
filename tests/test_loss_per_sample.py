"""Full objectives retain the batch axis for independent sample rejection."""
import unittest

import torch

from infinihand.training.losses import (
    LingBotMapHandLoss,
    _slice_loss_batch,
    axis_angle_to_matrix,
    compute_temporal_smoothness_loss,
    compute_betas_consistency_loss,
)


def fixture(batch_size=2, frames=4):
    torch.manual_seed(19)
    shape = (batch_size, frames, 2)
    valid = torch.ones(shape, dtype=torch.bool)
    gt_mano = torch.zeros(*shape, 61)
    gt_mano[..., 60] = 0.7
    gt_joints = torch.randn(*shape, 21, 3) * 0.01
    gt_joints[..., 2] += 0.7
    camera = torch.eye(4).repeat(batch_size, frames, 1, 1)
    camera[..., 0, 3] = torch.arange(frames) * 0.02
    intrinsics = torch.tensor([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]]).repeat(batch_size, frames, 1, 1)
    enc = torch.zeros(batch_size, frames, 9)
    enc[..., 6] = 1.
    enc[..., 0] = -torch.arange(frames) * 0.025
    enc[..., 7] = 2 * torch.atan(torch.tensor(120. / 300.))
    enc[..., 8] = 2 * torch.atan(torch.tensor(160. / 300.))
    offsets = torch.arange(batch_size).reshape(batch_size, 1, 1, 1) * .025
    output = {
        'pred_hand_probability': torch.full(shape, .5),
        'pred_hand_mask_logits': torch.randn(*shape, 8, 8, requires_grad=True),
        'geometry_crop_valid': valid.clone(),
        'pred_mano_params': {
            'global_orient': axis_angle_to_matrix(torch.full((*shape, 1, 3), .01, requires_grad=True)),
            'hand_pose': axis_angle_to_matrix(torch.full((*shape, 15, 3), .02, requires_grad=True)),
            'betas': torch.randn(*shape, 10, requires_grad=True) * .1,
            'transl': (gt_mano[..., 58:61] + .01 + offsets).requires_grad_(),
        },
        'pred_keypoints_3d_camera': (gt_joints + .005 + offsets.unsqueeze(-1)).requires_grad_(),
        'pred_keypoints_2d': torch.rand(*shape, 21, 2, requires_grad=True) - .5,
        'pred_camera_pose_enc': enc.requires_grad_(),
        'mano_virtual_rotations': torch.eye(3).repeat(*shape, 1, 1),
        'camera_output_space': 'image_camera',
        'model_variant': 'mutual_joint_fusion',
    }
    gt_world_joints = gt_joints + camera[..., :3, 3][:, :, None, None]
    gt_world_mano = gt_mano.clone()
    gt_world_mano[..., 58:61] += camera[..., :3, 3][:, :, None]
    target = {
        'hand_valid': valid,
        'hand_masks': torch.ones(*shape, 8, 8),
        'hand_mask_valid': valid.clone(),
        'mano_params': gt_mano,
        'mano_valid': valid.clone(),
        'keypoints_3d': gt_joints,
        'keypoints_3d_joint_valid': valid[..., None].expand(*shape, 21),
        'keypoints_2d_crop': torch.cat((torch.zeros(*shape, 21, 2), torch.ones(*shape, 21, 1)), -1),
        'crop_virtual_rotations': torch.eye(3).repeat(*shape, 1, 1),
        'frame_ids': torch.arange(frames).repeat(batch_size, 1),
        'timestamps_seconds': torch.arange(frames).repeat(batch_size, 1) * .1,
        'timestamps_valid': torch.ones(batch_size, frames, dtype=torch.bool),
        'camera_poses': camera,
        'intrinsics': intrinsics,
        'image_hw': torch.tensor([240., 320.]),
        'world_joints': gt_world_joints,
        'world_joints_valid': valid[..., None].expand(*shape, 21),
        'world_mano': gt_world_mano,
        'world_mano_valid': valid.clone(),
        'world_geometry_valid': torch.ones(batch_size, frames, dtype=torch.bool),
        'camera_trajectory_valid': torch.ones(batch_size, frames, dtype=torch.bool),
        'wilor_camera_poses': camera.unsqueeze(2).repeat(1, 1, 2, 1, 1),
        'stage2_supervision_valid': torch.ones(batch_size, frames, dtype=torch.bool),
    }
    return output, target


def criterion(**kwargs):
    settings = dict(bbox_loss_enabled=False, hand_probability=0., hand_bbox_smooth_l1=0., hand_bbox_giou=0.,
        world_joints_loss_weight=1., world_mano_loss_weight=0., abs_pose_loss_weight=1., rel_pose_loss_weight=.2,
        velocity_smooth_weight=.3, betas_consistency_weight=.05)
    settings.update(kwargs)
    return LingBotMapHandLoss(**settings)


class PerSampleLossTests(unittest.TestCase):
    def test_all_terms_equal_independent_sample_forward(self):
        output, target = fixture()
        target['hand_valid'][0, :, 1] = False
        c = criterion()
        loss = c(output, target)
        expected = torch.stack([c(_slice_loss_batch(output, i, 2), _slice_loss_batch(target, i, 2))['objective'] for i in range(2)])
        self.assertTrue(loss['objective_per_sample'].requires_grad)
        self.assertEqual(tuple(loss['objective_per_sample'].shape), (2,))
        torch.testing.assert_close(loss['objective_per_sample'], expected)
        torch.testing.assert_close(loss['objective'], expected.mean())
        loss['objective'].backward()
        self.assertTrue(torch.isfinite(output['pred_camera_pose_enc'].grad).all())
        self.assertTrue(torch.isfinite(output['pred_keypoints_3d_camera'].grad).all())
        self.assertGreater(output['pred_camera_pose_enc'].grad.abs().sum(), 0.)
        self.assertGreater(loss['loss_camera_hand_weighted'], 0.)
        self.assertGreater(loss['loss_world_hand_weighted'], 0.)
        self.assertGreater(loss['loss_pose_weighted'], 0.)
        self.assertGreater(loss['betas_consistency_weighted'], 0.)

    def test_shared_image_sizes_not_sliced_for_batch_two(self):
        _, target = fixture()
        target['crop_image_hw'] = torch.tensor([256., 192.])
        sliced = _slice_loss_batch(target, 1, 2)
        self.assertEqual(sliced['image_hw'].tolist(), [240., 320.])
        self.assertEqual(sliced['crop_image_hw'].tolist(), [256., 192.])
        self.assertEqual(sliced['camera_poses'].shape[0], 1)

    def test_bad_mask_sample_does_not_hide_good_sample_loss(self):
        output, target = fixture()
        with torch.no_grad():
            output['pred_hand_mask_logits'][1, 0, 0, 0, 0] = float('nan')
        c = criterion()
        loss = c(output, target)
        self.assertTrue(torch.isfinite(loss['objective_per_sample'][0]))
        self.assertTrue(torch.isnan(loss['objective_per_sample'][1]))
        # Trainer reruns accepted samples to isolate their backward graph.
        single = c(_slice_loss_batch(output, 0, 2), _slice_loss_batch(target, 0, 2))
        torch.testing.assert_close(loss['objective_per_sample'][0], single['objective'])

    def test_bad_camera_sample_is_not_silently_masked(self):
        output, target = fixture()
        with torch.no_grad():
            output['pred_camera_pose_enc'][1, 1, 0] = float('inf')
        loss = criterion()(output, target)['objective_per_sample']
        self.assertTrue(torch.isfinite(loss[0]))
        self.assertFalse(torch.isfinite(loss[1]))

    def test_disabled_nonfinite_mask_does_not_poison_mano_camera(self):
        output, target = fixture()
        with torch.no_grad():
            output['pred_hand_mask_logits'][1] = float('nan')
        loss = criterion(hand_mask_bce=0., hand_mask_dice=0.)(output, target)
        self.assertTrue(torch.isfinite(loss['objective_per_sample']).all())

    def test_selected_geometry_does_not_change_mask_supervision(self):
        output, target = fixture(1)
        output['geometry_crop_valid'][..., 1] = False
        c = criterion()
        before = c(output, target)
        with torch.no_grad():
            output['pred_keypoints_3d_camera'][..., 1, :, :] += 100.
            output['pred_mano_params']['transl'][..., 1, :] += 100.
        after = c(output, target)
        torch.testing.assert_close(after['joints'], before['joints'])
        torch.testing.assert_close(after['trans'], before['trans'])
        torch.testing.assert_close(after['loss_world_hand_weighted'], before['loss_world_hand_weighted'])
        torch.testing.assert_close(after['hand_mask_bce_loss'], before['hand_mask_bce_loss'])
        grads = torch.autograd.grad(after['objective'], output['pred_hand_mask_logits'])[0]
        self.assertGreater(grads[..., 1, :, :].abs().sum(), 0.)


    def test_selection_mask_alone_gates_all_hand_losses(self):
        output, target = fixture(1)
        output['mano_selected_valid'] = torch.ones_like(target['hand_valid'])
        output['mano_selected_valid'][..., 1] = False
        c = criterion(world_mano_loss_weight=1.)
        before = c(output, target)
        with torch.no_grad():
            output['pred_keypoints_3d_camera'][..., 1, :, :] += 500.
            output['pred_mano_params']['transl'][..., 1, :] += torch.arange(4).square().reshape(1, 4, 1) * 100.
            output['pred_mano_params']['betas'][..., 1, :] += torch.arange(4).reshape(1, 4, 1) * 100.
            output['pred_mano_params']['hand_pose'][..., 1, :, :, :] *= 3.
            output['pred_mano_params']['global_orient'][..., 1, :, :, :] *= 3.
            output['pred_keypoints_2d'][..., 1, :, :] += 500.
        after = c(output, target)
        for key in ('joints', 'trans', 'go', 'hp', 'betas', 'joints2d_direct', 'reproj',
                    'velocity_smooth', 'betas_consistency', 'loss_world_hand_weighted'):
            torch.testing.assert_close(after[key], before[key], msg=key)
        torch.testing.assert_close(after['hand_mask_bce_loss'], before['hand_mask_bce_loss'])

    def test_temporal_missing_selection_breaks_triples(self):
        frames = 5
        translations = torch.zeros(1, frames, 2, 3)
        translations[0, :, 0, 0] = torch.tensor([0., 1., 1000., 9., 16.])
        translations[0, :, 1, 0] = -torch.arange(frames).square().float()
        rotations = torch.eye(3).repeat(1, frames, 2, 1, 1)
        selected = torch.zeros(1, frames, 2, dtype=torch.bool)
        selected[0, [0, 1, 3, 4], 0] = True
        selected[0, 2, 1] = True
        times = torch.arange(frames).reshape(1, -1) * .1
        # No hand has three consecutive selected observations. Compressing
        # selected observations or differencing across hand slots would fail.
        loss = compute_temporal_smoothness_loss(translations, rotations, times, selected)
        self.assertEqual(loss.item(), 0.)
        translations[~selected] = float('nan')
        rotations[~selected] = float('nan')
        torch.testing.assert_close(compute_temporal_smoothness_loss(translations, rotations, times, selected), loss)

    def test_shape_temporal_ignores_nonselected_placeholders(self):
        values = torch.zeros(1, 4, 2, 10, requires_grad=True)
        valid = torch.zeros(1, 4, 2, dtype=torch.bool)
        valid[0, [0, 2], 0] = True
        valid[0, [1, 3], 1] = True
        with torch.no_grad():
            values[0, 0, 0] = 1.
            values[0, 2, 0] = 3.
            values[0, 1, 1] = 5.
            values[0, 3, 1] = 7.
            values[~valid] = float('nan')
        loss = compute_betas_consistency_loss(values, valid)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(values.grad[~valid].abs().sum().item(), 0.)

    def test_nonfinite_valid_camera_target_is_sample_local(self):
        output, target = fixture()
        target['camera_poses'][1, 1, 0, 3] = float('nan')
        losses = criterion()(output, target)['objective_per_sample']
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertTrue(torch.isnan(losses[1]))
        # The same property holds for camera-only training, whose normal
        # annotation validity filter would otherwise hide this corrupt label.
        losses = criterion(world_joints_loss_weight=0., world_mano_loss_weight=0.,
                           camera_hand_loss_enabled=False)(output, target)['objective_per_sample']
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertTrue(torch.isnan(losses[1]))

    def test_shape_errors_are_not_swallowed(self):
        output, target = fixture()
        target['camera_poses'] = torch.zeros(2, 4, 3, 3)
        with self.assertRaises(ValueError):
            criterion()(output, target)


if __name__ == '__main__':
    unittest.main()
