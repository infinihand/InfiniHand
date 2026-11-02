import unittest
from types import SimpleNamespace
import torch
from torch import nn
from infinihand.models.lingbotmap_hand_v6 import (
    HandFeatureProvider, LingBotFeatureAdapter, LingBotMapHand, LingBotMapHandConfig,
    sample_one_valid_hand, scatter_hand_rows,
)


class RecordingProvider(HandFeatureProvider):
    def __init__(self):
        super().__init__(crop_size=8, output_dim=3, output_hw=(8, 8), chunk_size=2)
        self.calls = []

    def encode_crops(self, x):
        self.calls.append(x.detach().clone())
        assert torch.isfinite(x).all(), "An invalid crop entered the encoder"
        return x * 2


class RecordingMANO(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shapedirs', torch.ones(778, 3, 10))
        self.calls = []

    def forward(self, global_orient, hand_pose, betas, pose2rot):
        self.calls.append(betas.detach().clone())
        assert torch.isfinite(betas).all(), 'Invalid hand reached MANO decode'
        # Keep dependence on beta to verify compact scatter backward.
        j = betas[:, :1, None] * torch.arange(21., device=betas.device)[None, :, None]
        v = betas[:, :1, None] * torch.arange(778., device=betas.device)[None, :, None]
        return SimpleNamespace(joints=j.expand(-1, -1, 3), vertices=v.expand(-1, -1, 3))


class TinyProvider(HandFeatureProvider):
    def __init__(self):
        super().__init__(crop_size=256, output_dim=4, output_hw=(16, 12), chunk_size=8)
        self.proj = nn.Conv2d(3, 4, 1)
        self.calls = []
    def encode_crops(self, x):
        self.calls.append(x.shape[0])
        return torch.nn.functional.adaptive_avg_pool2d(self.proj(x[..., 32:-32]), (16, 12))


class TinyMANOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 10)
        self.calls = []
    def encode_hand_token(self, features):
        self.calls.append(features.shape[0])
        return features.mean((-1, -2))[:, None]
    def decode_hand_token(self, tokens):
        beta = self.proj(tokens[:, 0])
        m = beta.shape[0]
        return dict(global_orient=torch.eye(3).expand(m, 1, 3, 3),
                    hand_pose=torch.eye(3).expand(m, 15, 3, 3), betas=beta,
                    log_z=beta[:, :1] * .01 - .7)


class TinyDetection(nn.Module):
    def forward(self, tokens, patch_start, hw):
        return tokens[0]


class TinyMask(nn.Module):
    def __init__(self):
        super().__init__(); self.proj = nn.Conv2d(4, 2, 1)
    def forward(self, x, hw):
        b, t = x.shape[:2]
        return torch.nn.functional.interpolate(self.proj(x.flatten(0, 1)), size=hw).reshape(b, t, 2, *hw)


def tiny_model_and_inputs():
    from infinihand.models.mutual_hand_joint_fusion import JointHeatmapDescriptorHead, MutualHandJointFusion, JointOffsetHead
    from infinihand.models.lingbotmap_hand_v6 import HandFeatureFusion
    m = LingBotMapHand.__new__(LingBotMapHand); nn.Module.__init__(m)
    m.cfg = LingBotMapHandConfig(image_height=28, image_width=28, feature_dim=4,
                                lingbot_pixel_channels=2, lingbot_pixel_patch_size=2,
                                mano_context_dim=4, mano_gradient_to_mask_head=False)
    m.model_variant = 'mutual_joint_fusion'; m._bbox_rescale_range = None
    m.camera_scale_head = None; m.hand_detection_block = TinyDetection(); m.hand_mask_head = TinyMask()
    m.lingbot_adapter = LingBotFeatureAdapter(m.cfg); m.hand_encoder = TinyProvider()
    m.feature_fusion = HandFeatureFusion(4); m.joint_head = JointHeatmapDescriptorHead(4, 4)
    m.mano_head = TinyMANOHead(); m.mutual_fusion = MutualHandJointFusion(4, 1)
    m.joint_offset_head = JointOffsetHead(4)
    m.mano_right, m.mano_left = RecordingMANO(), RecordingMANO()
    feature = torch.randn(1, 3, 4, 4, 4)
    inputs = dict(images=torch.rand(1, 3, 3, 28, 28),
                  gt_hand_boxes=torch.tensor([.2, .2, .8, .8]).expand(1, 3, 2, 4),
                  gt_hand_valid=torch.tensor([[[1, 0], [0, 1], [0, 0]]], dtype=torch.bool),
                  use_gt_boxes=True,
                  intrinsics=torch.tensor([[30., 0, 13.5], [0, 30., 13.5], [0, 0, 1.]]).expand(1, 3, 3, 3),
                  backbone_output=SimpleNamespace(aggregated_tokens=[feature], patch_start_idx=0,
                                                  patch_features=feature, pose_enc=torch.ones(1, 3, 9)))
    return m, inputs


class HandSelectionTests(unittest.TestCase):
    def test_selection_only_valid_and_uniform(self):
        torch.manual_seed(17)
        valid = torch.tensor([[[1, 0], [0, 1], [0, 0], [1, 1]]], dtype=torch.bool)
        chosen = sample_one_valid_hand(valid)
        self.assertTrue(torch.equal(chosen.sum(-1), valid.any(-1).long()))
        self.assertFalse((chosen & ~valid).any())
        self.assertTrue(chosen[0, 0, 0] and chosen[0, 1, 1])
        both = sample_one_valid_hand(torch.ones(20000, 2, dtype=torch.bool))
        self.assertTrue(0.48 < both[:, 0].float().mean().item() < 0.52)

    def test_compact_encoder_slots_flip_and_gradients(self):
        p = RecordingProvider()
        images = torch.arange(3 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 3, 8, 8)
        images[0, 2] = float('nan')
        images.requires_grad_()
        boxes = torch.tensor([0., 0., 1., 1.]).expand(1, 3, 2, 4)
        ids = torch.tensor([0, 3])  # frame 0 right, frame 1 left
        out = p(images, boxes, hand_indices=ids)
        self.assertEqual(sum(x.shape[0] for x in p.calls), 2)
        self.assertTrue(torch.allclose(out.canonical_crops[0], images[0, 0], atol=1e-4))
        self.assertTrue(torch.allclose(out.canonical_crops[1], images[0, 1].flip(-1), atol=1e-4))
        restored = scatter_hand_rows(out.features, ids, 6).reshape(1, 3, 2, 3, 8, 8)
        self.assertEqual(restored[0, 0, 1].abs().sum().item(), 0)
        self.assertEqual(restored[0, 1, 0].abs().sum().item(), 0)
        restored.sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertEqual(images.grad[0, 2].abs().sum().item(), 0)

    def test_all_invalid_skips_encoder(self):
        p = RecordingProvider()
        images = torch.full((1, 2, 3, 8, 8), float('nan'))
        boxes = torch.zeros(1, 2, 2, 4)
        out = p(images, boxes, hand_indices=torch.empty(0, dtype=torch.long))
        self.assertEqual(p.calls, [])
        self.assertEqual(out.features.shape, (0, 3, 8, 8))
        self.assertTrue(torch.isfinite(scatter_hand_rows(out.features, torch.empty(0, dtype=torch.long), 4)).all())

    def test_compact_feature_adapter_matches_full_selected_crops(self):
        cfg = LingBotMapHandConfig(feature_dim=4, lingbot_pixel_channels=2,
                                   lingbot_pixel_patch_size=2, mano_context_dim=3)
        adapter = LingBotFeatureAdapter(cfg)
        features = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
        boxes = torch.tensor([0.1, 0.2, 0.8, 0.9]).expand(1, 2, 2, 4)
        full, full_crops = adapter(features, boxes)
        ids = torch.tensor([1, 2])
        compact, compact_crops = adapter(features, boxes, hand_indices=ids)
        self.assertTrue(torch.allclose(compact, full[ids], atol=1e-6))
        self.assertTrue(torch.allclose(compact_crops, full_crops[ids], atol=1e-6))
        empty, _ = adapter(features, boxes, hand_indices=torch.empty(0, dtype=torch.long))
        self.assertEqual(empty.shape, (0, 3, 16, 12))

    def test_mano_decode_only_selected_slots(self):
        model = LingBotMapHand.__new__(LingBotMapHand)
        nn.Module.__init__(model)
        model.mano_right, model.mano_left = RecordingMANO(), RecordingMANO()
        global_rot = torch.eye(3).expand(3, 2, 1, 3, 3)
        pose = torch.eye(3).expand(3, 2, 15, 3, 3)
        beta = torch.full((3, 2, 10), float('nan'))
        beta[0, 0] = .1
        beta[1, 1] = .2
        beta.requires_grad_()
        valid = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
        joints, vertices = model._decode_mano(global_rot, pose, beta, torch.tensor([0, 0, 0]), valid)
        self.assertEqual([len(model.mano_right.calls), len(model.mano_left.calls)], [1, 1])
        self.assertTrue(torch.isfinite(joints).all() and torch.isfinite(vertices).all())
        self.assertTrue(torch.equal(joints[~valid], torch.zeros_like(joints[~valid])))
        joints.sum().backward()
        self.assertTrue(torch.isfinite(beta.grad).all())
        self.assertEqual(beta.grad[~valid].abs().sum().item(), 0)
        model.mano_right.calls.clear(); model.mano_left.calls.clear()
        model._decode_mano(global_rot, pose, beta, selected_valid=torch.zeros_like(valid))
        self.assertEqual(model.mano_right.calls + model.mano_left.calls, [])

    def test_real_forward_compacts_heads_and_retains_full_frame_outputs(self):
        m, args = tiny_model_and_inputs(); m.train()
        out = m(**args, mano_training_selection=True)
        self.assertEqual(m.hand_encoder.calls, [2]); self.assertEqual(m.mano_head.calls, [2])
        self.assertTrue(torch.equal(out['mano_selected_valid'], args['gt_hand_valid']))
        self.assertEqual(out['pred_hand_masks'].shape[:3], (1, 3, 2))
        self.assertEqual(out['pred_camera_pose_enc'].shape[:2], (1, 3))
        self.assertEqual(out['pred_mano_transl_camera'].shape, (1, 3, 2, 3))
        self.assertTrue(torch.isfinite(out['pred_keypoints_3d_camera']).all())
        (out['pred_mano_transl_camera'][out['mano_selected_valid']].square().mean()
         + out['pred_hand_mask_logits'].mean()).backward()
        self.assertTrue(torch.isfinite(m.mano_head.proj.weight.grad).all())
        # Evaluation ignores training-only selection, continuing to infer both hand slots.
        m.eval(); m.hand_encoder.calls.clear(); m.mano_head.calls.clear()
        out = m(**args, mano_training_selection=True)
        self.assertEqual(m.hand_encoder.calls, [6]); self.assertEqual(m.mano_head.calls, [6])
        self.assertNotIn('mano_selected_valid', out)

    def test_real_forward_empty_hand_batch_skips_all_hand_networks(self):
        m, args = tiny_model_and_inputs(); m.train()
        args['gt_hand_valid'] = torch.zeros_like(args['gt_hand_valid'])
        def forbidden(*args):
            raise AssertionError('An empty hand batch entered a network')
        hooks = [module.register_forward_pre_hook(forbidden) for module in
                 (m.feature_fusion, m.joint_head, m.mutual_fusion, m.joint_offset_head)]
        out = m(**args, mano_training_selection=True)
        self.assertEqual(m.hand_encoder.calls, []); self.assertEqual(m.mano_head.calls, [])
        self.assertEqual(m.mano_right.calls + m.mano_left.calls, [])
        self.assertFalse(out['geometry_crop_valid'].any())
        self.assertTrue(torch.isfinite(out['pred_mano_transl_camera']).all())
        out['pred_hand_mask_logits'].mean().backward()
        for hook in hooks: hook.remove()

    def test_scatter_keeps_identity_for_unselected_rotations(self):
        value = (torch.eye(3) * 2).reshape(1, 1, 3, 3).requires_grad_()
        out = scatter_hand_rows(value, torch.tensor([1]), 4, identity_rotation=True)
        self.assertTrue(torch.equal(out[0, 0], torch.eye(3)))
        self.assertTrue(torch.equal(out[1], value[0]))
        out.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))


if __name__ == '__main__':
    unittest.main()
