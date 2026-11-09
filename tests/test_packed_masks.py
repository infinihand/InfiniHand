"""Exact packed-raster contracts. CUDA tests need Triton, but no MANO assets."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from infinihand._postprocess_gpu import rotvec, expmap


class RotationTests(unittest.TestCase):
    def test_so3_matches_scipy(self):
        r = Rotation.random(128, random_state=12).as_matrix()
        v = rotvec(torch.tensor(r))
        np.testing.assert_allclose(v.numpy(), Rotation.from_matrix(r).as_rotvec(), atol=1e-12)
        np.testing.assert_allclose(expmap(v).numpy(), r, atol=1e-12)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedMaskTests(unittest.TestCase):
    def test_exact_coverage_and_score_with_padding(self):
        import triton
        from infinihand._postprocess_raster import gpu_mask
        from infinihand._postprocess_packed import raster_packed, pack_masks, gpu_score
        for height, width in ((31, 47), (47, 31)):
            res = max(height, width)
            uv = torch.tensor([[[-3., 4.], [38., 5.], [12., 50.], [5., 9.]],
                               [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]], device='cuda')
            faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device='cuda', dtype=torch.int32)
            dense = gpu_mask(uv, faces, res)
            dense[:, height:] = False
            dense[:, :, width:] = False
            words = triton.cdiv(res * res, 32)
            packed = torch.zeros((2, words), device='cuda', dtype=torch.int32)
            raster_packed[(2, 2)](uv, faces, packed, 2, 4, res, words, height, width, 128)
            px = torch.arange(res * res, device='cuda')
            unpacked = ((packed[:, px // 32] >> (px % 32)) & 1).reshape(2, res, res).bool()
            self.assertTrue(torch.equal(dense, unpacked))
            target = torch.zeros_like(dense)
            target[0, :height, :width] = True
            pt = pack_masks(target)
            decoded = ((pt[:, px // 32] >> (px % 32)) & 1).reshape_as(target).bool()
            self.assertTrue(torch.equal(target, decoded))
            i = (dense & target).sum((-1, -2))
            fp = (dense & ~target).sum((-1, -2))
            fn = (~dense & target).sum((-1, -2))
            for penalty in (.5, 1., 2.):
                score = gpu_score(uv, faces, res, height, width, pt,
                                  torch.arange(2, device='cuda'), target.sum((-1, -2)), penalty)
                expected = i.double() / (i + penalty * fp.double() + fn).clamp_min(1)
                torch.testing.assert_close(score, expected, atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
