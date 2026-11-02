import unittest
import torch
from infinihand.models.geometry import _invert_se3, _pose_encoding_to_w2c
from infinihand.models.mutual_hand_joint_fusion import decode_camera_translation_lsp
class GeometryTests(unittest.TestCase):
 def test_pose_and_inverse(self):
  enc=torch.tensor([[[1.,2.,3.,0.,0.,0.,1.,1.,1.]]]);T=_pose_encoding_to_w2c(enc)
  self.assertTrue(torch.allclose(T@_invert_se3(T),torch.eye(4)[None,None]))
  self.assertTrue(torch.allclose(T[0,0,:3,3],torch.tensor([1.,2.,3.])))
 def test_lsp_fixed_depth(self):
  torch.manual_seed(0);j=torch.randn(2,21,3)*.01;t=torch.tensor([[.04,-.02,.7],[-.03,.01,.8]])
  K=torch.tensor([[500.,0,96],[0,500.,128],[0,0,1]]).repeat(2,1,1)
  xyz=j+t[:,None];uv=xyz@K.transpose(-1,-2);uv=uv[:,:,:2]/uv[:,:,2:];uv/=torch.tensor([192.,256.])
  recovered=decode_camera_translation_lsp(uv,j,t[:,2].log(),K,(256,192))
  self.assertTrue(torch.allclose(recovered,t,atol=1e-6))
if __name__=='__main__':unittest.main()
