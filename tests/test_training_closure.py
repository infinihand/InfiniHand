import unittest,importlib.util,os
from pathlib import Path
import torch,numpy as np
from torch import nn
from infinihand.models.lingbotmap_hand_v6 import LingBotMapHand
from infinihand.training.losses import _masked_mean,compute_temporal_smoothness_loss
from infinihand.training.data.datasets.compact import annotations
class Tests(unittest.TestCase):
 def test_mixed_basis_grad_and_root(self):
  r=Path(__file__).resolve().parents[1]
  assets=Path(os.environ.get('MANO_MODEL_PATH',r/'weights/mano'))
  if not all((assets/f'MANO_{side}.pkl').is_file() for side in ('RIGHT','LEFT')):self.skipTest('Licensed MANO assets required; set MANO_MODEL_PATH')
  spec=importlib.util.spec_from_file_location('mw',r/'third_party/WiLoR/wilor/models/mano_wrapper.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  model=LingBotMapHand.__new__(LingBotMapHand);nn.Module.__init__(model)
  for s in ['right','left']:setattr(model,'mano_'+s,m.MANO(model_path=str(assets),is_rhand=s=='right',use_pca=False,flat_hand_mean=True))
  ori=torch.eye(3).repeat(2,2,1,1,1);pose=torch.eye(3).repeat(2,2,15,1,1);beta=torch.ones(2,2,10,requires_grad=True);before=model.mano_left.shapedirs.clone()
  j,v=model._decode_mano(ori,pose,beta,torch.tensor([False,True]));self.assertTrue(torch.equal(j[:,:,0],torch.zeros_like(j[:,:,0])));self.assertTrue(torch.equal(j[0,0],j[1,0]));self.assertGreater(float((j[0,1]-j[1,1]).abs().max()),.001);self.assertTrue(torch.equal(before,model.mano_left.shapedirs));v.square().mean().backward();self.assertTrue(torch.isfinite(beta.grad).all())
 def test_invalid_mask_preserved(self):
  class D:
   def _absolute_frame_path(self,*a):return ''
   def _read_image_remote(self,p):return np.ones((8,8),np.uint8)*255
   def _source_to_canonical(self,x):return x[::-1].copy()
  f={'hands':{s:{'hand_valid':False} for s in ['left','right']}}
  a=annotations(D(),{},f,(8,8));self.assertEqual(a[0].sum(),128);self.assertFalse(a[-1].any())
 def test_all_invalid_mean_backward(self):
  v=torch.randn(2,3,requires_grad=True);z=_masked_mean(v,torch.zeros(2,dtype=torch.bool));z.backward();self.assertEqual(z.item(),0);self.assertTrue(torch.equal(v.grad,torch.zeros_like(v)))
 def test_masked_mean_counts(self):
  v=torch.tensor([1.,3.,100.],requires_grad=True);self.assertEqual(_masked_mean(v,torch.tensor([True,True,False])).item(),2.)
if __name__=='__main__':unittest.main()
