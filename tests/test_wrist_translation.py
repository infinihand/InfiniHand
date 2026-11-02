import unittest
import torch
from torch import nn
from infinihand.models.lingbotmap_hand_v6 import LingBotMapHand
from infinihand.models.mutual_hand_joint_fusion import decode_camera_translation_lsp
class TinyModel(LingBotMapHand):
 def __init__(self):
  nn.Module.__init__(self)
  self.register_buffer('_mano_translation_convention',torch.tensor(1,dtype=torch.int64))
  self.mano_head=nn.Module();self.mano_head.deccam=nn.Linear(3,1)
class WristTests(unittest.TestCase):
 def test_checkpoint_convention_guard_and_roundtrip(self):
  m=TinyModel();fresh=m.mano_head.deccam.weight.detach().clone()
  unversioned={k:v for k,v in m.state_dict().items() if k!='_mano_translation_convention'}
  with self.assertRaisesRegex(RuntimeError,'wrist-depth convention'):m.load_state_dict(unversioned)
  mismatched=dict(m.state_dict());mismatched['_mano_translation_convention']=torch.tensor(2)
  with self.assertRaisesRegex(RuntimeError,'wrist-depth convention'):m.load_state_dict(mismatched)
  restored=TinyModel();restored.load_state_dict(m.state_dict())
  self.assertTrue(torch.equal(restored.mano_head.deccam.weight,fresh))
 def test_wrist_depth_lsp(self):
  torch.manual_seed(123)
  raw=torch.randn(2,21,3)*.015+torch.tensor([.07,-.03,.012])
  joints=raw-raw[:,:1]
  wrist=torch.tensor([[.03,.02,.6],[-.04,.01,.8]])
  k=torch.tensor([[430.,0,95.5],[0,430.,127.5],[0,0,1.]]).expand(2,-1,-1)
  xyz=joints+wrist[:,None];uv=xyz@k.transpose(-1,-2);uv=uv[...,:2]/uv[...,2:]/torch.tensor([192.,256.])
  logz=wrist[:,2].log().requires_grad_()
  recovered=decode_camera_translation_lsp(uv,joints,logz,k,(256,192))
  self.assertTrue(torch.allclose(recovered,wrist,atol=1e-6))
  self.assertTrue(torch.equal((joints+recovered[:,None])[:,0],recovered))
  recovered.sum().backward();self.assertTrue(torch.isfinite(logz.grad).all())
if __name__=='__main__':unittest.main()
