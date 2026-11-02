import unittest
import numpy as np
import torch
from infinihand.models.lingbotmap_hand_v6 import expand_to_square_boxes,virtual_crop_camera_from_boxes,sample_perspective_features,HandFeatureProvider
from infinihand.models.mutual_hand_joint_fusion import uncanonicalize_crop_joints,decode_camera_translation_lsp
from infinihand.training.data.datasets.wilor_virtual_crop import prepare_wilor_virtual_crops
class Provider(HandFeatureProvider):
 def encode_crops(self,x):return x
class PerspectiveTests(unittest.TestCase):
 def setUp(self):
  self.hw=(378,518);self.K=torch.tensor([[[[400.,0,258.5],[0,410.,188.5],[0,0,1.]]]])
  self.box=torch.tensor([[[[.05,.1,.25,.4],[.65,.45,.9,.85]]]])
  self.exp=expand_to_square_boxes(self.box,self.hw,1.5)
  self.k,self.q=virtual_crop_camera_from_boxes(self.exp,self.K,self.hw)
  self.h=self.k@self.q@torch.linalg.inv(self.K)[:,:,None]
 def test_projection_and_roundtrip(self):
  torch.manual_seed(4);j=torch.randn(1,1,2,21,3)*.04;j[...,2]+=.7
  jc=torch.einsum('bnkij,bnkpj->bnkpi',self.q,j)
  back=jc@self.q
  self.assertLess((back-j).abs().max().item(),2e-7)
  a=(self.K[:,:,None,None]@j[...,None]).squeeze(-1);a=a/a[...,2:]
  b=(self.h[:,:,:,None]@a[...,None]).squeeze(-1);b=b[...,:2]/b[...,2:]
  c=(self.k[:,:,:,None]@jc[...,None]).squeeze(-1);c=c[...,:2]/c[...,2:]
  self.assertLess((b-c).abs().max().item(),1e-3)
  self.assertTrue(torch.equal(self.k[...,0,2],torch.full((1,1,2),95.5)))
 def test_feature_and_rgb_same_rays(self):
  yy,xx=torch.meshgrid(torch.linspace(0,1,378),torch.linspace(0,1,518),indexing='ij')
  rgb=torch.stack((xx,yy))[None].expand(2,-1,-1,-1)
  low=torch.nn.functional.interpolate(rgb,size=(54,74),mode='bilinear',align_corners=True)
  a=sample_perspective_features(rgb,self.h,self.hw,(256,192))
  b=sample_perspective_features(low,self.h,self.hw,(256,192))
  yy,xx=torch.meshgrid(torch.arange(256.),torch.arange(192.),indexing='ij')
  ray=torch.einsum('mij,hwj->mhwi',torch.linalg.inv(self.h.reshape(-1,3,3)),torch.stack((xx,yy,torch.ones_like(xx)),-1))
  uv=ray[...,:2]/ray[...,2:]
  interior=(uv[...,0]>20)&(uv[...,0]<498)&(uv[...,1]>20)&(uv[...,1]<358)
  self.assertLess((a-b).permute(0,2,3,1)[interior].abs().max().item(),1e-5)
 def test_gradient_and_flip(self):
  box=self.box.clone().requires_grad_();k,q=virtual_crop_camera_from_boxes(expand_to_square_boxes(box,self.hw,1.5),self.K,self.hw)
  H=k@q@torch.linalg.inv(self.K)[:,:,None]
  image=torch.rand(2,3,378,518)
  sample_perspective_features(image,H,self.hw,(256,192)).mean().backward()
  self.assertTrue(torch.isfinite(box.grad).all());self.assertGreater(box.grad.abs().sum().item(),0)
  points=torch.rand(1,1,2,21,2)
  self.assertTrue(torch.allclose(uncanonicalize_crop_joints(uncanonicalize_crop_joints(points)),points,atol=1e-7))
  points.zero_();self.assertAlmostEqual(uncanonicalize_crop_joints(points)[0,0,1,0,0].item()*192,191,places=4)
 def test_provider_square_trim(self):
  image=torch.rand(1,1,3,378,518);shift=torch.eye(3);shift[0,2]=32
  provider=Provider(crop_size=256,output_dim=3,output_hw=(256,256),chunk_size=2)
  out=provider(image,self.exp,shift@self.h).canonical_crops
  expected=sample_perspective_features(image.reshape(1,3,378,518).expand(2,-1,-1,-1),self.h,self.hw,(256,192)).reshape(1,1,2,3,256,192)
  self.assertTrue(torch.allclose(out[:,:,0,...,32:-32],expected[:,:,0],atol=5e-4))
  self.assertTrue(torch.allclose(out[:,:,1,...,32:-32].flip(-1),expected[:,:,1],atol=5e-4))
 def test_streaming_uses_perspective_geometry(self):
  from types import SimpleNamespace as NS
  from infinihand.models.stage2_streaming_inference_v6 import mano_forward
  captured={}
  class Adapter:
   output_hw=(1,1)
   def __call__(this,features,boxes,h,image_hw):
    captured['feature_h']=h
    return torch.zeros(2,1,1,1),torch.zeros(2,1,256,256)
  class Encoder:
   output_dim=1;output_hw=(1,1)
   def __call__(this,images,boxes,h):
    captured['rgb_h']=h
    return NS(features=torch.zeros(1,1,2,1,1,1),canonical_crops=torch.zeros(1,1,2,3,256,256))
  camera_joints=torch.ones(1,1,2,21,3)*.2
  camera_joints[...,2]=.8
  crop_joints=camera_joints@self.q.transpose(-1,-2)
  def head(output,*args):
   output.update(pred_keypoints_3d_crop_camera=crop_joints,pred_vertices_crop_camera=crop_joints,
    pred_mano_params={'global_orient':self.q[...,None,:,:]},
    pred_mano_transl_crop_camera=crop_joints[...,0,:],pred_wrist_transl_camera=crop_joints[...,0,:])
   return output
  model=NS(cfg=NS(image_height=378,image_width=518,lingbot_pixel_channels=1,wilor_crop_size=256,mano_context_dim=1),
   _sample_bbox_rescale=lambda boxes:1.5,lingbot_adapter=Adapter(),hand_feature_provider=Encoder(),
   feature_fusion=lambda a,b:a,_forward_mutual_joint_fusion=head)
  packet=NS(output={'pred_hand_boxes':self.box},gt_hand_boxes=None,gt_hand_valid=None,
   intrinsics=self.K,features=torch.zeros(1,1,1,1,1),images=torch.zeros(1,1,3,378,518))
  with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
   out=mano_forward(model,packet)
  self.assertTrue(torch.equal(captured['rgb_h'],captured['feature_h']))
  self.assertTrue(torch.allclose(out['mano_crop_homographies'],self.h,atol=1e-6))
  self.assertTrue(torch.allclose(out['pred_keypoints_3d_camera'],camera_joints,atol=1e-6))
  self.assertEqual(out['camera_output_space'],'image_camera')
if __name__=='__main__':unittest.main()
