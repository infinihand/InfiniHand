import torch,unittest
from infinihand.training.losses import LingBotMapHandLoss
class MaskValidityTests(unittest.TestCase):
 def run_case(self,pixels,supervise):
  logits=torch.zeros(1,4,2,8,8,requires_grad=True)
  o={'pred_hand_probability':torch.full((1,4,2),.5),'pred_hand_mask_logits':logits}
  t={'hand_valid':torch.zeros(1,4,2,dtype=torch.bool),'hand_masks':torch.full_like(logits,pixels),'hand_mask_valid':torch.full((1,4,2),supervise,dtype=torch.bool)}
  c=LingBotMapHandLoss(bbox_loss_enabled=False,hand_probability=0.,hand_bbox_smooth_l1=0.,hand_bbox_giou=0.,joints2d_direct_weight=0.,reproj_loss_weight=0.,camera_hand_loss_enabled=False)
  loss=c(o,t)['objective'];loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());return loss.item(),logits.grad
 def test_confirmed_empty(self):
  z,g=self.run_case(0.,True);self.assertGreater(z,0);self.assertTrue((g>0).all())
 def test_invalid_geometry_but_projected_mask(self):
  z,g=self.run_case(1.,True);self.assertGreater(z,0);self.assertTrue((g<0).all())
 def test_missing_annotation_not_negative(self):
  z,g=self.run_case(0.,False);self.assertEqual(z,0);self.assertEqual(g.abs().sum(),0)
if __name__=='__main__':unittest.main()
