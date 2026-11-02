import unittest
import torch
from infinihand.inference_checkpoint import camera_head_state

class CameraCheckpointTests(unittest.TestCase):
 def test_head_only(self):
  v=torch.ones(2,3)
  result=camera_head_state({'trainable_state':{'backbone.model.camera_head.weight':v,'_mano_translation_convention':torch.tensor(1)},'step':135000})
  self.assertTrue(torch.equal(result['weight'],v))
 def test_reject_shared_backbone(self):
  with self.assertRaises(ValueError):
   camera_head_state({'trainable_state':{'backbone.model.camera_head.weight':torch.ones(1),'backbone.model.encoder.weight':torch.ones(1)}})
 def test_reject_nan(self):
  with self.assertRaises(ValueError):
   camera_head_state({'backbone.model.camera_head.weight':torch.tensor([float('nan')])})
if __name__=='__main__':unittest.main()
