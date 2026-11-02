import unittest,torch
from infinihand.training.losses import compute_temporal_smoothness_loss as loss,temporal_time_seconds
class TemporalTests(unittest.TestCase):
 def evaluate(self,t,x,valid=None):
  p=x.reshape(1,-1,1,1).expand(1,-1,2,3).clone().requires_grad_();r=torch.eye(3).expand(1,len(t),2,3,3);v=torch.ones(1,len(t),2,dtype=torch.bool) if valid is None else valid
  z=loss(p,r,t[None],v);z.backward();self.assertTrue(torch.isfinite(p.grad).all());return z.item()
 def test_matches_vidihand(self):
  t=torch.arange(5)*.1;x=torch.tensor([0.,1.,3.,6.,10.]);self.assertAlmostEqual(self.evaluate(t,x),float((x[2:]-2*x[1:-1]+x[:-2]).abs().mean()*3),places=4)
 def test_nonuniform_constant_velocity(self):
  t=torch.tensor([0.,.1,.4,.8]);self.assertAlmostEqual(self.evaluate(t,2*t),0.,places=5)
 def test_nonuniform_acceleration(self):
  t=torch.tensor([0.,.1,.4,.8]);self.assertAlmostEqual(self.evaluate(t,t*t),.06,places=5)
 def test_empty_valid(self):
  t=torch.arange(4)*.1;self.assertEqual(self.evaluate(t,t*t,torch.zeros(1,4,2,dtype=torch.bool)),0)
 def test_short(self):
  t=torch.arange(2)*.1;self.assertEqual(self.evaluate(t,t),0)
 def test_time_fallback_and_gaps(self):
  ids=torch.tensor([[0,2,5]]);a=temporal_time_seconds({'frame_ids':ids});self.assertTrue(torch.allclose(a,torch.tensor([[0.,.2,.5]],dtype=torch.double)))
if __name__=='__main__':unittest.main()
