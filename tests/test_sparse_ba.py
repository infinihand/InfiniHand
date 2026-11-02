import unittest
import numpy as np
from infinihand.sparse_ba import BinaryKeyframePool, SparseBinaryBA, apply_pose_corrections

class PoolTests(unittest.TestCase):
 def test_toggle(self):
  p=BinaryKeyframePool(4,2)
  for i in range(4):self.assertFalse(p.append(i))
  self.assertTrue(p.append(4));self.assertEqual(p.selected(),[0,2,4]);p.commit([0,2,4])
  self.assertEqual(p.state(),[[0,1],[2,1],[4,1]])
  p.append(5);p.append(6);self.assertEqual(p.selected(),[0,4,6]);p.commit([0,4,6])
  self.assertEqual(p.state(),[[2,1],[6,1]])
 def test_failure_does_not_evict(self):
  def fail(ids,seeds):raise RuntimeError('optimizer failure')
  c=SparseBinaryBA(fail,2,1)
  c.append(np.eye(4));c.append(np.eye(4))
  with self.assertRaises(RuntimeError):c.append(np.eye(4))
  self.assertEqual(c.pool.state(),[[0,0],[1,0],[2,0]])
 def test_long_identity_and_retirement(self):
  seen=[]
  def opt(ids,seeds):seen.append(ids);return seeds,{}
  c=SparseBinaryBA(opt,8,2)
  raw=np.tile(np.eye(4),(100,1,1));raw[:,0,3]=np.arange(100)*.1+3
  for pose in raw:c.append(pose)
  np.testing.assert_allclose(c.finish(),raw,atol=1e-12)
  self.assertGreater(len(seen),10);self.assertEqual(seen[0],[0,2,4,6,8]);self.assertIn(0,seen[1]);self.assertNotIn(0,seen[2])
  self.assertTrue(all(len(e['pool_after'])<=8 for e in c.events))
 def test_preserve_nonidentity_world_gauge(self):
  def opt(ids,seeds):
   # Simulate an optimizer that fixes its own first camera at identity.
   return np.linalg.inv(seeds[0])@seeds,{}
  c=SparseBinaryBA(opt,4,2);raw=np.tile(np.eye(4),(20,1,1));raw[:,0,3]=10+np.arange(20)
  for p in raw:c.append(p)
  np.testing.assert_allclose(c.finish(),raw,atol=1e-12)
 def test_interpolation_and_world_points(self):
  raw=np.tile(np.eye(4),(5,1,1));keys=raw[[0,4]].copy();keys[1,0,3]=2
  result=apply_pose_corrections(raw,[0,4],keys)
  np.testing.assert_allclose(result[:,0,3],np.linspace(0,2,5))
  point=np.array([.1,.2,.3,1]);world=result@point
  back=np.einsum('tij,tj->ti',np.linalg.inv(result),world)
  np.testing.assert_allclose(back,np.tile(point,(5,1)))
 def test_invalid_capacity_and_duplicate(self):
  with self.assertRaises(ValueError):BinaryKeyframePool(3,4)
  p=BinaryKeyframePool(4,2);p.append(0)
  with self.assertRaises(ValueError):p.append(0)
 def test_stride_one(self):
  c=SparseBinaryBA(lambda ids,p:(p,{}),1,1)
  for _ in range(10):c.append(np.eye(4))
  self.assertEqual(len(c.finish()),10)
 def test_short_video_flush(self):
  c=SparseBinaryBA(lambda ids,p:(p,{}),160,20)
  for i in range(10):c.append(np.eye(4))
  self.assertEqual(len(c.finish()),10);self.assertEqual(c.events,[])

if __name__=='__main__':unittest.main()
