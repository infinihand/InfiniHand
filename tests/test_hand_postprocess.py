"""Numerical contracts for offline hand refinement (no assets or GPU required)."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from infinihand.postprocess import sg5_both, select_depth_path, segments
from infinihand.stabilize_shape import estimate_fixed_betas


class HandPostprocessTests(unittest.TestCase):
    def test_irregular_quadratic_and_rotations(self):
        t = np.array([0., .1, .2, .27, .4, .5, .6])
        x = np.broadcast_to((2*t*t+t)[:,None,None], (7,2,3)).copy()
        r = Rotation.from_rotvec(t[:,None]*np.array([[.1,.2,.3]])).as_matrix()
        r = np.broadcast_to(r[:,None,None],(7,2,16,3,3)).copy()
        a,b = sg5_both(x,r,np.ones((7,2),bool),t)
        np.testing.assert_allclose(a,x,atol=1e-12)
        np.testing.assert_allclose(b,r,atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(b),1,atol=1e-12)

    def test_missing_hands_and_gaps(self):
        t = np.r_[np.arange(5)*.1,2+np.arange(5)*.1]
        valid = np.ones((10,2),bool);valid[2,0]=False
        self.assertEqual(list(segments(valid[:,0],t)),[(0,2),(3,5),(5,10)])
        x = np.zeros((10,2,3));x[2,0]=999
        r = np.broadcast_to(np.eye(3),(10,2,16,3,3)).copy()
        a,b = sg5_both(x,r,valid,t)
        np.testing.assert_array_equal(a[~valid],x[~valid])
        np.testing.assert_array_equal(b[~valid],r[~valid])
        self.assertEqual(np.max(a[valid]),0)

    def test_per_hand_robust_beta(self):
        b = np.zeros((8,2,10));b[:,1]=2;b[-1,0]=100
        fixed,_,_,report = estimate_fixed_betas(b,np.ones((8,2),bool))
        np.testing.assert_array_equal(fixed[:,0],0)
        np.testing.assert_array_equal(fixed[:,1],2)
        self.assertEqual(report[0]['rejected_frames'],1)

    def test_depth_dp_resets_and_excludes_infeasible(self):
        scores = np.zeros((5,2,3));scores[:,:,1]=1
        feasible = np.ones_like(scores,dtype=bool);feasible[2]=False
        path = select_depth_path(scores,feasible,np.ones((5,2),bool),np.arange(5)/10,np.array([.3,.4,.5]))
        np.testing.assert_array_equal(path[[0,1,3,4]],1)
        np.testing.assert_array_equal(path[2],-1)


if __name__ == '__main__':
    unittest.main()
