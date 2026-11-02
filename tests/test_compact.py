import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from infinihand.training.data.datasets.compact import geometry

class Source:
    def __init__(self):
        self.p = np.zeros((2,61),np.float32)
        self.p[:,58:61] = [[1,0,2],[0,1,2]]
        self.j = np.repeat(self.p[:,None,58:61],21,axis=1)
    def _absolute_frame_path(self, scene, frame, kind): return kind
    def _read_npz_remote(self, path):
        return {"params_61": self.p} if path == "camera_mano" else {"joints": self.j}

class CompactTests(unittest.TestCase):
    def test_world_transform_and_missing_camera(self):
        source=Source()
        frame={"hands":{"left":{"hand_valid":True},"right":{"hand_valid":False}}}
        cm,cj,wm,wj=geometry(source,{},frame)
        self.assertIsNone(wm)
        self.assertIsNone(wj)
        self.assertEqual(cm["mano_valid"].tolist(),[True,False])
        C=np.eye(4); C[:3,:3]=Rotation.from_euler("z",90,degrees=True).as_matrix(); C[:3,3]=[0,0,1]
        frame.update(camera_valid=True,camera_to_world=C.tolist())
        cm,cj,wm,wj=geometry(source,{},frame)
        np.testing.assert_allclose(wm["params_61"][:,58:61],[[0,1,3],[0,0,0]],atol=1e-6)
        np.testing.assert_allclose(wm["params_61"][:,3:58],source.p[:,3:58])
        np.testing.assert_allclose(wj["joints"][:,0],wm["params_61"][:,58:61],atol=1e-6)
        self.assertFalse(wj["joints_valid"][1].any())
