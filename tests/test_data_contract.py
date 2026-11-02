import unittest
import numpy as np
from infinihand.training.data.datasets.compact import geometry, validate_metadata, annotations
from infinihand.training.data.datasets.doss_dataset_base import DossDatasetBase
from infinihand.training.data.composed_dataset import ComposedDataset, OPTIONAL_WORLD_FIELDS, OPTIONAL_CAMERA_TRAJECTORY_FIELDS

class Source:
    def __init__(self):
        self.p=np.zeros((2,61),np.float32);self.p[:,58:]=[0,0,1]
        self.j=np.repeat(self.p[:,None,58:],21,axis=1)
    def _absolute_frame_path(self, scene, frame, kind, side=None):return kind
    def _read_npz_remote(self,path):return {'params_61':self.p} if path=='camera_mano' else {'joints':self.j}
    def _read_image_remote(self,path):return np.zeros((8,8,3),np.uint8)
    def _source_to_canonical(self,x):return x[[1,0]]

def frame():
    return {'image_size_hw':[8,8],'intrinsics':[[10,0,4],[0,10,4],[0,0,1]],'camera_valid':False,'hands':{s:{'hand_valid':s=='left'} for s in ('left','right')}}

class ContractTests(unittest.TestCase):
    def test_invalid_nan_is_masked(self):
        s=Source();s.p[1]=np.nan;s.j[1]=np.nan
        cm,cj,_,_=geometry(s,{},frame())
        self.assertTrue(np.isfinite(cm['params_61']).all())
        self.assertFalse(cj['joints_valid'][1].any())
    def test_valid_nan_rejected(self):
        s=Source();s.p[0]=np.nan
        with self.assertRaises(ValueError):geometry(s,{},frame())
    def test_wrong_root_convention_rejected(self):
        s=Source();s.p[0,-1]=2
        with self.assertRaises(ValueError):geometry(s,{},frame())
    def test_empty_positive_mask_rejected(self):
        with self.assertRaises(ValueError):annotations(Source(),{},frame(),(8,8))
    def test_camera_valid_requires_real_pose(self):
        f=frame();f['camera_valid']=True
        m={'schema_version':'compact_virtual_camera_v10_20261012','units':{'3d':'meters'},'side_order':['left','right'],'frames':[f],'frame_count':1}
        with self.assertRaises(ValueError):validate_metadata(m,'scene')
    def test_optional_validity_preserved(self):
        b={'ids':np.arange(2),'world_geometry_valid':np.array([True,False]),'camera_trajectory_valid':np.array([True,False])}
        for k,shape in OPTIONAL_WORLD_FIELDS.items():b[k]=np.zeros((2,*shape),np.float32)
        for k,shape in OPTIONAL_CAMERA_TRAJECTORY_FIELDS.items():b[k]=np.zeros((2,*shape),np.float32)
        x=ComposedDataset._normalize_optional_fields(b)
        self.assertEqual(x['world_geometry_valid'].tolist(),[True,False])
        self.assertEqual(x['camera_trajectory_valid'].tolist(),[True,False])
    def test_missing_world_not_supervised(self):
        x=ComposedDataset._normalize_optional_fields({'ids':np.arange(2)})
        self.assertFalse(x['world_geometry_valid'].any())
        self.assertFalse(x['camera_trajectory_valid'].any())
    def test_explicit_frame_indices_not_clipped(self):
        d=object.__new__(DossDatasetBase);d.SINGLE_IMAGE_MULTIVIEW=False
        with self.assertRaises(ValueError):d._sample_ids('s',10,2,[0,15],True)
    def test_impossible_temporal_span_fails_clearly(self):
        d=object.__new__(DossDatasetBase);d.SINGLE_IMAGE_MULTIVIEW=False;d.view_sampler_cfg={'max_distance_between_views':8}
        with self.assertRaisesRegex(ValueError,'cannot contain'):d._sample_ids('s',100,36,None,False)
