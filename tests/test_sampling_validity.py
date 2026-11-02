import copy, unittest
import numpy as np
from infinihand.training.data.datasets.doss_dataset_base import DossDatasetBase

class ValiditySamplingTests(unittest.TestCase):
    def setup_case(self, count, ratio, n=36):
        d=object.__new__(DossDatasetBase);d.min_hand_frame_fraction=ratio;d.hand_sampling_attempts=8
        d.SINGLE_IMAGE_MULTIVIEW=False;d.dataset_name='synthetic';d.view_sampler_cfg={'max_distance_between_views':64}
        s={'scene_name':'sample','frame_count':n,'metadata':{'frames':[{'frame_id':i*3+11,'timestamp_ns':1000000000+i*100000000,'hands':{'left':{'hand_valid':i<count},'right':{'hand_valid':False}}} for i in range(n)]}}
        return d,s
    def test_stage1_rejects_one_empty_frame(self):
        d,s=self.setup_case(3,1.,4)
        with self.assertRaises(ValueError):d._sample_valid_ids(s,4,np.arange(4),True)
    def test_stage1_accepts_single_hand_every_frame(self):
        d,s=self.setup_case(4,1.,4)
        np.testing.assert_array_equal(d._sample_valid_ids(s,4,np.arange(4),True),np.arange(4))
    def test_stage2_requires_ceil_70_percent(self):
        d,s=self.setup_case(25,.7)
        with self.assertRaises(ValueError):d._sample_valid_ids(s,36,np.arange(36),True)
        d,s=self.setup_case(26,.7)
        np.testing.assert_array_equal(d._sample_valid_ids(s,36,np.arange(36),True),np.arange(36))
    def test_original_timeline_untouched(self):
        d,s=self.setup_case(26,.7);before=copy.deepcopy(s)
        d._sample_valid_ids(s,36,np.arange(36),True)
        self.assertEqual(s,before)
    def test_evaluation_keeps_empty_frames(self):
        d,s=self.setup_case(0,0.,4)
        np.testing.assert_array_equal(d._sample_valid_ids(s,4,np.arange(4),True),np.arange(4))
    def test_rejects_invalid_window_not_just_total_scene(self):
        d,s=self.setup_case(30,1.,40)
        with self.assertRaises(ValueError):d._sample_valid_ids(s,4,np.array([0,1,2,39]),True)
if __name__=='__main__':unittest.main()
