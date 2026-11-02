import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from infinihand.training.data.frame_sampling import allocate_budget, manifest_sampling_plan
from infinihand.training.data.datasets.doss_dataset_base import DossDatasetBase
from infinihand.training.data.hand_dataloader import resolve_dataset_manifest


class FrameSamplingTests(unittest.TestCase):
    def test_manifest_paths_support_repo_relative_and_absolute_sources(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(resolve_dataset_manifest('configs/training/stage1_mask.yaml', 'infinihand/training'),
                         str(repo / 'configs/training/stage1_mask.yaml'))
        self.assertEqual(resolve_dataset_manifest('/external/manifest.json', 'ignored'), '/external/manifest.json')
        self.assertEqual(resolve_dataset_manifest('missing.json', '/legacy'), '/legacy/missing.json')

    def manifest(self, root, name, split, counts):
        path = Path(root) / (name + '.json')
        path.write_text(json.dumps(dict(dataset_name=name, sampling_name=name, split=split,
                                        scenes=[dict(name=f's{i}', frame_count=n) for i, n in enumerate(counts)])))
        return path

    def test_sqrt_weights_and_boost_after_short_scene_filter(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.manifest(root, 'arctic', 'train', [25, 100])
            b = self.manifest(root, 'other', 'train', [400])
            weights, lengths, stats = manifest_sampling_plan([a, b], 36, dict(arctic=2, other=1), 100)
        self.assertEqual(weights, dict(arctic=.5, other=.5))
        self.assertEqual(lengths, dict(arctic=100, other=100))
        self.assertEqual(stats['arctic']['eligible_frames'], 100)
        self.assertEqual(stats['arctic']['total_frames'], 125)

    def test_training_rejects_held_out_manifests(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ('test', 'val', '', 'evaluation'):
                path = self.manifest(root, 'arctic', split, [100])
                with self.assertRaisesRegex(ValueError, 'Training requires'):
                    manifest_sampling_plan([path], 4, dict(arctic=1), 100)

    def test_updated_manifest_frames_recompute_weights(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.manifest(root, 'a', 'train', [100])
            b = self.manifest(root, 'b', 'train', [100])
            before = manifest_sampling_plan([a, b], 4, dict(a=1, b=1), 100)
            self.manifest(root, 'b', 'train', [900])
            after = manifest_sampling_plan([a, b], 4, dict(a=1, b=1), 100)
        self.assertEqual(before[0], dict(a=.5, b=.5))
        self.assertEqual(after[0], dict(a=.25, b=.75))
        self.assertNotEqual(before[2]['b']['manifest_sha256'], after[2]['b']['manifest_sha256'])

    def test_empty_or_ambiguous_pool_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.manifest(root, 'a', 'train', [20])
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                manifest_sampling_plan([a], 36, dict(a=1), 10)
            with self.assertRaisesRegex(ValueError, 'Split budget'):
                manifest_sampling_plan([a], 4, dict(a=1), 10, dict(train=4, test=1))

    def test_allocation_keeps_small_datasets_and_exact_budget(self):
        result = allocate_budget(dict(a=1e-9, b=10, c=3), 100)
        self.assertEqual(sum(result.values()), 100)
        self.assertTrue(all(v >= 1 for v in result.values()))

    def test_full_pool_not_limited_by_virtual_length_and_changes_each_draw(self):
        d = object.__new__(DossDatasetBase)
        d.random_full_scene_pool = True; d.num_scenes = 300000; d.len_train = 10000
        with patch('numpy.random.randint', side_effect=[299999, 200001, 30000]):
            self.assertEqual(d._scene_candidate_indices(0, 1), [299999])
            self.assertEqual(d._scene_candidate_indices(0, 1), [200001])
            self.assertEqual(d._scene_candidate_indices(9999, 1), [30000])
        np.random.seed(31)
        a = d._scene_candidate_indices(0, 8)
        np.random.seed(31)
        self.assertEqual(a, d._scene_candidate_indices(0, 8))
        self.assertEqual(len(set(a)), 8)

    def test_uniform_scene_draws_cover_all_filtered_scenes(self):
        d = object.__new__(DossDatasetBase)
        d.random_full_scene_pool = True; d.num_scenes = 20; d.len_train = 3
        np.random.seed(12)
        counts = np.bincount([d._scene_candidate_indices(0, 1)[0] for _ in range(20000)], minlength=20)
        self.assertTrue((counts > 850).all() and (counts < 1150).all())

    def test_explicit_scene_and_evaluation_keep_index_semantics(self):
        d = object.__new__(DossDatasetBase)
        d.random_full_scene_pool = False; d.num_scenes = 5
        self.assertEqual(d._scene_candidate_indices(4, 3), [4, 0, 1])
        d.random_full_scene_pool = True
        d.dataset_name = 'test'; d.dataset_retry_attempts = 8
        d.scenes = {'s4': {'scene_name': 's4', 'frame_count': 4}}
        d._ensure_scene_loaded = lambda s: s
        d.get_min_required_scene_frames = lambda: 4
        d._sample_valid_ids = lambda s, n, ids, explicit: np.arange(4)
        d._get_doss_data = lambda s, ids: dict(seq_name=s['scene_name'])
        d._sample_has_required_files = lambda out: (True, [])
        with patch('numpy.random.randint', side_effect=AssertionError('Explicit scene must not draw')):
            self.assertEqual(d.get_data(seq_name='s4', img_per_seq=4)['seq_name'], 's4')


if __name__ == '__main__':
    unittest.main()
