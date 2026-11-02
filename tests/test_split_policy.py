import json
from pathlib import Path
import tempfile
import unittest

from infinihand.training.data.split_policy import validate_training_manifest, validate_training_paths


class SplitPolicyTests(unittest.TestCase):
    def test_explicit_training_manifest(self):
        validate_training_manifest(dict(dataset_name='arctic', split='train', scenes=[]))

    def test_missing_and_held_out_splits(self):
        for split in (None, '', 'test', 'val', 'evaluation'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_training_manifest(dict(split=split))

    def test_conflicting_identifiers_and_scene_split(self):
        for extra in ({'sampling_name': 'arctic_test'}, {'raw_s3_root': '/data/val/arctic'},
                      {'scenes': [{'prefix': 's3://data/hot3d_test/scene'}]},
                      {'scenes': [{'split': 'test'}]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_training_manifest(dict(split='train', **extra))

    def test_neutral_filename_does_not_bypass_content_validation(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'data.json'
            p.write_text(json.dumps(dict(split='test')))
            with self.assertRaises(ValueError):
                validate_training_paths([p])

    def test_split_budget(self):
        with self.assertRaises(ValueError):
            validate_training_paths([], {'train': 1, 'test': 1})

    def test_all_training_recipes(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'configs/training').glob('*.yaml'):
            c = yaml.safe_load(path.read_text())
            with self.subTest(config=path.name):
                self.assertEqual(c['sampling_split_weights'], {'train': 1})
                names = {Path(p).stem for p in c['dataset_jsons']}
                self.assertTrue(names)
                self.assertTrue(all('/train/' in p for p in c['dataset_jsons']))
                self.assertEqual(names, set(c['sampling_weights'][c['dataset_sampling_version']]))
                self.assertTrue(all(p.startswith('/path/to/') for p in c['dataset_jsons']))

    def test_configuration_schema(self):
        from infinihand.training.engine import load_config
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'configs').rglob('*.yaml'):
            with self.subTest(config=str(path)):
                load_config(str(path))
