import unittest

from infinihand.training.data.mano_basis_policy import (
    DATASET_BASES, POLICY_VERSION, basis_for, resolve_basis,
)


class BasisPolicyTests(unittest.TestCase):
    def test_every_approved_dataset_and_test_split(self):
        for dataset, basis in DATASET_BASES.items():
            for name in (dataset, dataset + '_test', dataset.replace('_', '-')):
                with self.subTest(dataset=name):
                    self.assertEqual(basis_for(name), basis)
                    self.assertEqual(resolve_basis(name, {
                        'mano_left_basis': basis, 'mano_basis_policy': POLICY_VERSION,
                    }), basis)

    def test_unmarked_A_data_is_not_relabelled(self):
        for name in ('arctic', 'dexycb', 'hoi4d', 'ho3d'):
            for task in ('mask', 'mano', 'joint', 'full'):
                with self.subTest(dataset=name, task=task):
                    metadata = {'schema_version': 'compact_virtual_camera_v10_20261012'}
                    with self.assertRaisesRegex(ValueError, 'Re-export'):
                        resolve_basis(name, metadata, task)
                    self.assertNotIn('mano_left_basis', metadata)

    def test_known_B_export_is_compatible_only_for_approved_B_dataset(self):
        for name in ('hot3d', 'egodex_test', 'h2o3d', 'egoverse',
                     'egopro', 'egopro_body', 'egostandard', 'egostandard_body'):
            self.assertEqual(resolve_basis(name, {
                'schema_version': 'compact_virtual_camera_v10_20261012',
            }), 'B')
        with self.assertRaises(ValueError):
            resolve_basis('hot3d', {'schema_version': 'unknown'})

    def test_conflicting_basis_or_unproven_migration_rejected(self):
        for dataset, expected in DATASET_BASES.items():
            with self.assertRaises(ValueError):
                resolve_basis(dataset, {'mano_left_basis': 'B' if expected == 'A' else 'A',
                                        'mano_basis_policy': POLICY_VERSION})
            with self.assertRaises(ValueError):
                resolve_basis(dataset, {'mano_left_basis': expected})

    def test_camera_only_does_not_require_unused_geometry(self):
        for dataset, expected in DATASET_BASES.items():
            self.assertEqual(resolve_basis(dataset, {}, 'camera'), expected)

    def test_unknown_dataset_has_no_default_basis(self):
        with self.assertRaisesRegex(ValueError, 'No approved MANO basis'):
            resolve_basis('unregistered_dataset', {'mano_left_basis': 'B'})


if __name__ == '__main__':
    unittest.main()
