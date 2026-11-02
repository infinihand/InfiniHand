import unittest
import torch
from infinihand.export_weights import merge

class ExportTests(unittest.TestCase):
    def test_partial_keeps_frozen_base(self):
        base = {"frozen": torch.ones(2), "head": torch.zeros(1)}
        merged = merge(base, [{"head": torch.ones(1)}])
        self.assertTrue(torch.equal(merged["frozen"], base["frozen"]))
        self.assertEqual(merged["head"].item(), 1)
        self.assertEqual(base["head"].item(), 0)
    def test_rejects_wrong_architecture(self):
        with self.assertRaises(ValueError):
            merge({"head": torch.zeros(1)}, [{"head": torch.zeros(2)}])
        with self.assertRaises(ValueError):
            merge({"head": torch.zeros(1)}, [{"other": torch.zeros(1)}])

class PublicWeightTests(unittest.TestCase):
    def model(self):
        m = torch.nn.Module()
        m.head = torch.nn.Linear(2, 1)
        m.mano_left = torch.nn.Module()
        m.mano_left.register_buffer('v_template', torch.ones(3, 3))
        return m

    def test_assets_remain_local_and_weights_are_complete(self):
        from infinihand.model_weights import without_mano_assets, load_inference_weights
        m = self.model()
        state = without_mano_assets(m.state_dict())
        self.assertNotIn('mano_left.v_template', state)
        load_inference_weights(m, state)
        self.assertTrue(torch.equal(m.mano_left.v_template, torch.ones(3, 3)))
        state.pop('head.weight')
        with self.assertRaises(RuntimeError):
            load_inference_weights(m, state)
