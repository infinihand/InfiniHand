import tempfile
from pathlib import Path
import unittest
import torch
from infinihand.training.engine import TrainConfig, save_checkpoint
from infinihand.training.checkpoint_utils import extract_model_state
from infinihand.training.sample_filter import RollingLossThreshold


class FilterCheckpointTest(unittest.TestCase):
    def test_partial_checkpoint_keeps_geometry_and_loss_history(self):
        model = torch.nn.Module()
        model.add_module('hand_mask_head', torch.nn.Linear(2, 2))
        model.register_buffer('_mano_translation_convention', torch.tensor(1))
        optimizer = torch.optim.AdamW(model.parameters())
        window = RollingLossThreshold(1000, 2)
        window.observe(4., 2)
        window.observe(100., 1)
        window.observe(0., 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'last.pt'
            save_checkpoint(path, model, optimizer, 3, TrainConfig(), [],
                            extra={'loss_filter_state': window.state_dict()})
            saved = torch.load(path, weights_only=False)
        self.assertEqual(int(extract_model_state(saved)['_mano_translation_convention']), 1)
        restored = RollingLossThreshold(1000, 2, state=saved['loss_filter_state'])
        self.assertEqual(restored.threshold(4)[0], 51.)
        self.assertEqual(list(restored.history), [2., 100., None])

if __name__ == '__main__': unittest.main()
