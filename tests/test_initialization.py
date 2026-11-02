import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from infinihand.training.initialization import validate_initialization, save_initial_state
from infinihand.cli import parser


class InitializationTests(unittest.TestCase):
    def config(self, root):
        for name in ('lingbot.pt', 'wilor.ckpt'):
            (root / name).touch()
        return SimpleNamespace(finetune_checkpoint='', resume_checkpoint='',
                               require_finetune_checkpoint=False,
                               lingbotmap_ckpt=str(root / 'lingbot.pt'),
                               wilor_ckpt=str(root / 'wilor.ckpt'),
                               load_wilor_checkpoint=True)

    def test_fresh_initialization_needs_both_foundation_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp)); validate_initialization(cfg)
            cfg.lingbotmap_ckpt = ''
            with self.assertRaisesRegex(ValueError, 'LingBot-Map'):
                validate_initialization(cfg)
            cfg = self.config(Path(tmp)); cfg.load_wilor_checkpoint = False
            with self.assertRaisesRegex(ValueError, 'WiLoR'):
                validate_initialization(cfg)

    def test_explicit_joint_initialization_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = self.config(root)
            cfg.require_finetune_checkpoint = True
            with self.assertRaisesRegex(ValueError, 'training stage'):
                validate_initialization(cfg)
            base = root / 'current_model.pt'; base.touch()
            cfg.finetune_checkpoint = str(base); validate_initialization(cfg)
            cfg.resume_checkpoint = str(root / 'missing.pt')
            with self.assertRaises(FileNotFoundError): validate_initialization(cfg)
            cfg.resume_checkpoint = str(base); validate_initialization(cfg)

    def test_full_initial_state_is_exact_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(3, 2)
            path = save_initial_state(model, tmp)
            state = torch.load(path, weights_only=True)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(state[key], value))
            with self.assertRaises(FileExistsError): save_initial_state(model, tmp)

    def test_inference_checkpoint_is_explicit(self):
        checkpoint = next(a for a in parser()._actions if a.dest == 'checkpoint')
        self.assertTrue(checkpoint.required)
        self.assertIsNone(checkpoint.default)


if __name__ == '__main__':
    unittest.main()
