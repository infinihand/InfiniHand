"""CPU-only regression coverage for partial-batch and empty-rank updates."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from infinihand.training.gradient_sync import (
    gradients_are_finite, sync_finite_sample_gradients,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.optional = torch.nn.Parameter(torch.ones(7, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor(5.0))

    def forward(self, inputs):
        return self.weight * inputs


def distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2)
    try:
        model = DDP(TinyModel(), find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        parameters = list(model.parameters())
        for mode in ("partial", "empty_rank"):
            with torch.no_grad():
                model.module.weight.fill_(2.0)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                if mode == "partial":
                    inputs = torch.tensor([1.0, float("nan")]) if rank == 0 else torch.tensor([3.0, 100.0])
                else:
                    inputs = torch.tensor([float("nan"), float("inf")]) if rank == 0 else torch.tensor([1.0, 2.0])
                # Discard the first graph; re-forward only accepted samples.
                with model.no_sync():
                    losses = model(inputs).square()
                    keep = torch.isfinite(losses) & (losses <= 50.0)
                    del losses
                with model.no_sync():
                    if keep.any():
                        model(inputs[keep]).square().sum().backward()
                count, finite = sync_finite_sample_gradients(parameters, int(keep.sum()), bucket_bytes=16)
                assert count == 2 and finite
                old = model.module.weight.item()
                expected_gradient = 2 * old * (10 if mode == "partial" else 5) / 2
                assert torch.allclose(model.module.weight.grad, torch.tensor(expected_gradient))
                assert model.module.optional.grad is None
                assert model.module.unused.grad is None
                optimizer.step()
                assert abs(model.module.weight.item() - (old - 0.01 * expected_gradient)) < 1e-6

        # Mixed dtype and tensors larger than the bucket; missing local grads
        # receive only the other rank's contribution.
        optimizer.zero_grad(set_to_none=True)
        model.module.optional.grad = torch.arange(1, 8, dtype=torch.float64) if rank == 0 else None
        count, finite = sync_finite_sample_gradients(parameters, 1, bucket_bytes=16)
        assert count == 2 and finite
        assert torch.equal(model.module.optional.grad, torch.arange(1, 8, dtype=torch.float64) / 2)
        assert model.module.weight.grad is None and model.module.unused.grad is None

        # A bad rank must not poison the finite rank's existing gradients.
        optimizer.zero_grad(set_to_none=True)
        model.module.weight.grad = torch.tensor(float("nan") if rank == 0 else 7.0)
        count, finite = sync_finite_sample_gradients(parameters, 1)
        assert count == 2 and not finite
        if rank == 1:
            assert model.module.weight.grad.item() == 7.0
        optimizer.zero_grad(set_to_none=True)
        count, finite = sync_finite_sample_gradients(parameters, 0)
        assert count == 0 and finite and all(p.grad is None for p in parameters)
    finally:
        dist.destroy_process_group()


class GradientSyncTests(unittest.TestCase):
    def test_single_process_and_finiteness(self):
        parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
        parameter.grad = torch.arange(8, dtype=torch.float64)
        count, finite = sync_finite_sample_gradients([parameter], 2, bucket_bytes=16)
        self.assertEqual(count, 2)
        self.assertTrue(finite)
        self.assertTrue(torch.equal(parameter.grad, torch.arange(8, dtype=torch.float64) / 2))
        parameter.grad[3] = float("inf")
        self.assertFalse(gradients_are_finite([parameter]))
        self.assertEqual(sync_finite_sample_gradients([parameter], 1), (1, False))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires CPU gloo")
    def test_two_rank_partial_and_empty_batches(self):
        with tempfile.TemporaryDirectory(prefix="infinihand-gradient-sync-") as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
