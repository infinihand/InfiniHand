"""Collect auxiliary tensors without changing geometry or precision."""
import torch


class AuxiliaryCapture:
    def __init__(self, storage, frames, anchor_frames=4):
        if storage not in ('cpu', 'cpu_async', 'cuda'):
            raise ValueError('storage must be cpu, cpu_async or cuda')
        self.storage = storage
        self.frames = frames
        self.anchor_frames = anchor_frames
        self.buffers = {}

    def __call__(self, key, value, order):
        """Read CPU results only after the owning CUDA stream has completed."""
        value = value.detach()
        if self.storage == 'cuda':
            return value
        if self.storage == 'cpu':
            return value.cpu()
        if key not in self.buffers:
            self.buffers[key] = torch.empty(
                (self.frames, *value.shape[1:]), dtype=value.dtype,
                device='cpu', pin_memory=True)
        first = 0 if order == 0 else self.anchor_frames + order - 1
        target = self.buffers[key][first:first + len(value)]
        target.copy_(value, non_blocking=True)
        return target
