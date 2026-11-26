"""Explicit gradient averaging for training with independently rejected samples.

The caller must put *both* DDP forward and backward inside ``no_sync()``.
Backward the sum of accepted sample objectives locally, then call this helper
on every rank, including ranks with no accepted samples. DDP must not also
reduce these gradients. The returned count is the global accepted-sample count.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.distributed as dist


def gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Check existing gradients, with at most one host read per device."""
    checks: dict[torch.device, list[torch.Tensor]] = defaultdict(list)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            checks[values.device].append(torch.isfinite(values).all())
    return all(bool(torch.stack(values).all().item()) for values in checks.values())


def _control_device(parameters: list[torch.nn.Parameter]) -> torch.device:
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return next((p.device for p in parameters if p.is_cuda),
                    torch.device("cuda", torch.cuda.current_device()))
    return torch.device("cpu")


def sync_finite_sample_gradients(
    trainable: Iterable[torch.nn.Parameter],
    accepted_local_samples: int,
    *,
    bucket_bytes: int = 25 * 1024 * 1024,
) -> tuple[int, bool]:
    """Average local accepted-loss-sum gradients over all accepted samples.

    All ranks must pass parameters in the same order. A non-finite local
    gradient is reported collectively *before* any gradient bucket is reduced,
    so callers can retry samples individually without poisoning other ranks.
    Parameters unused everywhere retain ``grad=None``; this also prevents
    unwanted AdamW decay. Parameters used only elsewhere receive zero local
    contributions. Dense floating-point gradients are supported.

    Returns ``(global_count, global_finite)``. Only step the optimizer when
    ``global_count > 0 and global_finite``. A zero local count contributes no
    gradients, even if the caller retained finite dummy gradients. Post-reduce
    finiteness is checked too, guarding arithmetic overflow in the collective.
    """
    parameters = list(trainable)
    local_count = int(accepted_local_samples)
    if local_count < 0:
        raise ValueError("accepted_local_samples must be nonnegative")
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    control_device = _control_device(parameters)
    state = torch.tensor(
        [local_count, int(gradients_are_finite(parameters))],
        dtype=torch.int64, device=control_device,
    )
    if distributed:
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
    global_count, finite_ranks = (int(value) for value in state.tolist())
    if finite_ranks != world_size:
        return global_count, False
    if global_count == 0:
        for parameter in parameters:
            parameter.grad = None
        return 0, True

    presence = torch.tensor(
        [int(local_count > 0 and p.grad is not None) for p in parameters],
        dtype=torch.int32, device=control_device,
    )
    if distributed and presence.numel():
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    globally_used = presence.tolist()
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = defaultdict(list)
    for parameter, used in zip(parameters, globally_used):
        if not used:
            parameter.grad = None
            continue
        if local_count == 0 or parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter, memory_format=torch.contiguous_format)
        elif parameter.grad.layout != torch.strided:
            raise ValueError("sample gradient synchronization requires dense gradients")
        elif not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()
        groups[(parameter.grad.device, parameter.grad.dtype)].append(parameter.grad.view(-1))

    def flush(parts: list[torch.Tensor]) -> None:
        if not parts:
            return
        flat = torch.cat(parts)
        # Dividing before SUM avoids overflowing a finite global mean.
        flat.div_(global_count)
        if distributed:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        cursor = 0
        for destination in parts:
            destination.copy_(flat[cursor:cursor + destination.numel()])
            cursor += destination.numel()

    with torch.no_grad():
        for gradients in groups.values():
            capacity = max(1, int(bucket_bytes) // gradients[0].element_size())
            parts: list[torch.Tensor] = []
            filled = 0
            for gradient in gradients:
                offset = 0
                while offset < gradient.numel():
                    size = min(capacity - filled, gradient.numel() - offset)
                    parts.append(gradient[offset:offset + size])
                    offset += size
                    filled += size
                    if filled == capacity:
                        flush(parts)
                        parts, filled = [], 0
            flush(parts)

    finite = torch.tensor(int(gradients_are_finite(parameters)),
                          dtype=torch.int32, device=control_device)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return global_count, bool(finite.item())
