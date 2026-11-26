"""Per-sequence loss rejection with isolated replay of accepted samples."""
from collections import deque
from dataclasses import dataclass
import math
import time

import torch

from .gradient_sync import gradients_are_finite


class RollingLossThreshold:
    """History contains one globally observed finite raw mean per attempted step.

    Rejected finite observations stay in the window. Empty/nonfinite steps occupy
    a position (None); they never turn into zero observations. The current step
    is appended only after its acceptance decision.
    """
    def __init__(self, window=1000, bootstrap_steps=1000, bootstrap_threshold=10.0,
                 multiplier=1.0, state=None):
        self.window = int(window)
        self.bootstrap_steps = int(bootstrap_steps)
        self.bootstrap_threshold = float(bootstrap_threshold)
        self.multiplier = float(multiplier)
        self.history = deque(maxlen=max(1, self.window))
        if state and state.get("protocol") == "finite_raw_step_mean_v1":
            for value in state.get("step_means", []):
                self.history.append(float(value) if value is not None and math.isfinite(value) else None)

    @property
    def mean(self):
        values = [x for x in self.history if x is not None]
        return sum(values) / len(values) if values else None

    def threshold(self, step):
        if self.window <= 0 or self.multiplier <= 0:
            return math.inf, "disabled"
        if step <= self.bootstrap_steps or self.mean is None:
            return self.bootstrap_threshold, "absolute_bootstrap"
        return self.mean * self.multiplier, "recent_raw_mean"

    def observe(self, finite_sum, finite_count):
        value = float(finite_sum) / int(finite_count) if finite_count else None
        self.history.append(value if value is not None and math.isfinite(value) else None)

    def state_dict(self):
        return {"protocol": "finite_raw_step_mean_v1", "step_means": list(self.history)}


def slice_batch_tree(value, indices, batch_size, key=""):
    """Slice batch-axis tensors without slicing static image dimension tuples."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch_size and key not in {"image_hw", "crop_image_hw"}:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, dict):
        return {k: slice_batch_tree(v, indices, batch_size, k) for k, v in value.items()}
    # The model/loss use tensors. Preserve metadata lists unchanged, including
    # default_collate's transposed lists of source paths and frame identifiers.
    return value


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    return value


@dataclass
class FilteredStep:
    output: dict
    target: dict
    losses: dict
    raw_losses: torch.Tensor
    accepted_indices: torch.Tensor
    rejected_indices: torch.Tensor
    gradient_rejected_indices: list
    forward_loss_seconds: float
    backward_seconds: float
    replay_count: int
    objective_sum: float
    diagnostics_scope: str


def filtered_sample_backward(evaluate, batch_size, device, threshold, parameters):
    """evaluate(indices) returns (output, target, full per-sample losses).

    Normal batches do one forward/backward. Any rejected sample discards the
    entire graph and replays only accepted inputs. Merely masking a NaN loss is
    unsafe: a zero upstream derivative can still multiply NaN activations.
    Finite-loss/nonfinite-gradient cases are diagnosed with isolated per-sample
    replay, accumulating gradients from only finite-gradient samples.
    Gradients are SUMs; the caller synchronizes/divides by global accepted count.
    """
    parameters = list(parameters)
    indices = torch.arange(batch_size, device=device)
    active = indices
    raw_losses = None
    diagnostics = None
    forward_seconds = backward_seconds = 0.0
    replay_count = 0
    objective_sum = 0.0
    gradient_rejected = []
    diagnostics_scope = "accepted_subbatch"
    while active.numel():
        t0 = time.perf_counter()
        output, target, losses = evaluate(active)
        forward_seconds += time.perf_counter() - t0
        values = losses["objective_per_sample"]
        if values.shape != (active.numel(),):
            raise ValueError("criterion must return complete objective_per_sample [B]")
        if raw_losses is None:
            raw_losses = values.detach().float().clone()
        keep = torch.isfinite(values.detach()) & (values.detach() <= threshold)
        diagnostics = (detach_tree(output), target, detach_tree(losses))
        if not bool(keep.all()):
            active = active[keep]
            # Do not retain graph references while running the clean replay.
            del output, target, losses, values
            replay_count += int(active.numel() > 0)
            continue
        objective_sum = float(values.detach().sum())
        t0 = time.perf_counter()
        values.sum().backward()
        backward_seconds += time.perf_counter() - t0
        del output, target, losses, values
        if gradients_are_finite(parameters):
            break
        # A finite objective can have an undefined derivative. Isolate samples
        # rather than losing the remaining valid members of this batch.
        for parameter in parameters:
            parameter.grad = None
        diagnostics_scope = "last_accepted_sample_after_gradient_replay"
        accumulated = [None] * len(parameters)
        accepted = []
        objective_sum = 0.0
        for index in active.tolist():
            one = indices.new_tensor([index])
            t0 = time.perf_counter()
            output, target, losses = evaluate(one)
            forward_seconds += time.perf_counter() - t0
            replay_count += 1
            value = losses["objective_per_sample"][0]
            finite_loss = bool(torch.isfinite(value.detach()) & (value.detach() <= threshold))
            if finite_loss:
                t0 = time.perf_counter()
                value.backward()
                backward_seconds += time.perf_counter() - t0
            if finite_loss and gradients_are_finite(parameters):
                accepted.append(index)
                objective_sum += float(value.detach())
                diagnostics = (detach_tree(output), target, detach_tree(losses))
                for i, parameter in enumerate(parameters):
                    if parameter.grad is not None:
                        if accumulated[i] is None:
                            accumulated[i] = parameter.grad.detach()
                        else:
                            accumulated[i].add_(parameter.grad.detach())
            else:
                gradient_rejected.append(index)
            for parameter in parameters:
                parameter.grad = None
            del output, target, losses, value
        for parameter, gradient in zip(parameters, accumulated):
            parameter.grad = gradient
        active = indices.new_tensor(accepted)
        break

    output, target, losses = diagnostics
    losses.pop("objective_per_sample", None)
    losses.pop("_camera_hand_objective_graph", None)
    losses["objective"] = torch.tensor(objective_sum / max(1, active.numel()), device=device)
    accepted_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    accepted_mask[active] = True
    return FilteredStep(output, target, losses, raw_losses, active,
                        indices[~accepted_mask], gradient_rejected,
                        forward_seconds, backward_seconds, replay_count, objective_sum,
                        diagnostics_scope if active.numel() else "rejected_batch_diagnostics_only")
