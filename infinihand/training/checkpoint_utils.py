from __future__ import annotations

import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn


def load_checkpoint_payload(path: str | Path) -> Any:
    """Load a checkpoint on CPU while remaining compatible with older torch."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(payload: Any) -> Mapping[str, torch.Tensor]:
    """Return model state from a full checkpoint or a model-only checkpoint."""
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("checkpoint payload is not a non-empty mapping")
    for key in ("trainable_state", "state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate
    if all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise RuntimeError("checkpoint has no recognizable model state")


def infer_checkpoint_step(payload: Any, configured_step: int = -1) -> int:
    """Infer the LR-schedule step represented by a checkpoint.

    New checkpoints store ``schedule_step`` explicitly.  Older fine-tune
    checkpoints stored the global step in ``step`` and the local schedule step
    in one of the window fields, so those fields take precedence over ``step``.
    A plain model state has no metadata and therefore requires an explicit
    ``resume_checkpoint_step`` in the YAML.
    """
    if isinstance(payload, Mapping):
        for key in (
            "schedule_step",
            "checkpoint_step",
            "checkpoint_window_finetune_step",
            "best_window_finetune_step",
            "finetune_steps",
        ):
            value = payload.get(key)
            if value is not None and int(value) >= 0:
                return int(value)
        step = payload.get("step")
        if step is not None:
            config = payload.get("config")
            if isinstance(config, Mapping):
                offset = int(config.get("finetune_step_offset", 0) or 0)
                if offset > 0 and int(step) >= offset:
                    return int(step) - offset
            return int(step)
    if int(configured_step) >= 0:
        return int(configured_step)
    raise RuntimeError(
        "cannot infer checkpoint_step from model-only checkpoint; set "
        "resume_checkpoint_step explicitly"
    )


def infer_finetune_step_offset(payload: Any, checkpoint_step: int) -> int:
    """Recover the global-step display offset without changing the LR clock."""
    if not isinstance(payload, Mapping):
        return 0
    config = payload.get("config")
    if isinstance(config, Mapping) and "finetune_step_offset" in config:
        configured = int(config.get("finetune_step_offset", 0) or 0)
        if configured >= 0:
            return configured
    global_step = payload.get("global_step", payload.get("step"))
    if global_step is not None and int(global_step) >= int(checkpoint_step):
        return int(global_step) - int(checkpoint_step)
    return 0


def validate_resume_schedule(
    payload: Any,
    *,
    train_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
    weight_decay: float,
) -> None:
    """Reject accidental schedule changes when a full checkpoint has metadata."""
    if not isinstance(payload, Mapping):
        return
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return
    expected = {
        "train_steps": int(train_steps),
        "warmup_steps": int(warmup_steps),
        "lr": float(peak_lr),
        "min_lr": float(min_lr),
        "weight_decay": float(weight_decay),
    }
    mismatches = []
    for key, current in expected.items():
        if key not in config:
            continue
        previous = config[key]
        same = (
            int(previous) == current
            if key in ("train_steps", "warmup_steps")
            else math.isclose(float(previous), current, rel_tol=1e-12, abs_tol=0.0)
        )
        if not same:
            mismatches.append(f"{key}: checkpoint={previous!r}, current={current!r}")
    if mismatches:
        raise RuntimeError(
            "resume schedule/config mismatch; continuing would not preserve the "
            "original LR/optimizer schedule: " + "; ".join(mismatches)
        )


def accepted_objective_history(
    payload: Any,
    rank: int,
    window: int,
) -> Tuple[list[float], str]:
    """Restore a rank's recent accepted objectives used by outlier filtering."""
    size = max(1, int(window))
    if isinstance(payload, Mapping):
        by_rank = payload.get("recent_accepted_objectives_by_rank")
        if isinstance(by_rank, Sequence) and 0 <= int(rank) < len(by_rank):
            values = by_rank[int(rank)]
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                clean = [float(value) for value in values if value is not None]
                return clean[-size:], "checkpoint_rank_history"

        values = payload.get("recent_accepted_objectives")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            clean = [float(value) for value in values if value is not None]
            return clean[-size:], "checkpoint_history"

        history = payload.get("history")
        if isinstance(history, Sequence):
            clean = []
            for record in history:
                if not isinstance(record, Mapping):
                    continue
                if bool(record.get("outlier_skipped", False)):
                    continue
                objective = record.get("objective")
                if objective is not None:
                    clean.append(float(objective))
            return clean[-size:], "history_fallback"
    return [], "empty"


def restore_optimizer_state(optimizer: torch.optim.Optimizer, payload: Any) -> bool:
    """Restore optimizer state when present; return False for model-only files."""
    if not isinstance(payload, Mapping):
        return False
    state = None
    for key in ("optimizer", "optimizer_state_dict", "optimizer_state"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            state = candidate
            break
    if state is None:
        return False
    optimizer.load_state_dict(state)
    return True


def compute_cosine_learning_rate(
    step: int,
    train_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Compute LR at an absolute schedule step (1-based)."""
    total = int(train_steps)
    if total <= 0:
        raise ValueError(f"train_steps must be positive, got {train_steps}")
    warmup = max(0, min(int(warmup_steps), total))
    if warmup > 0 and int(step) <= warmup:
        progress = 1.0 if warmup == 1 else min(max((int(step) - 1) / (warmup - 1), 0.0), 1.0)
        return float(min_lr) + 0.5 * (float(peak_lr) - float(min_lr)) * (
            1.0 - math.cos(math.pi * progress)
        )
    decay_steps = max(total - warmup, 1)
    progress = min(max((int(step) - warmup) / decay_steps, 0.0), 1.0)
    return float(min_lr) + 0.5 * (float(peak_lr) - float(min_lr)) * (
        1.0 + math.cos(math.pi * progress)
    )


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write fully to a sibling temp file, then atomically replace the target."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy_file(source: str | Path, destination: str | Path) -> None:
    """Copy to a sibling temp file before atomically publishing it."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.parent / (
        f".{destination_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        shutil.copyfile(source_path, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_trainable_module_state(
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Collect complete state for every top-level module with trainable parameters.

    Saving the complete top-level module state also preserves buffers that can change
    during training, while avoiding frozen top-level modules such as the image backbone.
    """
    raw_model = model.module if hasattr(model, "module") else model
    trainable_parameter_names = {
        name for name, parameter in raw_model.named_parameters() if parameter.requires_grad
    }
    if not trainable_parameter_names:
        raise RuntimeError("Cannot checkpoint a model with no trainable parameters")

    trainable_module_roots = tuple(
        sorted({name.split(".", 1)[0] for name in trainable_parameter_names})
    )
    state = {
        key: value.detach().cpu()
        for key, value in raw_model.state_dict().items()
        if key.split(".", 1)[0] in trainable_module_roots
    }
    missing = sorted(trainable_parameter_names.difference(state))
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"Checkpoint is missing trainable parameters: {preview}")
    return state, trainable_module_roots
