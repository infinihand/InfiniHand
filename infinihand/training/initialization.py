"""Explicit foundation-model initialization and complete bases for new runs."""
from pathlib import Path


def validate_initialization(cfg, *, finetune_checkpoint=None, resume_checkpoint=None):
    base = cfg.finetune_checkpoint if finetune_checkpoint is None else finetune_checkpoint
    resume = cfg.resume_checkpoint if resume_checkpoint is None else resume_checkpoint
    if cfg.require_finetune_checkpoint and not base:
        raise ValueError(
            "This training stage requires a complete checkpoint exported from the "
            "preceding component runs. Set finetune_checkpoint explicitly."
        )
    if resume and not base:
        raise ValueError("Resume requires the complete base recorded by its training run.")
    for label, value in (("finetune_checkpoint", base), ("resume_checkpoint", resume)):
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"{label} does not exist: {value}")
    if not base:
        if not cfg.lingbotmap_ckpt:
            raise ValueError("Fresh training requires an official LingBot-Map checkpoint.")
        if not cfg.load_wilor_checkpoint or not cfg.wilor_ckpt:
            raise ValueError("Fresh training requires an official WiLoR checkpoint.")
        for value in (cfg.lingbotmap_ckpt, cfg.wilor_ckpt):
            if not Path(value).is_file():
                raise FileNotFoundError(f"Foundation checkpoint does not exist: {value}")


def save_initial_state(model, output_dir):
    """Keep the complete initial state so partial checkpoints can be resumed/exported."""
    from .checkpoint_utils import atomic_torch_save
    path = Path(output_dir) / "initial_model.pt"
    if path.exists():
        raise FileExistsError(f"Use a fresh output directory: {path} already exists")
    state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    atomic_torch_save(state, path)
    return path
