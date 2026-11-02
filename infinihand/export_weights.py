"""Compose partial training checkpoints into a weight-only inference state dict."""
import argparse
from pathlib import Path
import torch
from .training.checkpoint_utils import extract_model_state

def merge(base, updates):
    result = dict(base)
    for update in updates:
        if not update:
            raise ValueError("Empty checkpoint")
        for name, value in update.items():
            if name not in result or not isinstance(value, torch.Tensor):
                raise ValueError(f"Unknown/non-tensor weight: {name}")
            if value.shape != result[name].shape:
                raise ValueError(f"Shape mismatch: {name}")
        result.update(update)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Overlay in order; use only trusted training checkpoints")
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-mano-assets", action="store_true",
                   help="Omit licensed MANO model buffers from a public inference export")
    a = p.parse_args()
    dst = Path(a.output)
    if dst.exists():
        raise FileExistsError(dst)
    base = extract_model_state(torch.load(a.base, map_location="cpu", weights_only=False))
    for source in a.checkpoint:
        update = extract_model_state(torch.load(source, map_location="cpu", weights_only=False))
        base = merge(base, [update])
    if a.exclude_mano_assets:
        from .model_weights import without_mano_assets
        base = without_mano_assets(base)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    torch.save(base, tmp)
    tmp.rename(dst)
    print(f"Exported {len(base)} tensors to {dst}; optimizer and step metadata omitted")

if __name__ == "__main__":
    main()
