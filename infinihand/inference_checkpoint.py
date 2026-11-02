"""Apply a trusted Camera-only training checkpoint to a complete inference model."""
from pathlib import Path
import hashlib
import torch

PREFIX = 'backbone.model.camera_head.'


def camera_head_state(payload):
    from .training.checkpoint_utils import extract_model_state
    state = extract_model_state(payload)
    head = {k[len(PREFIX):]:v for k,v in state.items() if k.startswith(PREFIX)}
    # Never silently replace the shared backbone, hand, or mask parameters.
    if not head or any(not k.startswith(PREFIX) and k != '_mano_translation_convention' for k in state):
        raise ValueError('Expected a camera-head-only checkpoint, not a full/mixed model')
    for k,v in head.items():
        if not torch.is_tensor(v) or not torch.isfinite(v).all():
            raise ValueError(f'Invalid camera weight: {k}')
    return head


def load_camera_checkpoint(model, path):
    path=Path(path)
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    head=camera_head_state(payload)
    model.backbone.model.camera_head.load_state_dict(head,strict=True)
    model.backbone.model.camera_head.float().eval()
    # The camera-only trained state is small enough to hash by streaming.
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(chunk)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest(),
                schedule_step=payload.get('schedule_step',payload.get('step')),
                tensors=len(head),scope=PREFIX,dtype='float32',ignored_metadata=['_mano_translation_convention'] if '_mano_translation_convention' in payload.get('trainable_state',{}) else [])
