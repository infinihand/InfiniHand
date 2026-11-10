"""Scoped inference acceleration switches, restored after streams complete."""
from contextlib import contextmanager, ExitStack


@contextmanager
def inference_optimizations(model, inference_module, *, fused_rope=False,
                            compact_cache=False, mask_backend='torch'):
    if model.training:raise ValueError('Runtime optimizations require eval mode')
    if mask_backend not in ('torch','triton'):raise ValueError('Unknown mask backend')
    previous=inference_module.tight_mask_boxes
    with ExitStack() as stack:
        if fused_rope:
            from .rope import fused_rope as context
            stack.enter_context(context(model))
        if compact_cache:
            from .cache import compact_kv_cache
            stack.enter_context(compact_kv_cache(model))
        if mask_backend=='triton':
            from .mask_boxes import tight_mask_boxes_triton
            inference_module.tight_mask_boxes=tight_mask_boxes_triton
        try:
            yield dict(fused_rope=fused_rope,compact_kv_cache=compact_cache,
                       mask_box_backend=mask_backend)
        finally:
            inference_module.tight_mask_boxes=previous
