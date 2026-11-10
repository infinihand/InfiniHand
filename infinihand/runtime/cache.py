"""Scoped, opt-in removal of redundant read clones in the existing SDPA cache."""
from contextlib import contextmanager


@contextmanager
def readonly_kv_cache(model):
    if model.training:
        raise ValueError('Read-only KV optimization is inference-only')
    modules=[m for m in model.backbone.model.aggregator.modules()
             if type(m).__name__=='SDPAAttention']
    if not modules:
        raise ValueError('This optimization requires the existing SDPA cache backend')
    previous=[getattr(m,'inference_readonly_kv',False) for m in modules]
    try:
        for m in modules:m.inference_readonly_kv=True
        yield len(modules)
    finally:
        for m,value in zip(modules,previous):m.inference_readonly_kv=value


@contextmanager
def compact_kv_cache(model):
    if model.training:
        raise ValueError('Compact KV cache is inference-only')
    modules=[m for m in model.backbone.model.aggregator.modules()
             if type(m).__name__=='SDPAAttention']
    if not modules:
        raise ValueError('This optimization requires the existing SDPA cache backend')
    previous=[getattr(m,'inference_compact_kv',False) for m in modules]
    try:
        with readonly_kv_cache(model):
            for m in modules:m.inference_compact_kv=True
            yield len(modules)
    finally:
        for m,value in zip(modules,previous):m.inference_compact_kv=value
