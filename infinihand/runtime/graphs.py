"""Optional CUDA Graph capture for stateless image encoders."""
import time
import torch


def prepare_image_encoder_graphs(model, *, branch_bf16=False, anchor_frames=4) -> float:
    """Capture stateless image encoders for B=1 streaming and four anchor frames.

    Call once on an eval model before starting inference threads. Precision and
    temporal-cache behavior are unchanged; graph outputs are cloned before reuse.
    """
    if model.training or not torch.cuda.is_available():
        raise ValueError('CUDA graph preparation requires an eval model on CUDA')
    started = time.perf_counter()
    def clone_tree(value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: clone_tree(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(clone_tree(item) for item in value)
        if isinstance(value, list):
            return [clone_tree(item) for item in value]
        return value

    def capture(module, shapes, autocast, spatial_only=False):
        if getattr(module, '_inference_graphs_prepared', False):
            if module._inference_graph_autocast != autocast:
                raise ValueError('Graph precision changed: use a fresh model')
            return
        original = module.forward
        operation = (lambda x: module.forward_features(x, spatial_only=True)) if spatial_only else original
        cache = {}
        device = next(module.parameters()).device
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.inference_mode():
            for shape in shapes:
                static = torch.zeros(shape, device=device, dtype=torch.float32)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast, cache_enabled=False):
                    for _ in range(3):
                        operation(static)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast, cache_enabled=False):
                    output = operation(static)
                cache[shape] = (graph, static, output)
        def replay(x):
            if module.training or not torch.is_inference_mode_enabled() or tuple(x.shape) not in cache or torch.is_autocast_enabled('cuda') != autocast or (autocast and torch.get_autocast_dtype('cuda') != torch.bfloat16):
                return original(x)
            graph, static, output = cache[tuple(x.shape)]
            static.copy_(x)
            graph.replay()
            module._inference_graph_replays = getattr(module, "_inference_graph_replays", 0) + 1
            return clone_tree(output)
        module.forward = replay
        module._inference_graphs_prepared = True
        module._inference_graph_autocast = autocast

    h, w = model.cfg.image_height, model.cfg.image_width
    capture(model.backbone.model.aggregator.patch_embed, [(1, 3, h, w), (anchor_frames, 3, h, w)], True)
    provider = model.hand_encoder
    batches = sorted({min(provider.chunk_size, total-i) for total in (2, 2*anchor_frames)
                      for i in range(0, total, provider.chunk_size)})
    capture(provider.backbone, [(batch, 3, provider.crop_size, provider.crop_size-64) for batch in batches], branch_bf16, spatial_only=True)
    torch.cuda.synchronize(next(model.parameters()).device)
    return time.perf_counter() - started

