"""Single-frame 4+window cache update without concatenate-then-evict copies.

The reference appends a frame, then concatenates anchors and the retained
window. This path constructs exactly that final tensor once. Special tokens
from the evicted frame are preserved in the same order and format.
"""
import torch


def append_compact(attention, cache, layer, k, v, camera_token_idx, scale_token_idx):
    old_k,old_v=cache[f'k_{layer}'],cache[f'v_{layer}']
    window=attention.kv_cache_sliding_window;anchors=attention.kv_cache_scale_frames
    if (old_k is None or not torch.is_inference_mode_enabled() or attention.training
        or not attention.kv_cache_include_scale_frames or window<2 or anchors<1
        or k.shape[2]!=1 or k.shape[3]<=1 or old_k.shape[2]!=anchors+window):
        return False
    # The reference would evict precisely the oldest non-anchor frame.
    if attention.kv_cache_cross_frame_special:
        end=camera_token_idx+1 if attention.kv_cache_camera_only else scale_token_idx+1
        for key,old in [('k',old_k),('v',old_v)]:
            evicted=old[:,:,anchors:anchors+1,camera_token_idx:end,:].clone()
            special=f'{key}_{layer}_special'
            previous=cache.get(special)
            cache[special]=evicted if previous is None else torch.cat((previous,evicted),dim=2)
    cache[f'k_{layer}']=torch.cat((old_k[:,:,:anchors],old_k[:,:,-(window-1):],k),dim=2)
    cache[f'v_{layer}']=torch.cat((old_v[:,:,:anchors],old_v[:,:,-(window-1):],v),dim=2)
    return True
