"""Exact cache ordering, special-token preservation, and fallback checks."""
import types
import torch
from infinihand.lingbot_map.layers.attention import SDPAAttention
from infinihand.runtime.kv_append import append_compact
rows=[]
with torch.inference_mode():
 for window in [2,16]:
  for special in [False,True]:
   for onlycam in [False,True]:
    for prior in [False,True]:
     a=types.SimpleNamespace(training=False,kv_cache_sliding_window=window,kv_cache_scale_frames=4,kv_cache_include_scale_frames=True,kv_cache_cross_frame_special=special,kv_cache_camera_only=onlycam)
     c={f'{x}_0':torch.randn(1,2,4+window,8,4) for x in ['k','v']}
     if prior:
      c.update({f'{x}_0_special':torch.randn(1,2,3,1 if onlycam else 6,4) for x in ['k','v']})
     d={k:v.clone() for k,v in c.items()};k,v=[torch.randn(1,2,1,8,4) for _ in range(2)]
     d['k_0']=torch.cat((d['k_0'],k),2);d['v_0']=torch.cat((d['v_0'],v),2)
     SDPAAttention._apply_kv_cache_eviction(a,d,0,0,5,4)
     assert append_compact(a,c,0,k,v,0,5)
     assert c.keys()==d.keys() and all(torch.equal(c[x],d[x]) for x in c)
     rows.append(dict(window=window,special=special,camera_only=onlycam,prior_special=prior,bitwise=True))

assert len(rows) == 16
print("16 compact cache equivalence cases passed")
