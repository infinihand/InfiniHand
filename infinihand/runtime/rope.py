"""Inference-only fused 2D RoPE with explicit eager-equivalent rounding.

Keep the original frequency tables and BF16/FP32 multiplication boundaries.
FMA is disabled: fusing the two rounded products would change the model.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X,P,C,S,Y, H:tl.constexpr,N:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,
          X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,X3:tl.constexpr,
          P0:tl.constexpr,P1:tl.constexpr,P2:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=idx<TOTAL
    d=idx%D;n=idx//D%N;h=idx//(D*N)%H;b=idx//(D*N*H)
    axis=d//(D//2);j=d%(D//2);other=(j+D//4)%(D//2)+axis*(D//2)
    pos=tl.load(P+b*P0+n*P1+axis*P2,ok,other=0)
    x=tl.load(X+b*X0+h*X1+n*X2+d*X3,ok,other=0)
    v=tl.load(X+b*X0+h*X1+n*X2+other*X3,ok,other=0)
    c=tl.load(C+pos*(D//2)+j,ok,other=0);s=tl.load(S+pos*(D//2)+j,ok,other=0)
    dtype=X.dtype.element_ty
    a=(x.to(tl.float32)*c.to(tl.float32)).to(dtype).to(tl.float32)
    v=tl.where(j<D//4,-v,v)
    z=(v.to(tl.float32)*s.to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(Y+idx,a+z,ok)


def rope2d(module,tokens,positions):
    if tokens.dtype not in (torch.bfloat16,torch.float32) or not tokens.is_cuda or tokens.shape[-1]%4:
        raise ValueError('Fused RoPE requires CUDA BF16/FP32 and dimension divisible by four')
    b,h,n,d=tokens.shape
    if positions.shape!=(b,n,2):raise ValueError('positions must be [B,N,2]')
    cos,sin=module._compute_frequency_components(d//2,n,tokens.device,tokens.dtype)
    out=torch.empty(tokens.shape,device=tokens.device,dtype=tokens.dtype)
    _rope[(triton.cdiv(out.numel(),256),)](tokens,positions,cos,sin,out,h,n,d,out.numel(),
        *tokens.stride(),*positions.stride(),256,enable_fp_fusion=False)
    return out


@contextmanager
def fused_rope(model):
    if model.training:raise ValueError('Fused RoPE is inference-only')
    patches=[]
    try:
        for module in model.backbone.model.aggregator.modules():
            if type(module).__name__!='RotaryPositionEmbedding2D':continue
            original=module.forward
            def forward(tokens,positions,_module=module,_original=original):
                if not _module.training and torch.is_inference_mode_enabled():return rope2d(_module,tokens,positions)
                return _original(tokens,positions)
            patches.append((module,original));module.forward=forward
        if not patches:raise ValueError('No compatible 2D RoPE module found')
        yield len(patches)
    finally:
        for module,original in patches:module.forward=original
