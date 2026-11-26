"""Optional two-kernel CUDA/Triton reduction for tight binary-mask boxes.

Input/output conventions exactly match infer.tight_mask_boxes. No sigmoid,
threshold, resizing, hand assignment or presence policy is changed here.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(M, P, H:tl.constexpr, W:tl.constexpr, TILES:tl.constexpr, BLOCK:tl.constexpr):
    hand = tl.program_id(0); tile = tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    yes = tl.load(M + hand * H * W + i, i < H*W, other=0).to(tl.int1)
    x = i % W; y = i // W
    x1 = tl.min(tl.where(yes, x, W), 0); y1 = tl.min(tl.where(yes, y, H), 0)
    x2 = tl.max(tl.where(yes, x+1, 0), 0); y2 = tl.max(tl.where(yes, y+1, 0), 0)
    base = (hand * TILES + tile) * 4
    tl.store(P+base, x1); tl.store(P+base+1, y1)
    tl.store(P+base+2, x2); tl.store(P+base+3, y2)


@triton.jit
def _finish(P, B, V, H:tl.constexpr, W:tl.constexpr, TILES:tl.constexpr, BLOCK:tl.constexpr):
    hand = tl.program_id(0); i = tl.arange(0, BLOCK)
    base = (hand*TILES+i)*4
    x1=tl.min(tl.load(P+base,i<TILES,other=W),0)
    y1=tl.min(tl.load(P+base+1,i<TILES,other=H),0)
    x2=tl.max(tl.load(P+base+2,i<TILES,other=0),0)
    y2=tl.max(tl.load(P+base+3,i<TILES,other=0),0)
    valid=x2>0
    tl.store(V+hand,valid)
    tl.store(B+hand*4,tl.where(valid,x1.to(tl.float32)*(1.0/W),0.0))
    tl.store(B+hand*4+1,tl.where(valid,y1.to(tl.float32)*(1.0/H),0.0))
    tl.store(B+hand*4+2,tl.where(valid,x2.to(tl.float32)*(1.0/W),0.0))
    tl.store(B+hand*4+3,tl.where(valid,y2.to(tl.float32)*(1.0/H),0.0))


def tight_mask_boxes_triton(binary):
    if binary.ndim != 5 or binary.shape[2] != 2 or binary.dtype != torch.bool or not binary.is_cuda:
        raise ValueError('Require CUDA bool mask [B,T,2,H,W]')
    b,t,slots,h,w=binary.shape
    if h==0 or w==0:raise ValueError('Empty image dimensions')
    binary=binary.contiguous();count=b*t*slots;tiles=triton.cdiv(h*w,2048)
    partial=torch.empty((count,tiles,4),device=binary.device,dtype=torch.int32)
    boxes=torch.empty((b,t,slots,4),device=binary.device,dtype=torch.float32)
    valid=torch.empty((b,t,slots),device=binary.device,dtype=torch.bool)
    if count:
        _partial[(count,tiles)](binary,partial,h,w,tiles,2048)
        _finish[(count,)](partial,boxes,valid,h,w,tiles,triton.next_power_of_2(tiles))
    return boxes,valid
