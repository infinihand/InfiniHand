"""Lossless packed triangle silhouettes and exact popcount mask scores."""

import torch
import triton
import triton.language as tl


@triton.jit
def raster_packed(
    U,
    F,
    M,
    NF: tl.constexpr,
    NV: tl.constexpr,
    RES: tl.constexpr,
    WORDS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
):
    # Pixel p=y*RES+x occupies word p//32, bit p%32. Atomic OR gives
    # silhouette union coverage even when triangles overlap.
    face = tl.program_id(0)
    batch = tl.program_id(1)
    a = tl.load(F + face * 3)
    b = tl.load(F + face * 3 + 1)
    c = tl.load(F + face * 3 + 2)
    off = batch * NV * 2
    ax = tl.load(U + off + a * 2)
    ay = tl.load(U + off + a * 2 + 1)
    bx = tl.load(U + off + b * 2)
    by = tl.load(U + off + b * 2 + 1)
    cx = tl.load(U + off + c * 2)
    cy = tl.load(U + off + c * 2 + 1)
    xmin = tl.maximum(0, tl.floor(tl.minimum(ax, tl.minimum(bx, cx))).to(tl.int32))
    xmax = tl.minimum(W - 1, tl.ceil(tl.maximum(ax, tl.maximum(bx, cx))).to(tl.int32))
    ymin = tl.maximum(0, tl.floor(tl.minimum(ay, tl.minimum(by, cy))).to(tl.int32))
    ymax = tl.minimum(H - 1, tl.ceil(tl.maximum(ay, tl.maximum(by, cy))).to(tl.int32))
    width = tl.maximum(xmax - xmin + 1, 1)
    count = tl.maximum(xmax - xmin + 1, 0) * tl.maximum(ymax - ymin + 1, 0)
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    for start in range(0, tl.cdiv(count, B)):
        t = start * B + tl.arange(0, B)
        x = xmin + t % width
        y = ymin + t // width
        e0 = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        e1 = (cx - bx) * (y - by) - (cy - by) * (x - bx)
        e2 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | (
            (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
        )
        valid = (t < count) & inside & (tl.abs(area) > 1e-7)
        pixel = y * RES + x
        tl.atomic_or(
            M + batch * WORDS + pixel // 32,
            tl.full((), 1, tl.uint32) << (pixel % 32).to(tl.uint32),
            mask=valid,
            sem="relaxed",
        )


@triton.jit
def pack_target(T, P, PIX: tl.constexpr, WORDS: tl.constexpr, B: tl.constexpr):
    b = tl.program_id(0)
    w = tl.program_id(1) * B + tl.arange(0, B)
    bits = tl.arange(0, 32)
    i = w[:, None] * 32 + bits[None, :]
    v = tl.load(T + b * PIX + i, mask=(i < PIX) & (w[:, None] < WORDS), other=0).to(
        tl.uint32
    )
    packed = tl.sum(v << bits[None, :].to(tl.uint32), axis=1)
    tl.store(P + b * WORDS + w, packed, mask=w < WORDS)


@triton.jit
def popcount(v):
    # SWAR population count; interpret signed int32 storage as unsigned bits.
    v = v.to(tl.uint32)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def score_packed(
    M, T, FI, TS, S, WORDS: tl.constexpr, LAMBDA: tl.constexpr, B: tl.constexpr
):
    c = tl.program_id(0)
    i = tl.arange(0, B)
    f = tl.load(FI + c)
    m = tl.load(M + c * WORDS + i, mask=i < WORDS, other=0).to(tl.uint32)
    t = tl.load(T + f * WORDS + i, mask=i < WORDS, other=0).to(tl.uint32)
    inter = tl.sum(popcount(m & t), 0)
    area = tl.sum(popcount(m), 0)
    target = tl.load(TS + f)
    denom = (
        inter.to(tl.float64)
        + LAMBDA * (area - inter).to(tl.float64)
        + (target - inter).to(tl.float64)
    )
    tl.store(S + c, inter.to(tl.float64) / tl.maximum(denom, 1.0))


def pack_masks(target: torch.Tensor) -> torch.Tensor:
    """Pack CUDA bool masks [F, R, R] into int32 [F, ceil(R*R/32)].

    Padding bits in the last word are zero. Signed storage does not change the
    bit pattern; kernels reinterpret words as uint32 before bit operations.
    """
    pix = target.shape[-2] * target.shape[-1]
    words = triton.cdiv(pix, 32)
    out = torch.empty((target.shape[0], words), device=target.device, dtype=torch.int32)
    pack_target[(target.shape[0], triton.cdiv(words, 32))](
        target.contiguous(), out, pix, words, 32
    )
    return out


def gpu_score(
    uv, faces, res, height, width, packed_target, frame_ids, target_sum, lambda_fp
):
    """Rasterize candidate silhouettes and return float64 overlap scores.

    ``uv`` is [C, V, 2] in original pixel coordinates; ``faces`` is int32
    [F, 3]. ``frame_ids[C]`` maps candidates to ``packed_target`` rows.
    ``res`` is the square padded stride; only [0:height, 0:width] is valid.
    ``target_sum`` contains uncompressed target pixel counts. Full candidate
    masks are never materialized: the packed buffer is consumed directly.
    """
    words = triton.cdiv(res * res, 32)
    buf = torch.zeros((uv.shape[0], words), device=uv.device, dtype=torch.int32)
    raster_packed[(faces.shape[0], uv.shape[0])](
        uv.contiguous(),
        faces,
        buf,
        faces.shape[0],
        uv.shape[1],
        res,
        words,
        height,
        width,
        128,
    )
    score = torch.empty((uv.shape[0],), device=uv.device, dtype=torch.float64)
    score_packed[(uv.shape[0],)](
        buf,
        packed_target,
        frame_ids.contiguous(),
        target_sum,
        score,
        words,
        lambda_fp,
        triton.next_power_of_2(words),
        num_warps=8,
    )
    return score
