"""Batched CUDA implementation of offline hand refinement.

Public entry point: :func:`infinihand.postprocess.refine_hands`. The numerical
objective matches ``refine_hands_reference``. CUDA handles geometry, packed
silhouette scores, temporal depth selection and SO(3) smoothing. CPU handles
validation and scheduling; a small availability readback defines segments,
and final geometry/audit values are transferred at the return boundary.

See ``docs/GPU_POSTPROCESS.md`` for tensor conventions and timing boundaries.
"""

import time
import numpy as np
import torch
from .postprocess import segments


def rotvec(R: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, 3] rotation matrices to principal SO(3) log vectors."""
    # Markley matrix-to-quaternion conversion (SciPy convention, scalar last).
    d = R.diagonal(dim1=-2, dim2=-1)
    tr = d.sum(-1)
    choice = torch.cat((d, tr[..., None]), -1).argmax(-1)
    cand = []
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3
        q = torch.empty((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)
        q[..., i] = 1 - tr + 2 * d[..., i]
        q[..., j] = R[..., j, i] + R[..., i, j]
        q[..., k] = R[..., k, i] + R[..., i, k]
        q[..., 3] = R[..., k, j] - R[..., j, k]
        cand.append(q)
    cand.append(
        torch.stack(
            (
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
                1 + tr,
            ),
            -1,
        )
    )
    q = (
        torch.stack(cand, -2)
        .gather(-2, choice[..., None, None].expand(*choice.shape, 1, 4))
        .squeeze(-2)
    )
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-30)
    q = torch.where(q[..., 3:] < 0, -q, q)
    s = torch.linalg.vector_norm(q[..., :3], dim=-1)
    angle = 2 * torch.atan2(s, q[..., 3])
    scale = torch.where(
        angle < 1e-3,
        2 + angle**2 / 12 + 7 * angle**4 / 2880,
        angle / s.clamp_min(1e-30),
    )
    return q[..., :3] * scale[..., None]


def expmap(v: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3] SO(3) log vectors to rotation matrices stably near zero."""
    theta = torch.linalg.vector_norm(v, dim=-1)
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2 * torch.pi)) ** 2
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    M = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *v.shape[:-1], 3, 3
    )
    return (
        torch.eye(3, device=v.device, dtype=v.dtype)
        + a[..., None, None] * M
        + b[..., None, None] * (M @ M)
    )


@torch.inference_mode()
@torch.autocast(device_type="cuda", enabled=False)
def refine_hands_gpu(
    prediction,
    intrinsics,
    masks,
    timestamps,
    mano_model_path,
    *,
    lambda_fp=0.5,
    device="cuda",
    frame_batch=1024,
    raster_batch=4096,
    return_tensors=False,
):
    """Refine one clip; called after public API validation.

    The two batch limits govern temporary work, not the full clip residency.
    Returns NumPy predictions and a JSON-serializable audit. Geometry precision
    intentionally matches the reference: FP64 MANO/SG5, FP32 LSP/projection.
    """
    from .models.mutual_hand_joint_fusion import decode_camera_translation_lsp as lsp
    from .wilor.models.mano_wrapper import MANO
    from ._postprocess_packed import pack_masks, gpu_score

    torch.cuda.synchronize()
    start = time.perf_counter()
    events = []

    def mark(name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append((name, e))

    # 1. Upload the clip once and initialize the matching MANO assets.
    mark("begin")
    def host(a):
        return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    valid = host(prediction["valid"]).astype(bool)
    ts = np.asarray(timestamps, np.float64)
    n = len(valid)
    assert ts.shape == (n,) and np.isfinite(ts).all() and (np.diff(ts) > 0).all()
    assert intrinsics.shape == (n, 3, 3) and lambda_fp > 0

    def tensor(a, dtype=None):
        return torch.as_tensor(a, device=device, dtype=dtype)

    p = {k: tensor(v) for k, v in prediction.items()}
    q = {k: v.clone() for k, v in p.items()}
    K = tensor(intrinsics, torch.float32)
    mask = tensor(masks, torch.bool)
    H, W = mask.shape[-2:]
    res = max(H, W)
    assert mask.shape[:2] == (n, 2)
    mark("upload")
    models = [
        MANO(
            model_path=str(mano_model_path),
            is_rhand=h == 0,
            use_pca=False,
            flat_hand_mean=True,
        )
        .to(device)
        .double()
        .eval()
        for h in range(2)
    ]
    mark("model_setup")
    # 2. Fix per-hand beta with MAD filtering; verify the original MANO decode.
    errors = []
    shape_reports = []
    for h, model in enumerate(models):
        ids = np.flatnonzero(
            valid[:, h] & host(torch.isfinite(p["betas"][:, h]).all(-1))
        )
        ix = tensor(ids, torch.long)
        if not len(ids):
            shape_reports.append(
                dict(
                    hand=("right", "left")[h],
                    usable_frames=0,
                    invalid_or_nonfinite_frames=n,
                    method="unchanged_no_valid_samples",
                    retained_frames=0,
                    rejected_frames=0,
                    fixed_betas=None,
                )
            )
            continue
        x = p["betas"][ix, h].double()
        center = torch.quantile(x, 0.5, dim=0)
        scale = (1.4826 * torch.quantile((x - center).abs(), 0.5, dim=0)).clamp_min(
            0.001
        )
        keep = ((x - center).abs() / scale).amax(-1) <= 3.5
        mean = (x * keep[:, None]).sum(0) / keep.sum().clamp_min(1)
        fixed = torch.where((keep.sum() > 0) & (len(ids) >= 5), mean, center)
        q["betas"][ix, h] = fixed.to(q["betas"].dtype)
        shape_reports.append(
            dict(
                hand=("right", "left")[h],
                usable_frames=len(ids),
                invalid_or_nonfinite_frames=n - len(ids),
                fixed_betas=fixed,
                median=center,
                mad_scale=scale,
                retained_frames=keep.sum(),
                sample_count=len(ids),
            )
        )
        for st in range(0, len(ids), 128):
            ii = ix[st : st + 128]
            go = p["camera_global_orient"][ii, h, None].double()
            hp = p["hand_pose"][ii, h].double()
            w = p["camera_joints"][ii, h, :1].double()
            for original in (True, False):
                out = model(
                    global_orient=go,
                    hand_pose=hp,
                    betas=(p if original else q)["betas"][ii, h].double(),
                    pose2rot=False,
                )
                j = out.joints - out.joints[:, :1] + w
                v = out.vertices - out.joints[:, :1] + w
                if original:
                    errors.append(
                        torch.maximum(
                            (j - p["camera_joints"][ii, h]).abs().max(),
                            (v - p["camera_vertices"][ii, h]).abs().max(),
                        )
                    )
                else:
                    q["camera_joints"][ii, h] = j.to(q["camera_joints"].dtype)
                    q["camera_vertices"][ii, h] = v.to(q["camera_vertices"].dtype)
    mark("fixed_beta_decode")
    # 3. Search the unchanged camera-depth grid in batches of valid frames.
    zg = np.arange(0.10, 1.20001, 0.005, dtype=np.float32)
    z = tensor(zg)
    G = len(zg)
    scores = torch.zeros((n, 2, G), device=device, dtype=torch.float64)
    feasible = torch.zeros((n, 2, G), device=device, dtype=torch.bool)
    translations = torch.zeros((n, 2, G, 3), device=device)
    reproduce = []
    kernel_sections = []
    # One metadata selection on host avoids CUDA nonzero synchronization per batch.
    ready = (
        valid
        & host(mask.any(dim=(-1, -2)))
        & host(p["lsp_valid_mask"].bool().any(dim=-1))
    )
    for h, model in enumerate(models):
        faces = tensor(model.faces.astype(np.int32), torch.int32)
        ids = np.flatnonzero(ready[:, h])
        for st in range(0, len(ids), frame_batch):
            ii = tensor(ids[st : st + frame_batch], torch.long)
            B = len(ii)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            wrist = p["camera_joints"][ii, h, 0].float()
            rot = p["mano_virtual_rotations"][ii, h].float()
            ck = p["mano_crop_intrinsics"][ii, h].float()
            uv = p["pred_keypoints_2d"][ii, h].float() + 0.5
            use = p["lsp_valid_mask"][ii, h].bool()
            jc = (q["camera_joints"][ii, h].float() - wrist[:, None]) @ rot.transpose(
                -1, -2
            )
            vrel = q["camera_vertices"][ii, h].float() - wrist[:, None]
            oldjc = (
                p["camera_joints"][ii, h].float() - wrist[:, None]
            ) @ rot.transpose(-1, -2)

            def solve(depth, joints):
                C = depth.shape[1]
                out = lsp(
                    uv[:, None].expand(-1, C, -1, -1).reshape(B * C, -1, 2),
                    joints[:, None].expand(-1, C, -1, -1).reshape(B * C, -1, 3),
                    depth.reshape(-1).log(),
                    ck[:, None].expand(-1, C, -1, -1).reshape(B * C, 3, 3),
                    (256, 192),
                    use[:, None].expand(-1, C, -1).reshape(B * C, -1),
                )
                return out.reshape(B, C, 3) @ rot

            reproduce.append(
                (
                    solve((wrist[:, None] @ rot.transpose(-1, -2))[:, :, 2], oldjc)[
                        :, 0
                    ]
                    - wrist
                )
                .abs()
                .max()
            )
            low = (
                (0.051 - jc[:, :, 2].amin(1))
                .clamp_min(0.001)[:, None]
                .expand(-1, G)
                .clone()
            )
            high = torch.full_like(low, 3.0)
            allowed = (solve(low, jc)[:, :, 2] <= z) & (solve(high, jc)[:, :, 2] >= z)
            for _ in range(25):
                mid = (low + high) / 2
                current = solve(mid, jc)[:, :, 2]
                low = torch.where(current < z, mid, low)
                high = torch.where(current >= z, mid, high)
            trs = solve((low + high) / 2, jc)
            translations[ii, h] = trs
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            # Flatten (frame, depth) candidates; each target is packed only once.
            target = torch.zeros((B, res, res), device=device, dtype=torch.bool)
            target[:, :H, :W] = mask[ii, h]
            target_sum = target.sum((-1, -2))
            packed_target = pack_masks(target)
            flat_s = []
            flat_f = []
            for c in range(0, B * G, raster_batch):
                ind = torch.arange(c, min(c + raster_batch, B * G), device=device)
                fi = ind // G
                zi = ind % G
                vv = vrel[fi] + trs[fi, zi, None]
                good = (
                    allowed[fi, zi]
                    & (vv[:, :, 2].amin(1) > 0.005)
                    & torch.isfinite(vv).all((1, 2))
                )
                proj = vv @ K[ii[fi]].transpose(-1, -2)
                points = proj[:, :, :2] / proj[:, :, 2:].clamp_min(0.005)
                points = torch.where(good[:, None, None], points, 0)
                score = gpu_score(
                    points, faces, res, H, W, packed_target, fi, target_sum, lambda_fp
                )
                flat_s.append(score)
                flat_f.append(good)
            scores[ii, h] = torch.cat(flat_s).reshape(B, G)
            feasible[ii, h] = torch.cat(flat_f).reshape(B, G)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev2.record()
            kernel_sections.append((ev0, ev1, ev2))
    mark("lsp_raster_score")
    # 4. Select depth paths without copying per-candidate scores to the host.
    # One tiny availability transfer for segment topology; no per-frame .item().
    available = (p["valid"].bool() & feasible.any(-1)).cpu().numpy()
    selected = torch.full((n, 2), -1, device=device, dtype=torch.long)
    penalty = 0.02 * ((z[:, None] - z[None, :]) / 0.01) ** 2
    for h in range(2):
        for a, b in segments(available[:, h], ts):
            cost = -torch.where(feasible[a, h], scores[a, h], -1e6)
            backs = []
            for i in range(a + 1, b):
                # NumPy reference multiplies float32 penalty by scalar before float64 sum.
                mat = cost[:, None] + penalty * float((0.1 / (ts[i] - ts[i - 1])) ** 2)
                val, arg = mat.min(0)
                cost = val - torch.where(feasible[i, h], scores[i, h], -1e6)
                backs.append(arg)
            last = cost.argmin()
            path = [last]
            for arg in reversed(backs):
                last = arg.gather(0, last.reshape(1)).squeeze(0)
                path.append(last)
            selected[a:b, h] = torch.stack(path[::-1])
    xyz = q["camera_joints"][:, :, 0].clone()
    gather = translations.gather(
        2, selected.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)
    xyz = torch.where((selected >= 0)[:, :, None], gather, xyz)
    mark("depth_dp")
    # 5. Apply segment-local SG5 to wrist positions and SO(3) pose tangents.
    rotations = torch.cat(
        (q["camera_global_orient"][:, :, None], q["hand_pose"]), 2
    ).double()
    xyz = xyz.double()
    smoothx = xyz.clone()
    smoothr = rotations.clone()
    for h in range(2):
        # windows are scheduling metadata only; polynomial fitting is CUDA.
        for window in (3, 5):
            rows = []
            windows = []
            for a, b in segments(valid[:, h], ts):
                w = min(5, (b - a) if (b - a) % 2 else (b - a) - 1)
                if w != window:
                    continue
                for i in range(a, b):
                    first = max(a, min(i - w // 2, b - w))
                    rows.append(i)
                    windows.append(list(range(first, first + w)))
            if not rows:
                continue
            ii = tensor(rows, torch.long)
            jj = tensor(windows, torch.long)
            t = tensor(ts)
            dt = t[jj] - t[ii, None]
            A = torch.stack((torch.ones_like(dt), dt, dt**2), -1)
            weights = torch.linalg.pinv(A)[:, 0]
            smoothx[ii, h] = (weights[:, :, None] * xyz[jj, h]).sum(1)
            anchor = rotations[ii, h]
            rel = anchor[:, None].transpose(-1, -2) @ rotations[jj, h]
            rv = rotvec(rel)
            smoothr[ii, h] = anchor @ expmap((weights[:, :, None, None] * rv).sum(1))
    mark("sg5")
    # 6. Decode the final consistent joints/mesh and package the audit.
    for h, model in enumerate(models):
        ix = tensor(np.flatnonzero(valid[:, h]), torch.long)
        for st in range(0, len(ix), 128):
            ii = ix[st : st + 128]
            out = model(
                global_orient=smoothr[ii, h, :1],
                hand_pose=smoothr[ii, h, 1:],
                betas=q["betas"][ii, h].double(),
                pose2rot=False,
            )
            q["camera_joints"][ii, h] = (
                out.joints - out.joints[:, :1] + smoothx[ii, h, None]
            ).to(q["camera_joints"].dtype)
            q["camera_vertices"][ii, h] = (
                out.vertices - out.joints[:, :1] + smoothx[ii, h, None]
            ).to(q["camera_vertices"].dtype)
        q["camera_translation"][ix, h] = smoothx[ix, h].to(
            q["camera_translation"].dtype
        )
        q["camera_global_orient"][ix, h] = smoothr[ix, h, 0].to(
            q["camera_global_orient"].dtype
        )
        q["hand_pose"][ix, h] = smoothr[ix, h, 1:].to(q["hand_pose"].dtype)
    mark("final_decode")
    maxdecode = float(torch.stack(errors).max()) if errors else 0
    maxlsp = float(torch.stack(reproduce).max()) if reproduce else 0
    if not np.isfinite(maxdecode) or maxdecode > 2e-5:
        raise ValueError(f"MANO reconstruction mismatch: {maxdecode} m")
    if not np.isfinite(maxlsp) or maxlsp > 2e-5:
        raise ValueError(f"LSP input convention mismatch: {maxlsp} m")
    for info in shape_reports:
        if info["fixed_betas"] is not None:
            for key in ("fixed_betas", "median", "mad_scale"):
                info[key] = info[key].cpu().tolist()
            info["retained_frames"] = int(info["retained_frames"])
            info["rejected_frames"] = info["sample_count"] - info["retained_frames"]
            info["method"] = (
                "median_fallback_fewer_than_5_samples"
                if info.pop("sample_count") < 5
                else (
                    "mad_filtered_mean"
                    if info["retained_frames"]
                    else "median_fallback_no_inliers"
                )
            )
    result = q if return_tensors else {k: v.cpu().numpy() for k, v in q.items()}
    sel = selected.cpu().numpy()
    mark("download")
    torch.cuda.synchronize()
    return result, dict(
        enabled=True,
        backend="gpu_packed",
        lambda_fp=float(lambda_fp),
        formula="I/(I+lambda*FP+FN)",
        order=["per-hand fixed beta", "joint LSP depth", "SG5 wrist and SO3 pose"],
        shape=shape_reports,
        z_grid_m=[0.1, 1.2, 0.005],
        temporal_depth_penalty="0.02*(delta_z/0.01m)^2*(0.1s/dt)^2",
        smoothing_window_frames=5,
        polynomial_order=2,
        gap_policy="missing detections or >1.5 median frame intervals",
        input_valid_hands=int(valid.sum()),
        fallback_hands=int((valid & (sel < 0)).sum()),
        boundary_hands=int(((sel == 0) | (sel == G - 1)).sum()),
        max_lsp_reproduction_error_m=maxlsp,
        causal=False,
        gt_used=False,
        seconds=time.perf_counter() - start,
        stage_ms={
            events[i][0]: events[i - 1][1].elapsed_time(events[i][1])
            for i in range(1, len(events))
        },
        lsp_ms=sum(a.elapsed_time(b) for a, b, c in kernel_sections),
        raster_score_ms=sum(b.elapsed_time(c) for a, b, c in kernel_sections),
        frame_batch=frame_batch,
        raster_batch=raster_batch,
        max_decode_error_m=maxdecode,
        max_lsp_error_m=maxlsp,
        optimized_hands=int((sel >= 0).sum()),
        selected=sel.tolist(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
