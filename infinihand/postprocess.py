"""Offline fixed-shape, mask-guided joint LSP depth and SG5 hand refinement.

Inputs use right/left slots, wrist-centred MANO, image-camera metres and actual
frame timestamps. No ground-truth annotations are read by this module.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .stabilize_shape import stabilize_prediction


def segments(valid, timestamps):
    """Split missing detections and gaps > 1.5 nominal frame intervals."""
    dt = np.diff(timestamps)
    limit = 1.5 * np.median(dt) if len(dt) else float('inf')
    contiguous = (dt > 0) & (dt <= limit)
    starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1] | ~contiguous])
    for start in starts:
        end = start + 1
        while end < len(valid) and valid[end] and contiguous[end-1]:
            end += 1
        yield start, end


def sg5_both(xyz, rotations, valid, timestamps):
    """Local quadratic fit, five frames, using actual times and SO(3) tangents."""
    x, r = np.array(xyz, dtype=np.float64, copy=True), np.array(rotations, dtype=np.float64, copy=True)
    for hand in range(2):
        for start, end in segments(valid[:, hand], timestamps):
            n = end - start
            window = min(5, n if n % 2 else n - 1)
            if window < 3:
                continue
            for i in range(start, end):
                first = max(start, min(i-window//2, end-window))
                ix = np.arange(first, first+window)
                dt = timestamps[ix] - timestamps[i]
                weights = np.linalg.pinv(np.stack([np.ones_like(dt), dt, dt**2], axis=1))[0]
                x[i, hand] = weights @ xyz[ix, hand]
                for joint in range(rotations.shape[2]):
                    anchor = rotations[i, hand, joint]
                    tangent = Rotation.from_matrix(anchor.T @ rotations[ix, hand, joint]).as_rotvec()
                    r[i, hand, joint] = anchor @ Rotation.from_rotvec(weights @ tangent).as_matrix()
    return x, r


def select_depth_path(scores, feasible, valid, timestamps, z_grid):
    """Dynamic programming with the previously evaluated 0.02 depth penalty."""
    selected = np.full(valid.shape, -1, dtype=np.int64)
    penalty = .02 * ((z_grid[:, None]-z_grid[None, :])/.01)**2
    for hand in range(2):
        available = valid[:, hand] & feasible[:, hand].any(-1)
        for start, end in segments(available, timestamps):
            costs = -np.where(feasible[start, hand], scores[start, hand], -1e6)
            backs = []
            for i in range(start+1, end):
                mat = costs[:, None] + penalty * (.1/(timestamps[i]-timestamps[i-1]))**2
                arg = mat.argmin(0)
                costs = mat[arg, np.arange(len(z_grid))] - np.where(feasible[i, hand], scores[i, hand], -1e6)
                backs.append(arg)
            path = [int(costs.argmin())]
            for arg in backs[::-1]:
                path.append(int(arg[path[-1]]))
            selected[start:end, hand] = path[::-1]
    return selected


def refine_hands(prediction, intrinsics, masks, timestamps, mano_model_path, *, lambda_fp=.5, device='cuda'):
    """Return refined camera prediction and a JSON-serializable audit report.

    The four LSP auxiliary fields must come from the same forward pass. Masks
    must be binary [T,2,H,W] in the image coordinates of intrinsics. Empty masks
    or infeasible depth candidates retain the fixed-shape original wrist before
    SG5. Invalid hands stay invalid and unchanged. Fixed beta assumes one person
    per hand slot across the input clip; split identity changes before inference.
    """
    from .models.mutual_hand_joint_fusion import decode_camera_translation_lsp as lsp
    from .wilor.models.mano_wrapper import MANO
    from ._postprocess_raster import gpu_mask

    started = time.perf_counter()
    if not np.isfinite(lambda_fp) or lambda_fp <= 0:
        raise ValueError('lambda_fp must be positive and finite')
    p = prediction
    valid = np.asarray(p['valid'], dtype=bool)
    n = len(valid)
    ts = np.asarray(timestamps, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Require finite, strictly increasing timestamps [T]')
    if K.shape != (n,3,3) or not np.isfinite(K).all() or masks.ndim != 4 or masks.shape[:2] != (n,2):
        raise ValueError('Require intrinsics [T,3,3] and masks [T,2,H,W]')
    required = ('pred_keypoints_2d', 'mano_virtual_rotations', 'mano_crop_intrinsics', 'lsp_valid_mask')
    if any(key not in p for key in required):
        raise ValueError('LSP needs auxiliary predictions captured from the same forward pass')
    if torch.device(device).type != 'cuda':
        raise ValueError('Mask-guided depth search requires CUDA and Triton')
    q, _, _, shape_report = stabilize_prediction(p, mano_model_path)
    z = np.arange(.10, 1.20001, .005, dtype=np.float32)
    G = len(z)
    height, width = masks.shape[-2:]
    res = max(height, width)
    scores = np.zeros((n,2,G), dtype=np.float64)
    feasible = np.zeros((n,2,G), dtype=bool)
    translations = np.zeros((n,2,G,3), dtype=np.float32)
    models = [MANO(model_path=str(mano_model_path), is_rhand=h==0, use_pca=False, flat_hand_mean=True).double().eval() for h in range(2)]
    max_reproduction = 0.
    def tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, device=device, dtype=dtype)
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for hand in range(2):
            faces = tensor(models[hand].faces.astype(np.int32), torch.int32)
            for i in np.flatnonzero(valid[:,hand]):
                if not masks[i,hand].any() or not np.asarray(p['lsp_valid_mask'][i,hand]).any():
                    continue
                wrist = tensor(p['camera_joints'][i,hand,0])
                rot = tensor(p['mano_virtual_rotations'][i,hand])
                ck = tensor(p['mano_crop_intrinsics'][i,hand])
                uv = tensor(p['pred_keypoints_2d'][i,hand]+.5)
                use = tensor(p['lsp_valid_mask'][i,hand], torch.bool)
                jc = (tensor(q['camera_joints'][i,hand])-wrist) @ rot.T
                vrel = tensor(q['camera_vertices'][i,hand])-wrist
                oldjc = (tensor(p['camera_joints'][i,hand])-wrist) @ rot.T
                def solve(depth, joints):
                    count = len(depth)
                    return lsp(uv[None].expand(count,-1,-1), joints[None].expand(count,-1,-1), depth.log(), ck[None].expand(count,-1,-1), (256,192), use[None].expand(count,-1)) @ rot
                error = float((solve((wrist @ rot.T)[2:3], oldjc)[0]-wrist).abs().max())
                if not np.isfinite(error) or error > 2e-5:
                    raise ValueError(f'LSP input convention mismatch at frame {i}, hand {hand}: {error} m')
                max_reproduction = max(max_reproduction, error)
                low = torch.full((G,), max(.051-float(jc[:,2].min()), .001), device=device)
                high = torch.full((G,), 3., device=device)
                target_z = tensor(z)
                allowed = (solve(low,jc)[:,2] <= target_z) & (solve(high,jc)[:,2] >= target_z)
                for _ in range(25):
                    mid = (low+high)/2
                    current = solve(mid,jc)[:,2]
                    low = torch.where(current < target_z,mid,low)
                    high = torch.where(current >= target_z,mid,high)
                trs = solve((low+high)/2,jc)
                target = torch.zeros((res,res), device=device, dtype=torch.bool)
                target[:height,:width] = tensor(masks[i,hand],torch.bool)
                for start in range(0,G,16):
                    stop = min(start+16,G)
                    vv = vrel[None]+trs[start:stop,None]
                    good = allowed[start:stop] & (vv[:,:,2].amin(1)>.005) & torch.isfinite(vv).all((1,2))
                    proj = vv @ tensor(K[i]).T
                    points = proj[:,:,:2]/proj[:,:,2:].clamp_min(.005)
                    points[~good] = 0
                    rendered = gpu_mask(points,faces,res)
                    rendered[:,height:] = False
                    rendered[:,:,width:] = False
                    intersection = (rendered & target).sum((-1,-2))
                    fp = (rendered & ~target).sum((-1,-2))
                    fn = (~rendered & target).sum((-1,-2))
                    # Float64 scoring matches the offline lambda sweep.
                    score = intersection.double()/(intersection+lambda_fp*fp.double()+fn).clamp_min(1)
                    scores[i,hand,start:stop] = score.cpu().numpy()
                    feasible[i,hand,start:stop] = good.cpu().numpy()
                translations[i,hand] = trs.cpu().numpy()
    selected = select_depth_path(scores,feasible,valid,ts,z)
    xyz = q['camera_joints'][:,:,0].copy()
    for hand in range(2):
        ids = np.flatnonzero(selected[:,hand]>=0)
        xyz[ids,hand] = translations[ids,hand,selected[ids,hand]]
    rotations = np.concatenate([q['camera_global_orient'][:,:,None],q['hand_pose']],axis=2)
    xyz, rotations = sg5_both(xyz,rotations,valid,ts)
    for hand, model in enumerate(models):
        ids = np.flatnonzero(valid[:,hand])
        for start in range(0,len(ids),128):
            ix = ids[start:start+128]
            with torch.no_grad():
                out = model(global_orient=torch.tensor(rotations[ix,hand,:1],dtype=torch.float64), hand_pose=torch.tensor(rotations[ix,hand,1:],dtype=torch.float64), betas=torch.tensor(q['betas'][ix,hand],dtype=torch.float64), pose2rot=False)
            joints, verts = out.joints.numpy(), out.vertices.numpy()
            q['camera_joints'][ix,hand] = joints-joints[:,:1]+xyz[ix,hand,None]
            q['camera_vertices'][ix,hand] = verts-joints[:,:1]+xyz[ix,hand,None]
        q['camera_translation'][ids,hand] = xyz[ids,hand]
        q['camera_global_orient'][ids,hand] = rotations[ids,hand,0]
        q['hand_pose'][ids,hand] = rotations[ids,hand,1:]
    report = dict(enabled=True, lambda_fp=float(lambda_fp), formula='I/(I+lambda*FP+FN)', order=['per-hand fixed beta','joint LSP depth','SG5 wrist and SO3 pose'], shape=shape_report, z_grid_m=[.1,1.2,.005], temporal_depth_penalty='0.02*(delta_z/0.01m)^2*(0.1s/dt)^2', smoothing_window_frames=5, polynomial_order=2, gap_policy='missing detections or >1.5 median frame intervals', input_valid_hands=int(valid.sum()), optimized_hands=int((selected>=0).sum()), fallback_hands=int((valid & (selected<0)).sum()), boundary_hands=int(((selected==0)|(selected==G-1)).sum()), max_lsp_reproduction_error_m=max_reproduction, seconds=time.perf_counter()-started, causal=False, gt_used=False)
    return q, report
