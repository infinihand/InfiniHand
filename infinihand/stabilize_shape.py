"""Offline, per-sequence MANO shape stabilization; no GT or translation fitting.

Example:
    python -m infinihand.stabilize_shape --prediction /path/to/prediction.npz \
        --mano-model-path /path/to/mano --output /path/to/stabilized.npz

Requires one subject per hand slot. Uses the whole sequence, so this is not a
causal streaming filter. Invalid detections remain invalid and unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def estimate_fixed_betas(betas, valid, threshold=3.5, scale_floor=1e-3):
    """Reject frames with any robust coordinate z-score > threshold, then mean.

    betas: [T, 2, D], slots right/left. MAD is multiplied by 1.4826;
    scale_floor is in beta coefficient units and handles zero MAD. With fewer
    than 5 valid samples, or no retained samples, use the coordinate median
    and explicitly report the fallback. No samples leaves that slot unchanged.
    """
    b = np.asarray(betas)
    valid = np.asarray(valid, dtype=bool)
    if b.ndim != 3 or b.shape[1] != 2 or valid.shape != b.shape[:2]:
        raise ValueError('Expected betas [T,2,D] and valid [T,2]')
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError('threshold and scale_floor must be positive and finite')
    usable = valid & np.isfinite(b).all(axis=-1)
    stabilized = b.copy()
    inliers = np.zeros_like(valid)
    scores = np.full(valid.shape, np.nan, dtype=np.float64)
    report = []
    for side, name in enumerate(('right', 'left')):
        idx = np.flatnonzero(usable[:, side])
        info = dict(hand=name, usable_frames=len(idx), invalid_or_nonfinite_frames=len(b)-len(idx))
        if not len(idx):
            info.update(method='unchanged_no_valid_samples', retained_frames=0, rejected_frames=0, fixed_betas=None)
            report.append(info)
            continue
        x = b[idx, side].astype(np.float64)
        center = np.median(x, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(x-center), axis=0), scale_floor)
        score = np.max(np.abs(x-center)/scale, axis=-1)
        keep = score <= threshold
        method = 'mad_filtered_mean'
        if len(x) < 5:
            fixed = center
            method = 'median_fallback_fewer_than_5_samples'
        elif not keep.any():
            fixed = center
            method = 'median_fallback_no_inliers'
        else:
            fixed = x[keep].mean(axis=0)
        scores[idx, side] = score
        inliers[idx[keep], side] = True
        stabilized[idx, side] = fixed.astype(b.dtype)
        info.update(method=method, retained_frames=int(keep.sum()), rejected_frames=int((~keep).sum()),
                    fixed_betas=fixed.tolist(), median=center.tolist(), mad_scale=scale.tolist())
        report.append(info)
    return stabilized, inliers, scores, report


def stabilize_prediction(prediction, mano_model_path, threshold=3.5, scale_floor=1e-3, batch_size=128):
    """Re-decode camera-space geometry, keeping pose and wrist XYZ unchanged.

    Original-beta decode must reproduce input geometry (<=0.02 mm). This guards
    against wrong MANO assets, hand ordering or decoder conventions.
    """
    import torch
    from infinihand.wilor.models.mano_wrapper import MANO

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    required = {'betas', 'valid', 'hand_pose', 'camera_global_orient', 'camera_joints', 'camera_vertices'}
    if not required.issubset(prediction):
        raise ValueError(f'Missing prediction fields: {sorted(required-set(prediction))}')
    if any(k.startswith('world_') for k in prediction):
        raise ValueError('Pass camera-space prediction.npz, not world-space geometry')
    out = {k: np.array(v, copy=True) for k,v in prediction.items()}
    fixed, inliers, scores, report = estimate_fixed_betas(out['betas'], out['valid'], threshold, scale_floor)
    n = len(fixed)
    if out['hand_pose'].shape != (n,2,15,3,3) or out['camera_global_orient'].shape != (n,2,3,3):
        raise ValueError('Expected saved rotation-matrix MANO poses')
    for side in range(2):
        ids = np.flatnonzero(out['valid'][:,side] & np.isfinite(out['betas'][:,side]).all(axis=-1))
        if not len(ids):
            continue
        model = MANO(model_path=str(mano_model_path), is_rhand=side==0,
                     use_pca=False, flat_hand_mean=True).double().eval()
        max_error = 0.0
        for start in range(0,len(ids),batch_size):
            ii = ids[start:start+batch_size]
            go = torch.as_tensor(out['camera_global_orient'][ii,side],dtype=torch.float64)[:,None]
            hp = torch.as_tensor(out['hand_pose'][ii,side],dtype=torch.float64)
            wrist = out['camera_joints'][ii,side,0:1].astype(np.float64)
            def decode(beta):
                with torch.no_grad():
                    result = model(global_orient=go, hand_pose=hp,
                                   betas=torch.as_tensor(beta,dtype=torch.float64), pose2rot=False)
                j = result.joints.numpy(); v = result.vertices.numpy()
                return j-j[:,0:1]+wrist, v-j[:,0:1]+wrist
            oldj, oldv = decode(out['betas'][ii,side])
            error = max(np.max(np.abs(oldj-out['camera_joints'][ii,side])),
                        np.max(np.abs(oldv-out['camera_vertices'][ii,side])))
            if not np.isfinite(error) or error > 2e-5:
                raise ValueError(f'MANO reconstruction mismatch on hand {side}: {error} m')
            max_error = max(max_error,float(error))
            j,v = decode(fixed[ii,side])
            out['camera_joints'][ii,side] = j
            out['camera_vertices'][ii,side] = v
        report[side]['original_decode_max_error_m'] = max_error
    out['betas'] = fixed
    # Do not add detection validity or move any existing wrist/translation.
    if not np.array_equal(out['camera_joints'][:,:,0], prediction['camera_joints'][:,:,0], equal_nan=True):
        raise AssertionError('Wrist position changed')
    return out, inliers, scores, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mano-model-path', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=3.5)
    parser.add_argument('--scale-floor', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    paths = [args.output,args.output.with_suffix('.shape_report.json'),args.output.with_suffix('.shape_filter.npz')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Use a new output path; existing results are never overwritten')
    with np.load(args.prediction,allow_pickle=False) as data:
        out, keep, scores, report = stabilize_prediction(dict(data),args.mano_model_path,
                                                        args.threshold,args.scale_floor,args.batch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**out)
    np.savez_compressed(paths[2],inliers=keep,robust_scores=scores)
    paths[1].write_text(json.dumps(dict(input=str(args.prediction),threshold=args.threshold,
                                      scale_floor=args.scale_floor,hands=report),indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
