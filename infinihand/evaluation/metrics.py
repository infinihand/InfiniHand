"""NumPy-only training-checkpoint diagnostics; NOT the paper's bbox-matched protocol.

compute_sequence_metrics(arrays) consumes metres, OpenCV camera-to-world poses,
runtime [right,left] slots, and the same source frames for prediction and GT.
Missing predictions are counted by detection and geometry coverage. Conditional
geometry means carry ``matched`` in their names; complete-coverage means are null
if any evaluable GT hand is missing. No arbitrary zero pose enters an error mean.
"""
from __future__ import annotations

import numpy as np


def _ratio(n, d):
    return float(n / d) if d else None


def _mean(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()) if v.size else None


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(count=int(v.size), mean=_mean(v),
                median=float(np.median(v)) if v.size else None,
                rmse=float(np.sqrt(np.mean(v * v))) if v.size else None)


def _sim3(pred, gt):
    """Fit one orientation-preserving similarity: scale * pred @ R + t."""
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if len(p) < 3:
        return None
    pc, gc = p - p.mean(0), g - g.mean(0)
    if np.linalg.matrix_rank(pc, tol=1e-12) < 2 or np.linalg.matrix_rank(gc, tol=1e-12) < 2:
        return None
    u, s, vh = np.linalg.svd(pc.T @ gc)
    d = np.ones(3); d[-1] = -1. if np.linalg.det(u @ vh) < 0 else 1.
    r = (u * d) @ vh
    scale = float(np.sum(s * d) / np.sum(pc * pc))
    shift = g.mean(0) - scale * p.mean(0) @ r
    return scale, r, shift


def _geometry(pred, gt, gt_valid, pred_valid, point_valid):
    """Means give every scored hand equal weight, using its valid GT joints."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    gt_points = np.asarray(point_valid, bool) & np.isfinite(gt).all(-1)
    evaluable = np.asarray(gt_valid, bool) & gt_points.any(-1)
    finite_pred = np.isfinite(pred).all(axis=(-1, -2))
    matched = evaluable & np.asarray(pred_valid, bool) & finite_pred
    errors, relative, pa, roots, depths = [], [], [], [], []
    pck_hits = {20: 0, 50: 0, 100: 0}
    point_denominator = int(gt_points[evaluable].sum())
    pa_degenerate = 0
    for t, s in np.argwhere(matched):
        valid = gt_points[t, s]
        p, g = pred[t, s], gt[t, s]
        e = np.linalg.norm(p[valid] - g[valid], axis=-1) * 1000.
        errors.append(float(e.mean()))
        for limit in pck_hits:
            pck_hits[limit] += int((e < limit).sum())
        if valid[0]:
            roots.append(float(np.linalg.norm(p[0] - g[0]) * 1000.))
            depths.append(float(abs(p[0, 2] - g[0, 2]) * 1000.))
            relative.append(float(np.linalg.norm((p - p[:1] - g + g[:1])[valid], axis=-1).mean() * 1000.))
        fit = _sim3(p[valid], g[valid])
        if fit is not None:
            scale, r, shift = fit
            pa.append(float(np.linalg.norm(scale * p[valid] @ r + shift - g[valid], axis=-1).mean() * 1000.))
        else:
            pa_degenerate += 1
    ng, nm = int(evaluable.sum()), int(matched.sum())
    report = dict(gt_valid_hands=int(np.asarray(gt_valid, bool).sum()),
        evaluable_gt_hands=ng, matched_hands=nm, missing_gt_hands=ng-nm,
        gt_hands_without_valid_joints=int((np.asarray(gt_valid, bool) & ~gt_points.any(-1)).sum()),
        nonfinite_predicted_hands=int((evaluable & np.asarray(pred_valid, bool) & ~finite_pred).sum()),
        coverage=_ratio(nm, ng), mpjpe_matched_mm=_mean(errors),
        root_relative_mpjpe_matched_mm=_mean(relative), pa_mpjpe_matched_mm=_mean(pa),
        root_error_matched_mm=_mean(roots), wrist_depth_error_matched_mm=_mean(depths),
        pa_scored_hands=len(pa), pa_degenerate_hands=pa_degenerate,
        mpjpe_complete_coverage_mm=_mean(errors) if nm == ng and ng else None,
        pa_mpjpe_complete_coverage_mm=_mean(pa) if nm == ng and ng and len(pa) == ng else None,
        pck_all_gt={str(k)+'mm': _ratio(v, point_denominator) for k, v in pck_hits.items()},
        pck_denominator_joints=point_denominator,
        missing_policy='Missing/invalid predictions count as failed PCK and reduce coverage; matched means are conditional.',
        averaging='Mean per evaluable hand, then mean across hands; PA fits one Sim(3) per hand/frame.')
    return report, matched, gt_points


def _mask_and_detection(a):
    gt, pr, inter, union = [np.asarray(a[k], np.float64) for k in
        ('mask_gt_count', 'mask_pred_count', 'mask_intersection', 'mask_union')]
    known = np.asarray(a['mask_supervision_valid'], bool)
    if not (gt.shape == pr.shape == inter.shape == union.shape == known.shape):
        raise ValueError('Mask count and supervision arrays must have identical [T,2] shapes')
    if not np.isfinite(np.stack([gt, pr, inter, union])).all() or np.any(inter < 0) or np.any(inter > np.minimum(gt, pr)):
        raise ValueError('Invalid mask intersection/counts')
    if not np.allclose(union, gt + pr - inter):
        raise ValueError('Mask union must equal gt_count + pred_count - intersection')
    visible, detected = gt > 0, pr > 0
    tp, fn = known & visible & detected, known & visible & ~detected
    fp, tn = known & ~visible & detected, known & ~visible & ~detected
    ntp,nfn,nfp,ntn = [int(v.sum()) for v in (tp,fn,fp,tn)]
    precision, recall = _ratio(ntp, ntp+nfp), _ratio(ntp, ntp+nfn)
    frame_has_gt = (known & visible).any(-1)
    frame_error = (fn | fp).any(-1)
    fully_known = known.all(-1)
    positive = known & visible
    iou = np.divide(inter, union, out=np.ones_like(inter), where=union>0)
    dice = np.divide(2*inter, gt+pr, out=np.ones_like(inter), where=(gt+pr)>0)
    detection = dict(protocol='Same-side mask presence, threshold supplied by inference; no bbox/mask matching.',
        FAcc=_ratio(int((frame_has_gt & ~frame_error).sum()), int(frame_has_gt.sum())),
        Recall=recall, Precision=precision, F1=_ratio(2*ntp, 2*ntp+nfp+nfn),
        tp=ntp,fn=nfn,fp=nfp,tn=ntn, scored_side_frames=int(known.sum()),
        unknown_side_frames=int((~known).sum()), facc_scored_frames=int(frame_has_gt.sum()),
        frame_accuracy_all_fully_annotated=_ratio(int((fully_known & ~frame_error).sum()),int(fully_known.sum())),
        fully_annotated_frames=int(fully_known.sum()),
        gt_presence='Nonempty projected GT mask on sides with mask_supervision_valid=True.',
        facc_definition='All annotated side-presence decisions correct, on frames with at least one annotated GT-positive hand. Unknown sides ignored.')
    masks = dict(iou_gt_positive=_mean(iou[positive]), dice_gt_positive=_mean(dice[positive]),
        gt_positive_hands=int(positive.sum()), iou_all_known=_mean(iou[known]),
        dice_all_known=_mean(dice[known]), all_known_hands=int(known.sum()),
        gt_empty_hands=int((known & ~visible).sum()), false_positive_empty_hands=nfp,
        micro_iou=_ratio(float(inter[known].sum()), float(union[known].sum())),
        micro_dice=_ratio(float(2*inter[known].sum()),float((gt+pr)[known].sum())),
        empty_empty_rule='IoU/Dice=1 for both empty; GT-positive means include missed hands as zero.')
    return masks,detection


def _transform_joints(joints, c2w):
    return np.einsum('tij,thkj->thki',c2w[:,:3,:3],joints) + c2w[:,None,None,:3,3]


def _camera_world(a, gt, pred, control, gt_valid, pred_valid, control_valid, point_valid):
    if not all(k in a for k in ('gt_c2w','pred_c2w','camera_valid')):
        return dict(available=False, reason='No GT or predicted camera trajectory'),dict(available=False)
    gc, pc = np.asarray(a['gt_c2w'],np.float64),np.asarray(a['pred_c2w'],np.float64)
    declared_valid = np.asarray(a['camera_valid'],bool)
    if gc.shape != pc.shape or gc.shape != (len(gt),4,4):
        raise ValueError('Camera poses must be aligned [T,4,4] camera-to-world matrices')
    gt_pose_valid = declared_valid & np.isfinite(gc).all(axis=(-1,-2))
    pred_pose_valid = np.isfinite(pc).all(axis=(-1,-2))
    valid = gt_pose_valid & pred_pose_valid
    if not valid.any():
        return dict(available=False,reason='No valid paired GT/predicted camera pose'),dict(available=False)
    if not valid[0]:
        return dict(available=False,reason='Sequence first frame has no valid paired camera pose; no replacement anchor silently used'),dict(available=False)
    gp, pp = np.zeros_like(gc),np.zeros_like(pc)
    gp[gt_pose_valid] = np.linalg.inv(gc[0]) @ gc[gt_pose_valid]
    pp[pred_pose_valid] = np.linalg.inv(pc[0]) @ pc[pred_pose_valid]
    delta = pp[valid,:3,3] - gp[valid,:3,3]
    trans = np.linalg.norm(delta,axis=-1)*1000.
    rotations = pp[valid,:3,:3] @ np.swapaxes(gp[valid,:3,:3],-1,-2)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotations,axis1=-2,axis2=-1)-1)/2,-1,1)))
    cam = dict(available=True,frames=len(gt),valid_frames=int(valid.sum()),
        gt_valid_frames=int(gt_pose_valid.sum()), missing_predicted_pose_frames=int((gt_pose_valid & ~pred_pose_valid).sum()),anchor_frame_index=0,
        alignment='Each c2w trajectory expressed in its own first-camera frame; no fitted rotation, translation or scale.',
        translation_error_mm=_stats(trans),rotation_error_deg=_stats(angle),
        reference_frame_included=True, scale_alignment=False, sparse_BA=False)
    paired = valid[:-1] & valid[1:]
    if paired.any():
        indices=np.flatnonzero(paired)
        gd=np.linalg.inv(gp[indices])@gp[indices+1]
        pd=np.linalg.inv(pp[indices])@pp[indices+1]
        rd=np.linalg.inv(gd)@pd
        cam['relative_translation_error_mm']=_stats(np.linalg.norm(rd[:,:3,3],axis=-1)*1000.)
        cam['relative_rotation_error_deg']=_stats(np.rad2deg(np.arccos(np.clip((np.trace(rd[:,:3,:3],axis1=-2,axis2=-1)-1)/2,-1,1))))
        cam['relative_pair_definition']='Adjacent stored frames, not a fixed time interval; no velocity normalization.'
    gj,pj=_transform_joints(gt,gp),_transform_joints(pred,pp)
    wg = gt_valid & gt_pose_valid[:,None]
    wr,matched,points=_geometry(pj,gj,wg,pred_valid & pred_pose_valid[:,None],point_valid)
    wr.pop('wrist_depth_error_matched_mm')  # A world z coordinate is not camera depth.
    wr.update(available=True,frame='First camera',no_scale_alignment=True,no_BA=True,
        w_mpjpe_matched_mm=wr['mpjpe_matched_mm'],w_pa_mpjpe_matched_mm=wr['pa_mpjpe_matched_mm'],
        w_root_error_matched_mm=wr['root_error_matched_mm'],
        metric_note='W-MPJPE uses estimated camera plus predicted camera-space joints. Per-hand W-PA removes placement errors.')
    fit_points=matched[...,None]&points
    sim=_sim3(pj[fit_points],gj[fit_points]) if fit_points.any() else None
    wr['wa_mpjpe_matched_mm']=None
    if sim is not None:
        scale,r,t=sim
        aligned=scale*pj@r+t
        per_hand=[np.linalg.norm(aligned[i,s,points[i,s]]-gj[i,s,points[i,s]],axis=-1).mean()*1000.
                  for i,s in np.argwhere(matched)]
        wr['wa_mpjpe_matched_mm']=_mean(per_hand)
        wr['wa_alignment']=dict(type='One diagnostic Sim(3) fitted to all matched joints across the entire sequence, both hands together',scale=scale)
    return cam,wr


def compute_sequence_metrics(arrays):
    """Return JSON-safe diagnostics from the arrays contract in METRICS_INTERFACE.md."""
    a=arrays
    gt,pred,control=[np.asarray(a[k],np.float64) for k in ('gt_joints','pred_joints','control_joints')]
    if gt.shape != pred.shape or gt.shape != control.shape or gt.ndim!=4 or gt.shape[1:]!=(2,21,3):
        raise ValueError('Joint arrays must all have shape [T,2,21,3]')
    gv,pv,cv=[np.asarray(a[k],bool) for k in ('gt_valid','pred_valid','control_valid')]
    points=np.asarray(a['point_valid'],bool)
    if gv.shape!=gt.shape[:2] or pv.shape!=gv.shape or cv.shape!=gv.shape or points.shape!=gt.shape[:-1]:
        raise ValueError('Hand-valid [T,2] and point-valid [T,2,21] shapes must match joints')
    masks,detection=_mask_and_detection(a)
    main,matched,_=_geometry(pred,gt,gv,pv,points)
    ctrl,ctrl_matched,_=_geometry(control,gt,gv,cv,points)
    common=matched&ctrl_matched
    pred_common,_,_=_geometry(pred,gt,gv,common,points)
    control_common,_,_=_geometry(control,gt,gv,common,points)
    camera,world=_camera_world(a,gt,pred,control,gv,pv,cv,points)
    return dict(schema='infinihand_four_scene_diagnostics_v1',frames=len(gt),hand_slots=['right','left'],
        units='Input joints/camera translation: metres. Reported geometry: millimetres. Ratios: [0,1].',
        protocol='Training-checkpoint diagnostics; same-side presence and conditional geometry. Not paper Table 1 matching/penalty protocol.',
        masks=masks,detection=detection,camera_space=main,
        gt_bbox_control=ctrl,common_hand_comparison=dict(common_hands=int(common.sum()),
            pred_bbox=pred_common,gt_bbox=control_common),camera_trajectory=camera,world_space=world)
