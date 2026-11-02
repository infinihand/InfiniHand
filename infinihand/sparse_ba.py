"""Camera-pose validation, dense correction interpolation, and BA update rejection."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def validate_poses(poses):
    a = np.asarray(poses, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or not np.isfinite(a).all():
        raise ValueError('Expected finite c2w [T,4,4]')
    if not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous pose row')
    if not np.allclose(a[:, :3, :3].transpose(0, 2, 1) @ a[:, :3, :3], np.eye(3), atol=2e-4):
        raise ValueError('Non-orthogonal camera rotation')
    if not np.allclose(np.linalg.det(a[:, :3, :3]), 1, atol=2e-4):
        raise ValueError('Camera rotation must be proper')
    return a


def apply_pose_corrections(poses, ids, refined):
    """Interpolate left SE(3) corrections, preserving dense predicted motion.

    Before/after selected timestamps the endpoint correction is held constant.
    This is an implementation choice for non-keyframes, not specified by Eq.34/35.
    """
    poses = validate_poses(poses)
    refined = validate_poses(refined)
    ids = np.asarray(ids, dtype=int)
    if len(ids) < 2 or len(refined) != len(ids) or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= len(poses):
        raise ValueError('Need >=2 distinct ordered in-range keyframes')
    correction = refined @ np.linalg.inv(poses[ids])
    correction[:, :3, :3] = Rotation.from_matrix(correction[:, :3, :3]).as_matrix()
    correction[:, :3, 3] = refined[:, :3, 3] - np.einsum('nij,nj->ni', correction[:, :3, :3], poses[ids, :3, 3])
    t = np.clip(np.arange(len(poses)), ids[0], ids[-1])
    delta = np.tile(np.eye(4), (len(poses), 1, 1))
    delta[:, :3, :3] = Slerp(ids, Rotation.from_matrix(correction[:, :3, :3]))(t).as_matrix()
    for axis in range(3):
        delta[:, axis, 3] = np.interp(t, ids, correction[:, axis, 3])
    result = delta @ poses
    if not np.allclose(result[ids], refined, atol=2e-5):
        raise RuntimeError('Dense correction does not reproduce refined keyframes')
    return validate_poses(result)


def reject_bad_updates(seeds, refined, weight_history=()):
    """Reject implausible BA updates without ground truth; not an accuracy guarantee."""
    seeds, refined = validate_poses(seeds), validate_poses(refined)
    if seeds.shape != refined.shape:
        raise ValueError('Seed and refined pose shapes must match')
    refined = seeds[0] @ np.linalg.inv(refined[0]) @ refined
    distance = np.linalg.norm(refined[:, :3, 3] - seeds[:, :3, 3], axis=-1)
    angle = np.degrees(Rotation.from_matrix(
        seeds[:, :3, :3].transpose(0, 2, 1) @ refined[:, :3, :3]).magnitude())
    translation_limit = max(.1, 5 * float(np.median(distance)))
    rotation_limit = max(10., 5 * float(np.median(angle)))
    pose_bad = (distance > translation_limit) | (angle > rotation_limit)
    low_confidence = np.zeros(len(seeds), dtype=bool)
    weights = np.asarray(weight_history, dtype=float)
    if weights.size:
        if weights.ndim != 2 or weights.shape[1] != len(seeds) or not np.isfinite(weights).all():
            raise ValueError('Expected finite per-iteration keyframe weights')
        ratios = weights[:3] / np.maximum(np.median(weights[:3], axis=1, keepdims=True), 1e-9)
        low_confidence = (ratios < .01).sum(axis=0) >= 2
    bad = pose_bad | low_confidence
    bad[0] = False  # The first input pose defines the fixed world gauge.
    result = refined.copy()
    result[bad] = seeds[bad]
    return result, dict(rejected_indices=np.flatnonzero(bad).tolist(),
        pose_rejected_indices=np.flatnonzero(pose_bad).tolist(),
        low_confidence_indices=np.flatnonzero(low_confidence).tolist(),
        translation_limit_m=translation_limit, rotation_limit_deg=rotation_limit,
        translation_updates_m=distance.tolist(), rotation_updates_deg=angle.tolist())
