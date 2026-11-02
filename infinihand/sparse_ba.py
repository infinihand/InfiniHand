"""Paper Appendix C: chronological binary keyframe pool and c2w refinement.

This module has no CUDA/DROID imports. Optimization is injected as a callable.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Entry:
    frame_id: int
    retained: bool = False


class BinaryKeyframePool:
    """Select first, optimize successfully, then toggle and evict (Eq. 34/35)."""
    def __init__(self, capacity=160, stride=20):
        if not isinstance(capacity, int) or not isinstance(stride, int) or stride < 1 or capacity < stride:
            raise ValueError('Require integer capacity >= stride >= 1 (at least two BA frames at trigger)')
        self.capacity, self.stride = capacity, stride
        self.entries = []
        self.last_id = -1

    def append(self, frame_id):
        if frame_id <= self.last_id:
            raise ValueError('Frames must arrive once in strictly increasing temporal order')
        self.entries.append(Entry(int(frame_id)))
        self.last_id = int(frame_id)
        return len(self.entries) > self.capacity

    def selected(self):
        return [x.frame_id for x in self.entries[::self.stride]]

    def commit(self, selected):
        if list(selected) != self.selected():
            raise ValueError('Selection changed before commit')
        selected = set(selected)
        for entry in self.entries:
            if entry.frame_id in selected:
                entry.retained = not entry.retained
        self.entries = [x for x in self.entries if x.retained]

    def state(self):
        return [[x.frame_id, int(x.retained)] for x in self.entries]


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


class SparseBinaryBA:
    """Incremental pool; the optimizer receives selected IDs and current c2w poses.

    No frames/poses are popped until a successful finite BA result. The first
    selected seed fixes the world gauge, so repeated calls do not reset the world.
    """
    def __init__(self, optimizer, capacity=160, stride=20):
        self.pool = BinaryKeyframePool(capacity, stride)
        self.optimizer = optimizer
        self.raw, self.poses, self.events = [], [], []
        self.carry = np.eye(4)

    def append(self, pose):
        pose = validate_poses(np.asarray(pose)[None])[0]
        self.raw.append(pose.copy())
        self.poses.append(self.carry @ pose)
        if self.pool.append(len(self.raw)-1):
            self.refine('capacity')

    def refine(self, reason='flush'):
        ids = self.pool.selected()
        if len(ids) < 2:
            return False
        before = self.pool.state()
        seed = np.asarray(self.poses)[ids].copy()
        refined, diagnostics = self.optimizer(ids, seed.copy())
        refined = validate_poses(refined)
        if len(refined) != len(ids):
            raise ValueError('Backend returned wrong keyframe count')
        # A monocular backend has a free rigid gauge. Preserve the input world.
        refined = seed[0] @ np.linalg.inv(refined[0]) @ refined
        start = ids[0]
        updated = apply_pose_corrections(np.asarray(self.poses)[start:], np.asarray(ids)-start, refined)
        self.poses[start:] = list(updated)
        self.carry = self.poses[-1] @ np.linalg.inv(self.raw[-1])
        self.pool.commit(ids)
        self.events.append(dict(reason=reason,at_frame=len(self.raw)-1,selected=ids,
                                pool_before=before,pool_after=self.pool.state(),backend=diagnostics))
        return True

    def finish(self, flush=True):
        # Optional end-of-video flush; uses the same pool-index stride and toggle.
        if flush:
            self.refine('flush')
        return validate_poses(np.asarray(self.poses))
