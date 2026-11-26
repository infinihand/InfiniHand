"""Validated compact exports; file slots are left/right, model slots right/left."""
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_SIDES = ("left", "right")


def validate_metadata(metadata, scene_name):
    if not (str(metadata.get("schema_version", "")).startswith("compact_virtual_camera_") or (metadata.get("schema_version") == "compact_camera_mano_joints_v9_20261012" and metadata.get("mano_basis_policy") == "egohandbench_right_mirror_v3_to_native_B")):
        raise ValueError(f"{scene_name}: expected prepared virtual-camera metadata")
    if tuple(metadata.get("side_order", ())) != SOURCE_SIDES:
        raise ValueError(f"{scene_name}: source slots must be left, right")
    if metadata.get("units", {}).get("3d") != "meters":
        raise ValueError(f"{scene_name}: 3D units must be meters")
    frames = metadata.get("frames", [])
    if not frames or int(metadata.get("frame_count", -1)) != len(frames):
        raise ValueError(f"{scene_name}: inconsistent frame count")
    for frame in frames:
        hw = frame.get("image_size_hw", [])
        if len(hw) != 2 or min(hw) <= 0:
            raise ValueError("Invalid image_size_hw")
        K = np.asarray(frame.get("intrinsics"), dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        for side in SOURCE_SIDES:
            if not isinstance(frame["hands"][side].get("hand_valid"), bool):
                raise ValueError("hand_valid must be an explicit boolean")
        if not isinstance(frame.get("camera_valid"), bool):
            raise ValueError("camera_valid must be an explicit boolean")
        if frame["camera_valid"]:
            C = np.asarray(frame.get("camera_to_world"), dtype=np.float64)
            if C.shape != (4, 4) or not np.isfinite(C).all():
                raise ValueError("Valid camera requires finite camera_to_world")
            R = C[:3, :3]
            if not np.allclose(C[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(R @ R.T, np.eye(3), atol=5e-4) or not np.isclose(np.linalg.det(R), 1, atol=5e-4):
                raise ValueError("camera_to_world must be a rigid SE(3) transform")
            if "world_to_camera" in frame:
                W = np.asarray(frame["world_to_camera"], dtype=np.float64)
                if W.shape != (4,4) or not np.allclose(W @ C, np.eye(4), atol=5e-4):
                    raise ValueError("camera_to_world/world_to_camera are inconsistent")
            else:
                frame["world_to_camera"] = np.linalg.inv(C).tolist()


def geometry(dataset, scene, frame, *, include_world=True):
    mano = dataset._read_npz_remote(dataset._absolute_frame_path(scene, frame, "camera_mano"))
    joints = dataset._read_npz_remote(dataset._absolute_frame_path(scene, frame, "camera_joints"))
    p = np.asarray(mano["params_61"], dtype=np.float32).copy()
    j = np.asarray(joints["joints"], dtype=np.float32).copy()
    if p.shape != (2, 61) or j.shape != (2, 21, 3):
        raise ValueError("Expected MANO [2,61] and decoded joints [2,21,3]")
    valid = np.array([frame["hands"][side]["hand_valid"] for side in SOURCE_SIDES], bool)
    if not np.isfinite(p[valid]).all() or not np.isfinite(j[valid]).all():
        raise ValueError("Valid hand has non-finite MANO/joints")
    if not np.allclose(p[valid, 58:61], j[valid, 0], atol=1e-4):
        raise ValueError("MANO translation must equal decoded joint-0 root position")
    # Invalid annotations can contain NaNs; keep them out of all geometric math.
    p[~valid] = 0
    j[~valid] = 0
    jvalid = np.repeat(valid[:, None], 21, axis=1)
    cm = {"params_61": p, "joints": j, "mano_valid": valid}
    cj = {"joints": j, "joints_valid": jvalid}
    wm = wj = None
    if include_world and frame.get("camera_valid", False):
        C = np.asarray(frame["camera_to_world"], np.float64)
        wp = p.copy()
        wp[:, :3] = Rotation.from_matrix(C[:3, :3][None] @ Rotation.from_rotvec(p[:, :3]).as_matrix()).as_rotvec()
        wp[:, 58:61] = p[:, 58:61] @ C[:3, :3].T + C[:3, 3]
        world = (j @ C[:3, :3].T + C[:3, 3]).astype(np.float32)
        wp[~valid] = 0
        world[~valid] = 0
        wm = {"params_61": wp, "joints": world, "mano_valid": valid}
        wj = {"joints": world, "joints_valid": jvalid}
    return cm, cj, wm, wj


def annotations(dataset, scene, frame, image_hw):
    masks, boxes, valid = [], [], []
    for side in SOURCE_SIDES:
        mask = dataset._read_image_remote(dataset._absolute_frame_path(scene, frame, "masks", side))
        if mask.ndim == 3:
            if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(mask[..., 0], mask[..., 2]):
                raise ValueError("Mask must be grayscale, not a color visualization")
            mask = mask[..., 0]
        if tuple(mask.shape) != tuple(image_hw):
            raise ValueError("Compact mask/image dimensions mismatch")
        mask = mask > 0
        is_valid = frame["hands"][side]["hand_valid"]
        ys, xs = np.nonzero(mask)
        if is_valid and not len(xs):
            raise ValueError("hand_valid=true with an empty projected mask")
        boxes.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1] if len(xs) else [0, 0, 0, 0])
        masks.append(mask)
        valid.append(is_valid)
    perm = dataset._source_to_canonical
    return (perm(np.asarray(masks)), perm(np.asarray(boxes, np.float32)),
            np.ones(2, bool), np.ones(2, bool), perm(np.asarray(valid, bool)))
