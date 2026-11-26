import torch

def _invert_se3(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"SE(3) tensor must end in [4,4], got {transform.shape}")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse_rotation = rotation.transpose(-1, -2)
    inverse[..., :3, :3] = inverse_rotation
    inverse[..., :3, 3] = -torch.matmul(
        inverse_rotation, translation.unsqueeze(-1)
    ).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse

def _pose_encoding_to_w2c(pose_encoding: torch.Tensor) -> torch.Tensor:
    """Decode LingBotMap's [T, quat_xyzw, fov] camera encoding to W2C."""
    if pose_encoding.ndim != 3 or pose_encoding.shape[-1] != 9:
        raise ValueError(
            "pred_camera_pose_enc must be [B,T,9], got "
            f"{tuple(pose_encoding.shape)}"
        )
    translation = pose_encoding[..., :3]
    quaternion = pose_encoding[..., 3:7]
    i, j, k, r = quaternion.unbind(dim=-1)
    two_s = 2.0 / quaternion.square().sum(dim=-1).clamp_min(1.0e-8)
    rotation = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    w2c = pose_encoding.new_zeros(*pose_encoding.shape[:-1], 4, 4)
    w2c[..., :3, :3] = rotation
    w2c[..., :3, 3] = translation
    w2c[..., 3, 3] = 1.0
    return w2c
