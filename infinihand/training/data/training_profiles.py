"""Task-specific training I/O; perspective crops are owned by the model."""
from collections import defaultdict

import numpy as np
import torch


TASKS = {'full', 'mask', 'mano', 'camera', 'joint'}


def training_task(cfg):
    if cfg.train_camera_head_only or cfg.train_camera_scale_head_only:
        return 'camera'
    if cfg.train_probability_mask_heads_only or cfg.train_detection_modules_only:
        return 'mask'
    if cfg.train_mano_modules_only:
        return 'mano'
    return 'joint'


def frame_time(frame):
    for name, scale in (('timestamp_seconds', 1.), ('timestamp_s', 1.),
                        ('timestamp_ns', 1e-9), ('timestamp_us', 1e-6)):
        if frame.get(name) is not None:
            value = float(frame[name]) * scale
            return (value, True) if np.isfinite(value) else (0., False)
    source = frame.get('source_paths', {})
    if source.get('timestamp_ns') is not None:
        value = float(source['timestamp_ns']) * 1e-9
    elif source.get('source_fps') and float(source['source_fps']) > 0:
        value = float(frame['frame_id']) / float(source['source_fps'])
    else:
        return 0., False
    return (value, True) if np.isfinite(value) else (0., False)


def load_training_sample(dataset, scene, ids):
    """Read only this task's objects, preserving image-camera labels and slots."""
    from .datasets.compact import annotations, geometry
    task = dataset.annotation_task
    need_mano = task in {'mano', 'joint'}
    need_masks = task in {'mask', 'joint'}
    need_camera = task in {'camera', 'joint'}
    values = defaultdict(list)
    target_hw = (int(dataset.cfg.get('lingbotmap_image_height', 378)),
                 int(dataset.cfg.get('lingbotmap_image_width', 518)))
    from .mano_basis_policy import resolve_basis
    basis = resolve_basis(dataset.dataset_name, scene['metadata'], task)
    frames = scene['metadata']['frames']
    for index in np.asarray(ids, dtype=np.int64):
        frame = frames[int(index)]
        image = dataset._read_image_remote(dataset._absolute_image_path(scene, frame))
        if tuple(image.shape[:2]) != tuple(frame['image_size_hw']):
            raise ValueError('Prepared RGB, metadata and model dimensions must agree')
        values['images'].append(image.astype(np.uint8, copy=False))
        values['intrinsics'].append(dataset._frame_intrinsic(frame))
        hand_valid = np.array([frame['hands'][s]['hand_valid'] for s in ('right', 'left')], bool)
        values['hand_valid'].append(hand_valid)
        values['source_frame_ids'].append(int(frame['frame_id']))
        timestamp, time_valid = frame_time(frame)
        values['timestamps_seconds'].append(timestamp)
        values['timestamps_valid'].append(time_valid)
        if need_camera:
            camera_valid = bool(frame['camera_valid']) and dataset._camera_trajectory_enabled()
            values['camera_trajectory_valid'].append(camera_valid)
            values['camera_poses'].append(np.asarray(frame['camera_to_world'], np.float32)
                                          if camera_valid else np.eye(4, dtype=np.float32))
        if need_masks:
            masks, boxes, _, _, _ = annotations(dataset, scene, frame, tuple(image.shape[:2]))
            values['hand_masks'].append(masks.astype(bool))
            values['hand_detections'].append(boxes)
            flags = []
            for side in ('right', 'left'):
                hand = frame['hands'][side]
                flags.append(bool(hand.get('mask_supervision_valid',
                    hand['hand_valid'] or hand.get('invalid_reason') in (
                        'insufficient_in_image_projection', 'no_projected_area', 'confirmed_absent'))))
            values['hand_mask_supervision_valid'].append(np.asarray(flags, bool))
        elif need_mano:
            # The exporter computes these exact pixel-inclusive boxes from the
            # saved projected masks. No mask download/decode is needed for MANO.
            boxes = []
            for side in ('right', 'left'):
                hand = frame['hands'][side]
                box = hand.get('bbox_xyxy')
                if box is None:
                    if hand['hand_valid']:
                        raise ValueError('Valid hand requires its exported mask bbox')
                    box = [0, 0, 0, 0]
                box = np.asarray(box, np.float32)
                if box.shape != (4,) or not np.isfinite(box).all():
                    raise ValueError('Invalid exported bbox_xyxy')
                boxes.append(box)
            values['hand_detections'].append(np.asarray(boxes, np.float32))
        if need_mano:
            cm, cj, _, _ = geometry(dataset, scene, frame, include_world=False)
            values['camera_mano'].append(dataset._source_to_canonical(cm['params_61']))
            values['camera_mano_joints'].append(dataset._source_to_canonical(cj['joints']))
            values['mano_valid'].append(dataset._source_to_canonical(cm['mano_valid']))
            values['hand_points_valid'].append(dataset._source_to_canonical(cj['joints_valid']))
            values['mano_left_basis_b'].append(basis == 'B')
            if task == 'joint':
                values['world_geometry_valid'].append(camera_valid and dataset._world_geometry_enabled())
    result = {name: np.stack(items) for name, items in values.items()}
    result = resize_training_pixels(result, target_hw)
    result.update(seq_name=scene['scene_name'], ids=np.asarray(ids, np.int64),
                  frame_num=int(scene['frame_count']), dataset_name=dataset.dataset_name,
                  sampling_name=dataset.sampling_name, annotation_task=task)
    return result


def tensorize_training_sample(batch):
    """Keep RGB uint8 and binary masks bool through pinned-memory/H2D transfer."""
    result = {
        'seq_name': batch['seq_name'], 'dataset_name': batch['dataset_name'],
        'dataset_sampling_name': batch.get('sampling_name', batch['dataset_name']),
        'frame_num': torch.tensor(batch['frame_num'], dtype=torch.long),
    }
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            if key == 'images':
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            result[key] = tensor
    return result


def build_target(batch, cfg):
    """3D targets stay in image-camera coordinates; actual crop H is applied in loss."""
    images = batch['images']
    height, width = images.shape[-2:]
    device = images.device
    target = {'hand_valid': batch['hand_valid'].bool(),
              'intrinsics': batch['intrinsics'].float(),
              'image_hw': torch.tensor([height, width], device=device, dtype=torch.float32),
              'frame_ids': batch['ids'].long()}
    for key in ('timestamps_seconds', 'timestamps_valid'):
        if key in batch:
            target[key] = batch[key]
    if 'hand_detections' in batch:
        detections = batch['hand_detections'].float()
        target['hand_boxes'] = (detections / detections.new_tensor([width, height, width, height])).clamp(0, 1)
    if 'hand_masks' in batch:
        target['hand_masks'] = batch['hand_masks']  # loss casts once to prediction dtype
        target['hand_mask_valid'] = batch['hand_mask_supervision_valid'].bool()
    if 'camera_mano' in batch:
        valid = batch['hand_valid'].bool() & batch['mano_valid'].bool()
        joints = batch['camera_mano_joints'].float()
        point_valid = (torch.isfinite(joints).all(-1) & (joints[..., 2] > 1e-6)
                       & valid[..., None] & batch['hand_points_valid'].bool())
        target.update(mano_params=batch['camera_mano'].float(), mano_valid=valid,
                      keypoints_3d=joints, keypoints_3d_joint_valid=point_valid,
                      keypoints_3d_valid=valid & point_valid.all(-1))
        projected = (target['intrinsics'][:, :, None, None] @ joints[..., None]).squeeze(-1)
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
        target['keypoints_2d_image'] = torch.cat((uv / uv.new_tensor([width, height]),
                                                  point_valid[..., None].to(uv)), -1)
    world_enabled = cfg.compute_stage2_diagnostics or any(
        w != 0 for w in (cfg.world_joints_loss_weight, cfg.world_mano_loss_weight))
    pose_enabled = cfg.compute_stage2_diagnostics or any(
        w != 0 for w in (cfg.abs_pose_loss_weight, cfg.rel_pose_loss_weight))
    if world_enabled or pose_enabled:
        target['camera_poses'] = batch['camera_poses'].float()
        target['camera_trajectory_valid'] = batch['camera_trajectory_valid'].bool()
        camera = target['camera_poses']
        zero_available = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
        world_available = zero_available
        if world_enabled:
            target['world_geometry_valid'] = batch['world_geometry_valid'].bool()
            target['world_joints'] = (camera[:, :, None, None, :3, :3] @ joints[..., None]).squeeze(-1) + camera[:, :, None, None, :3, 3]
            target['world_joints_valid'] = point_valid
            world_available = (point_valid & target['world_geometry_valid'][..., None, None]).flatten(1).any(1)
            if cfg.compute_stage2_diagnostics or cfg.world_mano_loss_weight != 0:
                from scipy.spatial.transform import Rotation
                world_mano = target['mano_params'].clone()
                # Optional legacy diagnostics only; current joint training has
                # world_mano_loss_weight=0 and never performs this conversion.
                rotations = Rotation.from_rotvec(world_mano[..., :3].detach().cpu().numpy().reshape(-1, 3)).as_matrix()
                camera_rotations = camera[:, :, None, :3, :3].expand(*world_mano.shape[:-1], 3, 3).detach().cpu().numpy().reshape(-1, 3, 3)
                orient = Rotation.from_matrix(camera_rotations @ rotations).as_rotvec()
                world_mano[..., :3] = torch.as_tensor(orient, device=device, dtype=torch.float32).reshape(*world_mano.shape[:-1], 3)
                world_mano[..., 58:61] = (camera[:, :, None, :3, :3] @ world_mano[..., 58:61, None]).squeeze(-1) + camera[:, :, None, :3, 3]
                target.update(world_mano=world_mano, world_mano_valid=valid)
        pose_available = target['camera_trajectory_valid'].sum(1) >= (2 if cfg.rel_pose_loss_weight or cfg.compute_stage2_diagnostics else 1)
        target.update(stage2_world_joints_available=world_available,
                      stage2_world_mano_available=world_available if 'world_mano' in target else zero_available,
                      stage2_pose_available=pose_available,
                      stage2_supervision_valid=world_available | (pose_available if pose_enabled else zero_available))
    return target, None


def resize_training_pixels(result, target_hw):
    """Letterbox native RGB and masks; apply identical pixel affine to K/bboxes.

    Pixel-center mapping follows OpenCV resize. Camera-space geometry is unchanged.
    """
    import cv2
    images = result['images']; h,w = images.shape[1:3]; th,tw = target_hw
    if (h,w)==(th,tw): return result
    scale=min(tw/w, th/h); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    left=(tw-nw)//2; top=(th-nh)//2; sx=nw/w; sy=nh/h
    A=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1]],np.float32)
    def resize(a, nearest=False):
        dst=np.zeros((th,tw)+a.shape[2:],dtype=a.dtype)
        dst[top:top+nh,left:left+nw]=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR)
        return dst
    result['images']=np.stack([resize(im) for im in images])
    result['intrinsics']=A[None]@result['intrinsics']
    if 'hand_masks' in result:
        result['hand_masks']=np.stack([np.stack([resize(m.astype(np.uint8),True)>0 for m in pair]) for pair in result['hand_masks']])
    if 'hand_detections' in result:
        boxes=result['hand_detections'].copy(); nonempty=(boxes[...,2]>boxes[...,0])&(boxes[...,3]>boxes[...,1])
        boxes[...,0::2]=boxes[...,0::2]*sx+left
        boxes[...,1::2]=boxes[...,1::2]*sy+top
        boxes[~nonempty]=0
        if 'hand_masks' in result:
            for t,pair in enumerate(result['hand_masks']):
                for hand,m in enumerate(pair):
                    ys,xs=np.nonzero(m); boxes[t,hand]=[xs.min(),ys.min(),xs.max()+1,ys.max()+1] if len(xs) else 0
        result['hand_detections']=boxes
    return result
