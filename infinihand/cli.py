"""Video/dataset input and export CLI. Numerical inference lives in infer.py."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--dataset-manifest', type=Path, help='Prepared scene manifest; reads RGB and per-frame intrinsics via UnifiedHandDataset')
    p.add_argument('--scene', help='Exact scene name for dataset inference')
    p.add_argument('--dataset-fps', type=float, help='Processed-frame FPS fallback only when dataset timestamps are missing')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Complete weights exported from the current training runs')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/infinihand_v1.json')
    p.add_argument('--intrinsics', type=Path, help='JSON: fx,fy,cx,cy,image_width,image_height, in original video pixels')
    p.add_argument('--mask-source', choices=['base', 'sam3-tracking'], default='base')
    p.add_argument('--mask-threshold', type=float, default=0.9)
    p.add_argument('--sam3-python', default=sys.executable)
    p.add_argument('--sam3-source', type=Path, default=Path('/path/to/infinihand/third_party/sam3'))
    p.add_argument('--sam3-checkpoint', type=Path, default=Path('/path/to/weights/sam3.pt'))
    p.add_argument('--detector', type=Path, default=Path('/path/to/weights/detector.pt'))
    p.add_argument('--postprocess-storage', choices=['auto','cpu','cpu_async','cuda'], default='cpu', help='Auxiliary capture: GPU resident or asynchronous pinned CPU; auto bounds GPU mask storage')
    p.add_argument('--fused-rope', action='store_true', help='Use rounding-preserving Triton 2D RoPE')
    p.add_argument('--compact-kv-cache', action='store_true', help='Avoid append-then-evict KV copies without changing memory contents')
    p.add_argument('--mask-box-backend', choices=['torch','triton'], default='torch')
    p.add_argument('--frame-cache', choices=['png','memory'], default='png', help='Memory avoids PNG roundtrip; requires base masks and --no-ba')
    p.add_argument('--branch-precision', choices=['fp32', 'bf16'], default='fp32', help='Optional BF16 neural branches; geometry stays FP32/FP64, backbone uses BF16')
    p.add_argument('--cuda-graphs', action='store_true', help='Capture fixed-shape image encoders once for faster repeated inference; uses extra GPU memory')
    p.add_argument('--ba-python', default=sys.executable)
    p.add_argument('--droid-weights', type=Path, default=Path('/path/to/weights/droid.pth'))
    p.add_argument('--no-ba', action='store_true')
    p.add_argument('--no-postprocess', action='store_true', help='Disable offline fixed-beta, joint LSP depth and SG5 refinement')
    p.add_argument('--postprocess-lambda', type=float, default=0.5, help='FP weight in mask-guided depth score (default: 0.5)')
    p.add_argument('--postprocess-frame-batch', type=int, default=1024, help='Maximum frames per GPU LSP batch')
    p.add_argument('--postprocess-raster-batch', type=int, default=4096, help='Maximum packed-mask candidates rendered together; reduce for lower VRAM')
    p.add_argument('--ba-mode', choices=['fixed'], default='fixed', help='Fixed-stride keyframe BA')
    p.add_argument('--ba-stride', type=int, default=10)
    p.add_argument('--ba-depth-init', choices=['none', 'sparse'], default='none')
    p.add_argument('--ba-no-reject-outliers', action='store_true', help='Disable update rejection for ablations')
    p.add_argument('--max-frames', type=int, default=8192)
    return p

def main(inference):
    build_model = inference.build_model
    prepare_cuda_graphs = inference.prepare_cuda_graphs
    run_dual_stream = inference.run_dual_stream
    camera_to_predicted_global = inference.camera_to_predicted_global
    validate_contract = inference.validate_contract
    a = parser().parse_args()
    inference.BENCH_BRANCH_BF16 = a.branch_precision == "bf16"
    if a.postprocess_frame_batch < 1 or a.postprocess_raster_batch < 1:
        raise ValueError('Postprocessing batch sizes must be positive')
    if not np.isfinite(a.postprocess_lambda) or a.postprocess_lambda <= 0:
        raise ValueError('postprocess-lambda must be positive and finite')
    if not 0 < a.mask_threshold < 1:
        raise ValueError('mask threshold must be in (0,1)')
    if not 4 <= a.max_frames <= 8192:
        raise ValueError('max-frames must be between 4 and 8192')
    if a.video is not None and a.intrinsics is None:
        raise ValueError('Video inference requires --intrinsics')
    if a.dataset_manifest is not None and (not a.scene or a.intrinsics is not None):
        raise ValueError('Dataset inference requires --scene and reads intrinsics from metadata; omit --intrinsics')
    if a.video is not None and (a.scene is not None or a.dataset_fps is not None):
        raise ValueError('--scene and --dataset-fps are dataset-only options')
    required = [a.checkpoint, a.config] + ([a.video, a.intrinsics] if a.video is not None else [a.dataset_manifest])
    if a.ba_stride < 1:
        raise ValueError('Require ba-stride >= 1')
    if a.mask_source == 'sam3-tracking':
        required += [a.sam3_checkpoint, a.detector, a.sam3_source]
    if not a.no_ba:
        required += [a.droid_weights]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    a.output.mkdir(parents=True, exist_ok=True)
    D = a.output / 'frames'
    D.mkdir(exist_ok=True)
    if list(D.glob('*.png')):
        raise ValueError('Use a fresh output directory to avoid stale frame/mask mixing')
    if a.frame_cache == 'memory' and (a.mask_source != 'base' or not a.no_ba):
        raise ValueError('Memory frame cache requires --mask-source base --no-ba; SAM3/BA need original frames')
    network_frames = [] if a.frame_cache == 'memory' else None
    dataset_data = None
    if a.dataset_manifest is not None:
        from .inference_data import load_dataset_input
        dataset_data = load_dataset_input(a.dataset_manifest, a.scene, a.max_frames, a.dataset_fps)
        rgb, Ks = dataset_data['images'], dataset_data['intrinsics']
        n, h, w = rgb.shape[:3]
        fps = dataset_data['fps']
        timestamps = dataset_data['timestamps_seconds']
        if not a.no_ba and not np.allclose(Ks, Ks[:1], rtol=0, atol=1e-5):
            raise ValueError('Camera BA currently requires constant intrinsics; use --no-ba for this scene with varying intrinsics')
        K0 = Ks[0]
        Kd = dict(fx=float(K0[0,0]), fy=float(K0[1,1]), cx=float(K0[0,2]), cy=float(K0[1,2]), image_width=w, image_height=h)
        a.intrinsics = a.output / 'input_intrinsics.json'
        a.intrinsics.write_text(json.dumps(Kd, indent=2))
        for i, im in enumerate(rgb):
            if network_frames is not None:
                network_frames.append(cv2.resize(im[:,:,::-1], (518,378))[:,:,::-1])
            elif not cv2.imwrite(str(D / f'{i:06d}.png'), im[:,:,::-1]):
                raise IOError('Failed to save dataset RGB frame')
        np.savez_compressed(a.output / 'input_metadata.npz', intrinsics=Ks,
                            source_frame_ids=dataset_data['source_frame_ids'], timestamps_seconds=timestamps)
    else:
        Kd = json.loads(a.intrinsics.read_text())
        cap = cv2.VideoCapture(str(a.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError('Cannot open video or invalid FPS')
        n = 0
        while n < a.max_frames:
            (ok, im) = cap.read()
            if not ok:
                break
            (h, w) = im.shape[:2]
            if (w, h) != (Kd['image_width'], Kd['image_height']):
                raise ValueError('Video dimensions differ from intrinsics metadata; explicitly calibrate/rescale before inference')
            if network_frames is not None:
                network_frames.append(cv2.resize(im, (518,378))[:,:,::-1])
            else:
                cv2.imwrite(str(D / f'{n:06d}.png'), im)
            n += 1
        cap.release()
        if n < 4:
            raise ValueError('At least four frames required')
        if n > 8192:
            raise ValueError('Maximum supported temporal index is 8192')
        timestamps = np.arange(n, dtype=np.float64) / fps
    boxes = np.zeros((n, 2, 4), np.float32)
    valid = np.zeros((n, 2), bool)
    if a.mask_source == 'sam3-tracking':
        import yaml
        cfg = yaml.safe_load((ROOT / 'third_party/sam3_tracking/configs/default.yaml').read_text())
        cfg.update(output_fps=fps, write_visualization=False, wandb_mode='disabled')
        cf = a.output / 'sam3_config.yaml'
        cf.write_text(yaml.safe_dump(cfg))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'third_party/sam3_tracking/src')
        subprocess.run([a.sam3_python, '-m', 'sam3_tracking', 'frames', '--frames-dir', str(D.resolve()), '--output-dir', str((a.output / 'sam3').resolve()), '--config', str(cf.resolve()), '--detector-weight', str(a.detector.resolve()), '--sam3-checkpoint', str(a.sam3_checkpoint.resolve()), '--sam3-src', str(a.sam3_source.resolve())], env=env, check=True)
        records = [json.loads(line) for line in (a.output / 'sam3/masks.jsonl').read_text().splitlines()]
        assert len(records) == n * 2
        seen = set()
        for row in records:
            i = int(row['frame_idx'])
            s = {'right': 0, 'left': 1}[row['side_name']]
            assert 0 <= i < n and (i, s) not in seen
            seen.add((i, s))
            if row['mask_box_xyxy'] is not None and row['mask_area'] > 0:
                boxes[i, s] = np.array(row['mask_box_xyxy']) / [w, h, w, h]
                valid[i, s] = True
    (model, load_seconds) = build_model(a.checkpoint, torch.device('cuda:0'), a.config)
    images = np.stack(network_frames if network_frames is not None else [cv2.resize(cv2.imread(str(D / f'{i:06d}.png')), (518, 378))[:, :, ::-1] for i in range(n)])
    del network_frames
    images = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).cuda()
    if dataset_data is None:
        Ks = np.broadcast_to(np.array([[Kd['fx'], 0, Kd['cx']], [0, Kd['fy'], Kd['cy']], [0, 0, 1]], np.float32), (n,3,3)).copy()
    else:
        Ks = dataset_data['intrinsics'].copy()
    Ks[:,0] *= 518 / w
    Ks[:,1] *= 378 / h
    intr = torch.from_numpy(Ks).unsqueeze(0).cuda()
    storage = a.postprocess_storage
    if storage == 'auto':
        free_bytes, _ = torch.cuda.mem_get_info()
        storage = 'cuda' if n*2*378*518 < min(2*1024**3, free_bytes//8) else 'cpu_async'
    opts = dict(postprocess_storage=storage, bbox_source='gt' if a.mask_source == 'sam3-tracking' else 'mask', mask_threshold=a.mask_threshold, capture_postprocess=not a.no_postprocess)
    if a.mask_source == 'sam3-tracking':
        opts.update(gt_hand_boxes=torch.from_numpy(boxes)[None].cuda(), gt_hand_valid=torch.from_numpy(valid)[None].cuda())
    from .runtime.execution import inference_optimizations
    with inference_optimizations(model, inference, fused_rope=a.fused_rope,
                                compact_cache=a.compact_kv_cache, mask_backend=a.mask_box_backend) as runtime_options:
        graph_setup = prepare_cuda_graphs(model) if a.cuda_graphs else 0.0
        with torch.inference_mode():
            (out, stats, timing) = run_dual_stream(model, images, intr, **opts)
            (world, _) = camera_to_predicted_global(out)
    validate_contract(stats, n)
    auxiliary = {'pred_keypoints_2d', 'mano_crop_intrinsics', 'mano_virtual_rotations', 'lsp_valid_mask', 'postprocess_masks'}
    arr = {k: v.cpu().numpy() if v.dtype == torch.bool else v.float().cpu().numpy() for (k, v) in out.items() if storage != 'cuda' or k not in auxiliary}
    poses = world['pred_camera_to_anchor'].float().cpu().numpy()
    if a.mask_source == 'base':
        boxes = arr['mask_tight_boxes']
        valid = arr['presence_mask'] > 0.5
    for (key, value) in arr.items():
        if not np.isfinite(value).all():
            raise ValueError('Non-finite model output: ' + key)
    auxiliary = {'pred_keypoints_2d', 'mano_crop_intrinsics', 'mano_virtual_rotations', 'lsp_valid_mask', 'postprocess_masks'}
    post_report = dict(enabled=False)
    if not a.no_postprocess:
        from .postprocess import refine_hands
        masks = out['postprocess_masks'] if storage == 'cuda' else arr.pop('postprocess_masks')
        if a.mask_source == 'sam3-tracking':
            masks = np.zeros((n, 2, 378, 518), dtype=bool)
            for row in records:
                i = int(row['frame_idx'])
                side = {'right': 0, 'left': 1}[row['side_name']]
                mask_path = a.output / 'sam3' / row['mask_relpath']
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(mask_path)
                masks[i, side] = cv2.resize(mask, (518, 378), interpolation=cv2.INTER_NEAREST) > 0
        raw = {k: v for k, v in arr.items() if k not in auxiliary}
        np.savez_compressed(a.output / 'camera_space_raw.npz', **raw, valid=valid)
        mano_path = (ROOT / json.loads(a.config.read_text())['mano_model_path']).resolve()
        if storage == 'cuda':
            from .postprocess import refine_hands_tensors
            native = {k: v for k,v in out.items() if k != 'postprocess_masks'}
            native['valid'] = out['presence_mask'] > 0.5
            masks = torch.as_tensor(masks, device=images.device, dtype=torch.bool)
            arr, post_report = refine_hands_tensors(native, intr[0], masks,
                                        timestamps, mano_path,
                                        lambda_fp=a.postprocess_lambda,
                                        frame_batch=a.postprocess_frame_batch,
                                        raster_batch=a.postprocess_raster_batch,
                                        return_tensors=False)
        else:
            arr, post_report = refine_hands(dict(arr, valid=valid), Ks, masks,
                                        timestamps, mano_path,
                                        lambda_fp=a.postprocess_lambda,
                                        frame_batch=a.postprocess_frame_batch,
                                        raster_batch=a.postprocess_raster_batch)
        post_report['mask_source'] = a.mask_source
        (a.output / 'POSTPROCESS.json').write_text(json.dumps(post_report, indent=2))
        arr.pop('valid')
        arr = {k: v for k, v in arr.items() if k not in auxiliary}
        del masks
        for key, value in arr.items():
            if not np.isfinite(value).all():
                raise ValueError('Non-finite postprocessed output: ' + key)
    np.savez_compressed(a.output / 'camera_space.npz', **arr, valid=valid)
    np.savez_compressed(a.output / 'ba_inputs.npz', poses=poses, boxes=boxes)
    del out, world, model, images, intr
    torch.cuda.empty_cache()
    if not a.no_ba:
        subprocess.run([a.ba_python, '-m', 'infinihand.sparse_ba_backend', '--output', str(a.output.resolve()), '--intrinsics', str(a.intrinsics.resolve()), '--weights', str(a.droid_weights.resolve()), '--stride', str(a.ba_stride), '--depth-init', a.ba_depth_init] + (['--no-reject-outliers'] if a.ba_no_reject_outliers else []), cwd=ROOT, check=True)
        poses = np.load(a.output / 'trajectory.npz')['poses']
    if not np.isfinite(poses).all():
        raise ValueError('Non-finite trajectory')
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    np.savez_compressed(a.output / 'world_space.npz', camera_to_world=poses, world_joints=np.einsum('tij,thnj->thni', rot, arr['camera_joints']) + trans[:, None, None, :], world_vertices=np.einsum('tij,thnj->thni', rot, arr['camera_vertices']) + trans[:, None, None, :], world_global_orient=rot[:, None] @ arr['camera_global_orient'], world_translation=np.einsum('tij,thj->thi', rot, arr['camera_translation']) + trans[:, None, :], hand_pose=arr['hand_pose'], betas=arr['betas'], valid=valid)
    receipt = dict(branch_precision=a.branch_precision, runtime_options=runtime_options, frame_cache=a.frame_cache, input_mode="dataset" if dataset_data is not None else "video", scene=a.scene, time_source=dataset_data["time_source"] if dataset_data is not None else "video FPS", postprocess=post_report, postprocess_storage=storage, cuda_graphs=a.cuda_graphs, cuda_graph_setup_seconds=graph_setup, feed_forward_fps=n/timing["camera_estimation_seconds"], ba_depth_init=None if a.no_ba else a.ba_depth_init, ba_reject_outliers=not (a.no_ba or a.ba_no_reject_outliers), ba_mode=None if a.no_ba else 'fixed', status='complete', frames=n, fps=fps, hand_slots=['right', 'left'], mask_source=a.mask_source, mask_threshold=a.mask_threshold, ba_stride=None if a.no_ba else a.ba_stride, coordinate_frame='first predicted camera', checkpoint=str(a.checkpoint), model_load_seconds=load_seconds, timing=timing, cache=stats)
    (a.output / 'SUCCESS.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
