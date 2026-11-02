"""Streaming camera-space evaluation for component checkpoints.

Inputs are exported RGB/K/labels and cached SAM3+tracking outputs. Runtime hand
slots are [right,left]. GT geometry is used only for the GT-box control and scoring.
"""
import argparse
import hashlib
import dataclasses
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from infinihand.training.engine import load_config
from infinihand.models.lingbotmap_hand_v6 import LingBotMapHand, LingBotMapHandConfig
from infinihand.training.checkpoint_utils import load_checkpoint_payload, extract_model_state
from infinihand.training.data.datasets.unified_hand_dataset import UnifiedHandDataset
from infinihand.training.data.training_profiles import tensorize_training_sample, build_target

def js(p, d):
    p.write_text(json.dumps(d, indent=2))

def arr(x):
    return x.detach().float().cpu().numpy()

class CameraSpaceEvaluator:
    def __init__(self, config, checkpoints):
        self.config_sha256 = hashlib.sha256(Path(config).read_bytes()).hexdigest()
        self.cfg = load_config(str(config))
        self.plan = json.loads(Path(checkpoints).read_text())
        for tag in ('stage1', 'stage2', 'mask'):
            if not Path(self.plan['checkpoints'][tag]['path']).is_file():
                raise FileNotFoundError(self.plan['checkpoints'][tag]['path'])

    def export(self, item, O, limit):
        if (O / 'INPUT.json').exists():
            return
        ds = UnifiedHandDataset(common_conf=SimpleNamespace(training=False, annotation_task='joint', metadata_cache_dir=self.cfg.metadata_cache_dir, data_cache_mode='none'), data_json=item['manifest'], view_sampler_cfg={'max_sampled_views': 4}, len_train=1)
        s = ds._ensure_scene_loaded(next(iter(ds.scenes.values())))
        N = min(s['frame_count'], limit) if limit else s['frame_count']
        store = {}
        (O / 'frames').mkdir(exist_ok=True)
        (O / 'gt_masks').mkdir(exist_ok=True)
        for start in range(0, N, 4):
            ids = np.arange(start, min(start + 4, N))
            raw = ds._get_doss_data(s, ids)
            b = tensorize_training_sample(raw)
            b = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for (k, v) in b.items()}
            b['images'] = b['images'].float() / 255
            (target, _) = build_target(b, self.cfg)
            for (k, v) in dict(gt_boxes=arr(target['hand_boxes'])[0], gt_valid=arr(target['hand_valid'])[0].astype(bool), gt_joints=raw['camera_mano_joints'], point_valid=raw['hand_points_valid'], K=raw['intrinsics'], source_ids=raw['source_frame_ids'], timestamps=raw['timestamps_seconds'], timestamps_valid=raw['timestamps_valid'], mask_valid=raw['hand_mask_supervision_valid'], basis=arr(b['mano_left_basis_b'])[0]).items():
                store.setdefault(k, []).append(v)
            for (j, i) in enumerate(ids):
                im = raw['images'][j]
                assert im.ndim == 3
                if im.shape[0] == 3:
                    im = im.transpose(1, 2, 0)
                assert cv2.imwrite(str(O / 'frames' / f'{i:06d}.png'), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                masks = raw['hand_masks'][j]
                rgb = np.zeros((*masks.shape[-2:], 3), np.uint8)
                rgb[:, :, 0] = masks[0] * 255
                rgb[:, :, 1] = masks[1] * 255
                assert cv2.imwrite(str(O / 'gt_masks' / f'{i:06d}.png'), rgb)
        a = {k: np.concatenate(v) for (k, v) in store.items()}
        assert all((len(v) == N for v in a.values()))
        np.savez_compressed(O / 'labels.npz', **a)
        ts = a['timestamps']
        time_source = 'source timestamps'
        if not (a['timestamps_valid'].all() and np.isfinite(ts).all() and (np.diff(ts) > 0).all()):
            fps = float(s['metadata'].get('sampling_fps', 0))
            assert fps > 0
            ts = np.arange(N) / fps
            time_source = 'declared sampling_fps grid; source timestamps unavailable; original timestamps and validity retained in labels.npz'
        js(O / 'INPUT.json', dict(frames=N, size=[im.shape[1], im.shape[0]], source_ids=a['source_ids'].tolist(), output_times_seconds=ts.tolist(), slots=['right', 'left'], time_source=time_source, scene=item, protocol='identical lossless RGB for every branch; released test sampling'))

    def build_model(self, version):
        kw = {f.name: getattr(self.cfg, f.name) for f in dataclasses.fields(LingBotMapHandConfig) if hasattr(self.cfg, f.name)}
        kw.update(mano_model_path=str(Path(self.cfg.wilor_root) / 'mano_data'), wilor_feature_chunk_size=self.cfg.hand_feature_chunk_size, decode_mano=True, use_gt_boxes_for_mano=True, use_soft_mask_boxes_for_mano=False, mano_gradient_to_mask_head=False, lingbotmap_ckpt='', load_wilor_checkpoint=False, lingbot_enable_camera_head=True, step_memory_enabled=True, step_memory_scale_frames=4, step_memory_sliding_window=16, step_memory_max_frames=8192, mask_only_threshold=0.9)
        m = LingBotMapHand(LingBotMapHandConfig(**kw))
        missing = set()
        loaded = set()
        for tag in ['stage1'] + (['stage2'] if version == 'stage2' else []) + ['mask']:
            p = load_checkpoint_payload(self.plan['checkpoints'][tag]['path'])
            st = extract_model_state(p)
            res = m.load_state_dict(st, strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
            if tag == 'stage1':
                missing = set(res.missing_keys)
            loaded.update(st)
            print('LOADED', tag, self.plan['checkpoints'][tag].get('step'), len(st), flush=True)
            del st, p
            gc.collect()
        assert missing <= loaded, missing - loaded
        m.requires_grad_(False)
        m.cuda().eval()
        assert int(m._mano_translation_convention) == 1
        return m

    def run(self, O, version, maskonly=False):
        protocol = {'config_sha256': self.config_sha256, 'checkpoints': self.plan['checkpoints'], 'version': version}
        record = O / 'EVALUATION_PROTOCOL.json'
        if record.exists() and json.loads(record.read_text()) != protocol:
            raise ValueError('Output belongs to a different evaluation recipe; use a new scene directory')
        if not record.exists():
            if any((O / name).exists() for name in ('MASK_SUCCESS.json', 'stage1_SUCCESS.json', 'stage2_SUCCESS.json')):
                raise ValueError('Existing outputs have no verified recipe; use a new scene directory')
            js(record, protocol)
        marker = O / ('MASK_SUCCESS.json' if maskonly else version + '_SUCCESS.json')
        if marker.exists():
            return
        a = np.load(O / 'labels.npz')
        N = len(a['K'])
        m = self.build_model(version)
        result = {}
        boxes = {}
        valids = {}
        preview_ids = set(np.linspace(0, N - 1, min(10, N)).round().astype(int).tolist())
        if not maskonly:
            np.save(O / 'mano_faces.npy', np.asarray(m.mano_right.faces, dtype=np.int32))
        if maskonly:
            (O / 'stage2_masks').mkdir(exist_ok=True)
        else:
            sam = np.load(O / 'sam/BOXES.npz')
            shape = json.loads((O / 'INPUT.json').read_text())['size']
            sb = sam['boxes'][:, [1, 0]] / np.array(shape * 2, np.float32)
            sv = (sam['areas'][:, [1, 0]] > 0) & np.isfinite(sb).all(-1)
            boxes = {'gt': a['gt_boxes'], 'stage2_mask': np.load(O / 'mask_boxes.npz')['boxes'], 'sam3_tracking_1hz': np.nan_to_num(sb)}
            valids = {'gt': a['gt_valid'], 'stage2_mask': np.load(O / 'mask_boxes.npz')['valid'], 'sam3_tracking_1hz': sv}
        starts = [np.arange(4)] + [np.array([i]) for i in range(4, N)]
        assert N >= 4
        for ids in starts:
            ims = np.stack([cv2.cvtColor(cv2.imread(str(O / 'frames' / f'{i:06d}.png')), cv2.COLOR_BGR2RGB) for i in ids])
            x = torch.from_numpy(ims).permute(0, 3, 1, 2).unsqueeze(0).cuda().float() / 255
            K = torch.from_numpy(a['K'][ids]).unsqueeze(0).cuda().float()
            with torch.inference_mode():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    bb = m.backbone.begin_inference_stream(x) if ids[0] == 0 else m.backbone.inference_stream_step(x)
                if maskonly:
                    out = m(x, heads='mask', backbone_output=bb, intrinsics=K)
                    pm = arr(out['pred_hand_masks'])[0] > 0.9
                    tight = np.zeros((*pm.shape[:2], 4), np.float32)
                    for ti in range(len(ids)):
                        for side in range(2):
                            (yy, xx) = np.where(pm[ti, side])
                            if len(xx):
                                tight[ti, side] = np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]) / np.array([pm.shape[-1], pm.shape[-2]] * 2)
                    result.setdefault('boxes', []).append(tight)
                    result.setdefault('valid', []).append(pm.any((-1, -2)))
                    for (j, i) in enumerate(ids):
                        rgb = np.zeros((*pm.shape[-2:], 3), np.uint8)
                        rgb[:, :, 0] = pm[j, 0] * 255
                        rgb[:, :, 1] = pm[j, 1] * 255
                        cv2.imwrite(str(O / 'stage2_masks' / f'{i:06d}.png'), rgb)
                else:
                    for src in boxes:
                        out = m(x, heads='mano', backbone_output=bb, intrinsics=K, mano_left_basis_b=torch.from_numpy(a['basis'][ids]).unsqueeze(0).cuda().bool(), gt_hand_boxes=torch.from_numpy(boxes[src][ids]).unsqueeze(0).cuda().float(), gt_hand_valid=torch.from_numpy(valids[src][ids]).unsqueeze(0).cuda().bool(), use_gt_boxes=True, mano_training_selection=False)
                        j = arr(out['pred_keypoints_3d_camera'])[0]
                        v = arr(out['geometry_crop_valid'])[0].astype(bool) & valids[src][ids] & np.isfinite(j).all((-1, -2))
                        result.setdefault(src + '_joints', []).append(j)
                        result.setdefault(src + '_valid', []).append(v)
                        keep = [z for (z, i) in enumerate(ids) if int(i) in preview_ids]
                        if keep:
                            result.setdefault(src + '_preview_vertices', []).append(arr(out['pred_vertices_camera'])[0][keep])
            del out, bb, x
            gc.collect()
            if int(ids[-1]) % 50 == 0:
                print(version, 'mask' if maskonly else 'mano', int(ids[-1]) + 1, N, flush=True)
        stats = m.backbone.end_inference_stream()
        assert stats['total_frames'] == N and stats['cache_cleared_at_step_end']
        np.savez_compressed(O / ('mask_boxes.npz' if maskonly else version + '.npz'), **{k: np.concatenate(v) for (k, v) in result.items()})
        js(marker, dict(frames=N, version=version, stats=stats))
        del m
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['export', 'mask', 'mano'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--scene', type=Path, help='One scene descriptor JSON, for export')
    parser.add_argument('--version', choices=['stage1', 'stage2'], default='stage2')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.manual_seed(42)
    args.scene_dir.mkdir(parents=True, exist_ok=True)
    runner = CameraSpaceEvaluator(args.config, args.checkpoints)
    if args.operation == 'export':
        if args.scene is None:
            parser.error('--scene is required for export')
        runner.export(json.loads(args.scene.read_text()), args.scene_dir, args.limit)
    else:
        runner.run(args.scene_dir, args.version, maskonly=args.operation == 'mask')

if __name__ == '__main__':
    main()
