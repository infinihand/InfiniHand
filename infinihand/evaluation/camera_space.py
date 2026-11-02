"""Evaluate one released inference checkpoint on prepared test datasets."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from infinihand.infer import (
    build_model, run_dual_stream, validate_contract,
)
from infinihand.training.data.datasets.unified_hand_dataset import UnifiedHandDataset
from .metrics import _geometry


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = {'checkpoint', 'model_config', 'output_dir', 'datasets'}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError('Evaluation requires checkpoint, model_config, output_dir and datasets')
    if not isinstance(config['datasets'], dict) or not config['datasets']:
        raise ValueError('datasets must map dataset names to test manifests')
    limit = int(config.get('max_frames', 8192))
    if not 4 <= limit <= 8192:
        raise ValueError('max_frames must be between 4 and 8192')
    if not 0 < float(config.get('mask_threshold', .9)) < 1:
        raise ValueError('mask_threshold must be in (0, 1)')
    return config


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_scene(dataset, descriptor, limit):
    scene = dataset._ensure_scene_loaded(descriptor)
    count = min(scene['frame_count'], limit)
    if count < 4:
        raise ValueError(f'{scene["scene_name"]}: at least four frames required')
    values = {}
    keys = ('images', 'intrinsics', 'camera_mano_joints', 'hand_valid',
            'hand_points_valid', 'source_frame_ids')
    for start in range(0, count, 4):
        sample = dataset._get_doss_data(scene, np.arange(start, min(start + 4, count)))
        for key in keys:
            values.setdefault(key, []).append(sample[key])
    return scene, {key: np.concatenate(parts) for key, parts in values.items()}


def evaluate_scene(model, dataset, descriptor, config, output, protocol):
    if output.exists():
        raise FileExistsError(f'Use a fresh scene output directory: {output}')
    scene, data = load_scene(dataset, descriptor, int(config.get('max_frames', 8192)))
    output.mkdir(parents=True)
    images = torch.from_numpy(np.ascontiguousarray(data['images'])).permute(0, 3, 1, 2)
    images = images.unsqueeze(0).cuda().float().div_(255)
    intrinsics = torch.from_numpy(data['intrinsics']).unsqueeze(0).cuda().float()
    with torch.inference_mode():
        predictions, cache, timing = run_dual_stream(
            model, images, intrinsics, bbox_source='mask',
            mask_threshold=float(config.get('mask_threshold', .9)),
        )
    validate_contract(cache, len(data['images']))
    arrays = {key: value.detach().float().cpu().numpy() for key, value in predictions.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Non-finite model output')
    valid = arrays['presence_mask'] > .5
    metrics, _, _ = _geometry(arrays['camera_joints'], data['camera_mano_joints'],
                             data['hand_valid'], valid, data['hand_points_valid'])
    metrics['protocol'] = 'Camera-space same-side joint errors; predicted mask boxes; no GT box input or BA. Not the paper bbox-matching protocol.'
    np.savez_compressed(output / 'camera_space.npz', **arrays, valid=valid)
    np.savez_compressed(output / 'labels.npz', gt_joints=data['camera_mano_joints'],
                        gt_valid=data['hand_valid'], point_valid=data['hand_points_valid'],
                        K=data['intrinsics'], source_ids=data['source_frame_ids'])
    write_json(output / 'METRICS.json', metrics)
    write_json(output / 'SUCCESS.json', dict(
        status='complete', scene=scene['scene_name'], frames=len(valid),
        available_frames=scene['frame_count'], truncated=len(valid) < scene['frame_count'],
        protocol=protocol, timing=timing, cache=cache,
    ))
    del predictions, images, intrinsics
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Override the single checkpoint in the config')
    parser.add_argument('--dataset', help='Evaluate one configured dataset; omit for all')
    parser.add_argument('--scene', help='Evaluate one exact scene name within --dataset')
    parser.add_argument('--max-frames', type=int, help='Explicit per-scene prefix limit for smoke tests')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.checkpoint:
        config['checkpoint'] = str(args.checkpoint)
    if args.max_frames is not None:
        if not 4 <= args.max_frames <= 8192:
            parser.error('--max-frames must be between 4 and 8192')
        config['max_frames'] = args.max_frames
    if args.scene and not args.dataset:
        parser.error('--scene requires --dataset')
    if args.dataset and args.dataset not in config['datasets']:
        parser.error('Unknown configured dataset')
    selected = {args.dataset: config['datasets'][args.dataset]} if args.dataset else config['datasets']
    for path in [config['checkpoint'], config['model_config'], *selected.values()]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    datasets = {}
    for name, manifest in selected.items():
        payload = json.loads(Path(manifest).read_text())
        if payload.get('split') not in ('test', 'val', 'validation'):
            raise ValueError(f'{name}: expected an explicit evaluation split')
        datasets[name] = UnifiedHandDataset(
            common_conf=SimpleNamespace(training=False, annotation_task='mano',
                metadata_cache_dir=config.get('metadata_cache_dir', ''), data_cache_mode='none'),
            data_json=manifest, view_sampler_cfg={'max_sampled_views': 4}, len_train=1,
        )
    if args.check_config:
        print(json.dumps({'datasets': {name: len(ds.scenes) for name, ds in datasets.items()},
                          'checkpoint': config['checkpoint'], 'status': 'config_valid'}))
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, load_seconds = build_model(config['checkpoint'], torch.device('cuda:0'), config['model_config'])
    protocol = dict(config=config, model_config_sha256=hashlib.sha256(Path(config['model_config']).read_bytes()).hexdigest(),
                    mask_source='base', model_load_seconds=load_seconds)
    summary = {}
    for name, dataset in datasets.items():
        descriptors = list(dataset.scenes.values())
        if args.scene:
            descriptors = [scene for scene in descriptors if scene['scene_name'] == args.scene]
            if not descriptors:
                raise ValueError(f'Scene not found: {args.scene}')
        for descriptor in descriptors:
            scene_name = descriptor['scene_name']
            if Path(scene_name).is_absolute() or '..' in Path(scene_name).parts:
                raise ValueError('Unsafe scene name')
            output = Path(config['output_dir']) / name / scene_name
            metrics = evaluate_scene(model, dataset, descriptor, config, output, protocol)
            summary[f'{name}/{scene_name}'] = metrics
            print(name, scene_name, metrics['mpjpe_matched_mm'], flush=True)
    write_json(Path(config['output_dir']) / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
