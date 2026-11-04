"""Read ordered prepared scenes for inference; never supply GT to the model."""
from types import SimpleNamespace
import numpy as np


def load_dataset_input(manifest, scene_name, max_frames, fps=None):
    from .training.data.datasets.unified_hand_dataset import UnifiedHandDataset
    dataset = UnifiedHandDataset(
        common_conf=SimpleNamespace(training=False, annotation_task='mano',
                                    metadata_cache_dir='', data_cache_mode='none'),
        data_json=str(manifest), view_sampler_cfg={'max_sampled_views': 4}, len_train=1,
    )
    matches = [s for s in dataset.scenes.values() if s['scene_name'] == scene_name]
    if len(matches) != 1:
        raise ValueError(f'Require one exact scene match for {scene_name!r}, found {len(matches)}')
    scene = dataset._ensure_scene_loaded(matches[0])
    count = min(scene['frame_count'], max_frames)
    if count < 4:
        raise ValueError('At least four frames required')
    keys = ('images', 'intrinsics', 'source_frame_ids', 'timestamps_seconds', 'timestamps_valid')
    parts = {key: [] for key in keys}
    for start in range(0, count, 4):
        sample = dataset._get_doss_data(scene, np.arange(start, min(start + 4, count)))
        for key in keys:
            parts[key].append(sample[key])
    data = {key: np.concatenate(value) for key, value in parts.items()}
    times = data['timestamps_seconds']
    if data['timestamps_valid'].all() and np.isfinite(times).all() and np.all(np.diff(times) > 0):
        time_source = 'dataset timestamps'
        nominal_fps = float(1 / np.median(np.diff(times)))
    elif fps is not None and np.isfinite(fps) and fps > 0:
        times = np.arange(count, dtype=np.float64) / fps
        time_source = 'explicit processed-frame FPS override'
        nominal_fps = float(fps)
    else:
        raise ValueError('Scene lacks valid increasing timestamps; supply --dataset-fps with the processed-frame rate')
    data['timestamps_seconds'] = times
    data['fps'] = nominal_fps
    data['time_source'] = time_source
    return data
