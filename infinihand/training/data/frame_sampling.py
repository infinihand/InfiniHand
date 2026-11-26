"""Frame-based dataset weights for training manifests."""
import hashlib
import json
import math
from pathlib import Path


SQRT_FRAME_SAMPLING_VERSION = 'ego_data_v2_sqrt_frames'


def allocate_budget(weights, total):
    """Largest-remainder allocation with a positive slot for every dataset."""
    if total < len(weights) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Budget must cover every positive finite weight')
    summed = sum(weights.values())
    exact = {k: total * v / summed for k, v in weights.items()}
    result = {k: max(1, math.floor(v)) for k, v in exact.items()}
    while sum(result.values()) < total:
        key = max(result, key=lambda k: (exact[k] - result[k], k))
        result[key] += 1
    while sum(result.values()) > total:
        key = max((k for k in result if result[k] > 1), key=lambda k: (result[k] - exact[k], k))
        result[key] -= 1
    return result


def manifest_sampling_plan(paths, minimum_frames, multipliers, nominal_length, split_weights=None):
    """Use completed manifest frames; exclude scenes too short for this mode.

    Multipliers are applied after sqrt(frame_count).
    The virtual length controls dataset frequency, never the scene pool size.
    """
    from .split_policy import validate_training_paths
    validate_training_paths(paths, split_weights)
    if minimum_frames < 1 or nominal_length < 1:
        raise ValueError('Frame minimum and nominal length must be positive')
    statistics = {}
    for path in paths:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
        name = str(manifest.get('sampling_name', manifest['dataset_name'])).lower()
        if name in statistics:
            raise ValueError(f'Duplicate sampling name: {name}')
        split = str(manifest.get('split', '')).lower()
        if split != 'train':
            raise ValueError(f'Explicit training split required: {path}')
        multiplier = float(multipliers[name])
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f'Invalid frame multiplier for {name}')
        counts = []
        for scene in manifest['scenes']:
            count = scene.get('frame_count')
            if isinstance(count, bool) or count is None or int(count) != count or count < 0:
                raise ValueError(f'Invalid/missing completed frame_count: {path}')
            counts.append(int(count))
        eligible = [n for n in counts if n >= max(minimum_frames, int(manifest.get('min_scene_frames', 1))) ]
        frames = sum(eligible)
        if frames == 0:
            raise ValueError(f'No eligible frames in {name}; remove it from this training mode')
        statistics[name] = dict(manifest=str(path), manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                dataset=manifest['dataset_name'], split=split,
                                scene_count=len(counts), total_frames=sum(counts),
                                eligible_scenes=len(eligible), eligible_frames=frames,
                                multiplier=multiplier, raw_weight=multiplier * math.sqrt(frames))
    if not statistics:
        raise ValueError('At least one dataset is required')
    total = len(statistics) * nominal_length
    if split_weights:
        groups = {v['split'] for v in statistics.values()}
        if groups != set(split_weights):
            raise ValueError(f'Split budget {set(split_weights)} must match configured pools {groups}')
        quotas = allocate_budget({k: float(v) for k, v in split_weights.items()}, total)
    else:
        quotas = {'all': total}
    lengths, effective = {}, {}
    for split, quota in quotas.items():
        raw_weights = {k: v['raw_weight'] for k, v in statistics.items() if split == 'all' or v['split'] == split}
        lengths.update(allocate_budget(raw_weights, quota))
        for name, weight in raw_weights.items():
            effective[name] = quota / total * weight / sum(raw_weights.values())
    for name, record in statistics.items():
        record.update(probability=effective[name], virtual_length=lengths[name],
                      allocated_probability=lengths[name] / total)
    return effective, lengths, statistics
