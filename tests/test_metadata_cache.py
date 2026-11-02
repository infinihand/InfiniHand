import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import weakref

import lz4.frame
import numpy as np

from infinihand.training.data.datasets.metadata_cache import MetadataCache
from infinihand.training.data.datasets.unified_hand_dataset import UnifiedHandDataset


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def concurrent_reader(directory, source, raw, counter, start, results):
    def fetch():
        with counter.get_lock():
            counter.value += 1
        time.sleep(.03)
        return raw
    start.wait(10)
    try:
        value = MetadataCache(directory).read_json(source, fetch, sha(raw))
        results.put(value['frame_id'])
    except Exception as error:
        results.put(str(error))


def fixture():
    return dict(
        schema_version='compact_virtual_camera_v10', side_order=['left', 'right'],
        units={'3d': 'meters'}, frame_count=4, mano_left_basis='A',
        mano_basis_policy='dataset_source_v1_20261015',
        frames=[dict(frame_id=100 + i * 3, timestamp_seconds=i * .1,
                     image_size_hw=[378, 518],
                     intrinsics=[[400., 0., 259.], [0., 400., 189.], [0., 0., 1.]],
                     camera_valid=True, camera_to_world=np.eye(4).tolist(),
                     hands={s: {'hand_valid': True} for s in ('left', 'right')})
                for i in range(4)],
    )


class MetadataCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = MetadataCache(self.temp.name)
        self.source = 'poss:ego/example/metadata.json'
        self.raw = json.dumps({'frame_id': 123, 'value': 0.123456789012345,
                               'valid': False, 'name': '左右'}).encode('utf-8')

    def test_lossless_hit_avoids_source_and_returns_fresh_objects(self):
        fetch = Mock(return_value=self.raw)
        first = self.cache.read_json(self.source, fetch, sha(self.raw))
        first['value'] = -1
        second = self.cache.read_json(self.source, fetch, sha(self.raw))
        self.assertEqual(second, json.loads(self.raw))
        self.assertEqual(fetch.call_count, 1)
        compressed = self.cache.path_for(self.source, sha(self.raw)).read_bytes()
        self.assertEqual(lz4.frame.decompress(compressed), self.raw)
        self.assertEqual(set(vars(self.cache)), {'root'})

    def test_revision_at_same_uri_cannot_reuse_old_metadata(self):
        first = self.cache.read_json(self.source, lambda: self.raw, sha(self.raw))
        new_raw = json.dumps({'frame_id': 456}).encode()
        second = self.cache.read_json(self.source, lambda: new_raw, sha(new_raw))
        self.assertNotEqual(first['frame_id'], second['frame_id'])
        self.assertNotEqual(self.cache.path_for(self.source, sha(self.raw)),
                            self.cache.path_for(self.source, sha(new_raw)))

    def test_corrupt_and_wrong_content_cache_are_repaired(self):
        path = self.cache.path_for(self.source, sha(self.raw))
        path.parent.mkdir(parents=True)
        for invalid in (b'truncated', lz4.frame.compress(b'{}')):
            path.write_bytes(invalid)
            fetch = Mock(return_value=self.raw)
            self.assertEqual(self.cache.read_json(self.source, fetch, sha(self.raw)),
                             json.loads(self.raw))
            fetch.assert_called_once()

    def test_bad_source_digest_is_not_published(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.cache.read_json(self.source, lambda: b'{}', sha(self.raw))
        self.assertFalse(self.cache.path_for(self.source, sha(self.raw)).exists())

    def test_manifest_digest_is_required(self):
        with self.assertRaisesRegex(ValueError, 'requires a manifest SHA256'):
            self.cache.read_json(self.source, lambda: self.raw, None)

    def test_interrupted_publish_leaves_no_partial_file(self):
        with patch('infinihand.training.data.datasets.metadata_cache.os.replace',
                   side_effect=OSError('simulated filesystem failure')):
            with self.assertRaises(OSError):
                self.cache.read_json(self.source, lambda: self.raw, sha(self.raw))
        self.assertFalse(self.cache.path_for(self.source, sha(self.raw)).exists())
        self.assertFalse(list(Path(self.temp.name).rglob('.pending-*')))

    def test_concurrent_processes_download_once(self):
        context = mp.get_context('spawn')
        counter, start, results = context.Value('i', 0), context.Event(), context.Queue()
        workers = [context.Process(target=concurrent_reader,
                    args=(self.temp.name, self.source, self.raw, counter, start, results))
                   for _ in range(4)]
        for worker in workers:
            worker.start()
        start.set()
        try:
            for worker in workers:
                worker.join(30)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual([results.get(timeout=2) for _ in workers], [123] * 4)
            self.assertEqual(counter.value, 1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

    def test_scene_registry_never_retains_decoded_metadata_or_arrays(self):
        raw = json.dumps(fixture()).encode()
        ds = UnifiedHandDataset.__new__(UnifiedHandDataset)
        ds.dataset_name = 'arctic'
        ds.raw_s3_root = 'poss:ego/example'
        ds.annotation_task = 'camera'
        ds.detection_only_annotations = ds.camera_pose_only_annotations = False
        ds.cfg = {'camera_trajectory_available': True}
        ds._metadata_disk_cache = self.cache
        ds._read_bytes_cached = Mock(return_value=raw)
        descriptors = [dict(scene_name=str(i), prefix=f'poss:ego/example/{i}',
                            loaded=False, metadata_sha256=sha(raw)) for i in range(24)]
        for descriptor in descriptors:
            loaded = ds._ensure_scene_loaded(descriptor)
            self.assertIsNot(loaded, descriptor)
            self.assertNotIn('metadata', descriptor)
            self.assertNotIn('intrinsics', descriptor)
            self.assertFalse(descriptor['loaded'])
            self.assertEqual(loaded['metadata']['frames'][0]['frame_id'], 100)
            self.assertEqual(loaded['camera_poses'].dtype, np.float32)
            ref = weakref.ref(loaded['camera_poses'])
            del loaded
            gc.collect()
            self.assertIsNone(ref())
        ds._read_bytes_cached.reset_mock()
        ds._ensure_scene_loaded(descriptors[0])
        ds._read_bytes_cached.assert_not_called()
        normalized = ds._normalize_doss_scene(dict(name='test', prefix='poss:ego/test',
                                                 metadata_sha256=sha(raw)))
        self.assertEqual(normalized['metadata_sha256'], sha(raw))


if __name__ == '__main__':
    unittest.main()
