"""Shared lossless metadata files; no decoded scene is retained in this object."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import lz4.frame


class MetadataCache:
    """Content-versioned LZ4 JSON cache for a POSIX shared filesystem.

    Readers see an atomically published complete file. Misses and repairs use
    256 persistent striped advisory locks, avoiding one lock file per scene.
    The manifest digest is mandatory: source labels can change at the same URI.
    """

    def __init__(self, directory):
        self.root = Path(directory).expanduser().resolve() / 'metadata_lz4' / 'v1'

    def path_for(self, source, expected_sha256):
        digest = str(expected_sha256 or '').lower()
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Metadata disk cache requires a manifest SHA256: {source}')
        key = hashlib.sha256((str(source) + '\0' + digest).encode('utf-8')).hexdigest()
        return self.root / key[:2] / f'{key}.json.lz4'

    @staticmethod
    def _decode(raw, expected_sha256, source):
        if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
            raise ValueError(f'Metadata SHA256 differs from manifest: {source}')
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'Metadata must be a JSON object: {source}')
        return value

    def _read(self, path, expected_sha256, source):
        try:
            compressed = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            raw = lz4.frame.decompress(compressed)
            return self._decode(raw, expected_sha256, source)
        except (RuntimeError, ValueError, UnicodeError):
            # A corrupt cache is repairable. Source checksum errors below are not.
            return None

    def read_json(self, source, fetch_bytes, expected_sha256):
        path = self.path_for(source, expected_sha256)
        value = self._read(path, expected_sha256, source)
        if value is not None:
            return value

        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / '.write.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # Another rank or node may have populated the file while we waited.
            value = self._read(path, expected_sha256, source)
            if value is not None:
                return value
            raw = fetch_bytes()
            value = self._decode(raw, expected_sha256, source)
            compressed = lz4.frame.compress(raw, compression_level=0, content_checksum=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', prefix='.pending-',
                                                 dir=path.parent, delete=False) as handle:
                    temp_path = Path(handle.name)
                    handle.write(compressed)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            return value
