import json
import os
from configparser import ConfigParser
from pathlib import Path


def is_remote_path(path):
    path = str(path)
    return path.startswith('s3://') or (':' in path and not path.startswith('/'))


def normalize_remote_path(path):
    path = str(path)
    if path.startswith('s3://'):
        return path
    # Keep DOSS' named-rclone URI intact.  Converting it to ``s3://`` loses
    # the remote name, so there is no way to choose the matching endpoint and
    # credentials from rclone.conf.
    if (is_remote_path(path) and not path.startswith('s3://')):
        return path
    if ':' in path and not path.startswith('/'):
        return f's3://{path.split(":", 1)[1].lstrip("/")}'
    return path


def split_remote_path(path):
    path = normalize_remote_path(path)
    if (is_remote_path(path) and not path.startswith('s3://')):
        body = path.split(':', 1)[1].lstrip('/')
        bucket, _, key = body.partition('/')
        return bucket, key
    body = path[len('s3://'):] if path.startswith('s3://') else path
    bucket, _, key = body.strip('/').partition('/')
    return bucket, key


def join_remote(root, rel_path):
    root = normalize_remote_path(root).rstrip('/')
    return f'{root}/{str(rel_path).lstrip("/")}'


def _default_conf_path():
    candidates = [
        os.environ.get('PETRELOSS_CONF'),
        os.environ.get('GLD_PETREL_CONF'),
        str(Path.home() / 'petreloss.conf'),
    ]
    return next((p for p in candidates if p and Path(p).exists()), None)


def make_petrel_client(conf_path=None):
    try:
        from petrel_client.client import Client
    except ImportError as exc:
        raise ImportError('Petrel SDK reads require `petrel_client`; install petrel-oss-sdk in this environment.') from exc
    conf_path = conf_path or _default_conf_path()
    return Client(conf_path) if conf_path else Client()


class PetrelIO:
    def __init__(self, conf_path=None):
        self.client = make_petrel_client(conf_path)

    def read_bytes(self, path):
        blob = self.client.get(normalize_remote_path(path))
        if blob is None:
            raise FileNotFoundError(path)
        return blob

    def read_text(self, path):
        return self.read_bytes(path).decode('utf-8')

    def read_json(self, path):
        return json.loads(self.read_text(path))

    def exists(self, path):
        return self.client.contains(normalize_remote_path(path))

    def list(self, path):
        return self.client.list(normalize_remote_path(path).rstrip('/') + '/', no_paginate=False)


def _default_rclone_conf_path(conf_path=None):
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        conf_path,
        os.environ.get('RCLONE_CONFIG'),
        Path.home() / '.config/rclone/rclone.conf',
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        'No rclone.conf found. Set RCLONE_CONFIG or configure ~/.config/rclone/rclone.conf.'
    )


def _split_named_s3_uri(path):
    value = str(path)
    if ':' not in value or value.startswith('/'):
        raise ValueError(f'Expected a named S3 URI such as doss:bucket/key, got {path!r}')
    remote, body = value.split(':', 1)
    bucket, separator, key = body.lstrip('/').partition('/')
    if not remote or not bucket:
        raise ValueError(f'Invalid named S3 URI: {path!r}')
    return remote, bucket, key if separator else ''


class DossIO:
    """Read a named rclone S3 remote through one persistent boto3 client.

    The rclone config is treated as a credential source only; no secret value
    is logged or copied into the dataset configuration.  A persistent SDK
    client avoids spawning one ``rclone cat`` process for every training
    sample asset.
    """

    def __init__(self, conf_path=None, remote='doss'):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError('DOSS reads require boto3/botocore in the training environment.') from exc

        self.conf_path = _default_rclone_conf_path(conf_path)
        self.remote = str(remote)
        config = ConfigParser()
        if not config.read(self.conf_path):
            raise FileNotFoundError(self.conf_path)
        if self.remote not in config:
            raise KeyError(f'Rclone remote [{self.remote}] is missing from {self.conf_path}')
        section = config[self.remote]
        required = ('endpoint', 'access_key_id', 'secret_access_key')
        missing = [key for key in required if not section.get(key)]
        if missing:
            raise KeyError(f'Rclone remote [{self.remote}] is missing keys: {missing}')
        client_config = Config(
            retries={'max_attempts': 8, 'mode': 'standard'},
            max_pool_connections=64,
            proxies={},
            s3={'addressing_style': 'path'},
        )
        self.client = boto3.client(
            's3',
            endpoint_url=section['endpoint'],
            aws_access_key_id=section['access_key_id'],
            aws_secret_access_key=section['secret_access_key'],
            config=client_config,
        )

    def _parse(self, path):
        remote, bucket, key = _split_named_s3_uri(path)
        if remote != self.remote:
            raise ValueError(
                f'DossIO for [{self.remote}] cannot read named remote [{remote}]'
            )
        return bucket, key

    def read_bytes(self, path):
        bucket, key = self._parse(path)
        if not key:
            raise IsADirectoryError(path)
        try:
            return self.client.get_object(Bucket=bucket, Key=key)['Body'].read()
        except self.client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(path) from exc

    def read_text(self, path):
        return self.read_bytes(path).decode('utf-8')

    def read_json(self, path):
        return json.loads(self.read_text(path))

    def exists(self, path):
        bucket, key = self._parse(path)
        if not key:
            return True
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'get', lambda *_: {})(
                'ResponseMetadata', {}
            ).get('HTTPStatusCode')
            if status in {403, 404}:
                return False
            raise

    def list(self, path):
        bucket, key = self._parse(path)
        prefix = key.rstrip('/') + '/' if key else ''
        token = None
        while True:
            kwargs = {
                'Bucket': bucket,
                'Prefix': prefix,
                'Delimiter': '/',
                'MaxKeys': 1000,
            }
            if token:
                kwargs['ContinuationToken'] = token
            page = self.client.list_objects_v2(**kwargs)
            for entry in page.get('CommonPrefixes', []):
                yield f'{self.remote}:{bucket}/{entry["Prefix"]}'
            for entry in page.get('Contents', []):
                object_key = entry['Key']
                if object_key != prefix:
                    yield f'{self.remote}:{bucket}/{object_key}'
            if not page.get('IsTruncated'):
                break
            token = page.get('NextContinuationToken')


def normalize_list_item(root, item):
    root = normalize_remote_path(root).rstrip('/') + '/'
    item = str(item)
    if (is_remote_path(root) and not root.startswith('s3://')):
        if (is_remote_path(item) and not item.startswith('s3://')):
            return item
        return f'{root}{item.lstrip("/")}'
    bucket, root_key = split_remote_path(root)
    root_key = root_key.rstrip('/') + '/' if root_key else ''
    if item.startswith('s3://'):
        return item
    if item.startswith(f'{bucket}/'):
        return f's3://{item}'
    if root_key and item.startswith(root_key):
        return f's3://{bucket}/{item}'
    return f'{root}{item.lstrip("/")}'
