"""Storage backends for cached imagery and metadata.

Two backends behind one interface. Local is for development and for the
reproducibility check; S3 is what runs in the AWS Batch pipeline. Keeping them
interchangeable means Stage 1-3 code never learns which one it is talking to,
and the whole pipeline can be exercised offline.

Key layout is partitioned by sequence:

    imagery/seq={sequence_id}/{image_id}.jpg
    metadata/seq={sequence_id}/images.jsonl
    manifests/{area}/manifest.json

Partitioning by sequence rather than by geography is deliberate. Stage 3 reads
whole sequences at a time to build baselines, so this layout gives it
contiguous reads and lets Batch shard work by sequence with no cross-shard
chatter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def image_key(sequence_id: str | None, image_id: str, ext: str = "jpg") -> str:
    seq = sequence_id or "_orphan"
    return f"imagery/seq={seq}/{image_id}.{ext}"


def metadata_key(sequence_id: str) -> str:
    return f"metadata/seq={sequence_id}/images.jsonl"


def manifest_key(area: str) -> str:
    return f"manifests/{area}/manifest.json"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Storage(ABC):
    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]: ...

    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type)

    def put_json(self, key: str, obj: Any) -> None:
        self.put_text(key, json.dumps(obj, indent=2), "application/json")

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))


class LocalStorage(Storage):
    def __init__(self, root: str | Path) -> None:
        # Resolve the root exactly once, here. Every other path in this class
        # is compared against it, so the two must be in the same form.
        #
        # This is not hypothetical tidiness: on macOS /var is a symlink to
        # /private/var, so a tempfile directory arrives as /var/folders/...
        # while .resolve() on any key beneath it yields /private/var/... .
        # Storing the root unresolved and resolving keys made relative_to()
        # raise "not in the subpath of" for every listing. Linux temp dirs have
        # no such symlink, so it only ever failed on macOS.
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()

    def _path(self, key: str) -> Path:
        # Guard against traversal from any key that ever comes from an API.
        p = (self.root / key).resolve()
        # is_relative_to, not str.startswith: startswith would accept a
        # sibling directory whose name merely shares the prefix, e.g.
        # /data/cache-evil passing a check against /data/cache.
        if not p.is_relative_to(self.root):
            raise ValueError(f"key escapes storage root: {key}")
        return p

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot leave a truncated file
        # that a later run would treat as a valid cache hit.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        search = base if base.is_dir() else base.parent
        if not search.exists():
            return []
        out = []
        for p in search.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                rel = str(p.relative_to(self.root))
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def free_space_gb(self) -> float:
        return shutil.disk_usage(self.root).free / 1e9


class S3Storage(Storage):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is not None:
            self.s3 = client
        else:
            import boto3  # imported lazily so local runs need no AWS deps

            self.s3 = boto3.client(
                "s3", region_name=region or os.environ.get("AWS_REGION", "us-east-2")
            )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=data,
            ContentType=content_type,
        )

    def get_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        out: list[str] = []
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                out.append(obj["Key"][strip:])
        return out


def make_storage(uri: str) -> Storage:
    """Build a backend from a URI.

    Accepts `s3://bucket/optional/prefix` or a plain filesystem path.
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        return S3Storage(bucket=bucket, prefix=prefix)
    return LocalStorage(uri)