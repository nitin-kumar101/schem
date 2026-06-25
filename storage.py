"""Storage backends for schematic page and highlight images."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import quote


class StorageBackend(ABC):
    @abstractmethod
    def upload_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes and return a retrievable link (s3://, https://, or local path)."""

    @abstractmethod
    def download_bytes(self, link: str) -> bytes:
        """Download object bytes from a link returned by upload_bytes."""

    @abstractmethod
    def upload_file(
        self,
        key: str,
        file_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a local file and return its link."""


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = self.base_dir / key.replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        path = self._resolve(key)
        path.write_bytes(data)
        return str(path.resolve())

    def upload_file(
        self,
        key: str,
        file_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        src = Path(file_path)
        dest = self._resolve(key)
        dest.write_bytes(src.read_bytes())
        return str(dest.resolve())

    def download_bytes(self, link: str) -> bytes:
        return Path(link).read_bytes()


class S3Storage(StorageBackend):
    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str = "us-east-1",
        client: Any | None = None,
    ) -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.client = client or boto3.client("s3", region_name=region)

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def _to_link(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full_key(key)}"

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        full_key = self._full_key(key)
        self.client.upload_fileobj(
            io.BytesIO(data),
            self.bucket,
            full_key,
            ExtraArgs={"ContentType": content_type},
        )
        return self._to_link(full_key)

    def upload_file(
        self,
        key: str,
        file_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        full_key = self._full_key(key)
        self.client.upload_file(
            str(file_path),
            self.bucket,
            full_key,
            ExtraArgs={"ContentType": content_type},
        )
        return self._to_link(full_key)

    def download_bytes(self, link: str) -> bytes:
        if link.startswith("s3://"):
            _, _, remainder = link.partition("s3://")
            bucket, _, key = remainder.partition("/")
        else:
            raise ValueError(f"Unsupported S3 link: {link}")

        buffer = io.BytesIO()
        self.client.download_fileobj(bucket, key, buffer)
        return buffer.getvalue()


def build_storage(settings: Any) -> StorageBackend:
    if settings.storage_backend.lower() == "s3":
        if not settings.s3_bucket:
            raise ValueError("SCHEMATIC_S3_BUCKET is required when storage backend is s3")
        return S3Storage(
            settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.aws_region,
        )
    return LocalStorage(settings.local_storage_dir)
