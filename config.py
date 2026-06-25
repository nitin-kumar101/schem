"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Processing
    max_pixels: int = _env_int("SCHEMATIC_MAX_PIXELS", 4_000_000)
    max_workers: int | None = (
        _env_int("SCHEMATIC_MAX_WORKERS", 0) or None
    )  # 0 => auto (cpu_count)
    components_only: bool = _env_bool("SCHEMATIC_COMPONENTS_ONLY", False)

    # Storage
    storage_backend: str = os.getenv("SCHEMATIC_STORAGE_BACKEND", "local")  # local | s3
    local_storage_dir: str = os.getenv("SCHEMATIC_LOCAL_STORAGE_DIR", "storage")
    s3_bucket: str = os.getenv("SCHEMATIC_S3_BUCKET", "")
    s3_prefix: str = os.getenv("SCHEMATIC_S3_PREFIX", "schematics")
    aws_region: str = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")
    db_table: str = os.getenv("SCHEMATIC_DB_TABLE", "schematic_pages")

    # Local dev fallback (no PostgreSQL)
    local_db_path: str = os.getenv("SCHEMATIC_LOCAL_DB_PATH", "output/schematic_index.json")


def get_settings() -> Settings:
    return Settings()
