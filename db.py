"""PostgreSQL (and local JSON fallback) persistence for schematic pages."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PageRecord:
    id: int | str
    schematic_id: str
    smetadata: dict[str, Any]
    words_bboxes: dict[str, list[dict[str, Any]]]


class SchematicRepository(ABC):
    @abstractmethod
    def insert_page(
        self,
        schematic_id: str,
        smetadata: dict[str, Any],
        words_bboxes: dict[str, list[dict[str, Any]]],
    ) -> int | str:
        """Insert one page row and return its id."""

    @abstractmethod
    def find_pages_by_components(
        self,
        schematic_id: str,
        component_names: list[str],
        *,
        match_all: bool = False,
    ) -> list[PageRecord]:
        """Return page rows that contain the requested component names."""

    @abstractmethod
    def get_all_pages(self, schematic_id: str) -> list[PageRecord]:
        """Return all pages for a schematic."""

    @abstractmethod
    def delete_schematic(self, schematic_id: str) -> None:
        """Remove all rows for a schematic (used before re-processing)."""


class PostgresSchematicRepository(SchematicRepository):
    def __init__(self, database_url: str, table_name: str = "schematic_pages") -> None:
        import psycopg
        from psycopg.rows import dict_row

        self.table_name = table_name
        self._connect = lambda: psycopg.connect(database_url, row_factory=dict_row)
        self._ensure_table()

    def _ensure_table(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id BIGSERIAL PRIMARY KEY,
            schematic_id TEXT NOT NULL,
            smetadata JSONB NOT NULL,
            words_bboxes JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_schematic_id
            ON {self.table_name} (schematic_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_words_bboxes
            ON {self.table_name} USING GIN (words_bboxes);
        """
        with self._connect() as conn:
            conn.execute(ddl)
            conn.commit()

    def insert_page(
        self,
        schematic_id: str,
        smetadata: dict[str, Any],
        words_bboxes: dict[str, list[dict[str, Any]]],
    ) -> int:
        sql = f"""
            INSERT INTO {self.table_name} (schematic_id, smetadata, words_bboxes)
            VALUES (%s, %s::jsonb, %s::jsonb)
            RETURNING id
        """
        with self._connect() as conn:
            row = conn.execute(
                sql,
                (schematic_id, json.dumps(smetadata), json.dumps(words_bboxes)),
            ).fetchone()
            conn.commit()
            return int(row["id"])

    def find_pages_by_components(
        self,
        schematic_id: str,
        component_names: list[str],
        *,
        match_all: bool = False,
    ) -> list[PageRecord]:
        if not component_names:
            return []

        operator = "?&" if match_all else "?|"
        sql = f"""
            SELECT id, schematic_id, smetadata, words_bboxes
            FROM {self.table_name}
            WHERE schematic_id = %s
              AND words_bboxes {operator} %s::text[]
            ORDER BY (smetadata->>'page_number')::int
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (schematic_id, component_names)).fetchall()

        return [
            PageRecord(
                id=row["id"],
                schematic_id=row["schematic_id"],
                smetadata=row["smetadata"],
                words_bboxes=row["words_bboxes"],
            )
            for row in rows
        ]

    def get_all_pages(self, schematic_id: str) -> list[PageRecord]:
        sql = f"""
            SELECT id, schematic_id, smetadata, words_bboxes
            FROM {self.table_name}
            WHERE schematic_id = %s
            ORDER BY (smetadata->>'page_number')::int
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (schematic_id,)).fetchall()
        return [
            PageRecord(
                id=row["id"],
                schematic_id=row["schematic_id"],
                smetadata=row["smetadata"],
                words_bboxes=row["words_bboxes"],
            )
            for row in rows
        ]

    def delete_schematic(self, schematic_id: str) -> None:
        sql = f"DELETE FROM {self.table_name} WHERE schematic_id = %s"
        with self._connect() as conn:
            conn.execute(sql, (schematic_id,))
            conn.commit()


class LocalJsonRepository(SchematicRepository):
    """File-backed repository for local development without PostgreSQL."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self._data: dict[str, list[dict[str, Any]]] = json.loads(
                self.index_path.read_text(encoding="utf-8")
            )
        else:
            self._data = {}
        self._next_id = self._compute_next_id()

    def _compute_next_id(self) -> int:
        max_id = 0
        for pages in self._data.values():
            for page in pages:
                max_id = max(max_id, int(page.get("id", 0)))
        return max_id + 1

    def _save(self) -> None:
        self.index_path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    def insert_page(
        self,
        schematic_id: str,
        smetadata: dict[str, Any],
        words_bboxes: dict[str, list[dict[str, Any]]],
    ) -> int:
        page_id = self._next_id
        self._next_id += 1
        self._data.setdefault(schematic_id, []).append(
            {
                "id": page_id,
                "schematic_id": schematic_id,
                "smetadata": smetadata,
                "words_bboxes": words_bboxes,
            }
        )
        self._save()
        return page_id

    def find_pages_by_components(
        self,
        schematic_id: str,
        component_names: list[str],
        *,
        match_all: bool = False,
    ) -> list[PageRecord]:
        pages = self._data.get(schematic_id, [])
        results: list[PageRecord] = []
        wanted = set(component_names)

        for page in pages:
            keys = set(page["words_bboxes"].keys())
            if match_all:
                if wanted.issubset(keys):
                    results.append(self._to_record(page))
            else:
                if wanted & keys:
                    results.append(self._to_record(page))

        results.sort(key=lambda r: int(r.smetadata.get("page_number", 0)))
        return results

    def get_all_pages(self, schematic_id: str) -> list[PageRecord]:
        pages = self._data.get(schematic_id, [])
        records = [self._to_record(page) for page in pages]
        records.sort(key=lambda r: int(r.smetadata.get("page_number", 0)))
        return records

    def delete_schematic(self, schematic_id: str) -> None:
        self._data.pop(schematic_id, None)
        self._save()

    @staticmethod
    def _to_record(page: dict[str, Any]) -> PageRecord:
        return PageRecord(
            id=page["id"],
            schematic_id=page["schematic_id"],
            smetadata=page["smetadata"],
            words_bboxes=page["words_bboxes"],
        )


def build_repository(settings: Any) -> SchematicRepository:
    if settings.database_url:
        return PostgresSchematicRepository(
            settings.database_url,
            table_name=settings.db_table,
        )
    return LocalJsonRepository(settings.local_db_path)
