"""SQLite + FTS5 + hash-chained append-only event ledger storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctx_pager.models import (
    CheckpointRecord,
    EventRecord,
    LinkRelation,
    MemoryKind,
    MemoryLink,
    MemoryRecord,
    MemoryStatus,
)
from ctx_pager.tokenizer import build_fts_query, to_fts_document

_GENESIS_HASH = "0" * 64
_SCHEMA_VERSION = "1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compute_event_hash(
    prev_hash: str, turn: int, event_type: str, payload: dict[str, Any]
) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    raw = f"{prev_hash}|{turn}|{event_type}|{canonical}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MemoryStorage:
    """Three-tier backing store: indexed memory pages, link graph, and tamper-evident event log."""

    def __init__(self, db_path: str | Path = ".ctx_pager/memory.db") -> None:
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._init_schema()

    def _configure_connection(self) -> None:
        if not self._is_memory:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA busy_timeout=5000;")

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    turn INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn);

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    supersedes_id TEXT,
                    superseded_by_id TEXT,
                    turn_created INTEGER NOT NULL,
                    turn_updated INTEGER NOT NULL,
                    turn_last_accessed INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    source_events_json TEXT NOT NULL,
                    token_l0 INTEGER NOT NULL,
                    token_l1 INTEGER NOT NULL,
                    token_l2 INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_status_kind ON memories(status, kind);
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
                CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned, status);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    title_tokens,
                    summary_tokens,
                    detail_tokens,
                    entity_tokens,
                    tokenize="unicode61 tokenchars '_'"
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id, relation),
                    FOREIGN KEY (src_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (dst_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_links_dst ON memory_links(dst_id);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    turn INTEGER NOT NULL,
                    event_seq INTEGER NOT NULL,
                    goal TEXT NOT NULL,
                    state_summary TEXT NOT NULL,
                    active_open_loops_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_turn ON checkpoints(turn);
                """
            )
            defaults = {
                "schema_version": _SCHEMA_VERSION,
                "current_turn": "0",
                "next_mem_seq": "1",
                "next_cp_seq": "1",
                "next_evt_seq": "1",
            }
            for key, val in defaults.items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, val)
                )

    def close(self) -> None:
        self.conn.close()

    def _get_meta(self, key: str, default: str = "0") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_current_turn(self) -> int:
        return int(self._get_meta("current_turn", "0"))

    def set_turn(self, turn: int) -> int:
        with self.conn:
            self._set_meta("current_turn", str(max(0, turn)))
        return self.get_current_turn()

    def advance_turn(self, delta: int = 1) -> int:
        new_turn = self.get_current_turn() + delta
        return self.set_turn(new_turn)

    def _alloc_seq(self, key: str) -> int:
        val = int(self._get_meta(key, "1"))
        self._set_meta(key, str(val + 1))
        return val

    def next_memory_id(self) -> str:
        with self.conn:
            seq = self._alloc_seq("next_mem_seq")
        return f"M-{seq:04d}"

    def next_checkpoint_id(self) -> str:
        with self.conn:
            seq = self._alloc_seq("next_cp_seq")
        return f"CP-{seq:04d}"

    def _append_event_in_tx(
        self,
        turn: int,
        event_type: str,
        payload: dict[str, Any],
        created_at: str | None = None,
    ) -> EventRecord:
        last_row = self.conn.execute(
            "SELECT content_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = last_row["content_hash"] if last_row else _GENESIS_HASH
        evt_num = self._alloc_seq("next_evt_seq")
        event_id = f"E-{evt_num:06d}"
        ts = created_at or _utc_now()
        content_hash = _compute_event_hash(prev_hash, turn, event_type, payload)
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        cur = self.conn.execute(
            """
            INSERT INTO events(event_id, turn, event_type, payload_json, prev_hash, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, turn, event_type, payload_json, prev_hash, content_hash, ts),
        )
        seq = int(cur.lastrowid or evt_num)
        return EventRecord(
            seq=seq,
            event_id=event_id,
            turn=turn,
            event_type=event_type,
            payload=payload,
            prev_hash=prev_hash,
            content_hash=content_hash,
            created_at=ts,
        )

    def append_event(
        self, turn: int, event_type: str, payload: dict[str, Any]
    ) -> EventRecord:
        with self.conn:
            if turn > self.get_current_turn():
                self._set_meta("current_turn", str(turn))
            return self._append_event_in_tx(turn=turn, event_type=event_type, payload=payload)

    def _sync_fts(self, record: MemoryRecord) -> None:
        self.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (record.id,))
        self.conn.execute(
            """
            INSERT INTO memories_fts(memory_id, title_tokens, summary_tokens, detail_tokens, entity_tokens)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.id,
                to_fts_document(record.title),
                to_fts_document(record.summary),
                to_fts_document(record.detail),
                to_fts_document(" ".join(record.entities + record.tags)),
            ),
        )

    def _insert_memory_row(self, record: MemoryRecord) -> None:
        record.recompute_token_costs()
        self.conn.execute(
            """
            INSERT INTO memories(
                id, kind, scope, title, summary, detail, entities_json, tags_json,
                importance, confidence, status, pinned, supersedes_id, superseded_by_id,
                turn_created, turn_updated, turn_last_accessed, access_count,
                source_events_json, token_l0, token_l1, token_l2, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.kind.value,
                record.scope,
                record.title,
                record.summary,
                record.detail,
                json.dumps(record.entities, ensure_ascii=False),
                json.dumps(record.tags, ensure_ascii=False),
                record.importance,
                record.confidence,
                record.status.value,
                1 if record.pinned else 0,
                record.supersedes_id,
                record.superseded_by_id,
                record.turn_created,
                record.turn_updated,
                record.turn_last_accessed,
                record.access_count,
                json.dumps(record.source_events, ensure_ascii=False),
                record.token_l0,
                record.token_l1,
                record.token_l2,
                record.created_at,
                record.updated_at,
            ),
        )
        self._sync_fts(record)

    def _update_memory_row(self, record: MemoryRecord) -> None:
        record.recompute_token_costs()
        self.conn.execute(
            """
            UPDATE memories SET
                kind = ?, scope = ?, title = ?, summary = ?, detail = ?,
                entities_json = ?, tags_json = ?, importance = ?, confidence = ?,
                status = ?, pinned = ?, supersedes_id = ?, superseded_by_id = ?,
                turn_created = ?, turn_updated = ?, turn_last_accessed = ?,
                access_count = ?, source_events_json = ?,
                token_l0 = ?, token_l1 = ?, token_l2 = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                record.kind.value,
                record.scope,
                record.title,
                record.summary,
                record.detail,
                json.dumps(record.entities, ensure_ascii=False),
                json.dumps(record.tags, ensure_ascii=False),
                record.importance,
                record.confidence,
                record.status.value,
                1 if record.pinned else 0,
                record.supersedes_id,
                record.superseded_by_id,
                record.turn_created,
                record.turn_updated,
                record.turn_last_accessed,
                record.access_count,
                json.dumps(record.source_events, ensure_ascii=False),
                record.token_l0,
                record.token_l1,
                record.token_l2,
                record.updated_at,
                record.id,
            ),
        )
        self._sync_fts(record)

    def insert_memory(
        self, record: MemoryRecord, turn: int | None = None
    ) -> tuple[MemoryRecord, EventRecord]:
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = record.created_at or _utc_now()
        record.created_at = ts
        record.updated_at = record.updated_at or ts
        record.turn_created = record.turn_created or eff_turn
        record.turn_updated = eff_turn
        record.turn_last_accessed = max(record.turn_last_accessed, eff_turn)

        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            evt = self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.committed",
                payload={"record": record.to_dict()},
                created_at=ts,
            )
            if evt.event_id not in record.source_events:
                record.source_events.append(evt.event_id)
            self._insert_memory_row(record)
        return record, evt

    def update_memory(
        self, record: MemoryRecord, turn: int | None = None
    ) -> tuple[MemoryRecord, EventRecord]:
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        record.updated_at = ts
        record.turn_updated = eff_turn
        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            evt = self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.updated",
                payload={"record": record.to_dict()},
                created_at=ts,
            )
            if evt.event_id not in record.source_events:
                record.source_events.append(evt.event_id)
            self._update_memory_row(record)
        return record, evt

    def supersede_memory(
        self, old_id: str, new_record: MemoryRecord, turn: int | None = None
    ) -> tuple[MemoryRecord, MemoryRecord, EventRecord]:
        old_record = self.get_memory(old_id)
        if old_record is None:
            raise KeyError(f"Memory not found: {old_id}")

        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        new_record.created_at = new_record.created_at or ts
        new_record.updated_at = ts
        new_record.turn_created = new_record.turn_created or eff_turn
        new_record.turn_updated = eff_turn
        new_record.turn_last_accessed = max(new_record.turn_last_accessed, eff_turn)
        new_record.supersedes_id = old_id

        old_record.status = MemoryStatus.SUPERSEDED
        old_record.superseded_by_id = new_record.id
        old_record.pinned = False
        old_record.turn_updated = eff_turn
        old_record.updated_at = ts

        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            evt = self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.superseded",
                payload={
                    "old_id": old_id,
                    "new_record": new_record.to_dict(),
                },
                created_at=ts,
            )
            if evt.event_id not in new_record.source_events:
                new_record.source_events.append(evt.event_id)
            self._insert_memory_row(new_record)
            self._update_memory_row(old_record)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memory_links(src_id, dst_id, relation, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_record.id, old_id, LinkRelation.SUPERSEDES.value, 1.0, ts),
            )
        return old_record, new_record, evt

    def set_status(
        self, memory_id: str, status: MemoryStatus, turn: int | None = None
    ) -> MemoryRecord:
        record = self.get_memory(memory_id)
        if record is None:
            raise KeyError(f"Memory not found: {memory_id}")
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        record.status = status
        if status != MemoryStatus.ACTIVE:
            record.pinned = False
        record.turn_updated = eff_turn
        record.updated_at = ts
        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.status_changed",
                payload={
                    "id": memory_id,
                    "status": status.value,
                    "pinned": record.pinned,
                },
                created_at=ts,
            )
            self._update_memory_row(record)
        return record

    def set_pinned(
        self, memory_id: str, pinned: bool, turn: int | None = None
    ) -> MemoryRecord:
        record = self.get_memory(memory_id)
        if record is None:
            raise KeyError(f"Memory not found: {memory_id}")
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        record.pinned = pinned
        record.turn_updated = eff_turn
        record.updated_at = ts
        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.pinned_changed",
                payload={"id": memory_id, "pinned": pinned},
                created_at=ts,
            )
            self._update_memory_row(record)
        return record

    def record_access(self, memory_ids: list[str], turn: int) -> None:
        if not memory_ids:
            return
        with self.conn:
            for mid in memory_ids:
                self.conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        turn_last_accessed = MAX(turn_last_accessed, ?)
                    WHERE id = ?
                    """,
                    (turn, mid),
                )

    def add_link(
        self,
        src_id: str,
        dst_id: str,
        relation: LinkRelation,
        weight: float = 1.0,
        turn: int | None = None,
    ) -> MemoryLink:
        if self.get_memory(src_id) is None:
            raise KeyError(f"Source memory not found: {src_id}")
        if self.get_memory(dst_id) is None:
            raise KeyError(f"Target memory not found: {dst_id}")
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        clamped = max(0.0, min(1.0, float(weight)))
        with self.conn:
            self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.linked",
                payload={
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "relation": relation.value,
                    "weight": clamped,
                },
                created_at=ts,
            )
            self.conn.execute(
                """
                INSERT INTO memory_links(src_id, dst_id, relation, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(src_id, dst_id, relation)
                DO UPDATE SET weight = excluded.weight, created_at = excluded.created_at
                """,
                (src_id, dst_id, relation.value, clamped, ts),
            )
        return MemoryLink(
            src_id=src_id, dst_id=dst_id, relation=relation, weight=clamped, created_at=ts
        )

    def get_neighbors(self, memory_ids: list[str]) -> list[MemoryLink]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.conn.execute(
            f"""
            SELECT src_id, dst_id, relation, weight, created_at
            FROM memory_links
            WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
            """,
            memory_ids + memory_ids,
        ).fetchall()
        return [
            MemoryLink(
                src_id=row["src_id"],
                dst_id=row["dst_id"],
                relation=LinkRelation(row["relation"]),
                weight=float(row["weight"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def save_checkpoint(
        self,
        goal: str,
        state_summary: str,
        active_open_loops: list[str] | None = None,
        turn: int | None = None,
        checkpoint_id: str | None = None,
    ) -> CheckpointRecord:
        eff_turn = self.get_current_turn() if turn is None else turn
        cp_id = checkpoint_id or self.next_checkpoint_id()
        ts = _utc_now()
        loops = list(active_open_loops or [])
        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            evt = self._append_event_in_tx(
                turn=eff_turn,
                event_type="checkpoint.saved",
                payload={
                    "id": cp_id,
                    "turn": eff_turn,
                    "goal": goal,
                    "state_summary": state_summary,
                    "active_open_loops": loops,
                },
                created_at=ts,
            )
            self.conn.execute(
                """
                INSERT INTO checkpoints(id, turn, event_seq, goal, state_summary, active_open_loops_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cp_id,
                    eff_turn,
                    evt.seq,
                    goal,
                    state_summary,
                    json.dumps(loops, ensure_ascii=False),
                    ts,
                ),
            )
        return CheckpointRecord(
            id=cp_id,
            turn=eff_turn,
            event_seq=evt.seq,
            goal=goal,
            state_summary=state_summary,
            active_open_loops=loops,
            created_at=ts,
        )

    def get_latest_checkpoint(self) -> CheckpointRecord | None:
        row = self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY turn DESC, event_seq DESC LIMIT 1"
        ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def list_checkpoints(self) -> list[CheckpointRecord]:
        rows = self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY turn ASC, event_seq ASC"
        ).fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    def compact_group(
        self,
        archived_ids: list[str],
        digest_record: MemoryRecord,
        turn: int | None = None,
    ) -> tuple[MemoryRecord, EventRecord]:
        eff_turn = self.get_current_turn() if turn is None else turn
        ts = _utc_now()
        digest_record.created_at = digest_record.created_at or ts
        digest_record.updated_at = ts
        digest_record.turn_created = digest_record.turn_created or eff_turn
        digest_record.turn_updated = eff_turn
        digest_record.turn_last_accessed = max(digest_record.turn_last_accessed, eff_turn)

        with self.conn:
            if eff_turn > self.get_current_turn():
                self._set_meta("current_turn", str(eff_turn))
            evt = self._append_event_in_tx(
                turn=eff_turn,
                event_type="memory.compacted",
                payload={
                    "archived_ids": list(archived_ids),
                    "digest_record": digest_record.to_dict(),
                },
                created_at=ts,
            )
            if evt.event_id not in digest_record.source_events:
                digest_record.source_events.append(evt.event_id)
            self._insert_memory_row(digest_record)
            for aid in archived_ids:
                rec = self.get_memory(aid)
                if rec is None:
                    continue
                rec.status = MemoryStatus.ARCHIVED
                rec.pinned = False
                rec.turn_updated = eff_turn
                rec.updated_at = ts
                self._update_memory_row(rec)
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_links(src_id, dst_id, relation, weight, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (digest_record.id, aid, LinkRelation.REFINES.value, 0.5, ts),
                )
        return digest_record, evt

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def get_memories_by_ids(self, memory_ids: list[str]) -> list[MemoryRecord]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids
        ).fetchall()
        by_id = {row["id"]: self._row_to_memory(row) for row in rows}
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        kind: MemoryKind | None = None,
        scope: str | None = None,
        pinned: bool | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if pinned is not None:
            clauses.append("pinned = ?")
            params.append(1 if pinned else 0)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM memories {where} ORDER BY turn_created ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def fts_search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT memory_id, bm25(memories_fts, 0.0, 4.0, 2.0, 1.0, 3.5) AS raw_score
                FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY raw_score ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # SQLite FTS5 bm25() returns negative scores where more negative = stronger match.
        return [(row["memory_id"], max(0.0, -float(row["raw_score"]))) for row in rows]

    def list_events(self, up_to_turn: int | None = None) -> list[EventRecord]:
        if up_to_turn is None:
            rows = self.conn.execute("SELECT * FROM events ORDER BY seq ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE turn <= ? ORDER BY seq ASC", (up_to_turn,)
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def verify_event_chain(self) -> tuple[bool, list[str]]:
        events = self.list_events()
        errors: list[str] = []
        expected_prev = _GENESIS_HASH
        for evt in events:
            if evt.prev_hash != expected_prev:
                errors.append(
                    f"seq={evt.seq} ({evt.event_id}): prev_hash mismatch "
                    f"(expected {expected_prev[:12]}..., got {evt.prev_hash[:12]}...)"
                )
            recomputed = _compute_event_hash(
                evt.prev_hash, evt.turn, evt.event_type, evt.payload
            )
            if evt.content_hash != recomputed:
                errors.append(
                    f"seq={evt.seq} ({evt.event_id}): content_hash tampered "
                    f"(expected {recomputed[:12]}..., got {evt.content_hash[:12]}...)"
                )
            expected_prev = evt.content_hash
        return (len(errors) == 0, errors)

    def replay_events_to_storage(self, target_turn: int) -> "MemoryStorage":
        """Reconstruct deterministic memory state at `target_turn` in an in-memory store."""
        replayed = MemoryStorage(":memory:")
        events = self.list_events(up_to_turn=target_turn)
        max_turn = 0
        max_mem_seq = 0
        max_cp_seq = 0

        def _parse_seq(prefix_id: str) -> int:
            try:
                return int(prefix_id.split("-")[1])
            except (IndexError, ValueError):
                return 0

        with replayed.conn:
            for evt in events:
                max_turn = max(max_turn, evt.turn)
                replayed._append_event_in_tx(
                    turn=evt.turn,
                    event_type=evt.event_type,
                    payload=evt.payload,
                    created_at=evt.created_at,
                )
                etype = evt.event_type
                payload = evt.payload

                if etype == "memory.committed":
                    rec = self._dict_to_memory(payload["record"])
                    max_mem_seq = max(max_mem_seq, _parse_seq(rec.id))
                    replayed._insert_memory_row(rec)

                elif etype == "memory.updated":
                    rec = self._dict_to_memory(payload["record"])
                    replayed._update_memory_row(rec)

                elif etype == "memory.superseded":
                    old_id = payload["old_id"]
                    new_rec = self._dict_to_memory(payload["new_record"])
                    max_mem_seq = max(max_mem_seq, _parse_seq(new_rec.id))
                    replayed._insert_memory_row(new_rec)
                    old_rec = replayed.get_memory(old_id)
                    if old_rec is not None:
                        old_rec.status = MemoryStatus.SUPERSEDED
                        old_rec.superseded_by_id = new_rec.id
                        old_rec.pinned = False
                        old_rec.turn_updated = evt.turn
                        old_rec.updated_at = evt.created_at
                        replayed._update_memory_row(old_rec)
                    replayed.conn.execute(
                        """
                        INSERT OR REPLACE INTO memory_links(src_id, dst_id, relation, weight, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            new_rec.id,
                            old_id,
                            LinkRelation.SUPERSEDES.value,
                            1.0,
                            evt.created_at,
                        ),
                    )

                elif etype == "memory.status_changed":
                    rec = replayed.get_memory(payload["id"])
                    if rec is not None:
                        rec.status = MemoryStatus(payload["status"])
                        rec.pinned = bool(payload.get("pinned", False))
                        rec.turn_updated = evt.turn
                        rec.updated_at = evt.created_at
                        replayed._update_memory_row(rec)

                elif etype == "memory.pinned_changed":
                    rec = replayed.get_memory(payload["id"])
                    if rec is not None:
                        rec.pinned = bool(payload["pinned"])
                        rec.turn_updated = evt.turn
                        rec.updated_at = evt.created_at
                        replayed._update_memory_row(rec)

                elif etype == "memory.linked":
                    replayed.conn.execute(
                        """
                        INSERT OR REPLACE INTO memory_links(src_id, dst_id, relation, weight, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            payload["src_id"],
                            payload["dst_id"],
                            payload["relation"],
                            float(payload.get("weight", 1.0)),
                            evt.created_at,
                        ),
                    )

                elif etype == "checkpoint.saved":
                    cp_id = payload["id"]
                    max_cp_seq = max(max_cp_seq, _parse_seq(cp_id))
                    replayed.conn.execute(
                        """
                        INSERT INTO checkpoints(id, turn, event_seq, goal, state_summary, active_open_loops_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cp_id,
                            int(payload["turn"]),
                            evt.seq,
                            payload["goal"],
                            payload["state_summary"],
                            json.dumps(payload.get("active_open_loops", []), ensure_ascii=False),
                            evt.created_at,
                        ),
                    )

                elif etype == "memory.compacted":
                    digest_rec = self._dict_to_memory(payload["digest_record"])
                    max_mem_seq = max(max_mem_seq, _parse_seq(digest_rec.id))
                    replayed._insert_memory_row(digest_rec)
                    for aid in payload.get("archived_ids", []):
                        old = replayed.get_memory(aid)
                        if old is not None:
                            old.status = MemoryStatus.ARCHIVED
                            old.pinned = False
                            old.turn_updated = evt.turn
                            old.updated_at = evt.created_at
                            replayed._update_memory_row(old)
                        replayed.conn.execute(
                            """
                            INSERT OR IGNORE INTO memory_links(src_id, dst_id, relation, weight, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                digest_rec.id,
                                aid,
                                LinkRelation.REFINES.value,
                                0.5,
                                evt.created_at,
                            ),
                        )

            replayed._set_meta("current_turn", str(min(target_turn, max_turn) if events else target_turn))
            replayed._set_meta("next_mem_seq", str(max_mem_seq + 1))
            replayed._set_meta("next_cp_seq", str(max_cp_seq + 1))

        return replayed

    def export_events_jsonl(self, output_path: str | Path) -> int:
        events = self.list_events()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for evt in events:
                fh.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")
        return len(events)

    def export_snapshot_dict(self) -> dict[str, Any]:
        return {
            "current_turn": self.get_current_turn(),
            "memories": [m.to_dict() for m in self.list_memories()],
            "checkpoints": [c.to_dict() for c in self.list_checkpoints()],
            "events_count": len(self.list_events()),
        }

    def stats(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_status = {
            row["status"]: row["cnt"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM memories GROUP BY status"
            ).fetchall()
        }
        by_kind = {
            row["kind"]: row["cnt"]
            for row in self.conn.execute(
                "SELECT kind, COUNT(*) AS cnt FROM memories WHERE status = 'active' GROUP BY kind"
            ).fetchall()
        }
        links_count = self.conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
        events_count = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        checkpoints_count = self.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        token_totals = self.conn.execute(
            """
            SELECT
                COALESCE(SUM(token_l0), 0) AS sum_l0,
                COALESCE(SUM(token_l1), 0) AS sum_l1,
                COALESCE(SUM(token_l2), 0) AS sum_l2
            FROM memories WHERE status = 'active'
            """
        ).fetchone()
        return {
            "db_path": self.db_path,
            "current_turn": self.get_current_turn(),
            "total_memories": total,
            "by_status": by_status,
            "active_by_kind": by_kind,
            "links_count": links_count,
            "events_count": events_count,
            "checkpoints_count": checkpoints_count,
            "active_tokens_l0": int(token_totals["sum_l0"]),
            "active_tokens_l1": int(token_totals["sum_l1"]),
            "active_tokens_l2": int(token_totals["sum_l2"]),
        }

    def doctor(self) -> dict[str, Any]:
        integrity = self.conn.execute("PRAGMA integrity_check").fetchone()[0]
        mem_cnt = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fts_cnt = self.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        chain_ok, chain_errors = self.verify_event_chain()
        orphan_sup = self.conn.execute(
            """
            SELECT id FROM memories
            WHERE supersedes_id IS NOT NULL
              AND supersedes_id NOT IN (SELECT id FROM memories)
            """
        ).fetchall()
        issues: list[str] = []
        if integrity != "ok":
            issues.append(f"sqlite integrity_check={integrity}")
        if mem_cnt != fts_cnt:
            issues.append(f"FTS sync mismatch: memories={mem_cnt}, memories_fts={fts_cnt}")
        if not chain_ok:
            issues.extend(chain_errors)
        if orphan_sup:
            issues.append(f"Orphan supersedes_id references: {[r['id'] for r in orphan_sup]}")
        return {
            "ok": len(issues) == 0,
            "sqlite_integrity": integrity,
            "memories_count": mem_cnt,
            "fts_count": fts_cnt,
            "event_chain_ok": chain_ok,
            "issues": issues,
        }

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            scope=row["scope"],
            title=row["title"],
            summary=row["summary"],
            detail=row["detail"],
            entities=json.loads(row["entities_json"]),
            tags=json.loads(row["tags_json"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            status=MemoryStatus(row["status"]),
            pinned=bool(row["pinned"]),
            supersedes_id=row["supersedes_id"],
            superseded_by_id=row["superseded_by_id"],
            turn_created=int(row["turn_created"]),
            turn_updated=int(row["turn_updated"]),
            turn_last_accessed=int(row["turn_last_accessed"]),
            access_count=int(row["access_count"]),
            source_events=json.loads(row["source_events_json"]),
            token_l0=int(row["token_l0"]),
            token_l1=int(row["token_l1"]),
            token_l2=int(row["token_l2"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _dict_to_memory(data: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=data["id"],
            kind=MemoryKind(data["kind"]),
            scope=data["scope"],
            title=data["title"],
            summary=data["summary"],
            detail=data.get("detail", ""),
            entities=list(data.get("entities", [])),
            tags=list(data.get("tags", [])),
            importance=data.get("importance"),
            confidence=float(data.get("confidence", 1.0)),
            status=MemoryStatus(data.get("status", "active")),
            pinned=bool(data.get("pinned", False)),
            supersedes_id=data.get("supersedes_id"),
            superseded_by_id=data.get("superseded_by_id"),
            turn_created=int(data.get("turn_created", 0)),
            turn_updated=int(data.get("turn_updated", 0)),
            turn_last_accessed=int(data.get("turn_last_accessed", 0)),
            access_count=int(data.get("access_count", 0)),
            source_events=list(data.get("source_events", [])),
            token_l0=int(data.get("token_l0", 0)),
            token_l1=int(data.get("token_l1", 0)),
            token_l2=int(data.get("token_l2", 0)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            seq=int(row["seq"]),
            event_id=row["event_id"],
            turn=int(row["turn"]),
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            prev_hash=row["prev_hash"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row["id"],
            turn=int(row["turn"]),
            event_seq=int(row["event_seq"]),
            goal=row["goal"],
            state_summary=row["state_summary"],
            active_open_loops=json.loads(row["active_open_loops_json"]),
            created_at=row["created_at"],
        )
