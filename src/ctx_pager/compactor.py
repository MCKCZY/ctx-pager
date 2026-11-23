"""Deterministic turn ingestion (rule-based extraction, dedup, conflict revision) and lossless episodic compaction."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ctx_pager.models import (
    DEFAULT_IMPORTANCE,
    EventRecord,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
)
from ctx_pager.storage import MemoryStorage
from ctx_pager.tokenizer import extract_identifiers, tokenize_text
from ctx_pager.vectorizer import jaccard_similarity, record_similarity

_CLAUSE_SPLIT_RE = re.compile(r"(?:[。！？；\n]+|(?<=[.!?])\s+)")

_CONSTRAINT_CUES = (
    "必须",
    "不要",
    "禁止",
    "不能",
    "不得",
    "严禁",
    "限制",
    "只允许",
    "must ",
    "must not",
    "do not",
    "don't",
    "never ",
    "constraint",
    "forbidden",
    "prohibit",
    "require",
    "only use",
)
_REVISION_CUES = (
    "改为",
    "换成",
    "切换为",
    "更新为",
    "不再",
    "废弃",
    "替代",
    "instead of",
    "changed to",
    "changed from",
    "switched to",
    "updated to",
    "replaced with",
    "migrated to",
    "deprecated",
    "no longer",
)
_DECISION_CUES = _REVISION_CUES + (
    "决定",
    "采用",
    "选用",
    "确定使用",
    "定为",
    "decided",
    "decision:",
    "we will use",
    "chosen ",
)
_OPEN_LOOP_CUES = (
    "todo",
    "待办",
    "稍后",
    "尚未",
    "还没",
    "未完成",
    "待排查",
    "待确认",
    "pending",
    "follow up",
    "follow-up",
    "next step",
    "need to ",
    "open loop",
)
_PROCEDURE_CUES = (
    "步骤",
    "排查",
    "修复",
    "解决方式",
    "命令",
    "workaround",
    "fix:",
    "command:",
    "runbook",
    "step ",
)


@dataclass
class IngestResult:
    event: EventRecord
    created: list[MemoryRecord] = field(default_factory=list)
    updated: list[MemoryRecord] = field(default_factory=list)
    superseded: list[tuple[str, MemoryRecord]] = field(default_factory=list)


@dataclass
class CompactionReport:
    dry_run: bool
    groups_compacted: int
    archived_ids: list[str] = field(default_factory=list)
    digest_ids: list[str] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "groups_compacted": self.groups_compacted,
            "archived_ids": list(self.archived_ids),
            "digest_ids": list(self.digest_ids),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
        }


class TurnIngestor:
    """Extracts structured memories from raw dialogue turns without requiring an LLM."""

    def __init__(self, storage: MemoryStorage) -> None:
        self.storage = storage

    def ingest_turn(
        self,
        *,
        turn: int | None = None,
        role: str = "user",
        text: str,
        scope: str = "global",
    ) -> IngestResult:
        eff_turn = self.storage.advance_turn(1) if turn is None else turn
        if eff_turn > self.storage.get_current_turn():
            self.storage.set_turn(eff_turn)

        evt = self.storage.append_event(
            turn=eff_turn,
            event_type="turn.observed",
            payload={"role": role, "scope": scope, "text": text},
        )
        result = IngestResult(event=evt)

        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c and len(c.strip()) >= 6]
        for clause in clauses:
            kind = self._classify_clause(clause)
            if kind is None:
                continue

            entities = extract_identifiers(clause)
            title = self._make_title(clause)
            summary = clause if len(clause) <= 180 else clause[:177].rstrip() + "..."
            detail = clause if len(clause) > 120 else ""

            candidate = MemoryRecord(
                id="TMP",
                kind=kind,
                scope=scope,
                title=title,
                summary=summary,
                detail=detail,
                entities=entities,
                tags=[f"role:{role}", "auto_ingest"],
                importance=DEFAULT_IMPORTANCE[kind],
                pinned=(kind == MemoryKind.CONSTRAINT),
                turn_created=eff_turn,
                turn_updated=eff_turn,
                turn_last_accessed=eff_turn,
                source_events=[evt.event_id],
            )

            active_same_scope = self.storage.list_memories(
                status=MemoryStatus.ACTIVE, kind=kind, scope=scope
            )

            # 1. Near-duplicate check: merge rather than bloating index.
            dup_target = self._find_near_duplicate(candidate, active_same_scope)
            if dup_target is not None:
                if evt.event_id not in dup_target.source_events:
                    dup_target.source_events.append(evt.event_id)
                for ent in entities:
                    if ent not in dup_target.entities:
                        dup_target.entities.append(ent)
                if len(clause) > len(dup_target.detail or dup_target.summary):
                    dup_target.detail = clause
                dup_target.turn_last_accessed = max(dup_target.turn_last_accessed, eff_turn)
                updated, _ = self.storage.update_memory(dup_target, turn=eff_turn)
                result.updated.append(updated)
                continue

            # 2. Explicit revision / conflict detection within scope.
            if kind in (MemoryKind.DECISION, MemoryKind.CONSTRAINT, MemoryKind.FACT) and self._has_revision_cue(clause):
                sup_target = self._find_conflict_target(candidate, active_same_scope)
                if sup_target is not None:
                    new_id = self.storage.next_memory_id()
                    candidate.id = new_id
                    candidate.pinned = sup_target.pinned or (kind == MemoryKind.CONSTRAINT)
                    _, new_rec, _ = self.storage.supersede_memory(
                        sup_target.id, candidate, turn=eff_turn
                    )
                    result.superseded.append((sup_target.id, new_rec))
                    continue

            candidate.id = self.storage.next_memory_id()
            inserted, _ = self.storage.insert_memory(candidate, turn=eff_turn)
            result.created.append(inserted)

        return result

    @staticmethod
    def _classify_clause(clause: str) -> MemoryKind | None:
        low = clause.lower()
        if any(cue in low for cue in _CONSTRAINT_CUES):
            return MemoryKind.CONSTRAINT
        if any(cue in low for cue in _OPEN_LOOP_CUES):
            return MemoryKind.OPEN_LOOP
        if any(cue in low for cue in _DECISION_CUES):
            return MemoryKind.DECISION
        if any(cue in low for cue in _PROCEDURE_CUES):
            return MemoryKind.PROCEDURE
        if extract_identifiers(clause):
            return MemoryKind.FACT
        return None

    @staticmethod
    def _has_revision_cue(clause: str) -> bool:
        low = clause.lower()
        return any(cue in low for cue in _REVISION_CUES)

    @staticmethod
    def _make_title(clause: str, max_len: int = 64) -> str:
        one_line = " ".join(clause.split())
        if len(one_line) <= max_len:
            return one_line
        return one_line[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _find_near_duplicate(
        candidate: MemoryRecord, existing: list[MemoryRecord]
    ) -> MemoryRecord | None:
        cand_norm = candidate.title.strip().lower()
        for rec in existing:
            if rec.title.strip().lower() == cand_norm:
                return rec
            if record_similarity(candidate, rec) >= 0.78:
                return rec
        return None

    @staticmethod
    def _find_conflict_target(
        candidate: MemoryRecord, existing: list[MemoryRecord]
    ) -> MemoryRecord | None:
        cand_ids = set(e.lower() for e in candidate.entities)
        cand_toks = set(tokenize_text(f"{candidate.title} {candidate.summary}", filter_stopwords=True))
        best_rec: MemoryRecord | None = None
        best_overlap = 0.0

        for rec in existing:
            rec_ids = set(e.lower() for e in rec.entities)
            id_overlap = len(cand_ids & rec_ids)
            rec_toks = set(tokenize_text(f"{rec.title} {rec.summary}", filter_stopwords=True))
            tok_jaccard = jaccard_similarity(cand_toks, rec_toks)
            shared_toks = len(cand_toks & rec_toks)
            if id_overlap >= 1 or tok_jaccard >= 0.28 or shared_toks >= 2:
                score = id_overlap * 1.0 + tok_jaccard
                if score > best_overlap:
                    best_overlap = score
                    best_rec = rec
        return best_rec


class MemoryCompactor:
    """Losslessly consolidates aged, low-access episodic records into compact scope digests."""

    def __init__(self, storage: MemoryStorage) -> None:
        self.storage = storage

    def compact(
        self,
        *,
        older_than_turns: int = 15,
        min_group_size: int = 3,
        max_access_count: int = 1,
        current_turn: int | None = None,
        dry_run: bool = False,
    ) -> CompactionReport:
        turn = self.storage.get_current_turn() if current_turn is None else current_turn
        active_episodes = self.storage.list_memories(
            status=MemoryStatus.ACTIVE,
            kind=MemoryKind.EPISODE,
            pinned=False,
        )

        eligible = [
            rec
            for rec in active_episodes
            if (turn - rec.turn_last_accessed) >= older_than_turns
            and rec.access_count <= max_access_count
        ]

        by_scope: dict[str, list[MemoryRecord]] = defaultdict(list)
        for rec in eligible:
            by_scope[rec.scope].append(rec)

        report = CompactionReport(dry_run=dry_run, groups_compacted=0)

        for scope, records in sorted(by_scope.items()):
            if len(records) < min_group_size:
                continue

            records.sort(key=lambda r: (r.turn_created, r.id))
            tokens_before = sum(r.token_l1 for r in records)
            min_t = records[0].turn_created
            max_t = records[-1].turn_created

            merged_events: list[str] = []
            merged_entities: list[str] = []
            bullet_summaries: list[str] = []
            detail_lines: list[str] = []

            for rec in records:
                for eid in rec.source_events:
                    if eid not in merged_events:
                        merged_events.append(eid)
                for ent in rec.entities:
                    if ent not in merged_entities:
                        merged_entities.append(ent)
                bullet_summaries.append(rec.title)
                detail_lines.append(f"- {rec.id} (T{rec.turn_created}): {rec.summary}")

            summary_joined = "; ".join(bullet_summaries)
            if len(summary_joined) > 220:
                summary_joined = summary_joined[:217].rstrip() + "..."

            digest_id = "DRY-RUN" if dry_run else self.storage.next_memory_id()
            digest = MemoryRecord(
                id=digest_id,
                kind=MemoryKind.EPISODE,
                scope=scope,
                title=f"Compacted {len(records)} episodes ({scope}, T{min_t}-T{max_t})",
                summary=summary_joined,
                detail="\n".join(detail_lines),
                entities=merged_entities[:12],
                tags=["compacted_digest"],
                importance=DEFAULT_IMPORTANCE[MemoryKind.EPISODE],
                turn_created=min_t,
                turn_updated=turn,
                turn_last_accessed=turn,
                source_events=merged_events,
            )

            report.groups_compacted += 1
            report.archived_ids.extend(r.id for r in records)
            report.tokens_before += tokens_before
            report.tokens_after += digest.token_l1

            if not dry_run:
                created_digest, _ = self.storage.compact_group(
                    [r.id for r in records], digest, turn=turn
                )
                report.digest_ids.append(created_digest.id)
            else:
                report.digest_ids.append(digest_id)

        report.tokens_saved = max(0, report.tokens_before - report.tokens_after)
        return report
