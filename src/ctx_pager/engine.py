"""High-level ContextPager orchestration facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ctx_pager.compactor import CompactionReport, IngestResult, MemoryCompactor, TurnIngestor
from ctx_pager.models import (
    PRESET_PROFILES,
    BudgetProfile,
    CheckpointRecord,
    ContextFrame,
    LinkRelation,
    MemoryKind,
    MemoryLink,
    MemoryRecord,
    MemoryStatus,
)
from ctx_pager.packer import ContextPacker
from ctx_pager.retriever import HybridRetriever, RetrievalConfig, RetrievalWeights
from ctx_pager.storage import MemoryStorage
from ctx_pager.tokenizer import extract_identifiers


class ContextPager:
    """Unified engine managing write-through memory commits, paged recall, page-in, and replay."""

    def __init__(
        self,
        db_path: str | Path = ".ctx_pager/memory.db",
        *,
        weights: RetrievalWeights | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.storage = MemoryStorage(db_path)
        self.retriever = HybridRetriever(
            self.storage, weights=weights, config=retrieval_config
        )
        self.packer = ContextPacker()
        self.compactor = MemoryCompactor(self.storage)
        self.ingestor = TurnIngestor(self.storage)

    def close(self) -> None:
        self.storage.close()

    def __enter__(self) -> "ContextPager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def _merge_entities(explicit: list[str] | None, text: str) -> list[str]:
        out: list[str] = []
        seen_lower: list[str] = []
        for raw in explicit or []:
            cleaned = raw.strip()
            low = cleaned.lower()
            if cleaned and low not in seen_lower:
                out.append(cleaned)
                seen_lower.append(low)

        added = 0
        for ident in extract_identifiers(text):
            low = ident.lower()
            if any(low == ex or low in ex for ex in seen_lower):
                continue
            out.append(ident)
            seen_lower.append(low)
            added += 1
            if added >= 6:
                break
        return out

    def commit(
        self,
        *,
        kind: str | MemoryKind,
        title: str,
        summary: str,
        detail: str = "",
        scope: str = "global",
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        confidence: float = 1.0,
        pinned: bool = False,
        supersedes_id: str | None = None,
        turn: int | None = None,
    ) -> MemoryRecord:
        eff_kind = MemoryKind(kind) if isinstance(kind, str) else kind
        eff_entities = self._merge_entities(entities, f"{title}\n{summary}\n{detail}")

        mem_id = self.storage.next_memory_id()
        eff_turn = self.storage.get_current_turn() if turn is None else turn
        record = MemoryRecord(
            id=mem_id,
            kind=eff_kind,
            scope=scope,
            title=title.strip(),
            summary=summary.strip(),
            detail=detail.strip(),
            entities=eff_entities,
            tags=list(tags or []),
            importance=importance,
            confidence=confidence,
            pinned=pinned,
            turn_created=eff_turn,
            turn_updated=eff_turn,
            turn_last_accessed=eff_turn,
        )

        if supersedes_id:
            _, new_rec, _ = self.storage.supersede_memory(
                supersedes_id, record, turn=eff_turn
            )
            return new_rec

        inserted, _ = self.storage.insert_memory(record, turn=eff_turn)
        return inserted

    def supersede(
        self,
        old_id: str,
        *,
        title: str,
        summary: str,
        detail: str = "",
        scope: str | None = None,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        pinned: bool | None = None,
        turn: int | None = None,
    ) -> tuple[MemoryRecord, MemoryRecord]:
        old = self.storage.get_memory(old_id)
        if old is None:
            raise KeyError(f"Memory not found: {old_id}")

        eff_entities = self._merge_entities(entities, f"{title}\n{summary}\n{detail}")

        eff_turn = self.storage.get_current_turn() if turn is None else turn
        new_rec = MemoryRecord(
            id=self.storage.next_memory_id(),
            kind=old.kind,
            scope=scope or old.scope,
            title=title.strip(),
            summary=summary.strip(),
            detail=detail.strip(),
            entities=eff_entities,
            tags=list(tags if tags is not None else old.tags),
            importance=importance if importance is not None else old.importance,
            confidence=old.confidence,
            pinned=old.pinned if pinned is None else pinned,
            turn_created=eff_turn,
            turn_updated=eff_turn,
            turn_last_accessed=eff_turn,
        )
        old_updated, new_saved, _ = self.storage.supersede_memory(
            old_id, new_rec, turn=eff_turn
        )
        return old_updated, new_saved

    def recall(
        self,
        query: str = "",
        *,
        budget: int | None = None,
        profile: str | BudgetProfile = "small_8k",
        fmt: str | None = None,
        include_superseded: bool = False,
        multi_resolution: bool = True,
        turn: int | None = None,
        record_access: bool = True,
    ) -> ContextFrame:
        if isinstance(profile, str):
            if profile not in PRESET_PROFILES:
                raise ValueError(
                    f"Unknown profile '{profile}'. Expected one of {sorted(PRESET_PROFILES)}"
                )
            base_profile = PRESET_PROFILES[profile]
        else:
            base_profile = profile

        eff_profile = BudgetProfile(
            name=base_profile.name,
            total_tokens=budget if budget is not None else base_profile.total_tokens,
            pinned_ratio=base_profile.pinned_ratio,
            open_loop_ratio=base_profile.open_loop_ratio,
            max_l0_candidates=base_profile.max_l0_candidates,
            min_relevance=base_profile.min_relevance,
            high_relevance_l1_threshold=base_profile.high_relevance_l1_threshold,
            include_superseded=include_superseded or base_profile.include_superseded,
            format=fmt or base_profile.format,
        )

        eff_turn = self.storage.get_current_turn() if turn is None else turn
        pinned_records = self.storage.list_memories(status=MemoryStatus.ACTIVE, pinned=True)
        open_loops = self.storage.list_memories(
            status=MemoryStatus.ACTIVE, kind=MemoryKind.OPEN_LOOP
        )
        retrieval = self.retriever.retrieve(
            query,
            current_turn=eff_turn,
            include_superseded=eff_profile.include_superseded,
            limit=max(40, eff_profile.max_l0_candidates * 2),
        )
        checkpoint = self.storage.get_latest_checkpoint()

        frame = self.packer.pack(
            query=query,
            turn=eff_turn,
            profile=eff_profile,
            pinned_records=pinned_records,
            open_loop_records=open_loops,
            retrieval_result=retrieval,
            checkpoint=checkpoint,
            multi_resolution=multi_resolution,
        )

        if record_access and frame.items:
            self.storage.record_access([item.record.id for item in frame.items], eff_turn)

        return frame

    def page_in(
        self,
        memory_ids: list[str],
        *,
        include_neighbors: bool = True,
        turn: int | None = None,
    ) -> dict[str, Any]:
        records = self.storage.get_memories_by_ids(memory_ids)
        eff_turn = self.storage.get_current_turn() if turn is None else turn
        if records:
            self.storage.record_access([r.id for r in records], eff_turn)

        links: list[MemoryLink] = []
        neighbor_records: list[MemoryRecord] = []
        if include_neighbors and records:
            links = self.storage.get_neighbors([r.id for r in records])
            target_set = {r.id for r in records}
            neighbor_ids: list[str] = []
            for link in links:
                if link.src_id not in target_set and link.src_id not in neighbor_ids:
                    neighbor_ids.append(link.src_id)
                if link.dst_id not in target_set and link.dst_id not in neighbor_ids:
                    neighbor_ids.append(link.dst_id)
            neighbor_records = self.storage.get_memories_by_ids(neighbor_ids)

        return {
            "records": records,
            "links": links,
            "neighbors": neighbor_records,
        }

    def link(
        self,
        src_id: str,
        dst_id: str,
        relation: str | LinkRelation,
        *,
        weight: float = 1.0,
        turn: int | None = None,
    ) -> MemoryLink:
        rel = LinkRelation(relation) if isinstance(relation, str) else relation
        return self.storage.add_link(src_id, dst_id, rel, weight=weight, turn=turn)

    def resolve(self, memory_id: str, *, turn: int | None = None) -> MemoryRecord:
        return self.storage.set_status(memory_id, MemoryStatus.RESOLVED, turn=turn)

    def pin(
        self, memory_id: str, *, pinned: bool = True, turn: int | None = None
    ) -> MemoryRecord:
        return self.storage.set_pinned(memory_id, pinned, turn=turn)

    def checkpoint(
        self,
        *,
        goal: str,
        state_summary: str,
        open_loops: list[str] | None = None,
        turn: int | None = None,
    ) -> CheckpointRecord:
        loops = list(open_loops) if open_loops is not None else [
            r.id
            for r in self.storage.list_memories(
                status=MemoryStatus.ACTIVE, kind=MemoryKind.OPEN_LOOP
            )
        ]
        return self.storage.save_checkpoint(
            goal=goal,
            state_summary=state_summary,
            active_open_loops=loops,
            turn=turn,
        )

    def compact(
        self,
        *,
        older_than_turns: int = 15,
        min_group_size: int = 3,
        max_access_count: int = 1,
        dry_run: bool = False,
    ) -> CompactionReport:
        return self.compactor.compact(
            older_than_turns=older_than_turns,
            min_group_size=min_group_size,
            max_access_count=max_access_count,
            dry_run=dry_run,
        )

    def ingest_turn(
        self,
        *,
        role: str = "user",
        text: str,
        scope: str = "global",
        turn: int | None = None,
    ) -> IngestResult:
        return self.ingestor.ingest_turn(turn=turn, role=role, text=text, scope=scope)

    def replay(
        self,
        target_turn: int,
        *,
        query: str = "",
        budget: int = 2048,
    ) -> dict[str, Any]:
        replayed_storage = self.storage.replay_events_to_storage(target_turn)
        try:
            replayed_retriever = HybridRetriever(
                replayed_storage,
                weights=self.retriever.weights,
                config=self.retriever.config,
            )
            pinned = replayed_storage.list_memories(status=MemoryStatus.ACTIVE, pinned=True)
            open_loops = replayed_storage.list_memories(
                status=MemoryStatus.ACTIVE, kind=MemoryKind.OPEN_LOOP
            )
            retrieval = replayed_retriever.retrieve(query, current_turn=target_turn)
            profile = BudgetProfile(name="replay", total_tokens=budget)
            frame = self.packer.pack(
                query=query,
                turn=target_turn,
                profile=profile,
                pinned_records=pinned,
                open_loop_records=open_loops,
                retrieval_result=retrieval,
                checkpoint=replayed_storage.get_latest_checkpoint(),
            )
            return {
                "target_turn": target_turn,
                "stats": replayed_storage.stats(),
                "memories": [m.to_dict() for m in replayed_storage.list_memories()],
                "frame": frame,
            }
        finally:
            replayed_storage.close()

    def stats(self) -> dict[str, Any]:
        return self.storage.stats()

    def doctor(self) -> dict[str, Any]:
        return self.storage.doctor()
