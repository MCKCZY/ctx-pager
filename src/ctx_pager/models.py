"""Core domain models, enums, budget profiles, and deterministic multi-resolution rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ctx_pager.tokenizer import estimate_tokens


class MemoryKind(str, Enum):
    CONSTRAINT = "constraint"
    DECISION = "decision"
    FACT = "fact"
    PROCEDURE = "procedure"
    EPISODE = "episode"
    OPEN_LOOP = "open_loop"


DEFAULT_IMPORTANCE: dict[MemoryKind, float] = {
    MemoryKind.CONSTRAINT: 0.95,
    MemoryKind.DECISION: 0.85,
    MemoryKind.OPEN_LOOP: 0.85,
    MemoryKind.PROCEDURE: 0.75,
    MemoryKind.FACT: 0.70,
    MemoryKind.EPISODE: 0.50,
}

DECAY_LAMBDA: dict[MemoryKind, float] = {
    MemoryKind.CONSTRAINT: 0.002,
    MemoryKind.DECISION: 0.010,
    MemoryKind.OPEN_LOOP: 0.005,
    MemoryKind.PROCEDURE: 0.015,
    MemoryKind.FACT: 0.020,
    MemoryKind.EPISODE: 0.055,
}


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class LinkRelation(str, Enum):
    DEPENDS_ON = "depends_on"
    RELATES_TO = "relates_to"
    CAUSED_BY = "caused_by"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    REFINES = "refines"


class Resolution(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


@dataclass
class MemoryRecord:
    """Single unit of paged agent memory with precomputed L0/L1/L2 token footprints."""

    id: str
    kind: MemoryKind
    scope: str
    title: str
    summary: str
    detail: str = ""
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    importance: float | None = None
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE
    pinned: bool = False
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    turn_created: int = 0
    turn_updated: int = 0
    turn_last_accessed: int = 0
    access_count: int = 0
    source_events: list[str] = field(default_factory=list)
    token_l0: int = 0
    token_l1: int = 0
    token_l2: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = MemoryKind(self.kind)
        if isinstance(self.status, str):
            self.status = MemoryStatus(self.status)
        if self.importance is None:
            self.importance = DEFAULT_IMPORTANCE[self.kind]
        else:
            self.importance = max(0.0, min(1.0, float(self.importance)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.recompute_token_costs()

    def recompute_token_costs(self) -> None:
        self.token_l0 = estimate_tokens(self.render(Resolution.L0))
        self.token_l1 = estimate_tokens(self.render(Resolution.L1))
        self.token_l2 = estimate_tokens(self.render(Resolution.L2))

    def render(self, resolution: Resolution) -> str:
        """Render the memory entry deterministically at L0 (stub), L1 (summary), or L2 (detail)."""
        status_tag = "" if self.status == MemoryStatus.ACTIVE else f"|{self.status.value}"
        l0 = f"[{self.id}|{self.kind.value}|{self.scope}{status_tag}] {self.title}"
        if resolution == Resolution.L0:
            return l0

        lines = [l0, f"  {self.summary}"]
        if self.entities:
            lines.append(f"  entities: {', '.join(self.entities)}")
        if resolution == Resolution.L1:
            return "\n".join(lines)

        detail_clean = self.detail.strip()
        if detail_clean and detail_clean != self.summary.strip():
            lines.append(f"  detail: {detail_clean}")
        return "\n".join(lines)

    def has_distinct_detail(self) -> bool:
        detail_clean = self.detail.strip()
        return bool(detail_clean and detail_clean != self.summary.strip())

    def token_cost(self, resolution: Resolution) -> int:
        if resolution == Resolution.L0:
            return self.token_l0
        if resolution == Resolution.L1:
            return self.token_l1
        return self.token_l2

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class MemoryLink:
    src_id: str
    dst_id: str
    relation: LinkRelation
    weight: float = 1.0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event_id: str
    turn: int
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    content_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "turn": self.turn,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    id: str
    turn: int
    event_seq: int
    goal: str
    state_summary: str
    active_open_loops: list[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turn": self.turn,
            "event_seq": self.event_seq,
            "goal": self.goal,
            "state_summary": self.state_summary,
            "active_open_loops": list(self.active_open_loops),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    total_tokens: int
    pinned_ratio: float = 0.25
    open_loop_ratio: float = 0.15
    max_l0_candidates: int = 24
    min_relevance: float = 0.08
    high_relevance_l1_threshold: float = 0.45
    include_superseded: bool = False
    format: str = "markdown"


PRESET_PROFILES: dict[str, BudgetProfile] = {
    "tiny_4k": BudgetProfile(
        name="tiny_4k",
        total_tokens=1024,
        pinned_ratio=0.25,
        open_loop_ratio=0.15,
        max_l0_candidates=16,
    ),
    "small_8k": BudgetProfile(
        name="small_8k",
        total_tokens=2048,
        pinned_ratio=0.25,
        open_loop_ratio=0.15,
        max_l0_candidates=24,
    ),
    "medium_16k": BudgetProfile(
        name="medium_16k",
        total_tokens=4096,
        pinned_ratio=0.22,
        open_loop_ratio=0.15,
        max_l0_candidates=32,
    ),
    "large_32k": BudgetProfile(
        name="large_32k",
        total_tokens=8192,
        pinned_ratio=0.20,
        open_loop_ratio=0.12,
        max_l0_candidates=48,
    ),
}


@dataclass(frozen=True)
class ScoreBreakdown:
    lexical: float
    identifier: float
    vector: float
    recency: float
    importance: float
    graph: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PackedItem:
    record: MemoryRecord
    resolution: Resolution
    slot: str
    score: float
    breakdown: ScoreBreakdown | None
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "kind": self.record.kind.value,
            "scope": self.record.scope,
            "resolution": self.resolution.value,
            "slot": self.slot,
            "score": round(self.score, 4),
            "tokens": self.tokens,
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
            "text": self.record.render(self.resolution),
        }


@dataclass
class ContextFrame:
    query: str
    turn: int
    budget_tokens: int
    used_tokens: int
    slot_tokens: dict[str, int]
    items: list[PackedItem]
    checkpoint: CheckpointRecord | None
    page_in_hints: list[str]
    dropped_count: int
    superseded_filtered_count: int
    rendered: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "turn": self.turn,
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "slot_tokens": dict(self.slot_tokens),
            "items": [item.to_dict() for item in self.items],
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "page_in_hints": list(self.page_in_hints),
            "dropped_count": self.dropped_count,
            "superseded_filtered_count": self.superseded_filtered_count,
            "rendered": self.rendered,
        }
