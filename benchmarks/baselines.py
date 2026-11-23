"""Baseline context strategies and ctx-pager configurations for quantitative comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmarks.datasets import MemoryOp, Scenario, TurnEntry
from ctx_pager.engine import ContextPager
from ctx_pager.models import BudgetProfile, MemoryRecord, Resolution
from ctx_pager.retriever import RetrievalConfig
from ctx_pager.tokenizer import estimate_tokens

_FIRST_CLAUSE_RE = re.compile(r"[。！？\n.!?]+")


@dataclass
class StrategyOutput:
    rendered_text: str
    used_tokens: int
    page_in_reachable_texts: list[str]


def build_engine_up_to_turn(
    scenario: Scenario,
    target_turn: int,
    *,
    retrieval_config: RetrievalConfig | None = None,
) -> ContextPager:
    """Construct an in-memory ContextPager replaying scenario operations up to `target_turn`."""
    engine = ContextPager(":memory:", retrieval_config=retrieval_config)
    alias_to_id: dict[str, str] = {}

    ops = sorted(
        (op for op in scenario.operations if op.turn <= target_turn),
        key=lambda o: o.turn,
    )
    for op in ops:
        engine.storage.set_turn(op.turn)
        if op.op == "commit":
            rec = engine.commit(
                kind=op.kind,
                scope=op.scope,
                title=op.title,
                summary=op.summary,
                detail=op.detail,
                entities=list(op.entities),
                pinned=op.pinned,
                turn=op.turn,
            )
            if op.alias:
                alias_to_id[op.alias] = rec.id
        elif op.op == "supersede":
            old_id = alias_to_id[op.target_alias]
            _, new_rec = engine.supersede(
                old_id,
                title=op.title,
                summary=op.summary,
                detail=op.detail,
                entities=list(op.entities),
                turn=op.turn,
            )
            if op.alias:
                alias_to_id[op.alias] = new_rec.id
        elif op.op == "link":
            src_id = alias_to_id[op.src_alias]
            dst_id = alias_to_id[op.dst_alias]
            engine.link(src_id, dst_id, op.relation, weight=op.weight, turn=op.turn)
        elif op.op == "resolve":
            engine.resolve(alias_to_id[op.target_alias], turn=op.turn)
        elif op.op == "checkpoint":
            engine.checkpoint(
                goal=op.goal, state_summary=op.state_summary, turn=op.turn
            )

    # Trigger realistic episodic compaction once sessions pass turn 55.
    if target_turn >= 55:
        engine.storage.set_turn(target_turn)
        engine.compact(older_than_turns=15, min_group_size=3, max_access_count=1)

    engine.storage.set_turn(target_turn)
    return engine


def run_sliding_window(turns: list[TurnEntry], query_turn: int, budget: int) -> StrategyOutput:
    """Keep the newest chronological turns fitting strictly inside `budget` tokens."""
    eligible = [t for t in turns if t.turn <= query_turn]
    selected_lines: list[str] = []
    used = 0
    for entry in reversed(eligible):
        line = f"[T{entry.turn} {entry.role}] {entry.text}"
        cost = estimate_tokens(line)
        if used + cost > budget:
            break
        selected_lines.append(line)
        used += cost
    selected_lines.reverse()
    rendered = "\n".join(selected_lines)
    return StrategyOutput(
        rendered_text=rendered,
        used_tokens=estimate_tokens(rendered),
        page_in_reachable_texts=[],
    )


def run_rolling_summary(turns: list[TurnEntry], query_turn: int, budget: int) -> StrategyOutput:
    """Allocate ~45% budget to recent verbatim turns and ~55% to rolling older turn digests."""
    eligible = [t for t in turns if t.turn <= query_turn]
    tail_budget = int(budget * 0.45)
    summary_budget = max(64, budget - tail_budget - 16)

    tail_lines: list[str] = []
    tail_used = 0
    split_idx = len(eligible)
    for idx in range(len(eligible) - 1, -1, -1):
        entry = eligible[idx]
        line = f"[T{entry.turn} {entry.role}] {entry.text}"
        cost = estimate_tokens(line)
        if tail_used + cost > tail_budget:
            break
        tail_lines.append(line)
        tail_used += cost
        split_idx = idx
    tail_lines.reverse()

    older = eligible[:split_idx]
    digest_lines: list[str] = []
    digest_used = 0
    for entry in older:
        first_clause = _FIRST_CLAUSE_RE.split(entry.text.strip())[0].strip()
        if not first_clause:
            continue
        snippet = f"- T{entry.turn}: {first_clause[:96]}"
        cost = estimate_tokens(snippet)
        if digest_used + cost > summary_budget:
            break
        digest_lines.append(snippet)
        digest_used += cost

    parts: list[str] = []
    if digest_lines:
        parts.append("# Rolling Summary\n" + "\n".join(digest_lines))
    if tail_lines:
        parts.append("# Recent Turns\n" + "\n".join(tail_lines))
    rendered = "\n".join(parts)
    while estimate_tokens(rendered) > budget and len(rendered) > 32:
        rendered = rendered[int(len(rendered) * 0.15) :]
    return StrategyOutput(
        rendered_text=rendered,
        used_tokens=estimate_tokens(rendered),
        page_in_reachable_texts=[],
    )


def run_flat_bm25(
    engine: ContextPager, query: str, budget: int
) -> StrategyOutput:
    """Flat FTS5 BM25 retrieval packing full L2 entries without conflict gating, graph, MMR, or L0/L1."""
    hits = engine.storage.fts_search(query, limit=40)
    records = engine.storage.get_memories_by_ids([mid for mid, _ in hits])
    # Fallback to recent order if BM25 returned too few records.
    if len(records) < 5:
        seen = {r.id for r in records}
        for rec in reversed(engine.storage.list_memories()):
            if rec.id not in seen:
                records.append(rec)

    lines: list[str] = []
    used = 0
    for rec in records:
        text = rec.render(Resolution.L2)
        cost = estimate_tokens(text)
        if used + cost <= budget:
            lines.append(text)
            used += cost
    rendered = "\n".join(lines)
    while estimate_tokens(rendered) > budget and lines:
        lines.pop()
        rendered = "\n".join(lines)
    return StrategyOutput(
        rendered_text=rendered,
        used_tokens=estimate_tokens(rendered),
        page_in_reachable_texts=[],
    )


def run_ctx_pager_strategy(
    engine: ContextPager,
    query: str,
    query_turn: int,
    budget: int,
    *,
    multi_resolution: bool,
) -> StrategyOutput:
    profile = BudgetProfile(name=f"bench_{budget}", total_tokens=budget)
    frame = engine.recall(
        query,
        profile=profile,
        multi_resolution=multi_resolution,
        turn=query_turn,
        record_access=False,
    )
    reachable_texts: list[str] = []
    if frame.page_in_hints:
        paged = engine.page_in(frame.page_in_hints, include_neighbors=False, turn=query_turn)
        for rec in paged["records"]:
            reachable_texts.append(rec.render(Resolution.L2))

    return StrategyOutput(
        rendered_text=frame.rendered,
        used_tokens=frame.used_tokens,
        page_in_reachable_texts=reachable_texts,
    )
