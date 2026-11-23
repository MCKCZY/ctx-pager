"""Multi-resolution token-budget packer for L0/L1/L2 memory allocation."""

from __future__ import annotations

import json
from xml.sax.saxutils import escape as xml_escape

from ctx_pager.models import (
    BudgetProfile,
    CheckpointRecord,
    ContextFrame,
    MemoryRecord,
    PackedItem,
    Resolution,
)
from ctx_pager.retriever import RetrievalCandidate, RetrievalResult
from ctx_pager.tokenizer import estimate_tokens

_UTILITY: dict[Resolution, float] = {
    Resolution.L0: 0.35,
    Resolution.L1: 0.82,
    Resolution.L2: 1.00,
}


class ContextPacker:
    """Packs pinned invariants, open loops, and ranked recall into a strict token budget."""

    def pack(
        self,
        *,
        query: str,
        turn: int,
        profile: BudgetProfile,
        pinned_records: list[MemoryRecord],
        open_loop_records: list[MemoryRecord],
        retrieval_result: RetrievalResult,
        checkpoint: CheckpointRecord | None = None,
        multi_resolution: bool = True,
    ) -> ContextFrame:
        total_budget = max(64, int(profile.total_tokens))
        fmt = profile.format if profile.format in ("markdown", "xml", "json") else "markdown"

        # Reserve framing overhead up front based on output format.
        base_overhead = 42 if fmt == "markdown" else (58 if fmt == "xml" else 85)
        usable_budget = max(16, total_budget - base_overhead)

        active_checkpoint: CheckpointRecord | None = None
        cp_tokens = 0
        if checkpoint is not None:
            cp_text = self._render_checkpoint_snippet(checkpoint)
            cp_cost = estimate_tokens(cp_text)
            cp_cap = max(48, int(total_budget * 0.16))
            if cp_cost <= cp_cap and cp_cost < usable_budget:
                active_checkpoint = checkpoint
                cp_tokens = cp_cost

        remaining_global = max(0, usable_budget - cp_tokens)
        packed_ids: set[str] = set()
        pinned_items: list[PackedItem] = []
        open_loop_items: list[PackedItem] = []
        recall_items: list[PackedItem] = []
        dropped_count = 0

        # Slot 1: Pinned invariants (sorted by importance desc, oldest first).
        pinned_cap = min(remaining_global, int(usable_budget * profile.pinned_ratio))
        pinned_sorted = sorted(
            pinned_records,
            key=lambda r: (-(r.importance or 0.0), r.turn_created, r.id),
        )
        pinned_used = 0
        for rec in pinned_sorted:
            if rec.id in packed_ids:
                continue
            slot_left = pinned_cap - pinned_used
            res: Resolution | None = None
            if rec.token_cost(Resolution.L1) <= slot_left:
                res = Resolution.L1
            elif rec.token_cost(Resolution.L0) <= slot_left:
                res = Resolution.L0
            if res is not None:
                cost = rec.token_cost(res)
                pinned_items.append(
                    PackedItem(
                        record=rec,
                        resolution=res,
                        slot="pinned",
                        score=1.0,
                        breakdown=None,
                        tokens=cost,
                    )
                )
                packed_ids.add(rec.id)
                pinned_used += cost
            else:
                dropped_count += 1

        remaining_global = max(0, remaining_global - pinned_used)

        # Slot 2: Active open loops.
        loop_cap = min(remaining_global, int(usable_budget * profile.open_loop_ratio))
        loops_sorted = sorted(
            open_loop_records,
            key=lambda r: (-(r.importance or 0.0), -r.turn_updated, r.id),
        )
        loop_used = 0
        for rec in loops_sorted:
            if rec.id in packed_ids:
                continue
            slot_left = loop_cap - loop_used
            res = None
            if rec.token_cost(Resolution.L1) <= slot_left:
                res = Resolution.L1
            elif rec.token_cost(Resolution.L0) <= slot_left:
                res = Resolution.L0
            if res is not None:
                cost = rec.token_cost(res)
                open_loop_items.append(
                    PackedItem(
                        record=rec,
                        resolution=res,
                        slot="open_loop",
                        score=0.95,
                        breakdown=None,
                        tokens=cost,
                    )
                )
                packed_ids.add(rec.id)
                loop_used += cost
            else:
                dropped_count += 1

        remaining_recall = max(0, remaining_global - loop_used)

        # Slot 3: Query-relevant recall candidates (gets spillover from Slots 1 & 2).
        filtered_cands: list[RetrievalCandidate] = [
            c
            for c in retrieval_result.candidates
            if c.record.id not in packed_ids and c.breakdown.total >= profile.min_relevance
        ][: profile.max_l0_candidates]

        recall_used = 0
        if multi_resolution:
            # Phase 3a: Admit top hits at L1 when high confidence; admit remaining at L0 stubs.
            for idx, cand in enumerate(filtered_cands):
                rec = cand.record
                score = cand.breakdown.total
                slot_left = remaining_recall - recall_used
                if (
                    idx < 3
                    and score >= profile.high_relevance_l1_threshold
                    and rec.token_cost(Resolution.L1) <= slot_left
                ):
                    res = Resolution.L1
                elif rec.token_cost(Resolution.L0) <= slot_left:
                    res = Resolution.L0
                else:
                    dropped_count += 1
                    continue

                cost = rec.token_cost(res)
                recall_items.append(
                    PackedItem(
                        record=rec,
                        resolution=res,
                        slot="recall",
                        score=score,
                        breakdown=cand.breakdown,
                        tokens=cost,
                    )
                )
                packed_ids.add(rec.id)
                recall_used += cost

            # Phase 3b: Marginal-utility greedy upgrades (L0 -> L1 -> L2).
            while True:
                slot_left = remaining_recall - recall_used
                if slot_left <= 0:
                    break
                best_item: PackedItem | None = None
                best_next_res: Resolution | None = None
                best_delta_t = 0
                best_gain_per_tok = 0.0
                did_free_upgrade = False

                for item in recall_items:
                    cur = item.resolution
                    if cur == Resolution.L0:
                        nxt = Resolution.L1
                        delta_u = _UTILITY[Resolution.L1] - _UTILITY[Resolution.L0]
                    elif cur == Resolution.L1 and item.record.has_distinct_detail():
                        nxt = Resolution.L2
                        delta_u = _UTILITY[Resolution.L2] - _UTILITY[Resolution.L1]
                        if item.breakdown and item.breakdown.identifier > 0.0:
                            delta_u *= 1.4
                    else:
                        continue

                    delta_t = item.record.token_cost(nxt) - item.tokens
                    if delta_t <= 0:
                        # Free resolution upgrade when text is identical.
                        if nxt == Resolution.L1 or item.record.has_distinct_detail():
                            item.resolution = nxt
                            item.tokens = item.record.token_cost(nxt)
                            did_free_upgrade = True
                        continue
                    if delta_t > slot_left:
                        continue

                    gain_per_tok = (max(0.05, item.score) * delta_u) / delta_t
                    if gain_per_tok > best_gain_per_tok:
                        best_gain_per_tok = gain_per_tok
                        best_item = item
                        best_next_res = nxt
                        best_delta_t = delta_t

                if best_item is None or best_next_res is None:
                    if did_free_upgrade:
                        continue
                    break
                best_item.resolution = best_next_res
                best_item.tokens = best_item.record.token_cost(best_next_res)
                recall_used += best_delta_t
        else:
            # Flat L2-only packing mode (used for ablation comparisons).
            for cand in filtered_cands:
                rec = cand.record
                cost = rec.token_cost(Resolution.L2)
                if cost <= (remaining_recall - recall_used):
                    recall_items.append(
                        PackedItem(
                            record=rec,
                            resolution=Resolution.L2,
                            slot="recall",
                            score=cand.breakdown.total,
                            breakdown=cand.breakdown,
                            tokens=cost,
                        )
                    )
                    packed_ids.add(rec.id)
                    recall_used += cost
                else:
                    dropped_count += 1

        # Render and strictly enforce used_tokens <= total_budget.
        rendered, active_checkpoint, dropped_extra = self._render_within_budget(
            fmt=fmt,
            query=query,
            turn=turn,
            total_budget=total_budget,
            checkpoint=active_checkpoint,
            pinned_items=pinned_items,
            open_loop_items=open_loop_items,
            recall_items=recall_items,
        )
        dropped_count += dropped_extra

        all_items = pinned_items + open_loop_items + recall_items
        page_in_hints = [
            item.record.id
            for item in all_items
            if item.resolution in (Resolution.L0, Resolution.L1)
            and item.record.has_distinct_detail()
        ]
        used_tokens = estimate_tokens(rendered)
        slot_tokens = {
            "pinned": sum(i.tokens for i in pinned_items),
            "open_loop": sum(i.tokens for i in open_loop_items),
            "recall": sum(i.tokens for i in recall_items),
            "overhead": max(0, used_tokens - sum(i.tokens for i in all_items)),
        }

        return ContextFrame(
            query=query,
            turn=turn,
            budget_tokens=total_budget,
            used_tokens=used_tokens,
            slot_tokens=slot_tokens,
            items=all_items,
            checkpoint=active_checkpoint,
            page_in_hints=page_in_hints,
            dropped_count=dropped_count,
            superseded_filtered_count=retrieval_result.superseded_filtered_count,
            rendered=rendered,
        )

    def _render_within_budget(
        self,
        *,
        fmt: str,
        query: str,
        turn: int,
        total_budget: int,
        checkpoint: CheckpointRecord | None,
        pinned_items: list[PackedItem],
        open_loop_items: list[PackedItem],
        recall_items: list[PackedItem],
    ) -> tuple[str, CheckpointRecord | None, int]:
        dropped_extra = 0
        cp = checkpoint

        while True:
            rendered = self._render_frame(
                fmt=fmt,
                query=query,
                turn=turn,
                total_budget=total_budget,
                checkpoint=cp,
                pinned_items=pinned_items,
                open_loop_items=open_loop_items,
                recall_items=recall_items,
            )
            if estimate_tokens(rendered) <= total_budget:
                return rendered, cp, dropped_extra

            changed, removed = self._trim_one_step(recall_items)
            if changed:
                if removed:
                    dropped_extra += 1
                continue
            changed, removed = self._trim_one_step(open_loop_items)
            if changed:
                if removed:
                    dropped_extra += 1
                continue
            changed, removed = self._trim_one_step(pinned_items)
            if changed:
                if removed:
                    dropped_extra += 1
                continue
            if cp is not None:
                cp = None
                continue
            # All items and checkpoint cleared; emit valid minimal fallback if formatting still exceeds tiny budget.
            if fmt == "json":
                minimal_json = json.dumps(
                    {"turn": turn, "budget_tokens": total_budget, "items": []},
                    separators=(",", ":"),
                )
                if estimate_tokens(minimal_json) <= total_budget:
                    return minimal_json, cp, dropped_extra
                return "{}", cp, dropped_extra
            if fmt == "xml":
                minimal_xml = f'<context_frame turn="{turn}" budget_tokens="{total_budget}"/>'
                if estimate_tokens(minimal_xml) <= total_budget:
                    return minimal_xml, cp, dropped_extra
                return "<context_frame/>", cp, dropped_extra
            while estimate_tokens(rendered) > total_budget and len(rendered) > 16:
                rendered = rendered[: max(16, int(len(rendered) * 0.8))]
            return rendered, cp, dropped_extra

    @staticmethod
    def _trim_one_step(items: list[PackedItem]) -> tuple[bool, bool]:
        """Downgrade the lowest-scoring item or remove the tail item to fit strict budget."""
        if not items:
            return False, False
        # First try downgrading L2 -> L1 or L1 -> L0 starting from the lowest-scoring tail.
        for idx in range(len(items) - 1, -1, -1):
            item = items[idx]
            if item.resolution == Resolution.L2:
                item.resolution = Resolution.L1
                item.tokens = item.record.token_cost(Resolution.L1)
                return True, False
        for idx in range(len(items) - 1, -1, -1):
            item = items[idx]
            if item.resolution == Resolution.L1:
                item.resolution = Resolution.L0
                item.tokens = item.record.token_cost(Resolution.L0)
                return True, False
        items.pop()
        return True, True

    @staticmethod
    def _render_checkpoint_snippet(cp: CheckpointRecord) -> str:
        loops = f" | open={','.join(cp.active_open_loops)}" if cp.active_open_loops else ""
        return f"[{cp.id}@T{cp.turn}] goal: {cp.goal} | state: {cp.state_summary}{loops}"

    def _render_frame(
        self,
        *,
        fmt: str,
        query: str,
        turn: int,
        total_budget: int,
        checkpoint: CheckpointRecord | None,
        pinned_items: list[PackedItem],
        open_loop_items: list[PackedItem],
        recall_items: list[PackedItem],
    ) -> str:
        all_items = pinned_items + open_loop_items + recall_items
        hints = [
            i.record.id
            for i in all_items
            if i.resolution in (Resolution.L0, Resolution.L1) and i.record.has_distinct_detail()
        ]

        if fmt == "json":
            payload = {
                "turn": turn,
                "budget_tokens": total_budget,
                "query": query,
                "checkpoint": checkpoint.to_dict() if checkpoint else None,
                "pinned": [i.to_dict() for i in pinned_items],
                "open_loops": [i.to_dict() for i in open_loop_items],
                "recall": [i.to_dict() for i in recall_items],
                "page_in_hints": hints,
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if fmt == "xml":
            lines = [f'<context_frame turn="{turn}" budget_tokens="{total_budget}">']
            if checkpoint is not None:
                lines.append(
                    f'  <checkpoint id="{xml_escape(checkpoint.id)}" turn="{checkpoint.turn}">'
                    f"{xml_escape(self._render_checkpoint_snippet(checkpoint))}</checkpoint>"
                )
            for tag, group in (
                ("pinned", pinned_items),
                ("open_loops", open_loop_items),
                ("recall", recall_items),
            ):
                if not group:
                    continue
                lines.append(f"  <{tag}>")
                for item in group:
                    lines.append(
                        f'    <memory id="{xml_escape(item.record.id)}" '
                        f'res="{item.resolution.value}" kind="{item.record.kind.value}">'
                        f"{xml_escape(item.record.render(item.resolution))}</memory>"
                    )
                lines.append(f"  </{tag}>")
            if hints:
                lines.append(f"  <page_in_hints>{','.join(hints)}</page_in_hints>")
            lines.append("</context_frame>")
            return "\n".join(lines)

        # Default markdown format.
        lines = [f"# Working Memory Frame (T{turn}, budget={total_budget})"]
        if checkpoint is not None:
            lines.append("## Checkpoint")
            lines.append(self._render_checkpoint_snippet(checkpoint))
        if pinned_items:
            lines.append("## Pinned Invariants")
            for item in pinned_items:
                lines.append(item.record.render(item.resolution))
        if open_loop_items:
            lines.append("## Open Loops")
            for item in open_loop_items:
                lines.append(item.record.render(item.resolution))
        if recall_items:
            lines.append("## Recalled Context")
            for item in recall_items:
                lines.append(item.record.render(item.resolution))
        if hints:
            lines.append(f"Page-in available: {', '.join(hints)}")
        return "\n".join(lines)
