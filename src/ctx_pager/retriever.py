"""Hybrid retriever with conflict gating, Ebbinghaus decay, 1-hop graph expansion, and MMR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ctx_pager.models import DECAY_LAMBDA, LinkRelation, MemoryRecord, MemoryStatus, ScoreBreakdown
from ctx_pager.storage import MemoryStorage
from ctx_pager.tokenizer import extract_identifiers, tokenize_text
from ctx_pager.vectorizer import Embedder, SparseTfIdfEmbedder, record_text


@dataclass(frozen=True)
class RetrievalWeights:
    lexical: float = 0.38
    identifier: float = 0.22
    vector: float = 0.16
    recency: float = 0.12
    importance: float = 0.12
    graph_boost: float = 0.28
    mmr_lambda: float = 0.78


@dataclass(frozen=True)
class RetrievalConfig:
    enable_conflict_gating: bool = True
    enable_graph_expansion: bool = True
    enable_mmr: bool = True
    enable_recency_decay: bool = True
    seed_limit: int = 5
    fts_limit: int = 60


@dataclass
class RetrievalCandidate:
    record: MemoryRecord
    breakdown: ScoreBreakdown


@dataclass
class RetrievalResult:
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    superseded_filtered_count: int = 0


class HybridRetriever:
    """Combines FTS5 BM25, exact identifier overlap, sparse vectors, decay, graph links, and MMR."""

    def __init__(
        self,
        storage: MemoryStorage,
        *,
        weights: RetrievalWeights | None = None,
        config: RetrievalConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.storage = storage
        self.weights = weights or RetrievalWeights()
        self.config = config or RetrievalConfig()
        self.custom_embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        current_turn: int | None = None,
        include_superseded: bool = False,
        limit: int = 40,
    ) -> RetrievalResult:
        turn = self.storage.get_current_turn() if current_turn is None else current_turn
        all_records = self.storage.list_memories()
        if not all_records:
            return RetrievalResult()

        gate_superseded = self.config.enable_conflict_gating and not include_superseded
        pool: list[MemoryRecord] = []
        superseded_pool: list[MemoryRecord] = []

        for rec in all_records:
            if rec.status == MemoryStatus.ACTIVE:
                pool.append(rec)
            elif rec.status == MemoryStatus.SUPERSEDED:
                if gate_superseded:
                    superseded_pool.append(rec)
                else:
                    pool.append(rec)

        if not pool:
            return RetrievalResult()

        query_clean = query.strip()
        if not query_clean:
            candidates = self._score_recency_importance_only(pool, turn)
            return RetrievalResult(candidates=candidates[:limit], superseded_filtered_count=0)

        fts_scores = dict(self.storage.fts_search(query_clean, limit=self.config.fts_limit))

        superseded_filtered_count = 0
        if superseded_pool:
            query_ids_check = set(extract_identifiers(query_clean))
            query_toks_check = set(tokenize_text(query_clean, filter_stopwords=True))
            for srec in superseded_pool:
                if srec.id in fts_scores:
                    superseded_filtered_count += 1
                    continue
                if query_ids_check and (query_ids_check & self._record_identifiers(srec)):
                    superseded_filtered_count += 1
                    continue
                if query_toks_check:
                    rec_toks = set(tokenize_text(f"{srec.title} {srec.summary}", filter_stopwords=True))
                    if len(query_toks_check & rec_toks) >= 2:
                        superseded_filtered_count += 1

        embedder = self.custom_embedder
        if embedder is None:
            tfidf = SparseTfIdfEmbedder()
            tfidf.fit([record_text(r) for r in pool])
            embedder = tfidf
        query_vec = embedder.embed(query_clean)
        query_ids = set(extract_identifiers(query_clean))

        w = self.weights
        scored_map: dict[str, ScoreBreakdown] = {}
        record_map = {r.id: r for r in pool}

        for rec in pool:
            raw_bm25 = fts_scores.get(rec.id, 0.0)
            lexical = raw_bm25 / (raw_bm25 + 2.5) if raw_bm25 > 0.0 else 0.0

            if query_ids:
                rec_ids = self._record_identifiers(rec)
                hits = len(query_ids & rec_ids)
                identifier = hits / len(query_ids)
            else:
                identifier = 0.0

            rec_vec = embedder.embed(record_text(rec))
            vector = embedder.similarity(query_vec, rec_vec) if query_vec and rec_vec else 0.0

            if self.config.enable_recency_decay:
                delta_t = max(0, turn - rec.turn_last_accessed)
                stability = 1.0 + 0.65 * math.log1p(max(0, rec.access_count))
                lam = DECAY_LAMBDA[rec.kind]
                recency = math.exp(-lam * delta_t / stability)
            else:
                recency = 0.5

            importance = rec.importance if rec.importance is not None else 0.7

            # Guard against unrelated high-importance memories floating solely on priors
            # when query has zero lexical, identifier, or vector signal.
            semantic_signal = lexical + identifier + vector
            base_total = (
                w.lexical * lexical
                + w.identifier * identifier
                + w.vector * vector
                + w.recency * recency
                + w.importance * importance
            )
            if semantic_signal <= 1e-6:
                base_total *= 0.25

            scored_map[rec.id] = ScoreBreakdown(
                lexical=lexical,
                identifier=identifier,
                vector=vector,
                recency=recency,
                importance=importance,
                graph=0.0,
                total=base_total,
            )

        if self.config.enable_graph_expansion:
            seeds = sorted(
                scored_map.items(),
                key=lambda kv: kv[1].total,
                reverse=True,
            )[: self.config.seed_limit]
            seed_scores = {
                rid: bd.total
                for rid, bd in seeds
                if (bd.lexical + bd.identifier + bd.vector) > 0.02 and bd.total > 0.12
            }
            if seed_scores:
                links = self.storage.get_neighbors(list(seed_scores.keys()))
                graph_boosts: dict[str, float] = {}
                for link in links:
                    if gate_superseded and link.relation == LinkRelation.SUPERSEDES:
                        continue
                    if link.src_id in seed_scores and link.dst_id in record_map:
                        boost = seed_scores[link.src_id] * link.weight * w.graph_boost
                        graph_boosts[link.dst_id] = max(graph_boosts.get(link.dst_id, 0.0), boost)
                    if link.dst_id in seed_scores and link.src_id in record_map:
                        boost = seed_scores[link.dst_id] * link.weight * w.graph_boost
                        graph_boosts[link.src_id] = max(graph_boosts.get(link.src_id, 0.0), boost)

                for rid, g_boost in graph_boosts.items():
                    old_bd = scored_map[rid]
                    # If a linked neighbor had zero direct keyword hits, lift its suppressed prior too.
                    base_without_penalty = (
                        w.lexical * old_bd.lexical
                        + w.identifier * old_bd.identifier
                        + w.vector * old_bd.vector
                        + w.recency * old_bd.recency
                        + w.importance * old_bd.importance
                    )
                    new_total = base_without_penalty + g_boost
                    scored_map[rid] = ScoreBreakdown(
                        lexical=old_bd.lexical,
                        identifier=old_bd.identifier,
                        vector=old_bd.vector,
                        recency=old_bd.recency,
                        importance=old_bd.importance,
                        graph=g_boost,
                        total=new_total,
                    )

        ranked = [
            RetrievalCandidate(record=record_map[rid], breakdown=bd)
            for rid, bd in sorted(scored_map.items(), key=lambda kv: kv[1].total, reverse=True)
        ]

        if self.config.enable_mmr and len(ranked) > 1:
            ranked = self._apply_mmr(ranked, limit=limit, mmr_lambda=w.mmr_lambda)
        else:
            ranked = ranked[:limit]

        return RetrievalResult(
            candidates=ranked,
            superseded_filtered_count=superseded_filtered_count,
        )

    def _score_recency_importance_only(
        self, pool: list[MemoryRecord], turn: int
    ) -> list[RetrievalCandidate]:
        w = self.weights
        out: list[RetrievalCandidate] = []
        for rec in pool:
            if self.config.enable_recency_decay:
                delta_t = max(0, turn - rec.turn_last_accessed)
                stability = 1.0 + 0.65 * math.log1p(max(0, rec.access_count))
                recency = math.exp(-DECAY_LAMBDA[rec.kind] * delta_t / stability)
            else:
                recency = 0.5
            imp = rec.importance if rec.importance is not None else 0.7
            total = w.recency * recency + w.importance * imp
            bd = ScoreBreakdown(
                lexical=0.0,
                identifier=0.0,
                vector=0.0,
                recency=recency,
                importance=imp,
                graph=0.0,
                total=total,
            )
            out.append(RetrievalCandidate(record=rec, breakdown=bd))
        out.sort(key=lambda c: c.breakdown.total, reverse=True)
        return out

    @staticmethod
    def _record_identifiers(rec: MemoryRecord) -> set[str]:
        ids: set[str] = set()
        for ent in rec.entities:
            ids.add(ent.lower())
            ids.update(extract_identifiers(ent))
        for tag in rec.tags:
            ids.add(tag.lower())
        ids.update(extract_identifiers(f"{rec.title}\n{rec.summary}\n{rec.detail}"))
        return ids

    @staticmethod
    def _apply_mmr(
        ranked: list[RetrievalCandidate], *, limit: int, mmr_lambda: float
    ) -> list[RetrievalCandidate]:
        pool = list(ranked[: max(limit * 2, 48)])
        max_total = max((c.breakdown.total for c in pool), default=1.0)
        if max_total <= 0.0:
            max_total = 1.0

        # Precompute token sets once instead of re-tokenizing O(N^2) times inside the MMR loop.
        token_sets: dict[str, set[str]] = {
            c.record.id: set(
                tokenize_text(f"{c.record.title} {c.record.summary}", filter_stopwords=True)
            )
            for c in pool
        }

        selected: list[RetrievalCandidate] = []
        remaining = list(pool)

        while remaining and len(selected) < limit:
            best_idx = 0
            best_mmr = -1e9
            for idx, cand in enumerate(remaining):
                rel = cand.breakdown.total / max_total
                if not selected:
                    mmr = rel
                else:
                    cand_toks = token_sets[cand.record.id]
                    max_sim = 0.0
                    for prev in selected:
                        prev_toks = token_sets[prev.record.id]
                        if not cand_toks or not prev_toks:
                            continue
                        inter = len(cand_toks & prev_toks)
                        if inter:
                            sim = inter / len(cand_toks | prev_toks)
                            if sim > max_sim:
                                max_sim = sim
                    mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected
