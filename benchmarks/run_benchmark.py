"""Run reproducible multi-turn evaluation benchmarks across strategies and budgets."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.baselines import (
    StrategyOutput,
    build_engine_up_to_turn,
    run_ctx_pager_strategy,
    run_flat_bm25,
    run_rolling_summary,
    run_sliding_window,
)
from benchmarks.datasets import EvalProbe, Scenario, load_scenarios
from ctx_pager.retriever import RetrievalConfig

STRATEGIES = (
    "sliding_window",
    "rolling_summary",
    "flat_bm25",
    "ctx_pager_ablated",
    "ctx_pager_full",
)


@dataclass
class ProbeEval:
    probe_id: str
    category: str
    direct_hit: bool
    effective_hit: bool
    stale_intruded: bool
    has_stale_check: bool
    used_tokens: int
    latency_ms: float


@dataclass
class AggregateMetrics:
    strategy: str
    budget: int
    total_probes: int
    direct_recall: float
    effective_recall: float
    constraint_survival: float
    exact_id_retention: float
    multi_hop_recall: float
    stale_intrusion_rate: float
    avg_used_tokens: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget": self.budget,
            "total_probes": self.total_probes,
            "direct_recall": round(self.direct_recall, 4),
            "effective_recall": round(self.effective_recall, 4),
            "constraint_survival": round(self.constraint_survival, 4),
            "exact_id_retention": round(self.exact_id_retention, 4),
            "multi_hop_recall": round(self.multi_hop_recall, 4),
            "stale_intrusion_rate": round(self.stale_intrusion_rate, 4),
            "avg_used_tokens": round(self.avg_used_tokens, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


def _evaluate_probe(probe: EvalProbe, out: StrategyOutput, latency_ms: float) -> ProbeEval:
    low_rendered = out.rendered_text.lower()
    direct_hit = all(sub.lower() in low_rendered for sub in probe.required_substrings)

    if direct_hit:
        effective_hit = True
    elif out.page_in_reachable_texts:
        combined_low = low_rendered + "\n" + "\n".join(out.page_in_reachable_texts).lower()
        effective_hit = all(sub.lower() in combined_low for sub in probe.required_substrings)
    else:
        effective_hit = False

    has_stale = len(probe.forbidden_stale_substrings) > 0
    stale_intruded = any(
        stale.lower() in low_rendered for stale in probe.forbidden_stale_substrings
    )
    return ProbeEval(
        probe_id=probe.id,
        category=probe.category,
        direct_hit=direct_hit,
        effective_hit=effective_hit,
        stale_intruded=stale_intruded,
        has_stale_check=has_stale,
        used_tokens=out.used_tokens,
        latency_ms=latency_ms,
    )


def _aggregate(strategy: str, budget: int, evals: list[ProbeEval]) -> AggregateMetrics:
    n = max(1, len(evals))
    direct_recall = sum(1 for e in evals if e.direct_hit) / n
    effective_recall = sum(1 for e in evals if e.effective_hit) / n

    c_evals = [e for e in evals if e.category == "buried_constraint"]
    constraint_survival = (
        sum(1 for e in c_evals if e.effective_hit) / len(c_evals) if c_evals else 0.0
    )

    id_evals = [e for e in evals if e.category == "exact_artifact"]
    exact_id_retention = (
        sum(1 for e in id_evals if e.effective_hit) / len(id_evals) if id_evals else 0.0
    )

    mh_evals = [e for e in evals if e.category == "multi_hop_context"]
    multi_hop_recall = (
        sum(1 for e in mh_evals if e.effective_hit) / len(mh_evals) if mh_evals else 0.0
    )

    stale_evals = [e for e in evals if e.has_stale_check]
    stale_rate = (
        sum(1 for e in stale_evals if e.stale_intruded) / len(stale_evals)
        if stale_evals
        else 0.0
    )

    avg_tokens = sum(e.used_tokens for e in evals) / n
    latencies = sorted(e.latency_ms for e in evals)
    p95_idx = min(len(latencies) - 1, int(0.95 * len(latencies)))
    p95_ms = latencies[p95_idx] if latencies else 0.0

    return AggregateMetrics(
        strategy=strategy,
        budget=budget,
        total_probes=len(evals),
        direct_recall=direct_recall,
        effective_recall=effective_recall,
        constraint_survival=constraint_survival,
        exact_id_retention=exact_id_retention,
        multi_hop_recall=multi_hop_recall,
        stale_intrusion_rate=stale_rate,
        avg_used_tokens=avg_tokens,
        p95_latency_ms=p95_ms,
    )


def run_benchmark(
    budgets: tuple[int, ...] = (1024, 2048, 4096),
) -> list[AggregateMetrics]:
    scenarios = load_scenarios()
    grouped: dict[tuple[str, int], list[ProbeEval]] = defaultdict(list)

    ablated_cfg = RetrievalConfig(
        enable_conflict_gating=False,
        enable_graph_expansion=False,
        enable_mmr=False,
        enable_recency_decay=False,
    )
    full_cfg = RetrievalConfig()

    for scenario in scenarios:
        # Group probes by turn so we build engines efficiently per turn.
        by_turn: dict[int, list[EvalProbe]] = defaultdict(list)
        for probe in scenario.probes:
            by_turn[probe.query_turn].append(probe)

        for q_turn, probes in sorted(by_turn.items()):
            eng_full = build_engine_up_to_turn(scenario, q_turn, retrieval_config=full_cfg)
            eng_ablated = build_engine_up_to_turn(scenario, q_turn, retrieval_config=ablated_cfg)
            try:
                for budget in budgets:
                    for probe in probes:
                        # 1. sliding_window
                        t0 = time.perf_counter()
                        out_sw = run_sliding_window(scenario.turns, q_turn, budget)
                        dt = (time.perf_counter() - t0) * 1000.0
                        grouped[("sliding_window", budget)].append(
                            _evaluate_probe(probe, out_sw, dt)
                        )

                        # 2. rolling_summary
                        t0 = time.perf_counter()
                        out_rs = run_rolling_summary(scenario.turns, q_turn, budget)
                        dt = (time.perf_counter() - t0) * 1000.0
                        grouped[("rolling_summary", budget)].append(
                            _evaluate_probe(probe, out_rs, dt)
                        )

                        # 3. flat_bm25
                        t0 = time.perf_counter()
                        out_bm = run_flat_bm25(eng_ablated, probe.query, budget)
                        dt = (time.perf_counter() - t0) * 1000.0
                        grouped[("flat_bm25", budget)].append(
                            _evaluate_probe(probe, out_bm, dt)
                        )

                        # 4. ctx_pager_ablated
                        t0 = time.perf_counter()
                        out_ab = run_ctx_pager_strategy(
                            eng_ablated,
                            probe.query,
                            q_turn,
                            budget,
                            multi_resolution=False,
                        )
                        dt = (time.perf_counter() - t0) * 1000.0
                        grouped[("ctx_pager_ablated", budget)].append(
                            _evaluate_probe(probe, out_ab, dt)
                        )

                        # 5. ctx_pager_full
                        t0 = time.perf_counter()
                        out_full = run_ctx_pager_strategy(
                            eng_full,
                            probe.query,
                            q_turn,
                            budget,
                            multi_resolution=True,
                        )
                        dt = (time.perf_counter() - t0) * 1000.0
                        grouped[("ctx_pager_full", budget)].append(
                            _evaluate_probe(probe, out_full, dt)
                        )
            finally:
                eng_full.close()
                eng_ablated.close()

    results: list[AggregateMetrics] = []
    for budget in budgets:
        for strategy in STRATEGIES:
            results.append(_aggregate(strategy, budget, grouped[(strategy, budget)]))
    return results


def format_markdown_report(metrics: list[AggregateMetrics]) -> str:
    by_budget: dict[int, list[AggregateMetrics]] = defaultdict(list)
    for m in metrics:
        by_budget[m.budget].append(m)

    lines: list[str] = [
        "# ctx-pager Benchmark Results (3 Bilingual Scenarios, 235 Turns, 27 Probes)",
        "",
    ]
    for budget in sorted(by_budget):
        lines.append(f"## Token Budget = {budget}")
        lines.append("")
        lines.append(
            "| Strategy | Direct Recall | Effective Recall (incl. L0 Page-In) | Constraint Survival | Exact ID Retention | Multi-Hop Recall | Stale Intrusion (Lower Better) | Avg Tokens | P95 Latency (ms) |"
        )
        lines.append(
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for m in by_budget[budget]:
            lines.append(
                f"| `{m.strategy}` | {m.direct_recall * 100:.1f}% | {m.effective_recall * 100:.1f}% | "
                f"{m.constraint_survival * 100:.1f}% | {m.exact_id_retention * 100:.1f}% | "
                f"{m.multi_hop_recall * 100:.1f}% | {m.stale_intrusion_rate * 100:.1f}% | "
                f"{m.avg_used_tokens:.0f} | {m.p95_latency_ms:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ctx-pager benchmark suite.")
    parser.add_argument(
        "--budgets",
        default="1024,2048,4096",
        help="Comma-separated token budgets (default: 1024,2048,4096)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save raw JSON metrics",
    )
    args = parser.parse_args(argv)
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())

    metrics = run_benchmark(budgets=budgets)
    print(format_markdown_report(metrics))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([m.to_dict() for m in metrics], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
