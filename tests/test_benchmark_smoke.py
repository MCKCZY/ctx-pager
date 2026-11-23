from __future__ import annotations

import unittest

from benchmarks.run_benchmark import run_benchmark


class TestBenchmarkSmoke(unittest.TestCase):
    def test_ctx_pager_outperforms_baselines_and_is_deterministic(self) -> None:
        m1 = run_benchmark(budgets=(1024, 2048))
        m2 = run_benchmark(budgets=(1024,))

        by_key = {(m.strategy, m.budget): m for m in m1}
        by_key_2 = {(m.strategy, m.budget): m for m in m2}

        for m in m1:
            self.assertLessEqual(m.avg_used_tokens, m.budget)

        for budget in (1024, 2048):
            full = by_key[("ctx_pager_full", budget)]
            sw = by_key[("sliding_window", budget)]
            bm = by_key[("flat_bm25", budget)]

            self.assertGreater(full.effective_recall, sw.effective_recall)
            self.assertGreater(full.constraint_survival, sw.constraint_survival)
            self.assertLess(full.stale_intrusion_rate, bm.stale_intrusion_rate)
            self.assertEqual(full.stale_intrusion_rate, 0.0)
            self.assertEqual(full.multi_hop_recall, 1.0)

        # Verify recall/stale/token results are bit-for-bit reproducible across runs.
        for strat in ("sliding_window", "rolling_summary", "flat_bm25", "ctx_pager_ablated", "ctx_pager_full"):
            a = by_key[(strat, 1024)]
            b = by_key_2[(strat, 1024)]
            self.assertEqual(a.direct_recall, b.direct_recall)
            self.assertEqual(a.effective_recall, b.effective_recall)
            self.assertEqual(a.stale_intrusion_rate, b.stale_intrusion_rate)
            self.assertEqual(a.avg_used_tokens, b.avg_used_tokens)


if __name__ == "__main__":
    unittest.main()
