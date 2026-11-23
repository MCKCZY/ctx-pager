from __future__ import annotations

import unittest

from benchmarks.run_benchmark import run_benchmark


class TestBenchmarkSmoke(unittest.TestCase):
    def test_ctx_pager_outperforms_baselines(self) -> None:
        metrics = run_benchmark(budgets=(1024, 2048))
        by_key = {(m.strategy, m.budget): m for m in metrics}

        for budget in (1024, 2048):
            full = by_key[("ctx_pager_full", budget)]
            sw = by_key[("sliding_window", budget)]
            bm = by_key[("flat_bm25", budget)]

            self.assertGreater(full.effective_recall, sw.effective_recall)
            self.assertGreater(full.constraint_survival, sw.constraint_survival)
            self.assertLess(full.stale_intrusion_rate, bm.stale_intrusion_rate)
            self.assertEqual(full.stale_intrusion_rate, 0.0)
            self.assertLessEqual(full.avg_used_tokens, budget)


if __name__ == "__main__":
    unittest.main()
