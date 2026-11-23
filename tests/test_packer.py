from __future__ import annotations

import json
import unittest

from ctx_pager.engine import ContextPager
from ctx_pager.models import BudgetProfile, MemoryKind, Resolution


class TestPacker(unittest.TestCase):
    def _populate_dense_store(self, eng: ContextPager) -> None:
        for i in range(6):
            eng.commit(
                kind=MemoryKind.CONSTRAINT,
                scope=f"mod{i}",
                title=f"Critical invariant #{i} forbidding unsafe queries",
                summary=f"Constraint {i}: always enforce strict schema boundary and timeout <= 3000ms.",
                detail=f"Extended rationale for constraint {i}: " + ("verbose detail " * 18),
                pinned=True,
                turn=1 + i,
            )
        for i in range(4):
            eng.commit(
                kind=MemoryKind.OPEN_LOOP,
                scope=f"todo{i}",
                title=f"Pending rollout verification #{i}",
                summary=f"Verify migration rollback script #{i} on staging before release.",
                detail=f"Checklist #{i}: " + ("verification step " * 12),
                turn=10 + i,
            )
        for i in range(32):
            eng.commit(
                kind=MemoryKind.FACT,
                scope="db" if i % 2 == 0 else "auth",
                title=f"Database connection pool and auth token note #{i}",
                summary=f"Service worker note #{i}: pool_size={20 + i}, jwt_leeway={10 + i}s.",
                detail=f"Full stack trace and YAML dump #{i}: " + ("trace_line_payload " * 25),
                entities=[f"ERR_CODE_{i:04d}", f"src/mod_{i}.py"],
                turn=15 + i,
            )
        eng.checkpoint(
            goal="Ship auth and db migration safely",
            state_summary="Investigating pool exhaustion and ERR_CODE_0005.",
            turn=50,
        )

    def test_strict_budget_invariant_across_budgets_and_formats(self) -> None:
        with ContextPager(":memory:") as eng:
            self._populate_dense_store(eng)
            for budget in (256, 512, 1024, 2048, 4096):
                for fmt in ("markdown", "xml", "json"):
                    prof = BudgetProfile(name=f"b_{budget}", total_tokens=budget, format=fmt)
                    frame = eng.recall(
                        "database connection pool ERR_CODE_0005", profile=prof, turn=50
                    )
                    self.assertLessEqual(
                        frame.used_tokens,
                        budget,
                        f"Budget overflow at budget={budget}, fmt={fmt}: {frame.used_tokens}",
                    )
                    if fmt == "json" and budget >= 512:
                        parsed = json.loads(frame.rendered)
                        self.assertIn("recall", parsed)

    def test_pinned_cap_prevents_starvation_and_multi_res_packs_more(self) -> None:
        with ContextPager(":memory:") as eng:
            self._populate_dense_store(eng)
            prof = BudgetProfile(name="tight_1024", total_tokens=1024)
            frame_multi = eng.recall(
                "database connection pool ERR_CODE_0005",
                profile=prof,
                multi_resolution=True,
                turn=50,
            )
            frame_flat = eng.recall(
                "database connection pool ERR_CODE_0005",
                profile=prof,
                multi_resolution=False,
                turn=50,
            )
            # Pinned slot respects cap and leaves plenty of room for recall items.
            self.assertLessEqual(frame_multi.slot_tokens["pinned"], int(1024 * 0.30))
            recall_multi = [i for i in frame_multi.items if i.slot == "recall"]
            recall_flat = [i for i in frame_flat.items if i.slot == "recall"]
            self.assertGreater(len(recall_multi), len(recall_flat))
            # Tight budget should retain L0/L1 stubs with valid page_in_hints.
            self.assertTrue(len(frame_multi.page_in_hints) > 0)
            self.assertTrue(
                any(i.resolution in (Resolution.L0, Resolution.L1) for i in recall_multi)
            )
            # Open loops slot populated.
            self.assertTrue(any(i.slot == "open_loop" for i in frame_multi.items))


if __name__ == "__main__":
    unittest.main()
