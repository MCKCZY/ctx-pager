from __future__ import annotations

import unittest

from ctx_pager.engine import ContextPager
from ctx_pager.models import MemoryKind, MemoryStatus


class TestCompactorAndIngestor(unittest.TestCase):
    def test_turn_ingestion_extracts_dedups_and_supersedes(self) -> None:
        with ContextPager(":memory:") as eng:
            r1 = eng.ingest_turn(
                turn=1,
                role="user",
                scope="auth",
                text=(
                    "必须禁止在日志中打印 Bearer token。"
                    "决定先使用端口 8080 启动 auth_srv。"
                    "待办：稍后补充 auth_srv 健康检查测试。"
                ),
            )
            kinds = {m.kind for m in r1.created}
            self.assertIn(MemoryKind.CONSTRAINT, kinds)
            self.assertIn(MemoryKind.DECISION, kinds)
            self.assertIn(MemoryKind.OPEN_LOOP, kinds)

            # Near-duplicate clause should update existing record rather than creating a clone.
            count_before = eng.stats()["total_memories"]
            r2 = eng.ingest_turn(
                turn=2,
                role="user",
                scope="auth",
                text="必须禁止在日志中打印 Bearer token。",
            )
            self.assertEqual(len(r2.created), 0)
            self.assertEqual(len(r2.updated), 1)
            self.assertEqual(eng.stats()["total_memories"], count_before)

            # Revision cue with shared entity/topic triggers auto-supersede.
            r3 = eng.ingest_turn(
                turn=4,
                role="user",
                scope="auth",
                text="决定把 auth_srv 监听端口改为 9443。",
            )
            self.assertEqual(len(r3.superseded), 1)
            old_id, new_rec = r3.superseded[0]
            old_rec = eng.storage.get_memory(old_id)
            self.assertIsNotNone(old_rec)
            assert old_rec is not None
            self.assertEqual(old_rec.status, MemoryStatus.SUPERSEDED)
            self.assertIn("9443", new_rec.summary)

    def test_lossless_compaction_protects_constraints_and_open_loops(self) -> None:
        with ContextPager(":memory:") as eng:
            c = eng.commit(
                kind=MemoryKind.CONSTRAINT,
                scope="ops",
                title="Immutable rule",
                summary="Never run DROP TABLE in prod",
                pinned=True,
                turn=1,
            )
            ol = eng.commit(
                kind=MemoryKind.OPEN_LOOP,
                scope="ops",
                title="Verify staging backup",
                summary="Pending backup restore check",
                turn=2,
            )
            eps = []
            for t in (3, 5, 7, 9):
                eps.append(
                    eng.commit(
                        kind=MemoryKind.EPISODE,
                        scope="ops",
                        title=f"Canary tick T{t}",
                        summary=f"Routine smoke check at turn {t} passed.",
                        turn=t,
                    )
                )

            eng.storage.set_turn(30)
            dry = eng.compact(older_than_turns=15, min_group_size=3, dry_run=True)
            self.assertEqual(dry.groups_compacted, 1)
            self.assertEqual(len(dry.archived_ids), 4)
            # Dry run leaves originals active.
            for ep in eps:
                self.assertEqual(eng.storage.get_memory(ep.id).status, MemoryStatus.ACTIVE)

            real = eng.compact(older_than_turns=15, min_group_size=3, dry_run=False)
            self.assertEqual(real.groups_compacted, 1)
            self.assertEqual(len(real.digest_ids), 1)
            digest = eng.storage.get_memory(real.digest_ids[0])
            self.assertIsNotNone(digest)
            assert digest is not None
            # Source events preserved losslessly.
            self.assertGreaterEqual(len(digest.source_events), 4)
            for ep in eps:
                self.assertEqual(eng.storage.get_memory(ep.id).status, MemoryStatus.ARCHIVED)
            # Constraint & open_loop remain untouched.
            self.assertEqual(eng.storage.get_memory(c.id).status, MemoryStatus.ACTIVE)
            self.assertEqual(eng.storage.get_memory(ol.id).status, MemoryStatus.ACTIVE)
            self.assertTrue(eng.doctor()["ok"])


if __name__ == "__main__":
    unittest.main()
