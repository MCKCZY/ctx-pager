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

    def test_cross_kind_revision_and_todo_precedence(self) -> None:
        with ContextPager(":memory:") as eng:
            # Explicit TODO prefix with "must" classifies as OPEN_LOOP, not pinned CONSTRAINT.
            r_todo = eng.ingest_turn(
                turn=1,
                role="user",
                scope="db",
                text="TODO: must verify migrations/004_tokens.sql rollback timing。待办：必须检查回滚锁表耗时。",
            )
            self.assertEqual(len(r_todo.created), 2)
            for rec in r_todo.created:
                self.assertEqual(rec.kind, MemoryKind.OPEN_LOOP)
                self.assertFalse(rec.pinned)

            # FACT followed by DECISION with revision cue supersedes the prior FACT in same scope.
            r_fact = eng.ingest_turn(
                turn=2,
                role="user",
                scope="gateway",
                text="gateway_srv 目前运行在端口 8080 上。",
            )
            self.assertEqual(r_fact.created[0].kind, MemoryKind.FACT)

            r_rev = eng.ingest_turn(
                turn=3,
                role="user",
                scope="gateway",
                text="决定将 gateway_srv 端口改为 9443。",
            )
            self.assertEqual(len(r_rev.superseded), 1)
            self.assertEqual(r_rev.superseded[0][0], r_fact.created[0].id)

    def test_lossless_compaction_protects_constraints_and_replays_cleanly(self) -> None:
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
            with self.assertRaises(ValueError):
                eng.compact(older_than_turns=-1)
            with self.assertRaises(ValueError):
                eng.compact(min_group_size=1)

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

            # Replay before compaction vs after compaction matches state transitions.
            rep_before = eng.replay(10)
            rep_after = eng.replay(30)
            before_ep_statuses = {
                m["id"]: m["status"] for m in rep_before["memories"] if m["id"].startswith("M-")
            }
            after_ep_statuses = {
                m["id"]: m["status"] for m in rep_after["memories"] if m["id"].startswith("M-")
            }
            self.assertEqual(before_ep_statuses[eps[0].id], "active")
            self.assertEqual(after_ep_statuses[eps[0].id], "archived")

    def test_compaction_non_inflation_guard(self) -> None:
        with ContextPager(":memory:") as eng:
            # Tiny 1-token titles in a small group where digest overhead exceeds sum of L1s.
            for t in (1, 2):
                eng.commit(
                    kind=MemoryKind.EPISODE,
                    scope="tiny",
                    title="ok",
                    summary="ok",
                    turn=t,
                )
            eng.storage.set_turn(30)
            rep = eng.compact(older_than_turns=10, min_group_size=2)
            self.assertEqual(rep.groups_compacted, 0)


if __name__ == "__main__":
    unittest.main()
