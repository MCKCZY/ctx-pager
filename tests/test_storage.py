from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ctx_pager.models import LinkRelation, MemoryKind, MemoryRecord, MemoryStatus
from ctx_pager.storage import MemoryStorage


class TestStorage(unittest.TestCase):
    def test_crud_fts_sync_and_supersede(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "store.db"
            st = MemoryStorage(db)
            try:
                r1 = MemoryRecord(
                    id=st.next_memory_id(),
                    kind=MemoryKind.DECISION,
                    scope="auth",
                    title="Listen on 8080",
                    summary="Use port 8080 with HS256",
                    entities=["8080", "HS256"],
                    pinned=True,
                    turn_created=1,
                )
                st.insert_memory(r1, turn=1)
                self.assertEqual(st.fts_search("8080")[0][0], r1.id)

                r2 = MemoryRecord(
                    id=st.next_memory_id(),
                    kind=MemoryKind.DECISION,
                    scope="auth",
                    title="Switch to 9443",
                    summary="Use TLS port 9443 with RS256",
                    entities=["9443", "RS256"],
                    pinned=True,
                    turn_created=5,
                )
                old_updated, new_saved, _ = st.supersede_memory(r1.id, r2, turn=5)
                self.assertEqual(old_updated.status, MemoryStatus.SUPERSEDED)
                self.assertEqual(old_updated.superseded_by_id, new_saved.id)
                self.assertFalse(old_updated.pinned)
                self.assertEqual(new_saved.supersedes_id, r1.id)

                links = st.get_neighbors([new_saved.id])
                self.assertTrue(
                    any(
                        l.src_id == new_saved.id
                        and l.dst_id == r1.id
                        and l.relation == LinkRelation.SUPERSEDES
                        for l in links
                    )
                )
                self.assertTrue(st.doctor()["ok"])
            finally:
                st.close()

    def test_event_hash_chain_tamper_detection_and_replay(self) -> None:
        st = MemoryStorage(":memory:")
        try:
            r1 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.CONSTRAINT,
                scope="db",
                title="MySQL 5.7",
                summary="Use MySQL 5.7 utf8mb4",
                pinned=True,
            )
            st.insert_memory(r1, turn=2)
            d1 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Port 8080",
                summary="Listen port 8080",
            )
            st.insert_memory(d1, turn=5)
            d2 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Port 9443",
                summary="Switch listen port to 9443",
            )
            st.supersede_memory(d1.id, d2, turn=15)
            st.save_checkpoint(goal="Auth TLS", state_summary="Port 9443 active", turn=20)

            ok, errs = st.verify_event_chain()
            self.assertTrue(ok, errs)

            # Time-travel replay at turn 10 should see d1 active and d2 absent.
            rep_t10 = st.replay_events_to_storage(10)
            try:
                mems_10 = {m.id: m for m in rep_t10.list_memories()}
                self.assertIn(d1.id, mems_10)
                self.assertNotIn(d2.id, mems_10)
                self.assertEqual(mems_10[d1.id].status, MemoryStatus.ACTIVE)
                self.assertIsNone(rep_t10.get_latest_checkpoint())
            finally:
                rep_t10.close()

            # Replay at turn 20 should see d1 superseded, d2 active, and checkpoint present.
            rep_t20 = st.replay_events_to_storage(20)
            try:
                mems_20 = {m.id: m for m in rep_t20.list_memories()}
                self.assertEqual(mems_20[d1.id].status, MemoryStatus.SUPERSEDED)
                self.assertEqual(mems_20[d2.id].status, MemoryStatus.ACTIVE)
                self.assertIsNotNone(rep_t20.get_latest_checkpoint())
            finally:
                rep_t20.close()

            # Export JSONL check.
            with tempfile.TemporaryDirectory() as tmp:
                out_jsonl = Path(tmp) / "events.jsonl"
                count = st.export_events_jsonl(out_jsonl)
                self.assertGreaterEqual(count, 4)
                lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), count)
                self.assertIn("content_hash", json.loads(lines[0]))

            # Tamper directly with an event payload row -> chain verification MUST fail.
            with st.conn:
                st.conn.execute(
                    "UPDATE events SET payload_json = '{\"tampered\":true}' WHERE seq = 2"
                )
            ok_after, errs_after = st.verify_event_chain()
            self.assertFalse(ok_after)
            self.assertGreater(len(errs_after), 0)
            self.assertFalse(st.doctor()["ok"])
        finally:
            st.close()


if __name__ == "__main__":
    unittest.main()
