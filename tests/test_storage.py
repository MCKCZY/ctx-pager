from __future__ import annotations

import json
import math
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

    def test_reopen_persistence_and_sequence_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "persist.db"
            s1 = MemoryStorage(db)
            try:
                m1 = MemoryRecord(
                    id=s1.next_memory_id(),
                    kind=MemoryKind.CONSTRAINT,
                    scope="db",
                    title="Rule 1",
                    summary="Keep utf8mb4",
                )
                s1.insert_memory(m1, turn=4)
                m2 = MemoryRecord(
                    id=s1.next_memory_id(),
                    kind=MemoryKind.FACT,
                    scope="db",
                    title="Fact 2",
                    summary="Pool limit 150",
                )
                s1.insert_memory(m2, turn=9)
            finally:
                s1.close()

            s2 = MemoryStorage(db)
            try:
                self.assertEqual(s2.get_current_turn(), 9)
                next_id = s2.next_memory_id()
                self.assertEqual(next_id, "M-0003")
                m3 = MemoryRecord(
                    id=next_id,
                    kind=MemoryKind.FACT,
                    scope="db",
                    title="Fact 3",
                    summary="Reopened cleanly",
                )
                s2.insert_memory(m3, turn=12)
                self.assertTrue(s2.doctor()["ok"])
                self.assertEqual(len(s2.list_memories()), 3)
            finally:
                s2.close()

    def test_bit_for_bit_replay_parity_including_source_events(self) -> None:
        st = MemoryStorage(":memory:")
        try:
            c1 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.CONSTRAINT,
                scope="db",
                title="MySQL 5.7",
                summary="Use MySQL 5.7 utf8mb4",
                pinned=True,
            )
            st.insert_memory(c1, turn=1)
            d1 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Port 8080",
                summary="Listen port 8080",
            )
            st.insert_memory(d1, turn=2)
            d1.summary = "Listen port 8080 (updated)"
            st.update_memory(d1, turn=4)
            d2 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Port 9443",
                summary="Switch listen port to 9443",
            )
            st.supersede_memory(d1.id, d2, turn=10)
            st.add_link(d2.id, c1.id, LinkRelation.DEPENDS_ON, weight=0.9, turn=11)
            st.save_checkpoint(goal="Auth TLS", state_summary="Port 9443 active", turn=12)

            ep_ids: list[str] = []
            for t in (3, 5, 6):
                ep = MemoryRecord(
                    id=st.next_memory_id(),
                    kind=MemoryKind.EPISODE,
                    scope="ops",
                    title=f"Smoke T{t}",
                    summary=f"Smoke check passed at turn {t}",
                )
                st.insert_memory(ep, turn=t)
                ep_ids.append(ep.id)

            digest = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.EPISODE,
                scope="ops",
                title="Compacted 3 episodes (ops, T3-T6)",
                summary="Smoke T3; Smoke T5; Smoke T6",
                detail="Archived episode digest",
                source_events=[],
            )
            st.compact_group(ep_ids, digest, turn=25)

            replayed = st.replay_events_to_storage(25)
            try:
                self.assertTrue(replayed.doctor()["ok"])
                live_mems = {m.id: m for m in st.list_memories()}
                rep_mems = {m.id: m for m in replayed.list_memories()}
                self.assertEqual(set(live_mems), set(rep_mems))
                for mid, lm in live_mems.items():
                    rm = rep_mems[mid]
                    self.assertEqual(lm.status, rm.status)
                    self.assertEqual(lm.pinned, rm.pinned)
                    self.assertEqual(lm.source_events, rm.source_events)
                    self.assertEqual(lm.supersedes_id, rm.supersedes_id)
                    self.assertEqual(lm.superseded_by_id, rm.superseded_by_id)
                self.assertEqual(len(replayed.list_events()), len(st.list_events()))
                self.assertEqual(replayed.get_current_turn(), 25)
                self.assertEqual(
                    replayed.get_latest_checkpoint().goal,
                    st.get_latest_checkpoint().goal,
                )
            finally:
                replayed.close()
        finally:
            st.close()

    def test_state_guardrails_reject_invalid_operations(self) -> None:
        st = MemoryStorage(":memory:")
        try:
            r1 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Old",
                summary="Old summary",
            )
            st.insert_memory(r1, turn=1)
            r2 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="New",
                summary="New summary",
            )
            st.supersede_memory(r1.id, r2, turn=2)

            # Double supersede rejected.
            r3 = MemoryRecord(
                id=st.next_memory_id(),
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Third",
                summary="Third summary",
            )
            with self.assertRaises(ValueError):
                st.supersede_memory(r1.id, r3, turn=3)

            # Self-supersede rejected.
            with self.assertRaises(ValueError):
                st.supersede_memory(r2.id, r2, turn=3)

            # Archived supersede rejected.
            st.set_status(r2.id, MemoryStatus.ARCHIVED, turn=4)
            with self.assertRaises(ValueError):
                st.supersede_memory(r2.id, r3, turn=5)

            # Self-link, NaN weight, and negative turn rejected.
            with self.assertRaises(ValueError):
                st.add_link(r1.id, r1.id, LinkRelation.RELATES_TO)
            with self.assertRaises(ValueError):
                st.add_link(r1.id, r2.id, LinkRelation.RELATES_TO, weight=math.nan)
            with self.assertRaises(ValueError):
                st.insert_memory(r3, turn=-1)
        finally:
            st.close()

    def test_event_hash_chain_tamper_and_fts_id_divergence_detection(self) -> None:
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

            # Export JSONL check.
            with tempfile.TemporaryDirectory() as tmp:
                out_jsonl = Path(tmp) / "events.jsonl"
                count = st.export_events_jsonl(out_jsonl)
                self.assertGreaterEqual(count, 4)
                lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), count)
                self.assertIn("content_hash", json.loads(lines[0]))

            # Same FTS row count but divergent IDs must be flagged by doctor().
            with st.conn:
                st.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (d2.id,))
                st.conn.execute(
                    "INSERT INTO memories_fts(memory_id, title_tokens, summary_tokens, detail_tokens, entity_tokens) "
                    "VALUES ('M-9999', 'ghost', 'ghost', '', '')"
                )
            doc_fts = st.doctor()
            self.assertFalse(doc_fts["ok"])
            self.assertTrue(any("divergence" in issue for issue in doc_fts["issues"]))

            # Tamper prev_hash and payload_json -> verify_event_chain() catches both.
            with st.conn:
                st.conn.execute("UPDATE events SET prev_hash = 'deadbeef' WHERE seq = 2")
                st.conn.execute(
                    "UPDATE events SET payload_json = '{\"tampered\":true}' WHERE seq = 3"
                )
            ok_after, errs_after = st.verify_event_chain()
            self.assertFalse(ok_after)
            self.assertTrue(any("prev_hash" in e for e in errs_after))
            self.assertTrue(any("content_hash" in e for e in errs_after))
        finally:
            st.close()


if __name__ == "__main__":
    unittest.main()
