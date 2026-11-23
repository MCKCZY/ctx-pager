from __future__ import annotations

import unittest

from ctx_pager.engine import ContextPager
from ctx_pager.models import LinkRelation, MemoryKind
from ctx_pager.retriever import HybridRetriever, RetrievalConfig


class TestRetriever(unittest.TestCase):
    def test_bilingual_exact_id_and_conflict_gating(self) -> None:
        with ContextPager(":memory:") as eng:
            old = eng.commit(
                kind=MemoryKind.DECISION,
                scope="auth",
                title="JWT 签名采用 HS256",
                summary="鉴权接口使用 HS256 对称密钥",
                entities=["HS256"],
                turn=1,
            )
            new = eng.commit(
                kind=MemoryKind.FACT,
                scope="auth",
                title="ERR_JWT_4012 异常排查",
                summary="在 src/auth/jwt_verify.py:88 出现 ERR_JWT_4012",
                entities=["ERR_JWT_4012", "src/auth/jwt_verify.py:88"],
                turn=3,
            )
            _, revised = eng.supersede(
                old.id,
                title="JWT 签名改用 RS256",
                summary="安全规范要求鉴权算法改用 RS256 非对称验证",
                entities=["RS256"],
                turn=10,
            )

            res = eng.retriever.retrieve("当前 JWT 签名算法是什么？", current_turn=20)
            ids = [c.record.id for c in res.candidates]
            self.assertIn(revised.id, ids)
            self.assertNotIn(old.id, ids)
            self.assertGreaterEqual(res.superseded_filtered_count, 1)

            # Including superseded should surface the old node when requested.
            res_inc = eng.retriever.retrieve(
                "HS256 算法", current_turn=20, include_superseded=True
            )
            self.assertIn(old.id, [c.record.id for c in res_inc.candidates])

            # Exact code ID retrieval.
            res_id = eng.retriever.retrieve("ERR_JWT_4012", current_turn=20)
            self.assertEqual(res_id.candidates[0].record.id, new.id)
            self.assertGreater(res_id.candidates[0].breakdown.identifier, 0.0)

    def test_graph_expansion_surfaces_zero_keyword_root_cause(self) -> None:
        with ContextPager(":memory:") as eng:
            symptom = eng.commit(
                kind=MemoryKind.FACT,
                scope="gateway",
                title="502 Bad Gateway timeout burst",
                summary="High traffic produces 502 Bad Gateway errors",
                entities=["502"],
                turn=5,
            )
            # Notice root_cause intentionally shares zero tokens with "502 Bad Gateway".
            root_cause = eng.commit(
                kind=MemoryKind.FACT,
                scope="db",
                title="Pool limit max_connections=150 missing",
                summary="YAML missing max_connections=150 exhausted sockets",
                entities=["max_connections=150"],
                turn=6,
            )
            eng.link(symptom.id, root_cause.id, LinkRelation.CAUSED_BY, weight=0.95, turn=6)

            res = eng.retriever.retrieve("502 Bad Gateway", current_turn=20)
            cand_map = {c.record.id: c for c in res.candidates}
            self.assertIn(root_cause.id, cand_map)
            self.assertGreater(cand_map[root_cause.id].breakdown.graph, 0.0)

            # Without graph expansion, root_cause should not receive graph boost.
            no_graph = HybridRetriever(
                eng.storage, config=RetrievalConfig(enable_graph_expansion=False)
            )
            res_ng = no_graph.retrieve("502 Bad Gateway", current_turn=20)
            ng_map = {c.record.id: c for c in res_ng.candidates}
            self.assertEqual(ng_map[root_cause.id].breakdown.graph, 0.0)

    def test_ebbinghaus_decay_and_mmr_diversity(self) -> None:
        with ContextPager(":memory:") as eng:
            # Two near-duplicate notes matching all query terms closely vs one distinct note.
            d1 = eng.commit(
                kind=MemoryKind.FACT,
                scope="auth",
                title="RS256 public key permission 0600 staging",
                summary="Verify keys/jwt_pub.pem has chmod 0600 and rotation check on staging",
                entities=["keys/jwt_pub.pem", "0600"],
                turn=18,
            )
            d2 = eng.commit(
                kind=MemoryKind.FACT,
                scope="auth",
                title="RS256 public key permission 0600 prod",
                summary="Verify keys/jwt_pub.pem has chmod 0600 and rotation check on prod",
                entities=["keys/jwt_pub.pem", "0600"],
                turn=17,
            )
            distinct = eng.commit(
                kind=MemoryKind.PROCEDURE,
                scope="auth",
                title="JWKS cache TTL configuration",
                summary="Set keys/jwt_pub.pem JWKS cache-control max-age=300 for rotation",
                entities=["keys/jwt_pub.pem"],
                turn=16,
            )
            old_ep = eng.commit(
                kind=MemoryKind.EPISODE,
                scope="auth",
                title="Temporary key rotation test note",
                summary="Tested key rotation smoke script",
                turn=1,
            )
            eng.storage.record_access([d1.id, d2.id], turn=28)

            no_mmr = HybridRetriever(eng.storage, config=RetrievalConfig(enable_mmr=False))
            res_plain = no_mmr.retrieve("keys/jwt_pub.pem 0600 rotation", current_turn=30)
            plain_top2 = [c.record.id for c in res_plain.candidates[:2]]
            self.assertEqual(set(plain_top2), {d1.id, d2.id})

            res_mmr = eng.retriever.retrieve("keys/jwt_pub.pem 0600 rotation", current_turn=30)
            mmr_top2 = [c.record.id for c in res_mmr.candidates[:2]]
            self.assertIn(distinct.id, mmr_top2)
            self.assertNotEqual(set(mmr_top2), {d1.id, d2.id})

            by_id = {c.record.id: c for c in res_mmr.candidates}
            self.assertGreater(by_id[d1.id].breakdown.recency, by_id[old_ep.id].breakdown.recency)


if __name__ == "__main__":
    unittest.main()
