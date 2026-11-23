#!/usr/bin/env python3
"""End-to-end 35-turn debugging session under a strict 1200-token working-memory budget."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ctx_pager.engine import ContextPager
from ctx_pager.models import BudgetProfile, LinkRelation, MemoryKind, Resolution


def run_demo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo_session.db"
        with ContextPager(db_path) as engine:
            print("=== 1. Turn 1-4: Commit Early Constraints & Initial Design ===")
            c_mysql = engine.commit(
                kind=MemoryKind.CONSTRAINT,
                scope="db",
                title="MySQL 5.7 utf8mb4 index key <= 191 chars",
                summary="生产数据库固定 MySQL 5.7 + utf8mb4，单列索引前缀严禁超过 191 字符。",
                detail="InnoDB barracuda row_format in MySQL 5.7 caps indexed varchar at 191 under utf8mb4.",
                entities=["MySQL 5.7", "utf8mb4", "191"],
                pinned=True,
                turn=1,
            )
            c_sec = engine.commit(
                kind=MemoryKind.CONSTRAINT,
                scope="auth",
                title="Never log Bearer tokens or refresh_secret",
                summary="禁止在日志和报错响应中输出 Bearer token 或 refresh_secret 原文。",
                entities=["Bearer", "refresh_secret"],
                pinned=True,
                turn=2,
            )
            d_port_old = engine.commit(
                kind=MemoryKind.DECISION,
                scope="auth",
                title="Initial auth listen port 8080 with HS256",
                summary="初期鉴权服务监听 8080 端口，使用 HS256 对称签名。",
                entities=["8080", "HS256"],
                turn=4,
            )
            print(f"Committed pinned constraints {c_mysql.id}, {c_sec.id} and initial decision {d_port_old.id}")

            print("\n=== 2. Turn 6-15: Debugging Noise, Causal Graph & Decision Superseding ===")
            f_symptom = engine.commit(
                kind=MemoryKind.FACT,
                scope="gateway",
                title="502 Bad Gateway under 400 RPS load test",
                summary="压测 400 RPS 时网关出现 14% 502 Bad Gateway。",
                detail="Nginx upstream premature close on /api/v1/login during burst.",
                entities=["502 Bad Gateway", "/api/v1/login"],
                turn=7,
            )
            f_root = engine.commit(
                kind=MemoryKind.FACT,
                scope="db",
                title="Connection pool exhaustion missing max_connections=150",
                summary="config/db_pool.yaml 未设 max_connections=150，连接数打满后被 wait_timeout=60s 断开。",
                detail="Fix in config/db_pool.yaml: set max_connections=150, pool_recycle=45, connect_timeout=3.",
                entities=["config/db_pool.yaml", "max_connections=150", "pool_recycle=45"],
                turn=9,
            )
            engine.link(f_symptom.id, f_root.id, LinkRelation.CAUSED_BY, weight=0.95, turn=9)

            # Simulate mid-session status episodes across turns 10-16.
            for t in (10, 12, 14, 16):
                engine.commit(
                    kind=MemoryKind.EPISODE,
                    scope="ops",
                    title=f"Routine canary build check T{t}",
                    summary=f"Canary smoke check at turn {t} passed without regressions.",
                    detail=f"Detailed stdout from turn {t}: 16 checks green, cpu=22%, mem=198MB.",
                    turn=t,
                )

            # Turn 18: Security review supersedes port 8080 / HS256 with port 9443 / RS256.
            old_rec, new_rec = engine.supersede(
                d_port_old.id,
                title="Auth TLS port 9443 with RS256 public key",
                summary="安全评审决定鉴权改为 HTTPS 端口 9443，算法使用 RS256（keys/jwt_pub.pem）。",
                detail="Asymmetric RS256 verification via keys/jwt_pub.pem with 24h JWKS rotation on port 9443.",
                entities=["9443", "RS256", "keys/jwt_pub.pem"],
                turn=18,
            )
            print(f"Superseded {old_rec.id} (status={old_rec.status.value}) -> {new_rec.id}")

            # Turn 24: Commit an active open loop and an exact error trace.
            engine.commit(
                kind=MemoryKind.FACT,
                scope="auth",
                title="ERR_JWT_4012 in src/auth/jwt_verify.py:88",
                summary="偶发 401 错误码 ERR_JWT_4012，位于 src/auth/jwt_verify.py:88。",
                detail="Traceback in src/auth/jwt_verify.py:88: leeway=0 failed under 500ms NTP drift; patch leeway=15.",
                entities=["ERR_JWT_4012", "src/auth/jwt_verify.py:88"],
                turn=24,
            )
            engine.commit(
                kind=MemoryKind.OPEN_LOOP,
                scope="db",
                title="Validate migrations/004_tokens.sql rollback duration",
                summary="待在 staging 实测 migrations/004_tokens.sql 回滚锁表时间。",
                entities=["migrations/004_tokens.sql"],
                turn=26,
            )
            engine.checkpoint(
                goal="Ship auth TLS 9443 + RS256 and resolve load-test 502",
                state_summary="9443/RS256 active; pool fix ready; migrations/004_tokens.sql rollback check open.",
                turn=30,
            )

            print("\n=== 3. Turn 35: Lossless Episodic Compaction ===")
            engine.storage.set_turn(35)
            comp_report = engine.compact(older_than_turns=15, min_group_size=3, max_access_count=1)
            print(
                f"Compacted {comp_report.groups_compacted} group(s), "
                f"archived={comp_report.archived_ids}, saved={comp_report.tokens_saved} tokens"
            )

            print("\n=== 4. Turn 35: Packed Working Frame (Strict 1200-Token Budget) ===")
            profile = BudgetProfile(name="demo_1200", total_tokens=1200)
            frame = engine.recall(
                "为什么压测出现 502 Bad Gateway？当前鉴权端口和算法是什么？",
                profile=profile,
                turn=35,
            )
            print(f"Budget: {frame.used_tokens}/{frame.budget_tokens} tokens | Superseded filtered: {frame.superseded_filtered_count}")
            print(f"Slot breakdown: {frame.slot_tokens}")
            print("--- Rendered Frame ---")
            print(frame.rendered)
            print("----------------------")
            assert frame.used_tokens <= frame.budget_tokens
            assert "9443" in frame.rendered and "RS256" in frame.rendered
            assert "8080" not in frame.rendered and "HS256" not in frame.rendered

            print("\n=== 5. On-Demand Page-In ===")
            target_id = frame.page_in_hints[0] if frame.page_in_hints else f_root.id
            paged = engine.page_in([target_id], include_neighbors=True, turn=35)
            for rec in paged["records"]:
                print(rec.render(Resolution.L2))

            print("\n=== 6. Time-Travel Replay (Turn 10 vs Turn 35) & Integrity Doctor ===")
            replay_t10 = engine.replay(10, query="当前鉴权端口是什么？", budget=1200)
            print(f"At Turn 10: {replay_t10['stats']['total_memories']} memories; port 8080 active? {'8080' in replay_t10['frame'].rendered}")
            print(f"At Turn 35: {engine.stats()['total_memories']} memories; port 9443 active? {'9443' in frame.rendered}")

            doc = engine.doctor()
            print(f"Doctor check: ok={doc['ok']}, memories={doc['memories_count']}, fts={doc['fts_count']}, hash_chain={doc['event_chain_ok']}")
            assert doc["ok"] is True


if __name__ == "__main__":
    run_demo()
