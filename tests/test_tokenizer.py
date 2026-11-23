from __future__ import annotations

import sqlite3
import unittest

from ctx_pager.tokenizer import (
    build_fts_query,
    char_ngrams,
    estimate_tokens,
    extract_identifiers,
    to_fts_document,
    tokenize_text,
)


class TestTokenizer(unittest.TestCase):
    def test_cjk_bigrams_and_unigrams(self) -> None:
        toks = tokenize_text("数据库连接池超时")
        for expected in ("数据", "据库", "库连", "连接", "接池", "池超", "超时", "库"):
            self.assertIn(expected, toks)
        # Single-char CJK noun in mixed text should survive.
        mixed = tokenize_text("Redis 锁超时")
        self.assertIn("redis", mixed)
        self.assertIn("锁", mixed)

    def test_code_and_identifier_extraction(self) -> None:
        text = (
            "Error ERR_JWT_4012 in src/auth/jwt_verify.py:88 using jwtRefreshToken, "
            "MySQL 5.7 on port 9443 with rate-limit `utf8mb4` CIDR 10.20.0.0/16 CVE-2026-1182."
        )
        ids = extract_identifiers(text)
        self.assertIn("err_jwt_4012", ids)
        self.assertIn("src/auth/jwt_verify.py:88", ids)
        self.assertIn("jwtrefreshtoken", ids)
        self.assertIn("5.7", ids)
        self.assertIn("9443", ids)
        self.assertIn("rate-limit", ids)
        self.assertIn("utf8mb4", ids)
        self.assertIn("10.20.0.0/16", ids)
        self.assertIn("cve-2026-1182", ids)

        toks = tokenize_text("jwtRefreshToken err_jwt_4012")
        for expected in ("jwtrefreshtoken", "jwt", "refresh", "token", "err_jwt_4012", "4012"):
            self.assertIn(expected, toks)

    def test_timestamp_does_not_trigger_port_false_positive(self) -> None:
        ids = extract_identifiers("Logged event at 14:30:05 UTC with 127.0.0.1:8080 and localhost:9443")
        self.assertNotIn("30", ids)
        self.assertNotIn("05", ids)
        self.assertNotIn("14:30", ids)
        self.assertIn("127.0.0.1:8080", ids)
        self.assertIn("localhost:9443", ids)

        zh_port = extract_identifiers("服务监听端口 9443 已开启")
        self.assertIn("9443", zh_port)

    def test_fts5_query_safety_and_max_terms_prioritization(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                'CREATE VIRTUAL TABLE t USING fts5(id UNINDEXED, body, tokenize="unicode61 tokenchars \'_\'")'
            )
            conn.execute(
                "INSERT INTO t VALUES ('1', ?)",
                (to_fts_document("foo-bar:8080 NOT AND OR (err_jwt_4012) 数据库"),),
            )
            malicious_queries = [
                'foo-bar:8080 "unclosed',
                "NOT AND OR NEAR/3 (test)*",
                "::: --- /// [] {} ^ $",
                "ERR_JWT_4012 OR (DROP TABLE)",
                "",
                "   \n\t ",
            ]
            for q in malicious_queries:
                fts_q = build_fts_query(q)
                if fts_q:
                    rows = conn.execute(
                        "SELECT id FROM t WHERE t MATCH ?", (fts_q,)
                    ).fetchall()
                    self.assertIsInstance(rows, list)
        finally:
            conn.close()

        capped = build_fts_query("锁 数据库连接池连接超时配置 ERR_JWT_4012", max_terms=4)
        terms = capped.split(" OR ")
        self.assertLessEqual(len(terms), 4)
        self.assertIn('"err_jwt_4012"', terms)

    def test_empty_inputs_and_token_estimation_bounds(self) -> None:
        self.assertEqual(extract_identifiers(""), [])
        self.assertEqual(tokenize_text(""), [])
        self.assertEqual(to_fts_document("   "), "")
        self.assertEqual(char_ngrams("", n=3), [])
        self.assertEqual(char_ngrams("abc", n=0), [])
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("x"), 0)
        zh = estimate_tokens("禁止在日志中打印认证凭据原文")
        en = estimate_tokens("Never print raw auth tokens in logs")
        self.assertGreater(zh, 10)
        self.assertGreater(en, 6)
        self.assertGreater(estimate_tokens("a " * 40), estimate_tokens("a " * 10))
        self.assertTrue(char_ngrams("database"))


if __name__ == "__main__":
    unittest.main()
