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
        for expected in ("数据", "据库", "库连", "连接", "接池", "池超", "超时", "库", "锁"[:0]):
            if expected:
                self.assertIn(expected, toks)
        # Single-char CJK noun in mixed text should survive.
        mixed = tokenize_text("Redis 锁超时")
        self.assertIn("redis", mixed)
        self.assertIn("锁", mixed)

    def test_code_and_identifier_extraction(self) -> None:
        text = (
            "Error ERR_JWT_4012 in src/auth/jwt_verify.py:88 using jwtRefreshToken, "
            "MySQL 5.7 on port 9443 with rate-limit `utf8mb4`."
        )
        ids = extract_identifiers(text)
        self.assertIn("err_jwt_4012", ids)
        self.assertIn("src/auth/jwt_verify.py:88", ids)
        self.assertIn("jwtrefreshtoken", ids)
        self.assertIn("5.7", ids)
        self.assertIn("9443", ids)
        self.assertIn("rate-limit", ids)
        self.assertIn("utf8mb4", ids)

        toks = tokenize_text("jwtRefreshToken err_jwt_4012")
        for expected in ("jwtrefreshtoken", "jwt", "refresh", "token", "err_jwt_4012", "4012"):
            self.assertIn(expected, toks)

    def test_fts5_query_safety_against_special_chars(self) -> None:
        conn = sqlite3.connect(":memory:")
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
        ]
        for q in malicious_queries:
            fts_q = build_fts_query(q)
            if fts_q:
                rows = conn.execute(
                    "SELECT id FROM t WHERE t MATCH ?", (fts_q,)
                ).fetchall()
                self.assertIsInstance(rows, list)
        conn.close()

    def test_token_estimation_bounds(self) -> None:
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
