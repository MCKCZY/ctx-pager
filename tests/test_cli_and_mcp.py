from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ctx_pager.cli import main as cli_main
from ctx_pager.engine import ContextPager
from ctx_pager.mcp_server import McpServer
from ctx_pager.tokenizer import estimate_tokens


class TestCliSkillAndMcp(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_main(argv)
        return code, out_buf.getvalue(), err_buf.getvalue()

    def test_cli_full_lifecycle_and_error_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cli.db")

            code, out, _ = self._run_cli(["--db", db, "--json", "init"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["doctor"]["ok"])

            code, out, _ = self._run_cli(
                [
                    "commit",
                    "--db",
                    db,
                    "--json",
                    "--kind",
                    "constraint",
                    "--scope",
                    "db",
                    "--pin",
                    "--title",
                    "MySQL 5.7 limit",
                    "--summary",
                    "Use utf8mb4 with index prefix <= 191",
                    "--entities",
                    "MySQL 5.7,utf8mb4",
                    "--turn",
                    "1",
                ]
            )
            self.assertEqual(code, 0)
            c_id = json.loads(out)["id"]

            code, out, _ = self._run_cli(
                [
                    "--db",
                    db,
                    "--json",
                    "commit",
                    "--kind",
                    "decision",
                    "--scope",
                    "auth",
                    "--title",
                    "Port 8080",
                    "--summary",
                    "Listen 8080",
                    "--turn",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            d1_id = json.loads(out)["id"]

            code, out, _ = self._run_cli(
                [
                    "--db",
                    db,
                    "--json",
                    "supersede",
                    d1_id,
                    "--title",
                    "Port 9443",
                    "--summary",
                    "Listen TLS 9443",
                    "--detail",
                    "Full TLS config details on 9443",
                    "--turn",
                    "6",
                ]
            )
            self.assertEqual(code, 0)
            d2_id = json.loads(out)["new"]["id"]

            code, _, _ = self._run_cli(
                ["--db", db, "link", d2_id, c_id, "--relation", "depends_on", "--weight", "0.8"]
            )
            self.assertEqual(code, 0)

            code, out, _ = self._run_cli(
                [
                    "--db",
                    db,
                    "--json",
                    "commit",
                    "--kind",
                    "open_loop",
                    "--title",
                    "Check TLS cert",
                    "--summary",
                    "Verify cert chain",
                    "--turn",
                    "7",
                ]
            )
            ol_id = json.loads(out)["id"]

            code, _, _ = self._run_cli(
                [
                    "--db",
                    db,
                    "checkpoint",
                    "--goal",
                    "Deploy TLS",
                    "--state",
                    "Port 9443 ready",
                    "--turn",
                    "8",
                ]
            )
            self.assertEqual(code, 0)

            code, out, _ = self._run_cli(
                ["--db", db, "--json", "recall", "--query", "TLS port", "--budget", "1024", "--turn", "10"]
            )
            self.assertEqual(code, 0)
            frame = json.loads(out)
            self.assertLessEqual(frame["used_tokens"], 1024)
            self.assertIn("9443", frame["rendered"])
            self.assertNotIn("8080", frame["rendered"])

            code, out, _ = self._run_cli(["--db", db, "--json", "page-in", d2_id, "M-9999"])
            self.assertEqual(code, 0)
            paged = json.loads(out)
            self.assertIn("Full TLS config details", paged["records"][0]["detail"])
            self.assertEqual(paged["missing_ids"], ["M-9999"])

            # All-missing page-in exits 1 with stderr warning.
            code, _, err = self._run_cli(["--db", db, "page-in", "M-9999"])
            self.assertEqual(code, 1)
            self.assertIn("M-9999", err)

            code, _, _ = self._run_cli(["--db", db, "resolve", ol_id, "--turn", "11"])
            self.assertEqual(code, 0)

            code, _, _ = self._run_cli(["--db", db, "pin", c_id, "--unpin"])
            self.assertEqual(code, 0)

            # Ingest from file + dry-run compact.
            ingest_file = Path(tmp) / "turn.txt"
            ingest_file.write_text("必须校验所有入参。决定启用 rate_limiter。", encoding="utf-8")
            code, out, _ = self._run_cli(
                ["--db", db, "--json", "ingest", "--file", str(ingest_file), "--scope", "api", "--turn", "12"]
            )
            self.assertEqual(code, 0)
            self.assertGreaterEqual(len(json.loads(out)["created"]), 2)

            code, out, _ = self._run_cli(["--db", db, "--json", "compact", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["dry_run"])

            code, out, _ = self._run_cli(["--db", db, "--json", "replay", "--turn", "3", "--query", "port"])
            self.assertEqual(code, 0)
            self.assertIn("8080", json.loads(out)["frame"]["rendered"])

            export_file = str(Path(tmp) / "events.jsonl")
            code, _, _ = self._run_cli(["--db", db, "export", "--output", export_file])
            self.assertEqual(code, 0)
            self.assertTrue(Path(export_file).is_file())

            export_json = Path(tmp) / "snapshot.json"
            code, _, _ = self._run_cli(
                ["--db", db, "export", "--format", "json", "--output", str(export_json)]
            )
            self.assertEqual(code, 0)
            self.assertIn("memories", json.loads(export_json.read_text(encoding="utf-8")))

            # Validation error paths.
            for bad_argv in (
                ["--db", db, "commit", "--kind", "fact", "--title", "   ", "--summary", "x"],
                ["--db", db, "recall", "--budget", "0"],
                ["--db", db, "supersede", "M-9999", "--title", "x", "--summary", "y"],
                ["--db", db, "resolve", "M-9999"],
            ):
                bad_code, _, _ = self._run_cli(bad_argv)
                self.assertEqual(bad_code, 1)

            code, out, _ = self._run_cli(["--db", db, "--json", "doctor"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["ok"])

            # Tamper DB -> doctor exits 1.
            with ContextPager(db) as eng:
                with eng.storage.conn:
                    eng.storage.conn.execute("UPDATE events SET payload_json = '{}' WHERE seq = 1")
            code, out, _ = self._run_cli(["--db", db, "--json", "doctor"])
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(out)["ok"])

    def test_skill_package_shim_and_installer(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        skill_md = repo_root / "skills" / "ctx-pager" / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
        self.assertLess(len(text.splitlines()), 140)
        self.assertLess(estimate_tokens(text), 1200)
        self.assertIn("name: ctx-pager", text)

        shim = repo_root / "skills" / "ctx-pager" / "scripts" / "ctx_pager.py"
        proc = subprocess.run(
            [sys.executable, str(shim), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ctx-pager", proc.stdout)

        sys.path.insert(0, str(repo_root))
        try:
            from scripts.install_skill import install_skill

            with tempfile.TemporaryDirectory() as tmp:
                copy_dest = Path(tmp) / "copied_skill"
                installed = install_skill(target="path", custom_path=str(copy_dest))
                self.assertTrue((installed / "scripts" / "_lib" / "ctx_pager" / "cli.py").is_file())
                with self.assertRaises(FileExistsError):
                    install_skill(target="path", custom_path=str(copy_dest), force=False)

                # --force replacement with --symlink
                sym_installed = install_skill(
                    target="path", custom_path=str(copy_dest), symlink=True, force=True
                )
                self.assertTrue(sym_installed.is_symlink())
        finally:
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))

    def test_mcp_server_protocol_tools_and_errors(self) -> None:
        with ContextPager(":memory:") as eng:
            reqs = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_commit",
                        "arguments": {
                            "kind": "constraint",
                            "scope": "api",
                            "title": "Rate limit 100 rpm",
                            "summary": "Enforce 100 requests/min",
                            "detail": "Configured via ratelimit_lua_v2",
                            "pinned": True,
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 41,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_commit",
                        "arguments": {
                            "kind": "open_loop",
                            "scope": "api",
                            "title": "Old limiter todo",
                            "summary": "Check ratelimit_v1",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_link",
                        "arguments": {
                            "src_id": "M-0002",
                            "dst_id": "M-0001",
                            "relation": "depends_on",
                            "weight": 0.9,
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 43,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_supersede",
                        "arguments": {
                            "old_id": "M-0002",
                            "title": "Updated limiter todo",
                            "summary": "Check ratelimit_lua_v2",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 44,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_resolve",
                        "arguments": {"memory_id": "M-0003"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 45,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_checkpoint",
                        "arguments": {"goal": "Ship limiter", "state_summary": "Ready"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 46,
                    "method": "tools/call",
                    "params": {"name": "ctx_compact", "arguments": {"dry_run": True}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 47,
                    "method": "tools/call",
                    "params": {"name": "ctx_stats", "arguments": {}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_recall",
                        "arguments": {"query": "Rate limit", "budget": 1024},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_page_in",
                        "arguments": {"ids": ["M-0001"]},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 61,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_page_in",
                        "arguments": {"ids": ["M-9999"]},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 62,
                    "method": "tools/call",
                    "params": {
                        "name": "ctx_supersede",
                        "arguments": {
                            "old_id": "M-0002",
                            "title": "Double supersede",
                            "summary": "Should fail",
                        },
                    },
                },
                {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "resources/read",
                    "params": {"uri": "ctx://working-set"},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 81,
                    "method": "resources/read",
                    "params": {"uri": "ctx://missing"},
                },
                {"jsonrpc": "2.0", "id": 9, "method": "unknown/method"},
                [1, 2, 3],
            ]
            raw_lines = "\n".join(json.dumps(r) for r in reqs) + "\n{broken json\n"
            stdin_buf = io.StringIO(raw_lines)
            stdout_buf = io.StringIO()
            server = McpServer(eng, stdin=stdin_buf, stdout=stdout_buf)
            server.serve()

            responses = [
                json.loads(line)
                for line in stdout_buf.getvalue().strip().splitlines()
                if line.strip()
            ]
            # 1 notification -> 20 responses for 21 reqs + 1 parse error = 21 lines.
            self.assertEqual(len(responses), 21)
            by_id = {r["id"]: r for r in responses if r.get("id") is not None}
            self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "ctx-pager")
            self.assertEqual(len(by_id[3]["result"]["tools"]), 9)
            self.assertFalse(by_id[4]["result"]["isError"])
            self.assertFalse(by_id[42]["result"]["isError"])
            self.assertFalse(by_id[43]["result"]["isError"])
            self.assertFalse(by_id[44]["result"]["isError"])
            self.assertFalse(by_id[45]["result"]["isError"])
            self.assertFalse(by_id[46]["result"]["isError"])
            self.assertFalse(by_id[47]["result"]["isError"])
            self.assertIn("100 requests/min", by_id[5]["result"]["content"][0]["text"])
            self.assertIn("_meta", by_id[5]["result"])
            self.assertIn("ratelimit_lua_v2", by_id[6]["result"]["content"][0]["text"])
            self.assertTrue(by_id[61]["result"]["isError"])
            self.assertTrue(by_id[62]["result"]["isError"])
            self.assertEqual(by_id[7]["result"]["resources"][0]["uri"], "ctx://working-set")
            self.assertIn("100 requests/min", by_id[8]["result"]["contents"][0]["text"])
            self.assertEqual(by_id[81]["error"]["code"], -32602)
            self.assertEqual(by_id[9]["error"]["code"], -32601)
            null_errors = [r["error"]["code"] for r in responses if r.get("id") is None]
            self.assertIn(-32600, null_errors)
            self.assertIn(-32700, null_errors)


if __name__ == "__main__":
    unittest.main()
