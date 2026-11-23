from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ctx_pager.cli import main as cli_main
from ctx_pager.engine import ContextPager
from ctx_pager.mcp_server import McpServer
from ctx_pager.tokenizer import estimate_tokens


class TestCliSkillAndMcp(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_cli_full_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cli.db")

            code, out = self._run_cli(["--db", db, "--json", "init"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["doctor"]["ok"])

            code, out = self._run_cli(
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

            code, out = self._run_cli(
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

            code, out = self._run_cli(
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

            code, _ = self._run_cli(
                ["--db", db, "link", d2_id, c_id, "--relation", "depends_on", "--weight", "0.8"]
            )
            self.assertEqual(code, 0)

            code, out = self._run_cli(
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

            code, _ = self._run_cli(
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

            code, out = self._run_cli(
                ["--db", db, "--json", "recall", "--query", "TLS port", "--budget", "1024", "--turn", "10"]
            )
            self.assertEqual(code, 0)
            frame = json.loads(out)
            self.assertLessEqual(frame["used_tokens"], 1024)
            self.assertIn("9443", frame["rendered"])
            self.assertNotIn("8080", frame["rendered"])

            code, out = self._run_cli(["--db", db, "--json", "page-in", d2_id])
            self.assertEqual(code, 0)
            self.assertIn("Full TLS config details", json.loads(out)["records"][0]["detail"])

            code, _ = self._run_cli(["--db", db, "resolve", ol_id, "--turn", "11"])
            self.assertEqual(code, 0)

            code, _ = self._run_cli(["--db", db, "pin", c_id, "--unpin"])
            self.assertEqual(code, 0)

            code, out = self._run_cli(["--db", db, "--json", "replay", "--turn", "3", "--query", "port"])
            self.assertEqual(code, 0)
            self.assertIn("8080", json.loads(out)["frame"]["rendered"])

            export_file = str(Path(tmp) / "events.jsonl")
            code, _ = self._run_cli(["--db", db, "export", "--output", export_file])
            self.assertEqual(code, 0)
            self.assertTrue(Path(export_file).is_file())

            code, out = self._run_cli(["--db", db, "--json", "doctor"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["ok"])

    def test_skill_package_and_shim(self) -> None:
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

    def test_mcp_server_protocol_and_tools(self) -> None:
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
                {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "resources/read",
                    "params": {"uri": "ctx://working-set"},
                },
                {"jsonrpc": "2.0", "id": 9, "method": "unknown/method"},
            ]
            stdin_buf = io.StringIO("\n".join(json.dumps(r) for r in reqs) + "\n")
            stdout_buf = io.StringIO()
            server = McpServer(eng, stdin=stdin_buf, stdout=stdout_buf)
            server.serve()

            responses = [
                json.loads(line)
                for line in stdout_buf.getvalue().strip().splitlines()
                if line.strip()
            ]
            # Notification produces no response -> 9 responses for 10 messages.
            self.assertEqual(len(responses), 9)
            by_id = {r["id"]: r for r in responses}
            self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "ctx-pager")
            self.assertGreaterEqual(len(by_id[3]["result"]["tools"]), 7)
            self.assertFalse(by_id[4]["result"]["isError"])
            self.assertIn("100 requests/min", by_id[5]["result"]["content"][0]["text"])
            self.assertIn("ratelimit_lua_v2", by_id[6]["result"]["content"][0]["text"])
            self.assertEqual(by_id[7]["result"]["resources"][0]["uri"], "ctx://working-set")
            self.assertIn("100 requests/min", by_id[8]["result"]["contents"][0]["text"])
            self.assertEqual(by_id[9]["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
