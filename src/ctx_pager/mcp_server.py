"""Pure-stdlib JSON-RPC 2.0 MCP server exposing ctx-pager over stdio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, TextIO

from ctx_pager.engine import ContextPager
from ctx_pager.models import LinkRelation, MemoryKind, Resolution

_PROTOCOL_VERSION = "2024-11-05"


def _tool_definitions() -> list[dict[str, Any]]:
    kinds = [k.value for k in MemoryKind]
    relations = [r.value for r in LinkRelation]
    return [
        {
            "name": "ctx_commit",
            "description": "Commit a structured memory entry (constraint, decision, fact, procedure, episode, or open_loop).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": kinds},
                    "scope": {"type": "string", "default": "global"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "detail": {"type": "string", "default": ""},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "pinned": {"type": "boolean", "default": False},
                    "supersedes_id": {"type": "string"},
                    "turn": {"type": "integer"},
                },
                "required": ["kind", "title", "summary"],
            },
        },
        {
            "name": "ctx_recall",
            "description": "Retrieve and pack a token-budgeted working memory frame using L0/L1/L2 resolutions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "budget": {"type": "integer"},
                    "profile": {
                        "type": "string",
                        "enum": ["tiny_4k", "small_8k", "medium_16k", "large_32k"],
                        "default": "small_8k",
                    },
                    "format": {"type": "string", "enum": ["markdown", "xml", "json"]},
                    "include_superseded": {"type": "boolean", "default": False},
                },
            },
        },
        {
            "name": "ctx_page_in",
            "description": "Page in full L2 detail and 1-hop linked neighbors for specific memory IDs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "include_neighbors": {"type": "boolean", "default": True},
                },
                "required": ["ids"],
            },
        },
        {
            "name": "ctx_supersede",
            "description": "Supersede an outdated memory with a new version and mark the old record superseded.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "detail": {"type": "string", "default": ""},
                    "scope": {"type": "string"},
                    "turn": {"type": "integer"},
                },
                "required": ["old_id", "title", "summary"],
            },
        },
        {
            "name": "ctx_checkpoint",
            "description": "Save a working-state checkpoint summarizing current goal, progress, and open loops.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "state_summary": {"type": "string"},
                    "open_loops": {"type": "array", "items": {"type": "string"}},
                    "turn": {"type": "integer"},
                },
                "required": ["goal", "state_summary"],
            },
        },
        {
            "name": "ctx_compact",
            "description": "Losslessly compact aged, low-access episode memories into scope digests.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "older_than_turns": {"type": "integer", "default": 15},
                    "min_group_size": {"type": "integer", "default": 3},
                    "dry_run": {"type": "boolean", "default": False},
                },
            },
        },
        {
            "name": "ctx_stats",
            "description": "Return storage counts, token footprints, and integrity status.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ctx_link",
            "description": "Create a typed relation link between two stored memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "src_id": {"type": "string"},
                    "dst_id": {"type": "string"},
                    "relation": {"type": "string", "enum": relations},
                    "weight": {"type": "number", "default": 1.0},
                    "turn": {"type": "integer"},
                },
                "required": ["src_id", "dst_id", "relation"],
            },
        },
        {
            "name": "ctx_resolve",
            "description": "Mark a memory (such as an open_loop) as resolved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "turn": {"type": "integer"},
                },
                "required": ["memory_id"],
            },
        },
    ]


class McpServer:
    """Line-delimited JSON-RPC 2.0 MCP server backed by ContextPager."""

    def __init__(
        self,
        engine: ContextPager,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.engine = engine
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def handle_request(self, req: Any) -> dict[str, Any] | None:
        if not isinstance(req, dict):
            return self._error(None, -32600, "Invalid Request")
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params")
        if params is None:
            params = {}

        if req_id is None and isinstance(method, str) and method.startswith("notifications/"):
            return None
        if not isinstance(method, str) or not method:
            return self._error(req_id, -32600, "Invalid Request")
        if not isinstance(params, dict):
            return self._error(req_id, -32602, "Invalid params")

        try:
            if method == "initialize":
                result: Any = {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "ctx-pager", "version": "0.1.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _tool_definitions()}
            elif method == "tools/call":
                raw_args = params.get("arguments")
                if raw_args is None:
                    raw_args = {}
                if not isinstance(raw_args, dict):
                    return self._error(req_id, -32602, "Invalid tool arguments")
                result = self._call_tool(params.get("name", ""), raw_args)
            elif method == "resources/list":
                result = {
                    "resources": [
                        {
                            "uri": "ctx://working-set",
                            "name": "Current Working Memory Frame",
                            "mimeType": "text/markdown",
                        }
                    ]
                }
            elif method == "resources/read":
                uri = params.get("uri", "")
                if uri != "ctx://working-set":
                    return self._error(req_id, -32602, f"Unknown resource URI: {uri}")
                frame = self.engine.recall("", profile="small_8k")
                result = {
                    "contents": [
                        {
                            "uri": "ctx://working-set",
                            "mimeType": "text/markdown",
                            "text": frame.rendered,
                        }
                    ]
                }
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")

            if req_id is None:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            if req_id is None:
                return None
            return self._error(req_id, -32603, str(exc))

    def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "ctx_commit":
                rec = self.engine.commit(
                    kind=args["kind"],
                    scope=args.get("scope", "global"),
                    title=args["title"],
                    summary=args["summary"],
                    detail=args.get("detail", ""),
                    entities=args.get("entities"),
                    tags=args.get("tags"),
                    pinned=bool(args.get("pinned", False)),
                    supersedes_id=args.get("supersedes_id"),
                    turn=args.get("turn"),
                )
                return self._tool_ok(rec.to_dict())

            if name == "ctx_recall":
                frame = self.engine.recall(
                    args.get("query", ""),
                    budget=args.get("budget"),
                    profile=args.get("profile", "small_8k"),
                    fmt=args.get("format"),
                    include_superseded=bool(args.get("include_superseded", False)),
                )
                return {
                    "content": [{"type": "text", "text": frame.rendered}],
                    "isError": False,
                    "_meta": {
                        "used_tokens": frame.used_tokens,
                        "budget_tokens": frame.budget_tokens,
                        "page_in_hints": frame.page_in_hints,
                    },
                }

            if name == "ctx_page_in":
                res = self.engine.page_in(
                    list(args["ids"]),
                    include_neighbors=bool(args.get("include_neighbors", True)),
                )
                if not res["records"]:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Memory not found: {', '.join(res['missing_ids'])}",
                            }
                        ],
                        "isError": True,
                    }
                lines = [r.render(Resolution.L2) for r in res["records"]]
                if res["neighbors"]:
                    lines.append("--- Neighbors ---")
                    lines.extend(n.render(Resolution.L1) for n in res["neighbors"])
                if res["missing_ids"]:
                    lines.append(f"[Missing IDs: {', '.join(res['missing_ids'])}]")
                return {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                    "isError": False,
                }

            if name == "ctx_supersede":
                old_rec, new_rec = self.engine.supersede(
                    args["old_id"],
                    title=args["title"],
                    summary=args["summary"],
                    detail=args.get("detail", ""),
                    scope=args.get("scope"),
                    turn=args.get("turn"),
                )
                return self._tool_ok({"old_id": old_rec.id, "new": new_rec.to_dict()})

            if name == "ctx_checkpoint":
                cp = self.engine.checkpoint(
                    goal=args["goal"],
                    state_summary=args["state_summary"],
                    open_loops=args.get("open_loops"),
                    turn=args.get("turn"),
                )
                return self._tool_ok(cp.to_dict())

            if name == "ctx_compact":
                rep = self.engine.compact(
                    older_than_turns=int(args.get("older_than_turns", 15)),
                    min_group_size=int(args.get("min_group_size", 3)),
                    dry_run=bool(args.get("dry_run", False)),
                )
                return self._tool_ok(rep.to_dict())

            if name == "ctx_stats":
                return self._tool_ok(
                    {"stats": self.engine.stats(), "doctor": self.engine.doctor()}
                )

            if name == "ctx_link":
                lnk = self.engine.link(
                    args["src_id"],
                    args["dst_id"],
                    args["relation"],
                    weight=float(args.get("weight", 1.0)),
                    turn=args.get("turn"),
                )
                return self._tool_ok(lnk.to_dict())

            if name == "ctx_resolve":
                rec = self.engine.resolve(args["memory_id"], turn=args.get("turn"))
                return self._tool_ok(rec.to_dict())

            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }
        except (KeyError, ValueError, TypeError) as exc:
            return {
                "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                "isError": True,
            }

    @staticmethod
    def _tool_ok(payload: Any) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, indent=2),
                }
            ],
            "isError": False,
        }

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def serve(self) -> None:
        for line in self.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                req = json.loads(stripped)
            except json.JSONDecodeError as exc:
                resp = self._error(None, -32700, f"Parse error: {exc}")
            else:
                resp = self.handle_request(req)
            if resp is not None:
                self.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                self.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctx-pager-mcp")
    parser.add_argument(
        "--db",
        default=os.environ.get("CTX_PAGER_DB", ".ctx_pager/memory.db"),
        help="Path to SQLite store",
    )
    args = parser.parse_args(argv)
    with ContextPager(args.db) as engine:
        server = McpServer(engine)
        server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
