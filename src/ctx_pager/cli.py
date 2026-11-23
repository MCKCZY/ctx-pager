"""Command-line interface for ctx-pager."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ctx_pager.engine import ContextPager
from ctx_pager.models import (
    PRESET_PROFILES,
    LinkRelation,
    MemoryKind,
    MemoryStatus,
    Resolution,
)


def _split_csv_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for val in values:
        for part in val.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db",
        default=argparse.SUPPRESS,
        help="Path to SQLite store (default: $CTX_PAGER_DB or .ctx_pager/memory.db)",
    )
    common.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit machine-readable JSON output",
    )

    parser = argparse.ArgumentParser(
        prog="ctx-pager",
        description="Three-tier paged context memory engine for small-context AI agents.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("CTX_PAGER_DB", ".ctx_pager/memory.db"),
        help="Path to SQLite store (default: $CTX_PAGER_DB or .ctx_pager/memory.db)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON output",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", parents=[common], help="Initialize database and verify FTS5")

    p_commit = sub.add_parser("commit", parents=[common], help="Commit a structured memory record")
    p_commit.add_argument("--kind", required=True, choices=[k.value for k in MemoryKind])
    p_commit.add_argument("--scope", default="global")
    p_commit.add_argument("--title", required=True)
    p_commit.add_argument("--summary", required=True)
    p_commit.add_argument("--detail", default="")
    p_commit.add_argument("--entities", action="append", help="Comma-separated entity identifiers")
    p_commit.add_argument("--tags", action="append", help="Comma-separated tags")
    p_commit.add_argument("--importance", type=float, default=None)
    p_commit.add_argument("--pin", action="store_true", default=False)
    p_commit.add_argument("--supersedes", dest="supersedes_id", default=None)
    p_commit.add_argument("--turn", type=int, default=None)

    p_recall = sub.add_parser("recall", parents=[common], help="Pack a budgeted working memory frame")
    p_recall.add_argument("--query", default="")
    p_recall.add_argument("--budget", type=int, default=None)
    p_recall.add_argument("--profile", default="small_8k", choices=sorted(PRESET_PROFILES))
    p_recall.add_argument("--format", dest="fmt", default=None, choices=["markdown", "xml", "json"])
    p_recall.add_argument("--include-superseded", action="store_true", default=False)
    p_recall.add_argument("--turn", type=int, default=None)

    p_page = sub.add_parser("page-in", parents=[common], help="Expand L2 detail for specific memory IDs")
    p_page.add_argument("ids", nargs="+")
    p_page.add_argument("--no-neighbors", action="store_true", default=False)

    p_sup = sub.add_parser("supersede", parents=[common], help="Replace an outdated memory with a new one")
    p_sup.add_argument("old_id")
    p_sup.add_argument("--title", required=True)
    p_sup.add_argument("--summary", required=True)
    p_sup.add_argument("--detail", default="")
    p_sup.add_argument("--entities", action="append")
    p_sup.add_argument("--scope", default=None)
    p_sup.add_argument("--turn", type=int, default=None)

    p_link = sub.add_parser("link", parents=[common], help="Create a typed relation link between memories")
    p_link.add_argument("src_id")
    p_link.add_argument("dst_id")
    p_link.add_argument("--relation", required=True, choices=[r.value for r in LinkRelation])
    p_link.add_argument("--weight", type=float, default=1.0)

    p_resolve = sub.add_parser("resolve", parents=[common], help="Mark a memory (e.g. open_loop) resolved")
    p_resolve.add_argument("memory_id")
    p_resolve.add_argument("--turn", type=int, default=None)

    p_pin = sub.add_parser("pin", parents=[common], help="Pin or unpin a memory in Slot 1")
    p_pin.add_argument("memory_id")
    p_pin.add_argument("--unpin", action="store_true", default=False)
    p_pin.add_argument("--turn", type=int, default=None)

    p_cp = sub.add_parser("checkpoint", parents=[common], help="Save a working-state checkpoint")
    p_cp.add_argument("--goal", required=True)
    p_cp.add_argument("--state", required=True)
    p_cp.add_argument("--open-loop", dest="open_loops", action="append", default=None)
    p_cp.add_argument("--turn", type=int, default=None)

    p_compact = sub.add_parser("compact", parents=[common], help="Compact aged low-access episodes")
    p_compact.add_argument("--older-than", type=int, default=15)
    p_compact.add_argument("--min-group", type=int, default=3)
    p_compact.add_argument("--max-access", type=int, default=1)
    p_compact.add_argument("--dry-run", action="store_true", default=False)

    p_ingest = sub.add_parser("ingest", parents=[common], help="Ingest raw dialogue text deterministically")
    p_ingest.add_argument("--role", default="user")
    src_grp = p_ingest.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--text", default=None)
    src_grp.add_argument("--file", default=None)
    p_ingest.add_argument("--scope", default="global")
    p_ingest.add_argument("--turn", type=int, default=None)

    p_replay = sub.add_parser("replay", parents=[common], help="Replay event log up to a historical turn")
    p_replay.add_argument("--turn", type=int, required=True)
    p_replay.add_argument("--query", default="")
    p_replay.add_argument("--budget", type=int, default=2048)

    p_inspect = sub.add_parser("inspect", parents=[common], help="List stored memory entries")
    p_inspect.add_argument("--status", choices=[s.value for s in MemoryStatus], default=None)
    p_inspect.add_argument("--kind", choices=[k.value for k in MemoryKind], default=None)
    p_inspect.add_argument("--scope", default=None)
    p_inspect.add_argument("--limit", type=int, default=50)

    sub.add_parser("stats", parents=[common], help="Show store statistics")

    p_export = sub.add_parser("export", parents=[common], help="Export event ledger or snapshot")
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--format", dest="export_fmt", choices=["jsonl", "json"], default="jsonl")

    sub.add_parser("doctor", parents=[common], help="Verify SQLite, FTS5 sync, and event hash chain")

    return parser


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    as_json: bool = bool(args.json_output)

    try:
        with ContextPager(args.db) as engine:
            cmd = args.command

            if cmd == "init":
                doc = engine.doctor()
                payload = {"db_path": engine.storage.db_path, "doctor": doc}
                if as_json:
                    _print_json(payload)
                else:
                    print(f"Initialized store: {engine.storage.db_path} (ok={doc['ok']})")
                return 0 if doc["ok"] else 1

            if cmd == "commit":
                rec = engine.commit(
                    kind=args.kind,
                    scope=args.scope,
                    title=args.title,
                    summary=args.summary,
                    detail=args.detail,
                    entities=_split_csv_list(args.entities),
                    tags=_split_csv_list(args.tags),
                    importance=args.importance,
                    pinned=args.pin,
                    supersedes_id=args.supersedes_id,
                    turn=args.turn,
                )
                if as_json:
                    _print_json(rec.to_dict())
                else:
                    print(f"Committed {rec.id} ({rec.kind.value}, L0={rec.token_l0}, L1={rec.token_l1}, L2={rec.token_l2} tok)")
                return 0

            if cmd == "recall":
                frame = engine.recall(
                    args.query,
                    budget=args.budget,
                    profile=args.profile,
                    fmt=args.fmt,
                    include_superseded=args.include_superseded,
                    turn=args.turn,
                )
                if as_json:
                    _print_json(frame.to_dict())
                else:
                    print(frame.rendered)
                return 0

            if cmd == "page-in":
                res = engine.page_in(
                    args.ids, include_neighbors=not args.no_neighbors
                )
                if as_json:
                    _print_json(
                        {
                            "records": [r.to_dict() for r in res["records"]],
                            "links": [l.to_dict() for l in res["links"]],
                            "neighbors": [n.to_dict() for n in res["neighbors"]],
                        }
                    )
                else:
                    for r in res["records"]:
                        print(r.render(Resolution.L2))
                    if res["links"]:
                        print("\nLinks:")
                        for l in res["links"]:
                            print(f"  {l.src_id} -[{l.relation.value} ({l.weight:.2f})]-> {l.dst_id}")
                    if res["neighbors"]:
                        print("\nLinked Neighbors (L1):")
                        for n in res["neighbors"]:
                            print(n.render(Resolution.L1))
                return 0

            if cmd == "supersede":
                old_rec, new_rec = engine.supersede(
                    args.old_id,
                    title=args.title,
                    summary=args.summary,
                    detail=args.detail,
                    entities=_split_csv_list(args.entities) if args.entities else None,
                    scope=args.scope,
                    turn=args.turn,
                )
                if as_json:
                    _print_json({"old": old_rec.to_dict(), "new": new_rec.to_dict()})
                else:
                    print(f"Superseded {old_rec.id} -> {new_rec.id}")
                return 0

            if cmd == "link":
                lnk = engine.link(
                    args.src_id, args.dst_id, args.relation, weight=args.weight
                )
                if as_json:
                    _print_json(lnk.to_dict())
                else:
                    print(f"Linked {lnk.src_id} -[{lnk.relation.value}]-> {lnk.dst_id}")
                return 0

            if cmd == "resolve":
                rec = engine.resolve(args.memory_id, turn=args.turn)
                if as_json:
                    _print_json(rec.to_dict())
                else:
                    print(f"Resolved {rec.id} (status={rec.status.value})")
                return 0

            if cmd == "pin":
                rec = engine.pin(args.memory_id, pinned=not args.unpin, turn=args.turn)
                if as_json:
                    _print_json(rec.to_dict())
                else:
                    print(f"Updated {rec.id} (pinned={rec.pinned})")
                return 0

            if cmd == "checkpoint":
                cp = engine.checkpoint(
                    goal=args.goal,
                    state_summary=args.state,
                    open_loops=args.open_loops,
                    turn=args.turn,
                )
                if as_json:
                    _print_json(cp.to_dict())
                else:
                    print(f"Saved checkpoint {cp.id} at turn {cp.turn}")
                return 0

            if cmd == "compact":
                report = engine.compact(
                    older_than_turns=args.older_than,
                    min_group_size=args.min_group,
                    max_access_count=args.max_access,
                    dry_run=args.dry_run,
                )
                if as_json:
                    _print_json(report.to_dict())
                else:
                    print(
                        f"Compacted {report.groups_compacted} group(s), "
                        f"archived={len(report.archived_ids)}, "
                        f"saved={report.tokens_saved} tokens (dry_run={report.dry_run})"
                    )
                return 0

            if cmd == "ingest":
                text = args.text
                if args.file is not None:
                    text = Path(args.file).read_text(encoding="utf-8")
                res = engine.ingest_turn(
                    role=args.role, text=text or "", scope=args.scope, turn=args.turn
                )
                payload = {
                    "event": res.event.to_dict(),
                    "created": [r.to_dict() for r in res.created],
                    "updated": [r.to_dict() for r in res.updated],
                    "superseded": [
                        {"old_id": oid, "new": rec.to_dict()}
                        for oid, rec in res.superseded
                    ],
                }
                if as_json:
                    _print_json(payload)
                else:
                    print(
                        f"Ingested turn {res.event.turn}: created={len(res.created)}, "
                        f"updated={len(res.updated)}, superseded={len(res.superseded)}"
                    )
                return 0

            if cmd == "replay":
                rep = engine.replay(args.turn, query=args.query, budget=args.budget)
                if as_json:
                    _print_json(
                        {
                            "target_turn": rep["target_turn"],
                            "stats": rep["stats"],
                            "memories": rep["memories"],
                            "frame": rep["frame"].to_dict(),
                        }
                    )
                else:
                    print(f"Replay state at turn {rep['target_turn']} (memories={len(rep['memories'])})")
                    print(rep["frame"].rendered)
                return 0

            if cmd == "inspect":
                records = engine.storage.list_memories(
                    status=MemoryStatus(args.status) if args.status else None,
                    kind=MemoryKind(args.kind) if args.kind else None,
                    scope=args.scope,
                    limit=args.limit,
                )
                if as_json:
                    _print_json([r.to_dict() for r in records])
                else:
                    for r in records:
                        pin_flag = " [PIN]" if r.pinned else ""
                        print(f"{r.render(Resolution.L0)}{pin_flag} (L1={r.token_l1}, L2={r.token_l2})")
                return 0

            if cmd == "stats":
                st = engine.stats()
                if as_json:
                    _print_json(st)
                else:
                    for k, v in st.items():
                        print(f"{k}: {v}")
                return 0

            if cmd == "export":
                out_path = Path(args.output)
                if args.export_fmt == "jsonl":
                    count = engine.storage.export_events_jsonl(out_path)
                    payload = {"format": "jsonl", "events_exported": count, "output": str(out_path)}
                else:
                    snap = engine.storage.export_snapshot_dict()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    payload = {
                        "format": "json",
                        "memories_exported": len(snap["memories"]),
                        "output": str(out_path),
                    }
                if as_json:
                    _print_json(payload)
                else:
                    print(f"Exported to {out_path} ({payload})")
                return 0

            if cmd == "doctor":
                doc = engine.doctor()
                if as_json:
                    _print_json(doc)
                else:
                    print(f"doctor ok={doc['ok']} memories={doc['memories_count']} fts={doc['fts_count']} chain={doc['event_chain_ok']}")
                    for issue in doc["issues"]:
                        print(f"  ISSUE: {issue}", file=sys.stderr)
                return 0 if doc["ok"] else 1

    except (KeyError, ValueError, OSError) as exc:
        if as_json:
            _print_json({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
