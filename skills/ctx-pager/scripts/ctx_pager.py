#!/usr/bin/env python3
"""Zero-install executable shim for invoking ctx-pager CLI directly from the skill folder."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    bundled_lib = script_dir / "_lib"
    repo_src = script_dir.parents[2] / "src"

    candidate = bundled_lib if (bundled_lib / "ctx_pager").is_dir() else repo_src
    if candidate.is_dir():
        cand_str = str(candidate)
        script_str = str(script_dir)
        # Remove script_dir and re-prepend candidate at index 0 so scripts/ctx_pager.py
        # never shadows the ctx_pager package even when PYTHONPATH already includes src/.
        cleaned = [p for p in sys.path if p not in (cand_str, script_str, "")]
        sys.path[:] = [cand_str] + cleaned


def main() -> int:
    _bootstrap_src()
    from ctx_pager.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
