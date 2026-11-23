---
name: ctx-pager
description: >-
  Paged working-memory protocol and CLI/MCP engine for agents operating under small
  or crowded context windows (4K-32K tokens). Use when a task spans many turns,
  requires preserving early constraints and revised decisions across context truncation,
  or needs resuming from prior checkpoints without bloating the prompt.
---

# ctx-pager: Paged Context Memory Protocol

Keep only what the current turn needs inside the context window. Persist durable constraints,
decisions, open loops, and verbatim details to SQLite via [ctx_pager.py](./scripts/ctx_pager.py),
then pull bounded `L0/L1/L2` working frames back on demand.

## Core Loop

### 1. Session Start or Resume
Initialize the store if missing, then load a budgeted working frame before acting:

```bash
python3 ./scripts/ctx_pager.py init
python3 ./scripts/ctx_pager.py recall --profile small_8k --query "<current task or file>"
```

If a checkpoint exists, continue from its `goal`, `state_summary`, and active `open_loops`.

### 2. Write-Through Discipline (During Work)
Never keep hard constraints or architectural pivots solely in raw chat history. Commit immediately:

- **Hard rules / boundaries** (pin selectively):
  `commit --kind constraint --scope <module> --pin --title "..." --summary "..."`
- **Decisions**:
  `commit --kind decision --scope <module> --title "..." --summary "..." --detail "<rationale>"`
- **Unfinished tasks**:
  `commit --kind open_loop --scope <module> --title "..." --summary "..."`
- **Plan or parameter changes** (never leave conflicting decisions active):
  `supersede <OLD_ID> --title "..." --summary "..." --detail "<why changed>"`
- **Causal chains** (connect symptoms, root causes, and fixes):
  `link <FIX_ID> <SYMPTOM_ID> --relation caused_by --weight 0.9`
- **Finished open loops**:
  `resolve <LOOP_ID>`

### 3. Paged Recall & On-Demand Page-In
Before starting a new subtask or when chat history grows noisy:

```bash
python3 ./scripts/ctx_pager.py recall --profile tiny_4k --query "<error code, path, or topic>"
```

The returned frame packs items at three resolutions:
- `L0`: single-line pointer (`[M-0007|fact|db] Connection pool exhaustion fix`)
- `L1`: summary + exact technical entities
- `L2`: verbatim detail (stack trace, SQL, config diff)

If `Page-in available: M-0007, ...` lists an ID whose verbatim detail is actually needed right now,
expand ONLY that ID instead of loading unrelated records:

```bash
python3 ./scripts/ctx_pager.py page-in M-0007
```

### 4. Milestone Checkpoint & Compaction
At subtask boundaries or handoff:

```bash
python3 ./scripts/ctx_pager.py checkpoint --goal "<goal>" --state "<verified state>"
python3 ./scripts/ctx_pager.py compact --older-than 15 --min-group 3
```

## Validation & Integrity
Verify store health, FTS5 synchronization, and the SHA-256 event hash chain:

```bash
python3 ./scripts/ctx_pager.py doctor
```

## References (Load Only When Needed)
- Memory taxonomy, L0/L1/L2 authoring rules, and JSON schema: [protocol.md](./references/protocol.md)
- Window sizing tables (4K/8K/16K/32K) and slot ratios: [budget_profiles.md](./references/budget_profiles.md)
- Troubleshooting pin bloat, stale echoes, and recall misses: [failure_modes.md](./references/failure_modes.md)
- Ultra-compact prompt for raw local models without native Skills: [bootstrap_prompt.md](./resources/bootstrap_prompt.md)
