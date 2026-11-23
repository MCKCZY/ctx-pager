# Failure Modes & Diagnostics

## 1. Pin Bloat (Slot 1 Starvation)
- **Symptom**: Too many records are committed with `--pin`, filling the 25% pinned cap and dropping
  late constraints.
- **Mitigation**: Only pin `constraint` invariants that apply across *every* turn (typically 2-5
  entries under 250 tokens total). Unpin module-local facts with `pin <ID> --unpin` and rely on
  hybrid recall when working inside that module.

## 2. Stale Decision Echo
- **Symptom**: Both the old port/library (`8080`, `Flask`) and the new value (`9443`, `FastAPI`)
  appear in recalled context, causing the model to revert to old behavior.
- **Root Cause**: The agent committed a new record instead of running `supersede <OLD_ID>`.
- **Fix**: Locate both records with `inspect`, then run `supersede` (or mark the obsolete one via
  Python API / re-commit with `--supersedes <OLD_ID>`).

## 3. Exact Identifier or Chinese Query Misses
- **Symptom**: Querying `ERR_JWT_4012` or `连接池超时` fails to rank the target record first.
- **Diagnosis**: Verify `entities` and `title` include the exact error code or path, and ensure the
  query is not overly generic. Check `recall --json` to inspect `ScoreBreakdown` (`lexical`,
  `identifier`, `vector`, `recency`, `importance`, `graph`).

## 4. Over-Growth of Episodic Noise
- **Symptom**: Dozens of turn-by-turn status episodes accumulate over 50+ turns.
- **Fix**: Run `compact --older-than 15 --min-group 3 --dry-run` to preview consolidation, then
  run without `--dry-run`. Active `constraint`, `decision`, `open_loop`, and `pinned` entries are
  exempt from compaction.

## 5. Database or Event Chain Corruption
- **Symptom**: `doctor` exits with code 1 citing FTS sync mismatch or hash-chain tamper errors.
- **Recovery**: Export the verified prefix of the event ledger (`export --format jsonl`) or use
  `replay --turn <N>` to reconstruct clean historical state up to the last valid turn.
