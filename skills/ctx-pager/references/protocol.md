# ctx-pager Memory Protocol Specification

## 1. Memory Taxonomy

Choose the narrowest accurate `kind` when committing a record. Retrieval decay rates (`lambda`)
and default importance priors depend directly on `kind`:

| Kind | Default Importance | Decay `lambda` | Use For | Compactable? |
| :--- | :--- | :--- | :--- | :--- |
| `constraint` | 0.95 | 0.002 | Hard invariants, forbidden actions, compatibility boundaries | Never |
| `open_loop` | 0.85 | 0.005 | Pending fixes, unverified TODOs, blocking follow-ups | Never |
| `decision` | 0.85 | 0.010 | Chosen library/port/schema/architecture and why | Never |
| `procedure` | 0.75 | 0.015 | Verified runbooks, build/deploy commands, troubleshooting steps | Never |
| `fact` | 0.70 | 0.020 | Observed metrics, schema facts, error codes, environment paths | Never |
| `episode` | 0.50 | 0.055 | Intermediate debugging attempts, transient status updates | Yes (when aged) |

## 2. Three-Resolution Authoring Guidelines (L0 / L1 / L2)

Every record is automatically pre-measured at three resolutions:

- **L0 (Title stub, target 10-20 tokens)**:
  Write `title` as a self-contained semantic handle. Include the module or key noun so an L0
  pointer tells the agent *what* can be paged in without guessing.
  - Good: `MySQL 5.7 utf8mb4 index length constraint`
  - Weak: `Database bug note`

- **L1 (Summary + entities, target 30-70 tokens)**:
  Put the actionable rule, decision outcome, or root conclusion in `summary`, and pass exact
  file paths, error codes, symbols, or ports in `--entities` (and text). Most tasks are solved
  at L1 without ever paying L2 token cost.

- **L2 (Verbatim detail, 80-400 tokens)**:
  Store exact stack traces, SQL migration snippets, command flags, or benchmark outputs in
  `detail`. Do not duplicate `summary` verbatim inside `detail` unless extra precision is needed.

## 3. Belief Revision (`supersede`) Protocol

Never call `commit` with a conflicting decision while leaving the old record `active`.
Instead, call `supersede <OLD_ID>` (or `commit --supersedes <OLD_ID>`):

1. Marks `<OLD_ID>` with `status = "superseded"`, `pinned = False`, and `superseded_by_id = <NEW_ID>`.
2. Inserts `<NEW_ID>` with `supersedes_id = <OLD_ID>` and records a `SUPERSEDES` graph edge.
3. Appends a single atomic `memory.superseded` event to the hash-chained ledger.

With default conflict gating enabled, `recall` excludes `superseded` records unless
`--include-superseded` is passed for audit/debugging.

## 4. Typed Graph Links

Link related records so 1-hop spreading activation lifts root causes or fixes when only a
symptom appears in the user query:

- `caused_by`: `<SYMPTOM_OR_FIX>` -> `<ROOT_CAUSE>` (most useful for debugging recall)
- `depends_on`: `<TASK_OR_DECISION>` -> `<PREREQUISITE>`
- `refines`: `<DETAIL_RECORD>` -> `<PARENT_RECORD>`
- `relates_to`: symmetric contextual association
- `contradicts`: explicit conflict marker when two facts need review

## 5. Machine-Readable Output (`--json`)

Every CLI command accepts `--json`. For `recall --json`, key top-level fields are:

```json
{
  "query": "ERR_JWT_4012",
  "turn": 35,
  "budget_tokens": 1200,
  "used_tokens": 948,
  "slot_tokens": {"pinned": 210, "open_loop": 118, "recall": 572, "overhead": 48},
  "page_in_hints": ["M-0012", "M-0019"],
  "dropped_count": 2,
  "superseded_filtered_count": 1,
  "items": [...],
  "rendered": "# Working Memory Frame ..."
}
```
