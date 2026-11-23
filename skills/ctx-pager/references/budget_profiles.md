# Budget Profiles & Context Window Allocation

`ctx-pager` budgets ONLY the injected working-memory frame, leaving room in the model's total
context window for system instructions, the active user turn, a short tail of recent turns, and
the completion reserve.

## Preset Profiles

| Profile | Model Window | Memory Frame Budget | Pinned Cap | Open-Loop Cap | Recall Cap (+Spillover) | Max L0 Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tiny_4k` | 4,096 tok | 1,024 tok (25%) | 25% (~245 tok) | 15% (~147 tok) | >= 60% | 16 |
| `small_8k` | 8,192 tok | 2,048 tok (25%) | 25% (~501 tok) | 15% (~300 tok) | >= 60% | 24 |
| `medium_16k` | 16,384 tok | 4,096 tok (25%) | 22% (~890 tok) | 15% (~607 tok) | >= 63% | 32 |
| `large_32k` | 32,768 tok | 8,192 tok (25%) | 20% (~1,629 tok) | 12% (~977 tok) | >= 68% | 48 |

## Full 4K / 8K Prompt Layout Example

For an 8,192-token local model (e.g., Qwen-7B / Llama-8B):
- System prompt + tool schemas + `ctx-pager` `SKILL.md`: ~1,300 tok
- `ctx-pager` working-memory frame (`small_8k`): <= 2,048 tok
- Recent verbatim turns (last 3-4 turns): ~2,800 tok
- Current user query + model output reserve: ~2,044 tok
- Total: 8,192 tok (zero silent truncation)

## Tuning Tips

1. **Custom strict budgets**: Pass `--budget <N>` to `recall` (tested down to 256 tokens).
2. **Spillover reuse**: Unused tokens from `pinned` or `open_loop` slots automatically roll into
   the `recall` slot and fund `L0 -> L1 -> L2` upgrades.
3. **Output format**: `markdown` has the lowest framing overhead (~42 tokens). Use `xml` when
   prompts rely on XML tag delimiters, or `json` for programmatic pipelines.
