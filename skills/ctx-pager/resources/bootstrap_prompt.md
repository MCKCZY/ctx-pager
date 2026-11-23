# Minimal Bootstrap Prompt (Local / Raw API Agents, ~190 tokens)

Use this prompt snippet when running small local models (Ollama, vLLM, llama.cpp) without
native Skill loading:

```text
You have an external paged memory tool `ctx-pager` (CLI or MCP). Never rely on long chat history
to remember rules or changing plans.
1. Start: run `ctx-pager recall --profile small_8k --query "<task>"` and read any checkpoint.
2. Write-through:
   - Rule/limit -> `ctx-pager commit --kind constraint --pin ...`
   - Choice/design -> `ctx-pager commit --kind decision ...`
   - Pending item -> `ctx-pager commit --kind open_loop ...`
   - Plan changed -> `ctx-pager supersede <OLD_ID> ...` (never leave conflicting decisions active)
   - Task done -> `ctx-pager resolve <LOOP_ID>`
3. Paging: if `Page-in available: <ID>` appears and you need exact logs/code, run `ctx-pager page-in <ID>`.
4. Handoff: run `ctx-pager checkpoint --goal "..." --state "..."`.
```
