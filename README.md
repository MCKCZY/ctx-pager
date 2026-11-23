# ctx-pager

面向受限上下文窗口（4K–32K tokens）AI 智能体的**三级分页上下文记忆 Skill 与零依赖检索引擎**。

当任务跨越数十轮对话、中间夹杂大量工具日志与状态输出时，小上下文模型普遍会遭遇三类退化：**早期硬约束被滑动窗口挤出**、**方案中途推翻后新旧决策同时残留导致模型回退**、以及**整块 RAG 召回吃光预算却仍遗漏关键报错行号**。`ctx-pager` 将智能体记忆拆分为“工作集页帧 + SQLite FTS5/关联图页表 + 不可变哈希链事件账本”，在纯 Python 标准库内为受限智能体提供确定性的跨轮记忆与可控预算管理。

## 系统架构

```text
[Agent Skill / CLI / Stdio MCP Server]
                  |
                  v
+-------------------------------------------------------------------+
| Tier 0: Working Frame (严格保证 used_tokens <= budget_tokens)     |
|   Checkpoint (<=16%) + Pinned Invariants (<=25%)                  |
|   + Open Loops (<=15%) + Multi-Res Recall (L0 / L1 / L2) + Hints  |
+-------------------------------------------------------------------+
         ^ recall() / page-in()                | commit() / supersede()
         | (边际效用贪心调页)                   v (写穿透落盘)
+-------------------------------------------------------------------+
| Tier 1: Indexed Pages & Causal Graph (SQLite WAL + FTS5)          |
|   memories (L0/L1/L2 token cost) | memories_fts (四栏加权检索)     |
|   memory_links (caused_by/depends_on/supersedes) | checkpoints    |
+-------------------------------------------------------------------+
         ^ replay(turn=T) / doctor()           | 同事务追加
         | (时间回放与防篡改校验)               v
+-------------------------------------------------------------------+
| Tier 2: Hash-Chained Event Ledger (`events`, SHA-256)             |
+-------------------------------------------------------------------+
```

## 核心机制

- **L0 / L1 / L2 多分辨率预算打包器 (`packer.py`)**：
  每条记忆按“标题指针 (L0) / 摘要+实体 (L1) / 完整细节 (L2)”预估成本。打包时优先保护 Pinned 硬约束与 Open Loops 未闭环待办（未使用配额自动回流给召回区），再用单位 token 边际收益 `score * delta_U / delta_tokens` 贪心升级高价值条目（命中精确错误码或路径时享有 `1.4x` 详情升级权重），并在渲染后强校验 `used_tokens <= budget_tokens`。
- **双语混合检索与 1-Hop 图扩散 (`tokenizer.py`, `retriever.py`, `vectorizer.py`)**：
  融合 FTS5 加权 BM25（`title=4.0`, `entities=3.5`, `summary=2.0`, `detail=1.0`）、精确技术标识符抽取（文件路径、错误码、端口号、驼峰/蛇形命名）、CJK 重叠二元组+单字、稀疏 TF-IDF + char 3-gram 向量、Ebbinghaus/ACT-R 访问稳定度衰减、1-hop 因果图扩散激活（`caused_by` / `depends_on`）以及预分词 MMR 多样性重排。
- **显式信念修订与冲突门控 (`supersede`)**：
  需求或架构变化时通过 `supersede` 原子替换旧记忆（置为 `superseded` 并建立 `SUPERSEDES` 边），检索默认过滤失效结论，防止过时端口、旧库版本或废弃方案污染提示词。
- **哈希链事件溯源与无损压缩 (`storage.py`, `compactor.py`)**：
  所有状态变更写入 SHA-256 前向哈希链账本 `events`。支持 `replay --turn N` 任意时间点状态重建、`doctor` 校验 FTS5 同步及篡改检测，以及仅对低频老旧 `episode` 执行的无损聚合压缩（永久豁免硬约束与待办）。
- **标准 Skill + 零依赖 CLI + Stdlib MCP Server**:
  `skills/ctx-pager/SKILL.md` 自身严格控制在 82 行，遵循渐进式披露原则；提供零安装可执行包装脚本与纯标准库 JSON-RPC 2.0 stdio MCP 服务。

## 基准评测摘要

在 3 个中英双语长程工程场景（共 235 轮对话、27 个探针）上的实测对比（完整数据与消融分析见 [docs/BENCHMARKS.md](./docs/BENCHMARKS.md)）：

| Budget | Strategy | Direct Recall | Effective Recall (incl. L0 Page-In) | Constraint Survival | Multi-Hop Recall | Stale Intrusion (Lower Better) | P95 (ms) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | `sliding_window` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.53 |
| 1024 | `rolling_summary` | 11.1% | 11.1% | 50.0% | 0.0% | 50.0% | 0.89 |
| 1024 | `flat_bm25` | 96.3% | 96.3% | 100.0% | 66.7% | 100.0% | 4.10 |
| 1024 | `ctx_pager_ablated` | 96.3% | 96.3% | 100.0% | 66.7% | 100.0% | 12.16 |
| 1024 | **`ctx_pager_full`** | **92.6%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **13.55** |
| 2048 | `sliding_window` | 14.8% | 14.8% | 0.0% | 0.0% | 16.7% | 1.09 |
| 2048 | `flat_bm25` | 96.3% | 96.3% | 100.0% | 66.7% | 100.0% | 4.80 |
| 2048 | **`ctx_pager_full`** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **12.59** |

## 快速开始

### 1. 环境要求与 35 轮会话演示
仅需 Python 3.10+（需系统 SQLite 支持 FTS5，已验证 Python 3.14 + SQLite 3.46.1），零第三方 pip 依赖：

```bash
git clone <your-repo-url> ctx-pager
cd ctx-pager
PYTHONPATH=src python3 examples/demo_small_context_session.py
```

### 2. CLI 日常工作流（无需 pip install）

```bash
# 初始化并检查 FTS5 与哈希链状态
python3 skills/ctx-pager/scripts/ctx_pager.py init

# 写入硬约束（固定在 Slot 1）与初期决策
python3 skills/ctx-pager/scripts/ctx_pager.py commit \
  --kind constraint --scope db --pin \
  --title "MySQL 5.7 utf8mb4 index <= 191" \
  --summary "生产库固定 MySQL 5.7，单列索引前缀不超过 191 字符"

python3 skills/ctx-pager/scripts/ctx_pager.py commit \
  --kind decision --scope auth \
  --title "Auth port 8080" --summary "初期鉴权监听 8080 端口"

# 方案变更时显式推翻旧记忆，避免 stale echo
python3 skills/ctx-pager/scripts/ctx_pager.py supersede M-0002 \
  --title "Auth TLS port 9443 RS256" \
  --summary "鉴权切至 9443 端口并使用 RS256" \
  --detail "公钥路径 keys/jwt_pub.pem，关闭 8080"

# 在严格 1024 tokens 预算内装配工作页帧
python3 skills/ctx-pager/scripts/ctx_pager.py recall \
  --profile tiny_4k --query "当前鉴权端口与数据库索引规范"

# 对提示中以 L0/L1 出现的条目按需展开 L2 详情与邻接图谱
python3 skills/ctx-pager/scripts/ctx_pager.py page-in M-0003

# 回放历史第 2 轮的状态并执行健康自检
python3 skills/ctx-pager/scripts/ctx_pager.py replay --turn 2 --query "鉴权端口"
python3 skills/ctx-pager/scripts/ctx_pager.py doctor
```

### 3. 安装为 Agent Skill 或启动 MCP Server

```bash
# 安装 Skill 到工作区 (.agents/skills/ctx-pager) 或 Claude/Codex 目录
python3 scripts/install_skill.py --target antigravity
# 或: python3 scripts/install_skill.py --target claude --symlink

# 启动纯标准库 stdio MCP Server（示例配置见 examples/mcp_config.example.json）
PYTHONPATH=src python3 -m ctx_pager.mcp_server --db .ctx_pager/memory.db
```

### 4. Python API 调用

```python
from ctx_pager import ContextPager, MemoryKind

with ContextPager(".ctx_pager/memory.db") as pager:
    pager.commit(
        kind=MemoryKind.CONSTRAINT,
        scope="db",
        title="MySQL 5.7 index cap",
        summary="utf8mb4 single-column index <= 191 chars",
        pinned=True,
    )
    frame = pager.recall("MySQL index", profile="tiny_4k")
    print(f"{frame.used_tokens}/{frame.budget_tokens} tokens\n{frame.rendered}")
```

## 目录结构

```text
ctx-pager/
├── src/ctx_pager/              # 核心引擎包 (models/tokenizer/storage/vectorizer/retriever/packer/compactor/cli/mcp_server)
├── skills/ctx-pager/           # Agent Skill 标准包 (82 行精简 SKILL.md + references/ + resources/ + scripts/)
├── benchmarks/                 # 3 场景 235 轮基准数据集、5 基线对比与评测脚本
├── tests/                      # 17 项 unittest 单元测试与集成测试
├── examples/                   # 35 轮受限上下文完整演示与 MCP 配置示例
├── scripts/                    # Skill 一键安装器
└── docs/                       # ARCHITECTURE.md / DESIGN_TRADEOFFS.md / BENCHMARKS.md
```

## 测试与工程验证

```bash
make check   # 编译检查 + 全部 17 项单测 + 35 轮端到端演示
make bench   # 重新生成完整定量评测报告
```

## 已知局限

- 默认零依赖模式依靠稀疏 TF-IDF + char 3-gram + FTS5 + 1-hop 图扩散检索；在无任何共同词汇或实体的纯粹同义跨语言查询上，需通过实现 `ctx_pager.vectorizer.Embedder` 接入外部稠密向量后端。
- 面向单智能体/本地低并发 SQLite WAL 场景设计，不适用于高并发分布式多写环境。
- 详细取舍说明见 [docs/DESIGN_TRADEOFFS.md](./docs/DESIGN_TRADEOFFS.md)，系统内部实现细节见 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)。
