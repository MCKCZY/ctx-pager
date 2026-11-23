# 基准评测报告

本文档记录 `ctx-pager` 与常见基线方案在多轮长程任务上的定量对比结果与消融分析。所有数据均由仓库内置评测套件实时运行测得，无外部 API 依赖。

## 1. 评测方法与数据集

### 1.1 测试场景 (`benchmarks/datasets.py`)
评测集包含 3 个中英双语混合的真实工程长程场景，共计 **235 轮对话** 与 **27 个探测问题 (Probes)**：

1. `backend_auth_mysql` (80 轮，9 探针): MySQL 5.7 索引限制、日志脱敏约束、端口与 JWT 签名从 `8080/HS256` 切换至 `9443/RS256`、限流迁移至 `ratelimit_lua_v2`、以及 `502 Bad Gateway` 与连接池根因关联。
2. `gpu_llm_finetune` (75 轮，9 探针): 24GB 显存上限约束、评测集防泄露约束、优化器从 `fp32_adamw` 切换为 `paged_adamw_8bit`、基座模型更新、`ERR_CUDA_OOM_STEP300` 代码行号定位，以及 `perplexity=98.4` 与 `pad_token_id=151643` 因果排查。
3. `release_security_audit` (80 轮，9 探针): 非 root UID `10001` 约束、RPC/MQ 切换至 `gRPC_PROTOBUF + NATS_JetStream`、审计保留期升级至 `180_days_immutable_s3`、`CVE-2026-1182` 修复参数与 `webhook_timeout_alert` 的 Ingress 根因追踪。

### 1.2 探针类别与评价指标
- `buried_constraint`: 在第 1–4 轮确立、中间隔绝 70 轮工程噪声后才被查询的硬约束（对应指标 **Constraint Survival**）。
- `revised_decision`: 中途被显式推翻并替换的决策，同时校验新方案是否命中与旧方案是否漏出（对应指标 **Stale Intrusion Rate**，越低越好）。
- `exact_artifact`: 藏在 `detail` 字段中的精确错误码、文件路径:行号、命令参数（对应指标 **Exact ID Retention**）。
- `multi_hop_context`: 查询仅提及症状，而根因依赖图边 (`caused_by`) 才能跨术语召回（对应指标 **Multi-Hop Recall**）。
- **Direct Recall vs. Effective Recall (incl. L0 Page-In)**: `Direct Recall` 要求所有答案子串直接出现在首轮装配出的 `ContextFrame` 文本中；`Effective Recall` 额外统计首轮以 `L0/L1` 暴露且可通过一次 `page-in` 获取 `L2` 详情的命中率。

### 1.3 对比基线 (`benchmarks/baselines.py`)
1. `sliding_window`: 按最新轮次倒序填充直至达到 Token 上限。
2. `rolling_summary`: 保留约 45% 最新原始轮次 + 55% 早期轮次首句滚动摘要。
3. `flat_bm25`: 使用相同记忆库运行纯 SQLite FTS5 BM25 检索，全量 L2 填充，关闭冲突门控、图扩展、MMR、槽位配额与多分辨率升级。
4. `ctx_pager_ablated`: 保留框架槽位，但关闭 `conflict_gating`、`graph_expansion`、`mmr`、`recency_decay` 和 `multi_resolution`（退化为平面 L2 打包）。
5. `ctx_pager_full`: 完整启用的三层分页记忆引擎。

## 2. 复现命令

```bash
PYTHONPATH=src python3 -m benchmarks.run_benchmark
```

## 3. 实测结果

### Token Budget = 1024 (`tiny_4k` 级别)

| Strategy | Direct Recall | Effective Recall (incl. L0 Page-In) | Constraint Survival | Exact ID Retention | Multi-Hop Recall | Stale Intrusion (Lower Better) | Avg Tokens | P95 Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sliding_window` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1002 | 0.53 |
| `rolling_summary` | 11.1% | 11.1% | 50.0% | 0.0% | 0.0% | 50.0% | 958 | 0.89 |
| `flat_bm25` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 761 | 4.10 |
| `ctx_pager_ablated` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 938 | 12.16 |
| `ctx_pager_full` | 92.6% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 942 | 13.55 |

### Token Budget = 2048 (`small_8k` 级别)

| Strategy | Direct Recall | Effective Recall (incl. L0 Page-In) | Constraint Survival | Exact ID Retention | Multi-Hop Recall | Stale Intrusion (Lower Better) | Avg Tokens | P95 Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sliding_window` | 14.8% | 14.8% | 0.0% | 25.0% | 0.0% | 16.7% | 2036 | 1.09 |
| `rolling_summary` | 22.2% | 22.2% | 50.0% | 0.0% | 0.0% | 66.7% | 2014 | 1.61 |
| `flat_bm25` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 1027 | 4.80 |
| `ctx_pager_ablated` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 1355 | 11.62 |
| `ctx_pager_full` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 1323 | 12.59 |

### Token Budget = 4096 (`medium_16k` 级别)

| Strategy | Direct Recall | Effective Recall (incl. L0 Page-In) | Constraint Survival | Exact ID Retention | Multi-Hop Recall | Stale Intrusion (Lower Better) | Avg Tokens | P95 Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sliding_window` | 51.9% | 51.9% | 0.0% | 58.3% | 33.3% | 100.0% | 4067 | 2.58 |
| `rolling_summary` | 37.0% | 37.0% | 50.0% | 16.7% | 0.0% | 83.3% | 3327 | 3.06 |
| `flat_bm25` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 1259 | 5.16 |
| `ctx_pager_ablated` | 96.3% | 96.3% | 100.0% | 100.0% | 66.7% | 100.0% | 1377 | 11.28 |
| `ctx_pager_full` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 1332 | 13.31 |

## 4. 结果拆解与消融结论

1. **滑动窗口与滚动摘要在长会话中必然崩塌**:
   当预算只有 1024 tokens 时，`sliding_window` 仅能装下最后约 12–15 轮状态日志，早期约束与历史决策召回率为 `0.0%`；即便预算放宽到 4096 tokens，早期约束存活率依然为 `0.0%`，且会把未淘汰的旧决策一起带入窗口（`100.0%` Stale Intrusion）。`rolling_summary` 虽能保住少量首句信息（`50.0%` Constraint Survival），但丢失了全部精确代码行号与多跳线索。
2. **冲突门控是消融中降噪收益最大的组件**:
   `flat_bm25` 与 `ctx_pager_ablated` 在直接关键词探针上都有 `96.3%` 的召回，但在涉及方案变更的 `revised_decision` 探针中均出现 **`100.0%` 的过时决策泄漏**（旧端口号、旧优化器、旧 RPC 协议与新方案同时出现在提示词中）。`ctx_pager_full` 通过显式 `supersede` 与冲突门控把这一污染率降为 **`0.0%`**。
3. **1-Hop 图扩散补齐跨术语因果链**:
   在 `multi_hop_context` 探针中，关闭图扩散的 `flat_bm25` 与 `ctx_pager_ablated` 均停留在 `66.7%`，而启用 `caused_by` 图扩散的 `ctx_pager_full` 稳定达到 **`100.0%`**。
4. **L0/L1/L2 分层打包在 1024 极紧预算下的换页取舍**:
   在 `1024` tokens 下，`ctx_pager_full` 优先保留了 Checkpoint、Pinned 约束与 Open Loops，并在召回区把部分低紧迫度或长详情条目先以 `L0/L1` 放进窗口，因此首轮 `Direct Recall` 为 `92.6%`（略低于 `flat_bm25` 的 `96.3%`），但通过尾部 `Page-in available` 暴露出的指针在 1 次 `page-in` 后即达到 **`100.0%` Effective Recall**。预算升至 `2048` 后，`Direct Recall` 直接升至 **`100.0%`** 且平均仅消耗 `1323` tokens。
5. **纯 Python 本地延迟可控**:
   经过 MMR 预分词优化后，`ctx_pager_full` 全流程（检索 + 图扩散 + MMR + 分辨率贪心打包 + 格式化验证 + `page_in` 查表）的 P95 延迟稳定在 **`13.3–13.9ms`**。

## 5. 评测边界声明

- 本基准验证的是**给定结构化写入轨迹下的记忆存储、冲突隔离、混合检索与预算打包能力**，不直接等同于真实大语言模型在闭环执行时的自主工具调用成功率（即若智能体自身忘记执行 `commit` 或 `supersede`，则无法触发结构化优势）。
- 评测集中的操作序列由人工编写的确定性事件构成，真实生产中的自然语言波动需结合具体模型提示词遵循能力评估。
