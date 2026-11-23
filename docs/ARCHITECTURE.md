# 系统架构说明

`ctx-pager` 把小上下文窗口智能体（4K–32K tokens）的上下文管理抽象为操作系统的三级分页存储体系。模型推理窗口只保留工作集页帧，全量结构化页面和不可变事件账本持久化在本地 SQLite 中。

## 1. 三级分层模型

```text
+-------------------------------------------------------------------------+
| Tier 0: Working Frame (RAM, 严格 <= BudgetProfile.total_tokens)          |
|   [Header + Latest Checkpoint (<=16%)]                                  |
|   [Slot 1] Pinned Invariants  (上限 20%-25%, 防止硬约束挤出召回区)       |
|   [Slot 2] Open Loops         (上限 12%-15%, 防止遗漏未闭环待办)         |
|   [Slot 3] Query Recall       (剩余预算 + Slot1/2 结余, L0/L1/L2 混排)  |
|   [Footer] Page-in Hints      (仅暴露含 detail 的 L0/L1 记录 ID)        |
+-------------------------------------------------------------------------+
                       ^  recall() / page-in()          |  commit() / supersede()
                       |  (按边际效用调页)               v  (Write-Through)
+-------------------------------------------------------------------------+
| Tier 1: Indexed Memory Pages & Link Graph (SQLite WAL + FTS5)           |
|   - memories:      L0/L1/L2 预计算 token 成本、状态机、访问计数          |
|   - memories_fts:  四栏加权 FTS5 虚拟表 (unicode61 tokenchars '_')       |
|   - memory_links:  caused_by / depends_on / refines / supersedes / ...  |
|   - checkpoints:   阶段目标与活跃 open_loop 快照                         |
+-------------------------------------------------------------------------+
                       ^  replay(turn=T) / doctor()     |  同事务追加
                       |  (任意历史轮次确定性重建)       v
+-------------------------------------------------------------------------+
| Tier 2: Append-Only Hash-Chained Event Ledger (`events`)                |
|   seq -> event_id -> turn -> event_type -> payload_json                 |
|   content_hash = SHA-256(prev_hash | turn | event_type | canonical_json)|
+-------------------------------------------------------------------------+
```

## 2. 存储层与 FTS5 事务同步 (`storage.py`)

### 2.1 表结构职责
- `memories`: 主记录表，预存储 `token_l0`、`token_l1`、`token_l2`，保证打包阶段无需重复渲染和估算全库文本。
- `memories_fts`: 使用 `tokenize="unicode61 tokenchars '_'"` 配置。列权重为 `title_tokens=4.0`, `entity_tokens=3.5`, `summary_tokens=2.0`, `detail_tokens=1.0`。放弃外部内容触发器（external content triggers），改为在写入与更新事务内部显式调用 `_sync_fts()` 删除旧索引并插入分词结果，避免批量重建或状态变更时的隐式不同步。
- `memory_links`: 有向带权关系表，主键 `(src_id, dst_id, relation)`，支持 `caused_by`、`depends_on`、`relates_to`、`refines`、`contradicts`、`supersedes`。
- `events`: 不可变账本。每次状态变更都在同一 SQLite 事务内计算前向 SHA-256 哈希链：
  `content_hash = sha256(f"{prev_hash}|{turn}|{event_type}|{canonical_json}")`。
  `doctor()` 遍历全表校验 `prev_hash` 连续性与 `content_hash` 一致性，可直接检测磁盘静默损坏或人工篡改。

### 2.2 状态机与确定性时间回放
记录支持四种状态：`active`、`superseded`、`resolved`、`archived`。
`replay_events_to_storage(target_turn)` 在内存中创建空库，按 `seq` 顺序重放所有 `turn <= target_turn` 的事件，可精确还原任意历史轮次的记忆状态与可检索快照，用于离线定位记忆污染问题。

## 3. 中英双语与代码标识符分词器 (`tokenizer.py`)

常规基于空格或纯英文词干的分词器在处理中文技术对话和代码报错时会失效。`tokenizer.py` 采用三路并行策略：

1. **精确标识符抽取 (`extract_identifiers`)**:
   通过正则抽取文件路径（`src/auth/jwt_verify.py:88`）、全大写错误码（`ERR_JWT_4012`）、`snake_case`、`camelCase`、`kebab-case`、版本号、IP:端口（`5.7`、`9443`）以及反引号包裹词元。
2. **混合索引分词 (`tokenize_text`)**:
   - ASCII 词元保留含下划线的完整标识符，同时按 `_` 与驼峰边界拆分为子词（如 `jwtRefreshToken` 拆出 `jwtrefreshtoken`, `jwt`, `refresh`, `token`）。
   - 连续汉字串切分为重叠二元组（bigrams），并保留过滤高频虚词（`的/了/在/是/...`）后的单字（unigrams），确保单字技术名词（如“锁”、“表”、“库”）与复合词均可在 BM25 中按 IDF 自然区分权重。
3. **FTS5 注入防护 (`build_fts_query`)**:
   所有查询词先清洗非字母数字/下划线/CJK 字符，再统一双引号包裹并用 `OR` 连接，避免查询中出现 `-`、`:`、`(`、`NOT` 等字符导致 `sqlite3.OperationalError`。同时对 SQLite `bm25()` 返回的负值取反再做饱和归一化。
4. **保守双语 Token 估算 (`estimate_tokens`)**:
   面向小窗口防溢出设计：汉字按 `1.15 tok/char`、ASCII 字母数字按 `ceil(len/4)`、换行与标点显式计入。相比乐观估算，宁可多算 5%–10% 也不触发模型端硬截断。

## 4. 六信号混合检索与重排 (`retriever.py`)

`HybridRetriever` 结合 `SparseTfIdfEmbedder`（字词 + 字符 3-gram 稀疏余弦相似度）与结构化图/衰减特征：

1. **冲突门控 (Conflict Gating)**:
   默认仅检索 `status = 'active'` 的记录；`superseded` 与 `archived` 节点被排除在外，同时记录被拦截的过时命中数 `superseded_filtered_count`。仅当传入 `include_superseded=True` 时开放审计检索。
2. **基础六信号评分**:
   - `lexical`: `raw_bm25 / (raw_bm25 + 2.5)`
   - `identifier`: `|Q_ids ∩ R_ids| / |Q_ids|`
   - `vector`: 稀疏 TF-IDF + char 3-gram 余弦相似度
   - `recency`: 受 Ebbinghaus / ACT-R 启发的间隔稳定度衰减：
     `S(m) = 1.0 + 0.65 * ln(1 + access_count)`，`recency = exp(-lambda[kind] * delta_turn / S(m))`。其中 `constraint` 的衰减系数 `lambda = 0.002`（几乎不遗忘），`episode` 为 `0.055`（快速衰减并等待压缩）。
   - `importance`: 类型先验或显式权重 `[0, 1]`。
   - 若某记录与当前查询的语义信号（`lexical + identifier + vector`）之和接近 0，其基础总分乘以惩罚系数 `0.25`，避免无关的高重要性记忆仅凭先验分霸占召回位。
3. **1-Hop 图扩散激活 (`graph_boost`)**:
   选取基础分最高的 `seed_limit=5` 个有效种子节点，沿 `memory_links` 双向扩散 `seed_score * edge.weight * graph_boost`（默认跳过 `SUPERSEDES` 边以免反向激活过时结论）。由此可通过症状（如 `502 Bad Gateway`）召回无共同关键词的根因节点（如 `max_connections=150`）。
4. **预分词 MMR 多样性重排**:
   在候选池上一次性预计算标题+摘要 token 集合，再执行贪心 Maximal Marginal Relevance（默认 `lambda=0.78`）抑制近重复日志刷屏。

## 5. 多分辨率边际效用打包 (`packer.py`)

`ContextPacker` 解决的核心问题是：在严格预算 $B$ 内最大化有效信息密度。

1. **槽位隔离与余量回流**:
   首先扣减格式开销与可选 `Checkpoint`（不超过预算 16%）；接着依次填充 `Slot 1 (Pinned, <=25%)` 与 `Slot 2 (Open Loops, <=15%)`，两者未用完的预算全额回流至 `Slot 3 (Recall)`。
2. **两阶段 L0/L1/L2 贪心升级**:
   - 定义分辨率效用：`U(L0) = 0.35`、`U(L1) = 0.82`、`U(L2) = 1.00`。
   - **准入阶段**：排名前 3 且相关性超过阈值的记录直接以 `L1` 准入；其余候选只要放得下均以 `L0` 指针形式准入，尽量保留全局视野。
   - **升级循环**：重复挑选单位 token 边际收益最高的可行升级（`L0 -> L1` 或 `L1 -> L2`）：
     `gain_per_token = (score * delta_U) / delta_tokens`。
     若记录命中了查询中的精确标识符（`breakdown.identifier > 0`）且包含独立 `detail`，其 `L1 -> L2` 边际收益获得 `1.4x` 加成，优先把报错堆栈和配置片段直接展开到 L2。
3. **强预算不变量校验**:
   渲染输出（支持 `markdown` / `xml` / `json`）后立即用 `estimate_tokens(rendered)` 校验总长。若格式化标签导致微幅超限，从低分尾部依次降级 `L2 -> L1 -> L0` 或裁剪末尾项，确保 `frame.used_tokens <= budget_tokens` 绝对成立。
4. **按需换页 (`page_in`)**:
   所有以 `L0` 或 `L1` 输出且含有非空 `detail` 的记录 ID 汇总在 `page_in_hints` 尾部提示中。智能体仅在确实需要原文或堆栈时调用 `page-in <ID>` 读取 `L2` 与 1-hop 邻接记录，同时递增访问次数 `access_count`，加固该记忆后续的间隔稳定度。

## 6. 确定性摄取与无损压缩 (`compactor.py`)

- **零 LLM 摄取 (`TurnIngestor`)**: 通过中英文规则子句切分与线索词匹配提取硬约束、决策、待办和含标识符的事实；在同 `scope + kind` 内执行近重复合并（相似度 >= 0.78）及显式修订冲突检测（命中修订词且实体/词面重叠时自动调用 `supersede`）。
- **无损情景压缩 (`MemoryCompactor`)**: 仅针对超出 `older_than_turns` 且低频访问（`access_count <= 1`）的 `episode` 记录按 `scope` 分组归档为摘要节点，汇总原 `source_events` 并保留 `REFINES` 关联边；`constraint`、`pinned` 与 `open_loop` 永久豁免压缩。
