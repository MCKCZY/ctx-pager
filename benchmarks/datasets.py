"""Deterministic bilingual multi-turn benchmark scenarios and probes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurnEntry:
    turn: int
    role: str
    text: str


@dataclass(frozen=True)
class MemoryOp:
    turn: int
    op: str  # "commit" | "supersede" | "link" | "resolve" | "checkpoint"
    alias: str = ""
    kind: str = "fact"
    scope: str = "global"
    title: str = ""
    summary: str = ""
    detail: str = ""
    entities: tuple[str, ...] = ()
    pinned: bool = False
    target_alias: str = ""
    src_alias: str = ""
    dst_alias: str = ""
    relation: str = "relates_to"
    weight: float = 1.0
    goal: str = ""
    state_summary: str = ""


@dataclass(frozen=True)
class EvalProbe:
    id: str
    category: str  # "buried_constraint" | "revised_decision" | "exact_artifact" | "multi_hop_context"
    query: str
    query_turn: int
    required_substrings: tuple[str, ...]
    forbidden_stale_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    turns: list[TurnEntry] = field(default_factory=list)
    operations: list[MemoryOp] = field(default_factory=list)
    probes: list[EvalProbe] = field(default_factory=list)


def _generate_noise_turns(
    *,
    start: int,
    end: int,
    skip: set[int],
    topic: str,
    lines_pool: list[str],
) -> list[TurnEntry]:
    out: list[TurnEntry] = []
    for t in range(start, end + 1):
        if t in skip:
            continue
        msg = lines_pool[t % len(lines_pool)]
        role = "user" if t % 2 == 1 else "assistant"
        out.append(
            TurnEntry(
                turn=t,
                role=role,
                text=(
                    f"[T{t} {topic}] Checked worker health metrics, log tail shard #{t % 7}, "
                    f"build pipeline step {(t * 3) % 11}: {msg} "
                    f"Latency sample={18 + (t % 14)}ms, rss={210 + (t % 45)}MB, queue_depth={t % 5}."
                ),
            )
        )
    return out


def _scenario_backend_auth_mysql() -> Scenario:
    turns: list[TurnEntry] = [
        TurnEntry(1, "user", "硬约束：生产数据库固定为 MySQL 5.7，所有表必须使用 utf8mb4，索引键单列不超过 191 字符。"),
        TurnEntry(2, "assistant", "已记录 MySQL 5.7 utf8mb4 索引长度约束（varchar<=191）。"),
        TurnEntry(3, "user", "硬约束：禁止在 HTTP 日志或报错响应中打印 Bearer Token 和 refresh_secret 原文。"),
        TurnEntry(4, "user", "鉴权服务初期决定监听端口 8080，JWT 签名算法先采用 HS256。"),
        TurnEntry(6, "user", "开始排查登录接口偶发 401 问题，错误码 ERR_JWT_4012 出现在 src/auth/jwt_verify.py:88。"),
        TurnEntry(8, "assistant", "在 src/auth/jwt_verify.py:88 定位到 clock skew 只有 0s，但尚未确定是否为唯一根因。"),
        TurnEntry(12, "user", "压力测试出现大量 502 Bad Gateway，Nginx upstream 报错连接重置。"),
        TurnEntry(15, "assistant", "排查后发现 502 Bad Gateway 的根因是 config/db_pool.yaml 中连接池未设 max_connections=150 上限，wait_timeout=60s 导致连接耗尽。"),
        TurnEntry(18, "user", "针对 MySQL 5.7 迁移脚本，创建复合索引 idx_user_email_tenant 时必须加前缀长度限制。"),
        TurnEntry(22, "user", "安全评审变更决定：鉴权服务放弃 8080 和 HS256，统一切换为 HTTPS 端口 9443，算法改为 RS256 非对称验签。"),
        TurnEntry(25, "assistant", "确认更新：监听端口已从 8080 改为 9443，JWT 算法从 HS256 改为 RS256（公钥路径 keys/jwt_pub.pem）。"),
        TurnEntry(28, "user", "限流模块初期使用内存字典限流 mem_dict_limiter，单机有效但多副本不一致。"),
        TurnEntry(34, "user", "修复 502 问题步骤：修改 config/db_pool.yaml 设置 max_connections=150 和 pool_recycle=45，并重启 gunicorn worker。"),
        TurnEntry(40, "user", "变更决定：限流组件不再使用 mem_dict_limiter，改为 Redis Lua 脚本滑动窗口 ratelimit_lua_v2。"),
        TurnEntry(45, "user", "待办：迁移脚本 migrations/004_tokens.sql 需要在 staging 验证回滚时间。"),
        TurnEntry(52, "user", "近重复测试日志：再次确认 RS256 公钥加载路径 keys/jwt_pub.pem 权限为 0600。"),
        TurnEntry(60, "assistant", "阶段检查点：鉴权与限流改造完成，准备进入上线前压测和审计核对。"),
    ]
    used_turns = {t.turn for t in turns}
    noise_pool = [
        "GET /healthz returned 200 OK across 4 replicas with zero 5xx spikes.",
        "CI lint job passed flake8 and mypy on unrelated billing helpers.",
        "Reviewed Prometheus scrape interval 15s and alertmanager silence rules.",
        "Docker layer cache hit 94% on base image python:3.12-slim.",
        "Rotated staging TLS test certs and verified OCSP stapling response.",
        "Cleaned temporary pytest artifacts and archived old container logs.",
    ]
    turns.extend(_generate_noise_turns(start=1, end=80, skip=used_turns, topic="auth-ops", lines_pool=noise_pool))
    turns.sort(key=lambda x: x.turn)

    ops = [
        MemoryOp(
            turn=1,
            op="commit",
            alias="c_mysql",
            kind="constraint",
            scope="db",
            title="MySQL 5.7 utf8mb4 index length <= 191",
            summary="生产库固定 MySQL 5.7，使用 utf8mb4，单列索引不得超过 191 字符",
            detail="InnoDB large prefix in MySQL 5.7 requires varchar(191) or explicit prefix index (191) under utf8mb4.",
            entities=("MySQL 5.7", "utf8mb4", "191"),
            pinned=True,
        ),
        MemoryOp(
            turn=3,
            op="commit",
            alias="c_redact",
            kind="constraint",
            scope="auth",
            title="Redact Bearer Token and refresh_secret in logs",
            summary="严禁在日志或报错响应中输出 Bearer Token 与 refresh_secret 原文",
            detail="Mask Authorization headers and refresh_secret fields before emitting structured JSON logs.",
            entities=("Bearer", "refresh_secret"),
            pinned=True,
        ),
        MemoryOp(
            turn=4,
            op="commit",
            alias="d_auth_v1",
            kind="decision",
            scope="auth",
            title="Initial auth listen port 8080 and HS256",
            summary="鉴权服务监听端口 8080，JWT 签名算法 HS256",
            detail="Temporary single-node dev config on port 8080 with shared HS256 secret.",
            entities=("8080", "HS256"),
        ),
        MemoryOp(
            turn=6,
            op="commit",
            alias="f_jwt_err",
            kind="fact",
            scope="auth",
            title="ERR_JWT_4012 in src/auth/jwt_verify.py:88",
            summary="登录偶发 401 报错 ERR_JWT_4012，位于 src/auth/jwt_verify.py:88",
            detail="Traceback in src/auth/jwt_verify.py:88 (verify_claims): leeway=0s rejects tokens when NTP drift > 500ms; set leeway=15.",
            entities=("ERR_JWT_4012", "src/auth/jwt_verify.py:88"),
        ),
        MemoryOp(
            turn=12,
            op="commit",
            alias="f_502",
            kind="fact",
            scope="gateway",
            title="Load test 502 Bad Gateway on upstream",
            summary="高压下出现 502 Bad Gateway，Nginx 报 upstream connection reset",
            detail="Observed 14% 502 Bad Gateway rate during 400 RPS spike against /api/v1/login.",
            entities=("502 Bad Gateway", "/api/v1/login"),
        ),
        MemoryOp(
            turn=15,
            op="commit",
            alias="f_pool_root",
            kind="fact",
            scope="db",
            title="DB pool exhaustion missing max_connections=150",
            summary="数据库连接池缺少上限导致耗尽，需在 config/db_pool.yaml 设 max_connections=150",
            detail="Root cause: unbounded pool opened >300 idle sockets until wait_timeout=60s dropped connections; config/db_pool.yaml needs max_connections=150 and pool_recycle=45.",
            entities=("config/db_pool.yaml", "max_connections=150", "pool_recycle=45"),
        ),
        MemoryOp(
            turn=15,
            op="link",
            src_alias="f_502",
            dst_alias="f_pool_root",
            relation="caused_by",
            weight=0.95,
        ),
        MemoryOp(
            turn=18,
            op="commit",
            alias="p_mig",
            kind="procedure",
            scope="db",
            title="Index idx_user_email_tenant migration rule",
            summary="创建 idx_user_email_tenant 复合索引时使用 email(191)",
            detail="ALTER TABLE users ADD INDEX idx_user_email_tenant (tenant_id, email(191));",
            entities=("idx_user_email_tenant", "191"),
        ),
        MemoryOp(
            turn=22,
            op="supersede",
            target_alias="d_auth_v1",
            alias="d_auth_v2",
            title="Auth port 9443 with RS256 public key",
            summary="鉴权服务切换至端口 9443，JWT 算法改为 RS256，公钥 keys/jwt_pub.pem",
            detail="Security review mandated TLS listener on 9443 and asymmetric RS256 verification via keys/jwt_pub.pem with 2048-bit key rotation.",
            entities=("9443", "RS256", "keys/jwt_pub.pem"),
        ),
        MemoryOp(
            turn=28,
            op="commit",
            alias="d_rate_v1",
            kind="decision",
            scope="ratelimit",
            title="In-memory mem_dict_limiter prototype",
            summary="限流暂用进程内 mem_dict_limiter 字典计数器",
            detail="Single-process dict token bucket; breaks consistency across gunicorn workers.",
            entities=("mem_dict_limiter",),
        ),
        MemoryOp(
            turn=34,
            op="commit",
            alias="p_502_fix",
            kind="procedure",
            scope="db",
            title="Apply pool_recycle=45 and restart workers",
            summary="在 config/db_pool.yaml 配置 max_connections=150 与 pool_recycle=45 后重启 worker",
            detail="1) Edit config/db_pool.yaml; 2) validate pool_recycle=45; 3) systemctl reload gunicorn-auth.",
            entities=("config/db_pool.yaml", "max_connections=150", "pool_recycle=45"),
        ),
        MemoryOp(
            turn=34,
            op="link",
            src_alias="p_502_fix",
            dst_alias="f_502",
            relation="relates_to",
            weight=0.90,
        ),
        MemoryOp(
            turn=40,
            op="supersede",
            target_alias="d_rate_v1",
            alias="d_rate_v2",
            title="Distributed Redis ratelimit_lua_v2 sliding window",
            summary="限流统一改为 Redis Lua 脚本 ratelimit_lua_v2 滑动窗口",
            detail="Atomic Redis EVALSHA sliding window script ratelimit_lua_v2 enforcing 100 req/min/ip across all gunicorn replicas.",
            entities=("ratelimit_lua_v2", "Redis"),
        ),
        MemoryOp(
            turn=45,
            op="commit",
            alias="ol_mig",
            kind="open_loop",
            scope="db",
            title="Verify migrations/004_tokens.sql rollback timing",
            summary="待在 staging 验证 migrations/004_tokens.sql 回滚耗时",
            detail="Measure lock duration on 2M-row tokens table during down-migration before prod deploy.",
            entities=("migrations/004_tokens.sql",),
        ),
        MemoryOp(
            turn=60,
            op="checkpoint",
            goal="Complete auth service TLS + pool + ratelimit pre-release checks",
            state_summary="Port 9443 RS256 active; ratelimit_lua_v2 deployed; 502 fixed via max_connections=150.",
        ),
    ]

    # Add realistic domain artifacts & near-duplicate noise to saturate flat 1024-token packing.
    extra_s1 = [
        (11, "fact", "auth", "Nginx proxy_read_timeout 30s config", "Nginx upstream proxy_read_timeout set to 30s in deploy/nginx/auth.conf", "proxy_connect_timeout 5s; proxy_read_timeout 30s; proxy_send_timeout 30s; keepalive 64; upstream auth_backend { server 127.0.0.1:9443 max_fails=3 fail_timeout=10s; }", ("deploy/nginx/auth.conf",)),
        (16, "procedure", "db", "Run pt-online-schema-change for users table", "大表变更使用 pt-online-schema-change 控制从库延迟 < 2s", "pt-online-schema-change --alter \"ADD COLUMN mfa_enabled TINYINT(1) DEFAULT 0\" D=prod,t=users --max-lag=2 --chunk-size=2000 --execute", ("pt-online-schema-change",)),
        (20, "fact", "auth", "OAuth2 PKCE S256 verifier requirement", "移动端登录强制开启 PKCE code_challenge_method=S256", "Reject plain code_challenge on /oauth/authorize; verify SHA256(code_verifier) length between 43 and 128 chars.", ("PKCE", "S256")),
        (23, "fact", "ratelimit", "Redis sentinel failover timeout 3000ms", "Redis sentinel down-after-milliseconds=3000ms，failover-timeout=10000ms", "sentinel monitor rate_master 10.20.4.12 6379 2; sentinel down-after-milliseconds rate_master 3000; client socket_timeout=0.4s.", ("sentinel",)),
        (26, "procedure", "auth", "Verify JWT key rotation JWKS cache TTL", "JWKS 公钥缓存 TTL 设为 300s，支持 kid 双密钥平滑轮换", "Publish both active and next kid under /.well-known/jwks.json with Cache-Control: public, max-age=300 for 24h before retiring old key.", ("JWKS",)),
        (29, "fact", "db", "MySQL slow query threshold 200ms audit", "开启 long_query_time=0.2s 记录慢 SQL 至 /var/log/mysql/slow.log", "log_queries_not_using_indexes=1; min_examined_row_limit=500; weekly pt-query-digest report archived to ops bucket.", ("long_query_time",)),
        (31, "fact", "auth", "Session audit log schema fields", "登录审计日志保留 user_id, ip_hash, user_agent, trace_id 字段", "Emit structured JSON audit line to Kafka topic auth_audit_v2; PII fields hashed with HMAC-SHA256 daily salt.", ("auth_audit_v2",)),
        (33, "fact", "auth", "RS256 public key permission check 0600", "确认 RS256 公钥文件 keys/jwt_pub.pem 权限 0600", "Verified chmod 0600 keys/jwt_pub.pem on staging-1 and staging-2; systemd unit runs under svc-auth user.", ("keys/jwt_pub.pem",)),
        (35, "fact", "auth", "Duplicate RS256 key permission check on prod", "复核生产节点 keys/jwt_pub.pem 文件属主与 0600 权限", "Checked prod-auth-01..04 ansible inventory: owner svc-auth, mode 0600, sha256 fingerprint verified.", ("keys/jwt_pub.pem",)),
        (38, "procedure", "gateway", "Drain gunicorn workers gracefully on SIGTERM", "部署发布发送 SIGTERM 并等待 graceful-timeout=25s 完成在途请求", "ExecReload=/bin/kill -s HUP $MAINPID; ExecStop=/bin/kill -s TERM $MAINPID with --graceful-timeout 25 --keep-alive 5.", ("SIGTERM",)),
        (44, "fact", "ratelimit", "Allowlist internal health probe CIDR 10.20.0.0/16", "内部探测网段 10.20.0.0/16 跳过限流计数器", "Lua script checks ngx.var.remote_addr prefix before INCR; prevents Prometheus and kubelet probes from hitting 429.", ("10.20.0.0/16",)),
        (50, "fact", "db", "Backup retention daily snapshot mysqldump check", "每日凌晨 03:00 执行逻辑快照并校验恢复耗时", "mysqldump --single-transaction --routines --triggers prod | zstd -T4 > /backups/prod-$(date +%F).sql.zst", ("mysqldump",)),
    ]
    for et, ek, esc, etit, esum, edet, eent in extra_s1:
        ops.append(
            MemoryOp(
                turn=et,
                op="commit",
                alias=f"ex1_{et}",
                kind=ek,
                scope=esc,
                title=etit,
                summary=esum,
                detail=edet,
                entities=eent,
            )
        )

    # Add realistic episodic status logs to test compaction & noise resistance.
    for ep_t in (10, 14, 19, 24, 30, 36, 42, 48):
        ops.append(
            MemoryOp(
                turn=ep_t,
                op="commit",
                alias=f"ep_{ep_t}",
                kind="episode",
                scope="auth-ops",
                title=f"Routine smoke test iteration T{ep_t}",
                summary=f"Executed auth smoke run at T{ep_t}; synthetic check passed with baseline latency.",
                detail=f"Full stdout from smoke run T{ep_t}: 12 assertions ok, 0 warnings.",
            )
        )

    probes = [
        EvalProbe(
            id="s1_p1",
            category="buried_constraint",
            query="创建新用户表索引时有什么数据库字段和字符集限制？",
            query_turn=75,
            required_substrings=("MySQL 5.7", "utf8mb4", "191"),
        ),
        EvalProbe(
            id="s1_p2",
            category="buried_constraint",
            query="打印认证失败调试日志时需要注意什么安全限制？",
            query_turn=76,
            required_substrings=("Bearer", "refresh_secret"),
        ),
        EvalProbe(
            id="s1_p3",
            category="revised_decision",
            query="当前鉴权服务对外监听端口和 JWT 签名算法是什么？",
            query_turn=77,
            required_substrings=("9443", "RS256"),
            forbidden_stale_substrings=("8080", "HS256"),
        ),
        EvalProbe(
            id="s1_p4",
            category="revised_decision",
            query="目前生产环境采用哪种限流实现方案？",
            query_turn=78,
            required_substrings=("ratelimit_lua_v2",),
            forbidden_stale_substrings=("mem_dict_limiter",),
        ),
        EvalProbe(
            id="s1_p5",
            category="exact_artifact",
            query="定位 ERR_JWT_4012 的精确代码位置和 leeway 修复参数",
            query_turn=78,
            required_substrings=("src/auth/jwt_verify.py:88", "leeway=15"),
        ),
        EvalProbe(
            id="s1_p6",
            category="exact_artifact",
            query="哪个迁移文件还在等待 staging 回滚验证？",
            query_turn=79,
            required_substrings=("migrations/004_tokens.sql",),
        ),
        EvalProbe(
            id="s1_p7",
            category="multi_hop_context",
            query="为什么压测时出现 502 Bad Gateway？根因配置在哪里？",
            query_turn=79,
            required_substrings=("config/db_pool.yaml", "max_connections=150"),
        ),
        EvalProbe(
            id="s1_p8",
            category="exact_artifact",
            query="RS256 公钥文件放在什么路径？",
            query_turn=80,
            required_substrings=("keys/jwt_pub.pem",),
        ),
        EvalProbe(
            id="s1_p9",
            category="exact_artifact",
            query="复合索引 idx_user_email_tenant 的前缀长度规范是多少？",
            query_turn=80,
            required_substrings=("idx_user_email_tenant", "191"),
        ),
    ]

    return Scenario(
        name="backend_auth_mysql",
        description="Backend auth service debugging, MySQL 5.7 migration constraints, and security revisions (80 turns)",
        turns=turns,
        operations=ops,
        probes=probes,
    )


def _scenario_gpu_llm_finetune() -> Scenario:
    turns: list[TurnEntry] = [
        TurnEntry(1, "user", "硬约束：单卡训练显存上限 24GB (RTX 4090)，严禁开启超过 2048 的 max_seq_len。"),
        TurnEntry(3, "user", "硬约束：评测集 data/eval_holdout_v3.jsonl 只允许最终验收读取，训练/调参禁止泄露。"),
        TurnEntry(5, "user", "初始方案：使用 AdamW 优化器 fp32_adamw，learning_rate=5e-4，LoRA rank=16。"),
        TurnEntry(9, "user", "训练日志第 300 步出现梯度爆炸后 NaN，loss=nan，报错 ERR_CUDA_OOM_STEP300 在 train/trainer_loop.py:214。"),
        TurnEntry(14, "assistant", "定位到 NaN/OOM 诱因：bf16 未开启且 batch micro_size=8 过大；同时发现分词器未对齐导致 eval perplexity 异常飙升。"),
        TurnEntry(18, "user", "调整决策：优化器从 fp32_adamw 改为 paged_adamw_8bit，学习率降为 2e-4。"),
        TurnEntry(23, "assistant", "排查出 eval perplexity 飙升到 98.4 的真正原因：tokenizer.pad_token_id 未设为 eos_token_id=151643，padding 被计入 cross_entropy。"),
        TurnEntry(28, "user", "微调基础模型最初锁定 Qwen2.5-7B-Base，后续需要对比指令对齐版本。"),
        TurnEntry(35, "user", "基线配置更新：基础模型切换为 Qwen2.5-7B-Instruct，LoRA target_modules 改为 all-linear。"),
        TurnEntry(41, "user", "导出部署脚本 scripts/export_awq.py 需要校准样本 128 条并启用 group_size=128。"),
        TurnEntry(47, "user", "待办：对比 vLLM 启动参数 --gpu-memory-utilization 0.85 与 0.90 的 TTFT P99。"),
        TurnEntry(55, "assistant", "Checkpoint：paged_adamw_8bit + pad_token_id=151643 后训练稳定，loss 收敛至 0.89。"),
    ]
    used_turns = {t.turn for t in turns}
    noise_pool = [
        "Monitored nvidia-smi: GPU temp 71C, power 330W/450W, PCIe tx normal.",
        "Saved intermediate LoRA checkpoint adapter_model.safetensors at step 600.",
        "Dataset shard checksum verified: sha256 prefix 8a91f4c matched manifest.",
        "Dataloader prefetch_factor=2 kept CPU utilization around 38%.",
        "Logged TensorBoard scalar grad_norm=0.74 and lr cosine decay step.",
    ]
    turns.extend(_generate_noise_turns(start=1, end=75, skip=used_turns, topic="ml-train", lines_pool=noise_pool))
    turns.sort(key=lambda x: x.turn)

    ops = [
        MemoryOp(
            turn=1,
            op="commit",
            alias="c_vram",
            kind="constraint",
            scope="train",
            title="24GB VRAM cap and max_seq_len <= 2048",
            summary="单卡 RTX 4090 显存上限 24GB，max_seq_len 严禁超过 2048",
            detail="Exceeding max_seq_len=2048 triggers allocator fragmentation and OOM on 24GB RTX 4090.",
            entities=("24GB", "2048", "max_seq_len"),
            pinned=True,
        ),
        MemoryOp(
            turn=3,
            op="commit",
            alias="c_holdout",
            kind="constraint",
            scope="eval",
            title="Protect data/eval_holdout_v3.jsonl from training leakage",
            summary="data/eval_holdout_v3.jsonl 仅用于最终验收，严禁混入训练或调参",
            detail="Check md5 against manifest before running final eval on data/eval_holdout_v3.jsonl.",
            entities=("data/eval_holdout_v3.jsonl",),
            pinned=True,
        ),
        MemoryOp(
            turn=5,
            op="commit",
            alias="d_opt_v1",
            kind="decision",
            scope="train",
            title="Initial fp32_adamw optimizer with lr 5e-4",
            summary="初期优化器使用 fp32_adamw，学习率 5e-4",
            detail="Baseline optimizer fp32_adamw consumed 14GB state alone.",
            entities=("fp32_adamw", "5e-4"),
        ),
        MemoryOp(
            turn=9,
            op="commit",
            alias="f_oom",
            kind="fact",
            scope="train",
            title="ERR_CUDA_OOM_STEP300 in train/trainer_loop.py:214",
            summary="第 300 步报错 ERR_CUDA_OOM_STEP300 于 train/trainer_loop.py:214",
            detail="Stacktrace at train/trainer_loop.py:214 backward(): micro_batch=8 + fp32_adamw overflowed 24GB; reduce micro_batch=2 with grad_accum=8.",
            entities=("ERR_CUDA_OOM_STEP300", "train/trainer_loop.py:214"),
        ),
        MemoryOp(
            turn=14,
            op="commit",
            alias="f_ppl_spike",
            kind="fact",
            scope="eval",
            title="Eval perplexity spike to 98.4",
            summary="验证集 eval perplexity 异常飙升至 98.4",
            detail="Validation loss stayed >4.5 despite training loss dropping steadily.",
            entities=("perplexity", "98.4"),
        ),
        MemoryOp(
            turn=18,
            op="supersede",
            target_alias="d_opt_v1",
            alias="d_opt_v2",
            title="Switch to paged_adamw_8bit with lr 2e-4",
            summary="优化器切换为 paged_adamw_8bit，学习率调整为 2e-4",
            detail="paged_adamw_8bit reduced optimizer state memory by 68% and kept training loss stable at lr=2e-4 with cosine warmup.",
            entities=("paged_adamw_8bit", "2e-4"),
        ),
        MemoryOp(
            turn=23,
            op="commit",
            alias="f_pad_root",
            kind="fact",
            scope="eval",
            title="Tokenizer pad_token_id must match eos_token_id=151643",
            summary="分词器未设 pad_token_id=eos_token_id=151643 导致 padding 污染 loss",
            detail="Fix in data/collator.py: assign tokenizer.pad_token_id = 151643 and mask labels=-100 on pad positions.",
            entities=("pad_token_id", "151643", "data/collator.py"),
        ),
        MemoryOp(
            turn=23,
            op="link",
            src_alias="f_ppl_spike",
            dst_alias="f_pad_root",
            relation="caused_by",
            weight=0.95,
        ),
        MemoryOp(
            turn=28,
            op="commit",
            alias="d_base_v1",
            kind="decision",
            scope="model",
            title="Base checkpoint Qwen2.5-7B-Base",
            summary="初期基座选择 Qwen2.5-7B-Base",
            detail="Required raw chat template wrapping from scratch.",
            entities=("Qwen2.5-7B-Base",),
        ),
        MemoryOp(
            turn=35,
            op="supersede",
            target_alias="d_base_v1",
            alias="d_base_v2",
            title="Base model updated to Qwen2.5-7B-Instruct all-linear",
            summary="基座改为 Qwen2.5-7B-Instruct，LoRA target_modules 设为 all-linear",
            detail="Qwen2.5-7B-Instruct converged 2.3x faster on structured tool-calling traces using all-linear LoRA adapters (r=16, alpha=32).",
            entities=("Qwen2.5-7B-Instruct", "all-linear"),
        ),
        MemoryOp(
            turn=41,
            op="commit",
            alias="p_awq",
            kind="procedure",
            scope="deploy",
            title="AWQ export via scripts/export_awq.py",
            summary="使用 scripts/export_awq.py 量化导出，calibration=128，group_size=128",
            detail="python3 scripts/export_awq.py --samples 128 --group-size 128 --bits 4",
            entities=("scripts/export_awq.py", "group_size=128"),
        ),
        MemoryOp(
            turn=47,
            op="commit",
            alias="ol_vllm",
            kind="open_loop",
            scope="deploy",
            title="Benchmark vLLM --gpu-memory-utilization 0.85 vs 0.90",
            summary="待对比 vLLM --gpu-memory-utilization 0.85 与 0.90 的 TTFT P99",
            detail="Run 50-concurrency locust load test on 4-bit AWQ weights.",
            entities=("--gpu-memory-utilization", "vLLM"),
        ),
        MemoryOp(
            turn=55,
            op="checkpoint",
            goal="Stabilize Qwen2.5-7B-Instruct LoRA training and prepare AWQ serving",
            state_summary="paged_adamw_8bit + lr 2e-4 stable; pad_token_id=151643 fixed perplexity; vLLM bench open.",
        ),
    ]
    extra_s2 = [
        (11, "fact", "train", "FlashAttention-2 dtype bf16 enabled", "启用 flash_attention_2 与 torch.bfloat16 降低显存峰值", "attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, gradient_checkpointing=True in configs/lora.yaml.", ("flash_attention_2", "configs/lora.yaml")),
        (15, "procedure", "train", "Deepspeed ZeRO-2 off fallback note", "单卡不启用 ZeRO offload 以免 PCIe 瓶颈拖慢吞吐", "Keep pure single-GPU bnb 4bit NF4 quantization with double_quant=True instead of CPU offload.", ("NF4",)),
        (21, "fact", "eval", "ChatML template special tokens check", "确认 <|im_start|> 与 <|im_end|> 未被拆分成子词", "Verified len(tokenizer.encode('<|im_end|>', add_special_tokens=False)) == 1 in tests/test_tokenizer_fmt.py.", ("tests/test_tokenizer_fmt.py",)),
        (25, "fact", "train", "Dataset deduplication minhash threshold 0.85", "训练集去重 MinHash 阈值 0.85，过滤 4120 条近重复样本", "scripts/dedup_sft.py --threshold 0.85 --num-perm 128 reduced train rows from 48,200 to 44,080.", ("scripts/dedup_sft.py",)),
        (30, "fact", "train", "Cosine LR warmup ratio 0.05", "学习率预热 warmup_ratio=0.05，weight_decay=0.01，max_grad_norm=1.0", "Standard cosine scheduler with min_lr_ratio=0.1 over 3 epochs.", ("warmup_ratio=0.05",)),
        (34, "fact", "deploy", "LoRA adapter merge verification norm diff", "合并 LoRA 权重后校验 FP16 输出余弦相似度 > 0.9995", "python3 scripts/merge_lora.py --check-cosine-sim 0.9995 --out artifacts/merged-fp16.", ("scripts/merge_lora.py",)),
        (39, "fact", "eval", "JSON schema strict validity rate metric", "工具调用验证集 JSON Schema 严格合法率需 >= 96%", "Evaluated with jsonschema.Draft202012Validator on 500 held-out dev prompts before merge.", ("Draft202012Validator",)),
        (43, "procedure", "deploy", "vLLM warmup request sequence before bench", "压测前先发 10 条预热请求触发 CUDA Graph 捕获", "Send 10 prompts (len 256-1024) to /v1/completions before recording P50/P95/P99 latency.", ("CUDA Graph",)),
        (49, "fact", "train", "Checkpoint retention limit save_total_limit=3", "训练产物磁盘上限限制 save_total_limit=3 防止 /workspace 写满", "save_steps=200, save_total_limit=3, load_best_model_at_end=True metric_for_best_model='eval_loss'.", ("save_total_limit=3",)),
        (52, "fact", "eval", "Seed determinism torch_manual_seed 42", "固定随机种子 seed=42 以保证消融实验可复现", "Set random, numpy, torch, and CUDA seeds to 42 in train/repro.py.", ("train/repro.py",)),
    ]
    for et, ek, esc, etit, esum, edet, eent in extra_s2:
        ops.append(
            MemoryOp(
                turn=et,
                op="commit",
                alias=f"ex2_{et}",
                kind=ek,
                scope=esc,
                title=etit,
                summary=esum,
                detail=edet,
                entities=eent,
            )
        )
    for ep_t in (8, 12, 16, 20, 26, 32, 38, 44):
        ops.append(
            MemoryOp(
                turn=ep_t,
                op="commit",
                alias=f"ep_ml_{ep_t}",
                kind="episode",
                scope="ml-train",
                title=f"Epoch telemetry snapshot T{ep_t}",
                summary=f"Recorded GPU utilization and dataloader throughput at T{ep_t}.",
                detail=f"Step {ep_t * 25}: throughput=1840 tok/s, temp=71C.",
            )
        )

    probes = [
        EvalProbe(
            id="s2_p1",
            category="buried_constraint",
            query="单卡训练显存和序列长度有哪些硬性约束？",
            query_turn=70,
            required_substrings=("24GB", "2048", "max_seq_len"),
        ),
        EvalProbe(
            id="s2_p2",
            category="buried_constraint",
            query="评测数据集 data/eval_holdout_v3.jsonl 的使用规范是什么？",
            query_turn=71,
            required_substrings=("data/eval_holdout_v3.jsonl",),
        ),
        EvalProbe(
            id="s2_p3",
            category="revised_decision",
            query="当前微调使用的优化器和学习率配置是什么？",
            query_turn=72,
            required_substrings=("paged_adamw_8bit", "2e-4"),
            forbidden_stale_substrings=("fp32_adamw", "5e-4"),
        ),
        EvalProbe(
            id="s2_p4",
            category="revised_decision",
            query="目前选定的基座模型名称和 LoRA 模块配置是什么？",
            query_turn=73,
            required_substrings=("Qwen2.5-7B-Instruct", "all-linear"),
            forbidden_stale_substrings=("Qwen2.5-7B-Base",),
        ),
        EvalProbe(
            id="s2_p5",
            category="exact_artifact",
            query="ERR_CUDA_OOM_STEP300 在哪个文件哪一行发生？怎么改 micro_batch？",
            query_turn=73,
            required_substrings=("train/trainer_loop.py:214", "micro_batch=2"),
        ),
        EvalProbe(
            id="s2_p6",
            category="exact_artifact",
            query="AWQ 量化导出脚本路径和 group_size 参数是多少？",
            query_turn=74,
            required_substrings=("scripts/export_awq.py", "group_size=128"),
        ),
        EvalProbe(
            id="s2_p7",
            category="multi_hop_context",
            query="为什么之前验证集 eval perplexity 飙升到 98.4？根因和修复文件在哪？",
            query_turn=74,
            required_substrings=("151643", "data/collator.py"),
        ),
        EvalProbe(
            id="s2_p8",
            category="exact_artifact",
            query="部署环节还有哪个 vLLM 参数测试待完成？",
            query_turn=75,
            required_substrings=("--gpu-memory-utilization",),
        ),
        EvalProbe(
            id="s2_p9",
            category="exact_artifact",
            query="collator 修复 padding label mask 的值设成多少？",
            query_turn=75,
            required_substrings=("labels=-100",),
        ),
    ]

    return Scenario(
        name="gpu_llm_finetune",
        description="24GB GPU LoRA fine-tuning, tokenizer debugging, and model/optimizer revisions (75 turns)",
        turns=turns,
        operations=ops,
        probes=probes,
    )


def _scenario_release_security_audit() -> Scenario:
    turns: list[TurnEntry] = [
        TurnEntry(2, "user", "硬约束：所有生产容器必须以非 root 用户 uid 10001 运行，禁止挂载 docker.sock。"),
        TurnEntry(4, "user", "硬约束：对外 Webhook 回调超时不得超过 3000ms，禁止同步阻塞主事务。"),
        TurnEntry(7, "user", "服务间通信第一版选型使用 REST_HTTP1_JSON，消息队列暂定 RabbitMQ。"),
        TurnEntry(11, "user", "灰度环境回调服务出现 webhook_timeout_alert，大量请求在网关层被丢弃。"),
        TurnEntry(16, "assistant", "追查发现 webhook_timeout_alert 的根因是 Ingress 默认 proxy_ignore_headers_underscores=off，丢弃了 X_Webhook_Sig 头导致下游反复重试阻塞。"),
        TurnEntry(21, "user", "架构评审修订：内部 RPC 放弃 REST_HTTP1_JSON，改用 gRPC_PROTOBUF；消息队列从 RabbitMQ 迁移到 NATS_JetStream。"),
        TurnEntry(27, "user", "安全审计发现 CVE-2026-1182 影响 deploy/charts/gateway/values.yaml 中的旧镜像 nginx-ingress:1.9.1。"),
        TurnEntry(33, "assistant", "升级方案：将 deploy/charts/gateway/values.yaml 镜像固定为 nginx-ingress:1.11.4 并开启 allowSnippetAnnotations=false。"),
        TurnEntry(39, "user", "合规要求补充：审计日志保留期最初设为 30_days_retention。"),
        TurnEntry(48, "user", "法务通知更新：审计日志保留期不再用 30_days_retention，改为 180_days_immutable_s3。"),
        TurnEntry(54, "user", "待办：在 scripts/verify_sbom.sh 中补齐 cosign 签名校验。"),
        TurnEntry(62, "assistant", "Checkpoint：gRPC 与 NATS_JetStream 切换完成，CVE-2026-1182 补丁就绪。"),
    ]
    used_turns = {t.turn for t in turns}
    noise_pool = [
        "Helm template diff showed 0 drift on unrelated prometheus-stack release.",
        "Trivy scan finished on base alpine image: 0 CRITICAL, 1 LOW.",
        "Verified PodDisruptionBudget minAvailable=2 for gateway deployment.",
        "Canary pod rollout 25% -> 50% completed with zero error budget burn.",
        "DNS CoreDNS cache hit ratio stabilized at 96.4% in cluster-prod.",
    ]
    turns.extend(_generate_noise_turns(start=1, end=80, skip=used_turns, topic="sec-rel", lines_pool=noise_pool))
    turns.sort(key=lambda x: x.turn)

    ops = [
        MemoryOp(
            turn=2,
            op="commit",
            alias="c_nonroot",
            kind="constraint",
            scope="k8s",
            title="Containers run as uid 10001; forbid docker.sock",
            summary="生产容器必须以非 root uid 10001 运行，严禁挂载 docker.sock",
            detail="PodSecurity restricted profile enforced: runAsUser=10001, allowPrivilegeEscalation=false, no docker.sock hostPath.",
            entities=("10001", "docker.sock"),
            pinned=True,
        ),
        MemoryOp(
            turn=4,
            op="commit",
            alias="c_webhook_sla",
            kind="constraint",
            scope="webhook",
            title="Webhook timeout <= 3000ms and async dispatch",
            summary="对外 Webhook 超时不超过 3000ms，禁止同步阻塞主事务",
            detail="Enqueue dispatch job outbox rather than blocking checkout DB transaction.",
            entities=("3000ms", "Webhook"),
            pinned=True,
        ),
        MemoryOp(
            turn=7,
            op="commit",
            alias="d_bus_v1",
            kind="decision",
            scope="infra",
            title="Initial REST_HTTP1_JSON and RabbitMQ bus",
            summary="服务间采用 REST_HTTP1_JSON，队列采用 RabbitMQ",
            detail="Initial prototype wiring before latency audit.",
            entities=("REST_HTTP1_JSON", "RabbitMQ"),
        ),
        MemoryOp(
            turn=11,
            op="commit",
            alias="f_wh_alert",
            kind="fact",
            scope="webhook",
            title="Canary webhook_timeout_alert gateway drops",
            summary="灰度环境触发 webhook_timeout_alert，请求在网关重试阻塞",
            detail="P99 dispatch jumped from 210ms to 3400ms after gateway cutover.",
            entities=("webhook_timeout_alert",),
        ),
        MemoryOp(
            turn=16,
            op="commit",
            alias="f_wh_root",
            kind="fact",
            scope="webhook",
            title="Ingress drops underscore header X_Webhook_Sig",
            summary="Ingress 默认丢弃下划线头 X_Webhook_Sig 引起鉴权失败和重试阻塞",
            detail="Nginx underscores_in_headers=off stripped X_Webhook_Sig; fix in deploy/ingress.conf by enabling underscores_in_headers on.",
            entities=("X_Webhook_Sig", "underscores_in_headers", "deploy/ingress.conf"),
        ),
        MemoryOp(
            turn=16,
            op="link",
            src_alias="f_wh_alert",
            dst_alias="f_wh_root",
            relation="caused_by",
            weight=0.95,
        ),
        MemoryOp(
            turn=21,
            op="supersede",
            target_alias="d_bus_v1",
            alias="d_bus_v2",
            title="Switch internal RPC to gRPC_PROTOBUF and NATS_JetStream",
            summary="内部通信改为 gRPC_PROTOBUF，消息队列迁移到 NATS_JetStream",
            detail="Internal service mesh standardized on gRPC_PROTOBUF over mTLS and NATS_JetStream durable consumers for deterministic replay.",
            entities=("gRPC_PROTOBUF", "NATS_JetStream"),
        ),
        MemoryOp(
            turn=27,
            op="commit",
            alias="f_cve",
            kind="fact",
            scope="security",
            title="CVE-2026-1182 in deploy/charts/gateway/values.yaml",
            summary="CVE-2026-1182 影响 deploy/charts/gateway/values.yaml 中的 nginx-ingress:1.9.1",
            detail="Upgrade image tag in deploy/charts/gateway/values.yaml from 1.9.1 to nginx-ingress:1.11.4 and set allowSnippetAnnotations=false.",
            entities=("CVE-2026-1182", "deploy/charts/gateway/values.yaml", "nginx-ingress:1.11.4"),
        ),
        MemoryOp(
            turn=39,
            op="commit",
            alias="d_ret_v1",
            kind="decision",
            scope="compliance",
            title="Audit log retention 30_days_retention",
            summary="初期审计日志保留策略为 30_days_retention",
            detail="Standard local ES rollover at 30 days.",
            entities=("30_days_retention",),
        ),
        MemoryOp(
            turn=48,
            op="supersede",
            target_alias="d_ret_v1",
            alias="d_ret_v2",
            title="Audit retention upgraded to 180_days_immutable_s3",
            summary="法务更新审计日志保留策略为 180_days_immutable_s3",
            detail="Object Lock WORM compliance mode enabled on S3 audit bucket for 180_days_immutable_s3 with KMS customer-managed key.",
            entities=("180_days_immutable_s3",),
        ),
        MemoryOp(
            turn=54,
            op="commit",
            alias="ol_cosign",
            kind="open_loop",
            scope="security",
            title="Add cosign verify to scripts/verify_sbom.sh",
            summary="待在 scripts/verify_sbom.sh 中补齐 cosign 镜像签名校验",
            detail="Verify Fulcio keyless cert issuer before Helm push.",
            entities=("scripts/verify_sbom.sh", "cosign"),
        ),
        MemoryOp(
            turn=62,
            op="checkpoint",
            goal="Pass release security audit and cut over gRPC + NATS_JetStream",
            state_summary="gRPC_PROTOBUF + NATS_JetStream live; CVE-2026-1182 patched; cosign open loop remaining.",
        ),
    ]
    extra_s3 = [
        (10, "fact", "k8s", "Readiness probe initialDelaySeconds 10s", "网关 readinessProbe /readyz initialDelaySeconds=10s, periodSeconds=5s", "httpGet path=/readyz port=8443; failureThreshold=3; preStop sleep 5s before SIGTERM.", ("/readyz",)),
        (15, "procedure", "infra", "NATS JetStream stream retention limits", "NATS JetStream stream_events 配置 max_bytes=20Gi 与 replicas=3", "nats stream add EVENTS --subjects='events.*' --storage=file --replicas=3 --max-bytes=20Gi --discard=old", ("JetStream",)),
        (22, "fact", "security", "OPA Gatekeeper policy reject hostNetwork", "OPA Gatekeeper 策略禁止 Pod 设置 hostNetwork: true 与 hostPID: true", "ConstraintTemplate k8spsphostnamespace enforced across prod-eu and prod-ap namespaces.", ("Gatekeeper",)),
        (26, "fact", "k8s", "HPA target CPU utilization 65% bounds", "HPA 弹性范围 minReplicas=3, maxReplicas=18, averageUtilization=65%", "scaleDown stabilizationWindowSeconds=300 to avoid thrashing during burst webhook retries.", ("stabilizationWindowSeconds",)),
        (30, "fact", "webhook", "HMAC signature header SHA-256 verification", "Webhook 下游签名校验采用 HMAC-SHA256 与 300s 时间窗防重放", "Compare constant-time hmac.compare_digest over timestamp + '.' + body; reject skew > 300s.", ("HMAC-SHA256",)),
        (35, "procedure", "security", "Rotate external KMS envelope key quarterly", "每季度执行 KMS 信封主密钥轮换并验证旧密文可读", "aws kms enable-key-rotation --key-id alias/prod-audit-cmk; alert if KeyRotationEnabled == false.", ("alias/prod-audit-cmk",)),
        (41, "fact", "infra", "gRPC keepalive_time_ms 30000 settings", "gRPC 服务端保活参数 keepalive_time_ms=30000, keepalive_timeout_ms=5000", "grpc.http2.min_ping_interval_without_data_ms=10000 to prevent Envoy GOAWAY ENHANCE_YOUR_CALM.", ("keepalive_time_ms=30000",)),
        (45, "fact", "security", "NetworkPolicy default-deny ingress egress", "命名空间 prod-core 默认应用 default-deny ingress/egress NetworkPolicy", "Explicitly allow DNS UDP/TCP 53 to kube-dns and egress 443 only for webhook dispatcher pods.", ("NetworkPolicy",)),
        (51, "fact", "compliance", "S3 KMS bucket policy require TLS 1.2+", "审计日志桶策略拒绝非 HTTPS 请求 (aws:SecureTransport=false)", "Bucket policy enforces Deny on s3:* when SecureTransport=false or s3:x-amz-server-side-encryption != aws:kms.", ("SecureTransport",)),
        (56, "procedure", "k8s", "Helm rollback dry-run verification step", "生产升级前先执行 helm upgrade --dry-run --debug 核对渲染清单", "Compare rendered manifests via helm-diff plugin against live release revision before applying.", ("helm-diff",)),
    ]
    for et, ek, esc, etit, esum, edet, eent in extra_s3:
        ops.append(
            MemoryOp(
                turn=et,
                op="commit",
                alias=f"ex3_{et}",
                kind=ek,
                scope=esc,
                title=etit,
                summary=esum,
                detail=edet,
                entities=eent,
            )
        )
    for ep_t in (9, 14, 19, 25, 31, 37, 43, 50):
        ops.append(
            MemoryOp(
                turn=ep_t,
                op="commit",
                alias=f"ep_rel_{ep_t}",
                kind="episode",
                scope="sec-rel",
                title=f"Cluster canary status tick T{ep_t}",
                summary=f"Observed canary rollout status at T{ep_t} with healthy pod readiness.",
                detail=f"Rollout revision {ep_t}: 4/4 pods Ready, 0 restarts.",
            )
        )

    probes = [
        EvalProbe(
            id="s3_p1",
            category="buried_constraint",
            query="生产容器运行用户 UID 和宿主机挂载有哪些安全约束？",
            query_turn=75,
            required_substrings=("10001", "docker.sock"),
        ),
        EvalProbe(
            id="s3_p2",
            category="buried_constraint",
            query="Webhook 回调的最大超时时间和事务要求是什么？",
            query_turn=76,
            required_substrings=("3000ms",),
        ),
        EvalProbe(
            id="s3_p3",
            category="revised_decision",
            query="目前服务间 RPC 协议和消息队列组件各自是什么？",
            query_turn=77,
            required_substrings=("gRPC_PROTOBUF", "NATS_JetStream"),
            forbidden_stale_substrings=("REST_HTTP1_JSON", "RabbitMQ"),
        ),
        EvalProbe(
            id="s3_p4",
            category="revised_decision",
            query="审计日志现在的合规保留周期策略是什么？",
            query_turn=78,
            required_substrings=("180_days_immutable_s3",),
            forbidden_stale_substrings=("30_days_retention",),
        ),
        EvalProbe(
            id="s3_p5",
            category="exact_artifact",
            query="CVE-2026-1182 涉及哪个 Helm values 文件，目标修复版本是什么？",
            query_turn=78,
            required_substrings=("deploy/charts/gateway/values.yaml", "nginx-ingress:1.11.4"),
        ),
        EvalProbe(
            id="s3_p6",
            category="exact_artifact",
            query="发布前还有哪个 SBOM 脚本待补全签名校验？",
            query_turn=79,
            required_substrings=("scripts/verify_sbom.sh", "cosign"),
        ),
        EvalProbe(
            id="s3_p7",
            category="multi_hop_context",
            query="为什么灰度会出现 webhook_timeout_alert？具体在哪个文件修？",
            query_turn=79,
            required_substrings=("X_Webhook_Sig", "deploy/ingress.conf"),
        ),
        EvalProbe(
            id="s3_p8",
            category="exact_artifact",
            query="Nginx gateway 安全加固除了镜像版本外还要设置哪个 snippet 参数？",
            query_turn=80,
            required_substrings=("allowSnippetAnnotations=false",),
        ),
        EvalProbe(
            id="s3_p9",
            category="exact_artifact",
            query="X_Webhook_Sig 被丢弃需要启用什么 Nginx 配置项？",
            query_turn=80,
            required_substrings=("underscores_in_headers",),
        ),
    ]

    return Scenario(
        name="release_security_audit",
        description="Kubernetes security rollout, CVE patching, and architecture revisions (80 turns)",
        turns=turns,
        operations=ops,
        probes=probes,
    )


def load_scenarios() -> list[Scenario]:
    """Return all deterministic benchmark scenarios (235 turns, 27 probes total)."""
    return [
        _scenario_backend_auth_mysql(),
        _scenario_gpu_llm_finetune(),
        _scenario_release_security_audit(),
    ]
