# AI Memory Gateway

Gateway 是“小家”的认知上下文与模型执行服务。它不拥有公开聊天事实，但持久保存由 Relay 已接受事实派生的 cognitive conversation partitions，并统一拥有 Memory ACL、Persona、Model Profile、供应商调用、Prompt Cache、摘要与 usage telemetry。

## 当前权责

```text
Relay
= canonical raw transcript / media bytes / factual authority

Gateway
= cognitive conversation partitions
  + scoped memory and summaries
  + actor Persona versions
  + Model Profiles and provider execution
  + prompt-cache/compression state
  + usage/cache telemetry

Orchestrator
= scheduling / ordering / preemption / cancellation / fence lifecycle
```

Gateway 不直接发布 final；Orchestrator 消费 Gateway stream 后仍需通过 Relay 的 fence/CAS publication 接口落定事实。

### 2026-09-07 回复修复与手动摘要 TEST 部署

基于 `bdd1b53`，功能提交 `8b4541e`、Docker 清单补充 `a9b20b4` 已推送并
部署 TEST（`6a9e642e3aa3b4323a8b595a`，Running/数据库 ready）：
记忆搜索日期转为 JSON 可序列化值；
供应商 delta 即时转发；空终态记为失败；失败日志只保留 generation/Profile、
HTTP 状态和异常类型。未回复的图片延续到后续文字请求，直到对应 actor 回复。
这修复了可复现缺陷，不代表已确定所有历史空回的具体异常。

`POST /api/conversation-compression` 复用管理鉴权及 Relay 管理代理，仅接受
actor 的私聊 room/conversation/current_event。Relay 仍拥有完整聊天事实；
Gateway 用当前主 Profile 为较早事实生成语义摘要，保留最近 48 条原文，
通过现有 anchored history revision CAS 原子保存，不写长期记忆、不删除原记录。
无新可压缩范围不调用模型；调用计入 `conversation_compression` usage，
不自动重试、不调用记忆工具。单实例一次只运行一个手动摘要。原有免费的
自动摘录压缩仍作长度保护；手动摘要只在用户点击并确认费用后调用模型。

本地相关测试 63 passed / 4 skipped（未启用隔离 PostgreSQL DSN）；
Tidal 本地跨服务测试 2 passed，包含新私聊窗口的发送、回复和历史隔离。
摘要验证使用人工数据/模拟供应商，无新增付费调用。线上摘要接口通过
Relay 代理对空 payload 返回 422；运行文件 hash 与发布提交一致。
完整部署证据以配对 Tidal 仓库 DEPLOYMENT.md 为准；真实付费回复/摘要
质量未重新验收，ASR 验收仍延期。本段收尾记录为部署后的本地文档提交。

## 唯一配置 Source of Truth

- 模型、协议、provider route、key reference、capabilities、cache strategy：Model Profile。
- 椒椒/老克身份提示：actor Persona version。
- conversation context：Relay accepted facts 同步形成的 cognitive partition。
- 长期记忆：scoped memory schema + pre-retrieval ACL。
- Dashboard 的显式新增记忆直接写入 scoped memory；删除使用数据库硬删除，历史整理产生的归档仍可恢复。
- cache/compression：actor + canonical conversation + Profile + prompt/runtime/room/tool versions 隔离。
- 记忆整理模型：独立的 `MEMORY_API_KEY`、`MEMORY_API_BASE_URL`、`MEMORY_MODEL`。

已退休且不再接受配置：

- 旧 `/v1/chat/completions` 与 `/v1/models` 执行入口；
- 旧 global API/model/systemPrompt/reasoning/cache 配置；
- 旧 `/api/partition/*` 会话线与滑动分区缓存；
- `system_prompt.txt` 全局 Persona。

历史对话查看、搜索、导入、导出和 Memory 管理仍保留。

Relay 派生事实（`fact_identity` 非空）不可通过旧 Conversations 编辑、删除、
批量删除或合并入口修改；这些入口返回 409 / `relay_derived_conversation_read_only`。
不含派生事实的旧会话仍可管理；Memory 编辑和永久删除语义不变。
执行前从 Relay 分页核对到当前事件的完整历史，不能仅凭同步水位判断完整。
缺失或冲突从 Relay 修复；来源缺少当前事件、重复或遗漏已接受事实时阻断执行，
不推进水位。修复事实与清空对应压缩摘要在同一 PostgreSQL 事务中完成，
沿用摘要 revision 拒绝旧压缩写回；会话 advisory lock 同时协调新缓存创建。
目前每轮核对成本随历史长度线性增长；只有 Relay 提供权威完整性证明后才能安全恢复增量读取。
对应永久回归：`tests/test_conversation_integrity.py`（使用显式授权的隔离 PostgreSQL）。

## 记忆整理与合并

整理先按 scope、confidential、perspective、memory_type 和 source_kind 分组，
每组独立请求整理模型；手动合并拒绝跨组来源。typed 替代复用 scoped writer，
保留证据并记录每条原记忆的来源；legacy_unscoped 仍隔离，不自动推断归属。
新记忆与实际覆盖来源的停用在同一事务完成；空结果返回 `no_changes`，
未覆盖来源保持活跃，重复/外部 ID、重复内容和中途写入失败不留下部分替代。
恢复归档要求所有来源仍存在且未被其他替代占用；永久删除语义不变。

Actor merge/supersede 使用已声明工具参数和原记忆分类，提交时拒绝已停用来源。
Dashboard 整理还会核验模型调用期间来源是否被编辑；Actor stage→commit 期间
仍活跃来源的内容编辑没有版本比较，不能把前者的保障泛化为全部 Actor 路径。
永久回归：`tests/test_memory_replacement.py` 使用真实隔离 PostgreSQL；
Actor staged rollback 与 Dashboard 无变更文案另有现有套件中的回归。

## Model Profiles

Profile 明确声明：

- `provider`、`protocol`、`base_url`、`route_id`、`model`；
- `credential_ref`（只保存环境变量名，不保存 Key）；
- `input_modalities` 与其它 capabilities；
- provider-specific cache strategy 与已验证 TTL；
- selectable/verified 状态和显式 ordered fallback。

支持的协议 adapter 包括：

- `openai_responses`
- `openai_chat_completions`
- `anthropic_messages`
- `anthropic_messages_compatible`

actor identity 与 Profile 解耦。Profile 切换不会改变 `actor_id`、Persona、Memory ACL 或历史。

Dashboard 编辑读取已保存 Profile；空白 key/header 编辑字段表示保留既有引用，不回显密钥。
修改配置必须提交下一 revision，并重新进入 unverified；旧页面写入返回 409，旧 revision 的探针不能认证新配置。
Actor 默认与 ordered fallback 使用一次带 revision 的原子保存；任一 Profile 无效或并发冲突时全部不变。
房间实际生效的 override 与 actor default 分开展示。读取失败必须显示错误，不能伪装成空配置。

## Conversation Cache Pin

用户可为椒椒私聊、老克私聊、Living Room 或 active Bedroom session 开启“保持这段对话的长上下文”。

- Pin 持久化在 Gateway。
- PostgreSQL 在同一条领取语句中核验启用/到期，推进下次时间并增加计数；
  竞争执行者仅一个发起保活。旧执行者按原领取状态条件写回，不覆盖新领取。
  创建 Pin 和 actor 状态使用同一事务；旧版中断留下的缺失 actor 行会补齐。
- `call_count` 现为保活领取次数，包含领取后、发送前崩溃；旧值沿用历史成功次数，
  不追溯重算。它不能证明供应商实际接收或收费。进程崩溃/取消保留下次时间，
  下一个原有周期可重新领取，Pin 不会自动关闭；这一等待可能错过缓存 TTL。
- 只有当前 verified Profile 明确支持 `anthropic_prefix_anchored_v1` + `1h` 时才约每 50 分钟发起一次最小 keepalive。
- 不支持的 Profile 保留 Pin，但状态为 `paused`。
- keepalive 不写公开 timeline，不产生 Relay final，不触发 Memory extraction。
- Bedroom session 正式结束后停止其 Pin。
- usage receipt 的 `execution_purpose=cache_keepalive`，Dashboard 显示 last/next/领取次数/cache read。
- `active/paused` 只描述保活运行状态，最近供应商回执另分 HIT / OBSERVED_MISS / UNOBSERVABLE；缺字段不是未命中，也不能以 active 宣称命中。编辑后未验证的 Profile 会暂停保活。
- Dashboard 诊断展示真实 receipt 的时间、Profile revision、conversation 和 prefix/version/cursor；不估算缺失的 usage。

这是一项会产生 provider 费用的用户显式设置；默认没有 Pin，也没有空闲保活。

## 逐尝试 usage

每轮实际 provider HTTP 尝试独立生成 `receipt_id`，保留同一逻辑请求的
`generation_request_id`。fallback、工具续轮、Pin 和显式双发缓存探针均记录；
不再只在最终成功时保存一份汇总。`succeeded` 仅表示这次供应商尝试完整返回，
不代表 Relay 已发布回复；失败/受控取消也保存已收到的 usage，未提供字段为 null。
同轮累计 usage 按快照更新，跨轮任一字段未知时，汇总该字段仍未知。

现有回执表移除 generation 唯一约束，继续用 receipt 主键去重；同 ID 的不同
payload 拒绝写入。迁移保留旧行及时间，不改写历史汇总为虚构逐次记录。
Dashboard 只统计当前最近列表（最多 200 条），区分逻辑生成请求和尝试行，
调度探针、Pin、缓存探针不计入逻辑生成请求；旧版行可能汇总多轮。
已产生多份回执后，旧版仅支持 generation 唯一的程序不能作为兼容回退版本，
不能盲目恢复唯一约束或删除新回执；发布回退必须保留逐次记账能力。

回执写入和取消后的 Memory 暂存清理各使用至多 5 秒的取消屏蔽范围。
记账失败/超时不会触发 provider fallback；取消仍向上层传播。
这覆盖正常异常和 ASGI/AnyIO 取消，不保证进程硬退出或数据库持续不可用时的
最终回执落库，也不代表准确账单或缓存命中率改善。

协议终止依据：[Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)、
[OpenAI Chat streaming](https://developers.openai.com/api/reference/resources/chat)、
[OpenAI Responses streaming](https://developers.openai.com/api/reference/resources/responses/streaming-events)。
错误事件和缺失终止信号不能作为成功；原成功测试响应现包含明确协议终止。

## Media 与 v1.1

`group-room.v1.1` 在保持 v1.0 bytes/SHA 不变的前提下，为 typed private 与 Group factual events 增加：

- image
- attachment
- sticker
- voice_message

Relay 保存媒体 bytes 和 factual metadata；Gateway cognitive history 只保存 `MediaReference`。provider adapter 只能根据当前 Profile 的 `input_modalities` 进行显式转换。Sticker 不等于 reaction，voice message 不等于 Voice Call。

Bedroom media 与 Group Voice Call 不在 v1.1 范围内。

## 主要环境变量

所有 feature flags 默认关闭；测试或部署必须显式启用。

| 变量 | 用途 |
|---|---|
| `DATABASE_URL` | PostgreSQL 持久状态 |
| `GATEWAY_SECRET` | 完整 Gateway 管理凭证 |
| `ACTOR_PERSONA_PROXY_SECRET` | Relay Persona-only 代理凭证 |
| `MEMORY_ENABLED` | Memory/cognitive persistence 总开关 |
| `MEMORY_API_KEY` | 独立记忆整理供应商 Key |
| `MEMORY_API_BASE_URL` | 独立记忆整理 endpoint |
| `MEMORY_MODEL` | 独立记忆整理模型 |
| `MODEL_EXECUTION_ENABLED` | Gateway 模型执行 |
| `MODEL_PROFILE_MANAGEMENT_ENABLED` | Profile 管理 API |
| `GATEWAY_GROUP_MEMORY_ENABLED` | Group scoped memory |
| `GATEWAY_BEDROOM_ENABLED` | Bedroom context/retention |
| `CONVERSATION_CACHE_PIN_INTERVAL_SECONDS` | Pin 调度间隔，默认 3000 秒 |
| `CONVERSATION_CACHE_PIN_POLL_SECONDS` | 到期扫描间隔，默认 60 秒 |

真实 provider Key 只能通过环境变量注入，不得写进仓库、Profile JSON、日志或 fixture。

## 主要接口

- `POST /internal/model-execution/probe`
- `POST /internal/model-execution/stream`
- `POST /internal/group/context-packs/probe`
- `POST /internal/group/context-packs/full`
- `POST /internal/bedroom/context-packs/full`
- `POST /internal/bedroom/retention`
- `GET|PUT /api/model-profiles`
- `GET|PUT /api/model-bindings`
- `GET /api/model-usage/summary`
- `GET|PUT /api/cache-pins`
- `GET|POST /api/actor-prompts...`
- Memory、cognitive Conversations、archive、import/export 管理接口

跨仓库 wire contract 以 `contracts/group-room/` 下的版本化 JSON schema、golden fixtures 和 `SHA256SUMS` 为准；两个仓库不得 import 对方的 Python dataclass。

## 本地验证

### 管理鉴权与健康状态（F01/F10 稳定化）

管理请求必须使用 `GATEWAY_SECRET`；未配置时返回 503，错误密钥返回 401。
`ACTOR_PERSONA_PROXY_SECRET` 仍只允许现有 Persona 方法和路径，不能读取或修改 Memory。
`/health` 是公开的进程存活检查；`/` 是数据库就绪检查，部署验收必须使用后者。
只有数据库初始化完成且当前查询成功时，`/` 返回 200 / `ready: true`。
数据库故障或持久化关闭返回 503 / `ready: false` / `memory_count: null`；
真正空库在就绪时返回 `memory_count: 0`。初始化失败不启动提取/Pin worker，
修正配置后重启以重新执行初始化；初始化后发生的短暂查询故障在数据库恢复后可恢复就绪。

永久回归：`tests/test_gateway_health_auth.py` 从 FastAPI HTTP 边界验证缺密钥拒绝、
管理员权限、初始化失败、查询失败/恢复和空库区分。Persona API 的现有测试继续覆盖限权。
`test_real_postgres_lifespan_and_readiness_recovery` 使用共享的 opt-in schema
夹具运行真实初始化、缺表故障与恢复；复用下述 PostgreSQL DSN/授权变量。
基线 `cba9d8e` 在该故障下错误返回 200，修复版返回 503，恢复后 200。
F06 本地 PostgreSQL 18.6 全量回归为 409 passed、零跳过（59.86 秒）；Tidal 跨进程模型验收 3 passed（108.15 秒）。尚未部署或真实付费验收。
这些本地结果不等于测试部署或生产验收。

```powershell
python -m pytest tests -v
python -m compileall .
git diff --check
```

真实 provider/cache probe 会产生费用，只能在用户明确授权后运行。普通单元、契约和 fake-provider tests 不访问生产、不调用付费 API。

模型配置浏览器回归在独立 Edge/Chromium 中使用真实 DOM，所有网络均拦截为测试响应。
`tests/test_postgres_model_profile_store.py` 的并发/重连验收需显式设置 `GATEWAY_MODEL_SETTINGS_TEST_DSN` 与
`GATEWAY_TEST_POSTGRES_APPROVED=true`，只允许已授权 test PostgreSQL；测试自建并清理临时 `group_e2e_<hex>` schema，不访问业务表。


## 固定构建输入（F13）

Dockerfile 使用官方 Python 3.12 镜像的不可变 manifest 摘要；
requirements.txt 固定直接与传递运行依赖，继续使用 pip。
.dockerignore 只允许逐项列出的运行文件，新增模块、模板或静态资源需明确加入。
本地密钥、数据库、缓存、测试、文档和未列出的新文件不会进入构建上下文。
外部配置仍由运行时引用或挂载提供，不要把真实配置加入白名单。

在仓库根目录执行 `docker build .`。平台若保存了 Dockerfile override，
必须核对它与仓库中的固定版本一致。依赖和镜像摘要更新必须重新安装、构建和测试。
Windows Python 3.13 的回归结果不等同于 Linux Python 3.12 镜像构建验收。
配套 Tidal 的 `tests/acceptance/test_group_build_inputs.py` 检查两仓输入；
跨服务测试通过 `GATEWAY_PYTHON` 显式选择本仓依赖环境。
