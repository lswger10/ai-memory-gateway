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

## 唯一配置 Source of Truth

- 模型、协议、provider route、key reference、capabilities、cache strategy：Model Profile。
- 椒椒/老克身份提示：actor Persona version。
- conversation context：Relay accepted facts 同步形成的 cognitive partition。
- 长期记忆：scoped memory schema + pre-retrieval ACL。
- cache/compression：actor + canonical conversation + Profile + prompt/runtime/room/tool versions 隔离。
- 记忆整理模型：独立的 `MEMORY_API_KEY`、`MEMORY_API_BASE_URL`、`MEMORY_MODEL`。

已退休且不再接受配置：

- 旧 `/v1/chat/completions` 与 `/v1/models` 执行入口；
- 旧 global API/model/systemPrompt/reasoning/cache 配置；
- 旧 `/api/partition/*` 会话线与滑动分区缓存；
- `system_prompt.txt` 全局 Persona。

历史对话查看、搜索、导入、导出和 Memory 管理仍保留。

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

## Conversation Cache Pin

用户可为椒椒私聊、老克私聊、Living Room 或 active Bedroom session 开启“保持这段对话的长上下文”。

- Pin 持久化在 Gateway。
- 只有当前 verified Profile 明确支持 `anthropic_prefix_anchored_v1` + `1h` 时才约每 50 分钟发起一次最小 keepalive。
- 不支持的 Profile 保留 Pin，但状态为 `paused`。
- keepalive 不写公开 timeline，不产生 Relay final，不触发 Memory extraction。
- Bedroom session 正式结束后停止其 Pin。
- usage receipt 的 `status=cache_keepalive`，Dashboard 显示 last/next/call count/cache read。

这是一项会产生 provider 费用的用户显式设置；默认没有 Pin，也没有空闲保活。

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
| `GROUP_MEMORY_ENABLED` | Group scoped memory |
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

```powershell
python -m pytest tests -v
python -m compileall .
git diff --check
```

真实 provider/cache probe 会产生费用，只能在用户明确授权后运行。普通单元、契约和 fake-provider tests 不访问生产、不调用付费 API。
