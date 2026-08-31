# 🧸 AI Memory Gateway

**让你的 AI 拥有长期记忆。**

一个轻量级转发网关，在你和 LLM 之间加一层记忆系统。支持任何 OpenAI 兼容客户端（Kelivo、ChatBox、NextChat 等）和任何 LLM 服务商（OpenRouter、OpenAI、本地 Ollama 等）。

Give your AI long-term memory. A lightweight proxy gateway that adds a memory layer between you and any LLM.

---

## ✨ 功能

- **自定义人设** — 把你的 system prompt 写在 `system_prompt.txt`，每次对话自动注入
- **长期记忆** — 自动从对话中提取关键信息，下次聊天时自动回忆相关内容
- **三层记忆架构** — 碎片（自动提取的原始记忆）→ 事件（整理合并后的完整事件）→ 核心（手动标记的重要记忆），支持 AI 自动整理、手动合并、撤回合并、查看合并来源
- **分区缓存** — 自动管理对话上下文，通过 A/B 区轮转 + 摘要压缩，利用 prompt caching 大幅节省 token 费用。兼容 tool 调用消息
- **对话线管理** — 固定 session ID 实现跨平台对话衔接，支持多对话线切换、摘要编辑
- **对话记录** — 浏览、搜索、批量管理历史对话，支持 session 合并
- **Token 统计** — 自动记录每次对话的 token 消耗，按 session 汇总显示
- **全端点鉴权** — 设置 `GATEWAY_SECRET` 环境变量后，所有 API 端点需要携带密钥。Dashboard 通过 URL 参数传递密钥，自动注入后续请求
- **预置记忆** — 把你想让 AI "一开始就知道"的事情批量导入
- **兼容性强** — 支持所有 OpenAI 格式的客户端和 API 服务商
- **记忆向量搜索（可选）** — 关键词 + 语义向量四维混合搜索，说"过年"能搜到"春节"。支持 OpenAI 兼容的 Embedding API
- **设置面板** — 在 Dashboard 中直接管理所有运行时配置，热更新无需重启。支持模型列表动态拉取、可搜索下拉选择
- **零成本起步** — 可部署在 Render、Zeabur 等平台的免费额度内

## 🏗️ 架构

### Tidal Group / typed-room execution（新主链）

在 Tidal 的新主链中，Gateway 不只是“记忆转发器”，而是唯一的模型执行面：它管理 actor-specific prompt、Memory ACL、Context Pack、Model Profiles、provider adapters、实际模型请求、Prompt Cache 和真实 usage/cache telemetry。Relay 仍是公开消息事实源；Orchestrator 只负责谁先说、串行、抢占/取消与 fence 生命周期，不能直接持有 provider key 或自行拼 prompt。

Model Profile 将 actor identity 与供应商分开。同一个 `jiao` 或 `laoke` 可以人工切换到任意已启用且已实测通过的 Profile；自动 fallback 只按人工批准的有序名单执行。Profile 的 `base_url`、协议、model、credential env、headers 与 capabilities 均显式配置，不因“compatible”字样猜测缓存、tools 或 TTL 能力。

Claude/Anthropic 风格长会话使用 `anthropic_prefix_anchored_v1`：静态 tools/system 与锚定历史位于缓存断点之前，动态 memory、当前事件、stance 和 Bedroom context 位于断点之后。历史使用 `compressed_up_to_event_id`，达到阈值时以一次原子覆盖式有界摘要推进游标；正常追加不移动游标。默认不做任何付费 Prompt Cache 保活，只复用 HTTP/TCP/TLS 连接。1h TTL 只有在该 Profile 对应真实 route 的双发探针观察到 cache-read 后才可标记通过。

相关开关默认关闭：`MODEL_EXECUTION_ENABLED=false`、`MODEL_PROFILE_MANAGEMENT_ENABLED=false`、`MODEL_PROFILE_PWA_ENABLED=false`、`ACTOR_PERSONA_MANAGEMENT_ENABLED=false`。密钥只通过 Profile 的 `credential_ref=env:...` 从 Gateway 服务端环境读取，不能写入 Git 或下发浏览器。

完整 actor Persona 不应写入仓库中的 `actor_prompt_profiles.json`。该文件只保留 `jiao.v1` / `laoke.v1` 的最小故障回滚文本。启用 `ACTOR_PERSONA_MANAGEMENT_ENABLED=true` 后，可通过受 `GATEWAY_SECRET` 保护的管理接口或 Tidal household Settings 上传 UTF-8 `.md`；Relay 代理应使用独立的 `ACTOR_PERSONA_PROXY_SECRET`，Gateway 仅接受它访问现有 Persona 列表、保存、启用/回滚和导出接口，不能访问其它 management API。未配置 `GATEWAY_SECRET` 时管理接口会拒绝运行。Gateway 校验原始 UTF-8 bytes 的 SHA 后，将正文作为不可变版本存入 PostgreSQL；列表只返回元数据，正文只能通过受鉴权导出取得。上传不会自动激活，切换使用 revision/CAS 防止并发覆盖，任何旧版本和内置 v1 都可回滚。每次模型执行/Context Pack 构建前会刷新活动 revision，使不同 Gateway worker 使用同一版本。`ACTOR_PROMPT_MAX_BYTES` 是单份 Markdown 的应用层上限，默认 262144 bytes。

```
你的客户端（Kelivo / ChatBox / ...）
        ↓
   AI Memory Gateway（本项目）
   ├── 注入 system prompt（人设）
   ├── 搜索相关记忆 → 注入上下文
   ├── 转发请求 → LLM API
   └── 后台提取新记忆 → 存入数据库
        ↓
   LLM API（OpenRouter / OpenAI / Ollama / ...）
```

## 🚀 快速开始

### 第一阶段：纯转发网关（不需要数据库）

最简单的起步方式——先跑通网关，确认你的客户端能通过网关和 AI 对话。

**1. 准备文件**

你只需要这几个文件：
- `main.py` — 网关主程序
- `system_prompt.txt` — 你的 AI 人设（可选）
- `requirements.txt` — Python 依赖
- `Dockerfile` — 容器配置

**2. 修改人设**

编辑 `system_prompt.txt`，写入你想要的 AI 性格设定。

**3. 部署到 Render（推荐）**

1. Fork 或上传代码到你的 GitHub 仓库
2. 注册 [Render](https://render.com)（免费层支持 Web Service，够用）
3. 创建 Web Service → 连接 GitHub 仓库 → Render 会自动检测 Dockerfile
4. 设置环境变量（Environment → Add Environment Variable）：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `API_KEY` | 你的 LLM API Key | `sk-or-v1-xxxx`（OpenRouter）|
| `API_BASE_URL` | LLM API 地址 | `https://openrouter.ai/api/v1/chat/completions` |
| `DEFAULT_MODEL` | 默认模型 | `anthropic/claude-sonnet-4.5` |
| `PORT` | 端口 | `8000` |
| `GATEWAY_SECRET`（可选） | 网关鉴权密钥，设置后所有 API 端点需要携带此密钥 | `your-secret-key` |
| `ACTOR_PERSONA_PROXY_SECRET`（可选） | Relay Persona 代理专用窄权限密钥；仅允许现有 Persona list/save/activate/rollback/export 接口 | `separate-persona-proxy-key` |

5. 部署，访问你的网关地址看到 `{"status":"running"}` 就成功了

> ⚠️ Render 免费层的服务在无活动时会休眠，第一次访问需要等几十秒唤醒，之后就正常了。其他支持 Docker 部署的平台（Zeabur、Railway、Fly.io 等）也可以，流程类似。

**4. 连接客户端**

以 Kelivo 为例：
- API 地址填：`https://你的网关地址.onrender.com/v1`
- API Key 填：随便填一个（网关会用自己的 key）
- 模型填：你在 `DEFAULT_MODEL` 里设的模型

### 第二阶段：加上记忆系统

在第一阶段基础上，加一个 PostgreSQL 数据库就能开启记忆功能。

**1. 创建数据库**

在 Render 中：Dashboard → New → PostgreSQL，创建一个免费的 PostgreSQL 实例，拿到连接字符串（Internal Database URL）。

> ⚠️ Render 免费 PostgreSQL 有 90 天有效期，到期前记得用导出功能备份数据。其他平台（如 [Neon](https://neon.tech)、[Supabase](https://supabase.com)）也提供免费 PostgreSQL，可按需选择。如果使用外部数据库，连接字符串末尾可能需要加 `?sslmode=require`。

**2. 添加环境变量**

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `DATABASE_URL` | PostgreSQL 连接字符串 | `postgresql://user:pass@host:port/db` |
| `MEMORY_ENABLED` | 开启记忆 | `true` |
| `MEMORY_MODEL` | 提取记忆用的模型（推荐便宜的小模型） | `anthropic/claude-haiku-4.5` |
| `MAX_MEMORIES_INJECT` | 每次注入的最大记忆条数 | `15` |
| `MIN_SCORE_THRESHOLD` | 记忆搜索最低分数阈值，低于此分数的记忆不注入（0=不过滤） | `0.15` |
| `MEMORY_EXTRACT_INTERVAL` | 自动记忆提取间隔（0=禁用自动批量/1=每轮/N=每个 session 各自每N轮；显式记忆请求不受此间隔限制） | `1` |
| `MEMORY_EXTRACT_ENABLED（可选）` | 记忆提取+注入总开关，false时只存消息不提取记忆 | `true` |
| `TIMEZONE_HOURS` | 时区偏移（小时），用于记忆注入时的日期显示 | `8`（UTC+8） |
| `FORCE_STREAM（可选）` | 强制所有请求走流式传输（解决部分客户端thinking不显示） | `false` |
| `REASONING_EFFORT（可选）` | 推理强度（low/medium/high），注入请求启用思维链。注意部分模型不支持 medium | 留空不注入 |

**3. 重新部署**

部署后访问 `https://你的网关地址/dashboard`，能正常打开管理页面就说明数据库连接成功。

**4. 导入预置记忆（可选）**

**方式一（推荐，不用碰代码）：** 写一个 `.txt` 文件，每行一条你想让 AI 知道的信息，然后打开 `https://你的网关地址/dashboard`，在「导入记忆」页面选择「纯文本导入」上传文件，系统会自动评估每条记忆的重要程度并导入。也可以勾选"跳过自动评分"节省 API 额度，之后在「记忆管理」页面手动调整权重。

**方式二（代码方式，开发者用）：**
1. 复制 `seed_memories_example.py` 为 `seed_memories.py`
2. 修改里面的记忆条目，写入你想让 AI 一开始就知道的信息
3. 部署后访问 `https://你的网关地址/import/seed-memories`，看到 `"status": "done"` 就导入成功了

**5. 管理记忆（可选）**

打开 `https://你的网关地址/dashboard` 可以查看所有记忆，支持搜索、编辑内容、调整权重、单条删除和批量删除，以及导入/导出备份。

> 💡 如果设置了 `GATEWAY_SECRET`，访问地址变为 `https://你的网关地址/dashboard?gateway_key=你的密钥`。客户端请求头需加 `X-Gateway-Key: 你的密钥`。

### 第三阶段：分区缓存（省 token 费）

分区缓存让网关自动管理对话上下文，通过 A/B 区轮转 + 摘要压缩利用 prompt caching，大幅降低 token 开销。

**工作原理：**

```
[人设区]    system prompt，永远不变     ← 缓存命中
[摘要区]    历史压缩摘要               ← 正常轮次命中
[历史A区]   15轮原始消息               ← 正常轮次命中
[历史B区]   当前周期消息               ← 通过lookback命中
[当前输入]  时间+记忆+用户消息         ← 不缓存（每次不同）
```

每聊 15 轮自动轮转一次：A 区压缩成摘要追加到摘要区，B 区升级为新的 A 区。正常轮次 90% 的 token 走缓存读取（0.1x 价格）。

**添加环境变量：**

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `CACHE_PARTITION_ENABLED` | 分区缓存开关 | `true` |
| `CACHE_PARTITION_X` | 轮转周期（轮数）。1轮 = 一次用户发言 + AI回复 | `15` |
| `CACHE_SUMMARY_MODEL` | 摘要模型。**留空 = 不生成摘要**，轮转时旧消息直接滑出上下文（纯轮转模式）。从旧版本升级的用户注意：旧版此项有默认模型，新版默认为空，需要摘要请显式配置。不建议使用推理模型（思考可能耗尽输出token导致摘要为空） | 空 |
| `PARTITION_SESSION_ID` | 固定的 session ID | `my-thread` |
| `CACHE_PARTITION_TRIGGER`（可选） | 轮转触发方式：`rounds`（按轮次，默认）或 `time`（按时间窗口，适合微信等消息频率高的场景） | `rounds` |
| `CACHE_PARTITION_WINDOW`（可选） | 时间窗口（分钟），仅 `trigger=time` 时生效。窗口内的消息不触发摘要压缩 | `30` |
| `CACHE_MAX_ROTATIONS`（可选） | 时间窗口模式下单次请求最大轮转次数 | `2` |
| `CACHE_TTL`（可选） | 缓存有效期：`5m`（默认）或 `1h`。Anthropic 官方定价：5m 写入 1.25x、1h 写入 2x，读取都是 0.1x。消息间隔经常超过 5 分钟的慢聊场景（比如挂着微信/TG 等回复）建议 `1h`，缓存不会中途过期。OpenRouter 会原样透传此参数。设置面板可热更新 | `5m` |

> 💡 **不需要记忆功能也能用分区缓存。** 设置 `MEMORY_ENABLED=true`（连数据库存消息）+ `MEMORY_EXTRACT_ENABLED=false`（关闭记忆提取）+ `CACHE_PARTITION_ENABLED=true`，就能只用分区缓存不用记忆系统。

**管理面板：**

部署后在 Dashboard 的「🔗 对话线」页面可以：
- 查看当前活跃对话线的状态（摘要长度、轮转进度）
- 重命名对话线 ID（关联的对话记录和 token 统计自动迁移）
- 查看、编辑、清空摘要内容
- 新建对话线（可选择继承已有摘要）
- 一键切换活跃对话线（运行时生效，不用重启）

### 第四阶段：关闭记忆（应急）

如果记忆系统出问题，把环境变量 `MEMORY_ENABLED` 改回 `false` 即可退回纯转发模式。不需要改代码。

## 📁 文件说明

```
ai-memory-gateway/
├── main.py                    # 网关主程序
├── database.py                # 数据库操作（PostgreSQL）
├── memory_extractor.py        # AI 记忆提取
├── system_prompt.txt          # 你的 AI 人设（自行编辑）
├── seed_memories_example.py   # 预置记忆示例
├── requirements.txt           # Python 依赖
├── Dockerfile                 # 容器配置
├── templates/                 # 页面模板（Dashboard 界面）
│   ├── dashboard.html         # 主控制台页面
│   └── ...
├── static/                    # 静态资源
│   ├── css/                   # 样式文件
│   └── js/                    # 前端脚本
├── LICENSE                    # MIT 许可证
└── README.md                  # 本文件
```

## 🔧 API 接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查，查看网关状态 |
| `/v1/chat/completions` | POST | 核心转发接口（OpenAI 兼容） |
| `/v1/models` | GET | 模型列表 |
| `/dashboard` | GET | 管理控制台（记忆、对话、对话线一体化界面） |
| `/import/seed-memories` | GET | 执行预置记忆导入（开发者用） |
| `/api/memories` | GET | 获取所有记忆（支持 `?layer=` `?active_only=` 筛选） |
| `/api/memories/consolidate` | POST | 手动触发记忆整理（异步，碎片 → 事件） |
| `/api/memories/consolidate/status` | GET | 查询整理任务状态 |
| `/api/memories/merge` | POST | 手动合并多条记忆 |
| `/api/memories/check-duplicate` | POST | 记忆去重检查 |
| `/api/memories/cleanup-fragments` | POST | 清理 N 天前的归档碎片 |
| `/api/memories/layer-stats` | GET | 获取各层记忆统计 |
| `/api/memories/{id}` | PUT | 更新记忆（支持 content / importance / title / layer） |
| `/api/memories/{id}` | DELETE | 删除记忆（`?soft=true` 软删除） |
| `/api/memories/{id}/promote` | POST | 升级为核心记忆 |
| `/api/memories/{id}/restore` | POST | 恢复已归档的记忆 |
| `/api/memories/{id}/revert-merge` | POST | 撤回合并，恢复原始碎片 |
| `/api/conversations` | GET | 分页获取对话列表（含 token 统计） |
| `/api/conversations/{id}/messages` | GET | 获取指定对话的消息列表 |
| `/api/conversations/{id}` | DELETE | 删除指定对话 |
| `/api/conversations/batch-delete` | POST | 批量删除对话 |
| `/api/admin/merge-sessions` | POST | 合并多个 session 到目标 session |
| `/api/admin/backfill-memory-embeddings` | POST | 启动记忆 embedding 补算（后台异步） |
| `/api/admin/backfill-memory-embeddings/status` | GET | 查询补算进度 |
| `/api/models` | GET | 获取可用模型列表（根据 API 服务商自动适配） |
| `/api/settings` | GET | 获取所有运行时配置（设置面板用） |
| `/api/settings` | PUT | 保存配置（写入数据库 + 热更新，立即生效无需重启） |
| `/api/partition/status` | GET | 获取分区缓存当前状态 |
| `/api/partition/threads` | GET | 列出所有对话线 |
| `/api/partition/summary` | PUT/DELETE | 编辑/清空对话线摘要 |
| `/api/partition/thread` | POST | 新建对话线 |
| `/api/partition/thread/rename` | PUT | 重命名对话线 ID |
| `/api/partition/switch` | POST | 切换活跃对话线 |

## 🌐 支持的 LLM 服务商

只要兼容 OpenAI 聊天格式就行。改 `API_BASE_URL` 环境变量即可切换：

| 服务商 | API_BASE_URL |
|--------|-------------|
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` |
| OpenAI | `https://api.openai.com/v1/chat/completions` |
| Ollama（本地） | `http://localhost:11434/v1/chat/completions` |
| 其他兼容服务 | 查阅对应文档 |

> ⚠️ 部分 Gemini preview 模型（如 `gemini-3-flash-preview`）可能存在流式输出兼容性问题导致空回复，建议使用正式版模型（如 `gemini-2.5-flash`）。

## 💡 记忆系统原理

1. **你发消息** → 网关从数据库搜索相关记忆
2. **记忆注入** → 相关记忆 + 记忆应用规则拼接到 system prompt 后面
3. **AI 回复** → 网关边转发边捕获完整回复
4. **后台提取** → 用小模型（如 Haiku）从完整对话上下文中提取关键信息
5. **存入数据库** → 下次对话时可以检索到

网关只为每个 session 在 PostgreSQL 保存一个“已处理到哪条 conversation message”的游标；尚未处理的新轮次直接从既有 `conversations` 记录重建，不复制保存整批消息，因此 Gateway 重启后也能继续累计。通过 `MEMORY_EXTRACT_INTERVAL` 可以控制自动提取频率：设为 0 禁用自动批量提取，设为 1 每轮都提，设为 N 则每个 session 各自每 N 轮提取一次（适合控制成本）。当用户以完整指令明确要求“记住”或“保存为长期记忆”时，该 session 当前尚未处理的轮次会立即送入同一个提取器；处理成功后游标推进，不会在下一次 interval 中重复发送。否定、询问记忆能力或普通文件保存请求不会触发该快速路径。

> **关于向量搜索：** 当前版本支持可选的记忆向量搜索功能。默认使用 jieba 中文分词 + 关键词匹配（ILIKE），适合大多数场景。如果需要语义搜索（说"过年"能搜到"春节"），可以设置 `MEMORY_VECTOR_ENABLED=true` + `EMBEDDING_API_KEY`，系统会同时走关键词和向量两路搜索，四维加权排序。支持任何 OpenAI 兼容的 Embedding API（OpenAI、Jina、Voyage、本地 Ollama 等）。如果数据库支持 pgvector 扩展会自动启用，否则回退到 Python 端计算余弦相似度。

**向量搜索环境变量（可选）：**

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `MEMORY_VECTOR_ENABLED` | 记忆向量搜索开关 | `false` |
| `EMBEDDING_API_KEY` | Embedding API Key（必需） | 无 |
| `EMBEDDING_BASE_URL` | Embedding API 地址 | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-3-small` |
| `EMBEDDING_DIM` | 向量维度 | `256` |
| `MEMORY_HW_KEYWORD` | 混合搜索：关键词权重 | `0.35` |
| `MEMORY_HW_SEMANTIC` | 混合搜索：语义相似度权重 | `0.35` |
| `MEMORY_HW_IMPORTANCE` | 混合搜索：重要程度权重 | `0.15` |
| `MEMORY_HW_RECENCY` | 混合搜索：时间衰减权重 | `0.15` |
| `MEMORY_SEMANTIC_THRESHOLD` | 向量相似度阈值 | `0.5` |

开启后，新记忆会自动计算 embedding。已有记忆可以在 Dashboard 记忆管理页面点击「开始补算」一键补算。

## ❓ 常见问题

**Q: 部署后访问显示 502 或服务无响应？**
A: 检查端口设置。Render 默认用 `PORT` 环境变量，确保设置为 `8000`（和 Dockerfile 里一致）。如果用其他平台，注意端口是否匹配。

**Q: 数据库连接失败？**
A: 如果数据库和网关不在同一个平台，连接字符串末尾可能需要加 `?sslmode=require`。

**Q: 记忆会越来越多影响性能吗？**
A: 每次最多注入 15 条记忆（可调），不会无限增长地消耗 token。自动提取只发送该 session 尚未处理的 interval 批次；可以调大 `MEMORY_EXTRACT_INTERVAL` 降低提取频率，显式记忆请求则立即处理当前待处理批次。

**Q: 能用免费额度跑吗？**
A: Render 免费层支持 Web Service + PostgreSQL，网关资源消耗很低，够用（注意免费 PostgreSQL 有 90 天期限）。也可以用 Neon 或 Supabase 的免费 PostgreSQL 作为长期方案。LLM API 费用另算（推荐 OpenRouter，按量付费）。

**Q: 怎么备份记忆？换平台会丢数据吗？**
A: 打开 `https://你的网关地址/dashboard`，在「导出备份」页面下载所有记忆的 JSON，建议定期备份。迁移到新平台后，在「导入记忆」页面选择「JSON 备份恢复」上传导出的文件即可。

**Q: 不会写代码能搞吗？**
A: 能。这个项目的第一个部署者就是不会写代码的——代码是 AI 写的，部署是她自己看文档搞定的。

## 📋 更新日志

### v3.7（2026-07-07）

- **缓存 TTL 可配置** — 新增 `CACHE_TTL` 变量（`5m` 默认 / `1h`），所有缓存断点统一带上对应有效期。消息间隔经常超过 5 分钟的慢聊场景用 `1h` 可避免缓存中途过期反复重建。支持设置面板热更新，OpenRouter 链路原样透传

### v3.6（2026-05-10）

- **时间窗口模式** — 分区缓存新增 `CACHE_PARTITION_TRIGGER=time` 模式。按时间而非轮次触发摘要压缩，适合微信等消息频率高的场景（一条消息算一轮，15轮可能就几分钟）。窗口时间通过 `CACHE_PARTITION_WINDOW` 配置，默认30分钟
- **非 Claude 模型兼容** — 分区缓存模式下自动检测模型，非 Anthropic Claude 系列的模型在发送前自动剥离 `cache_control` 字段并将 content 降级为纯字符串格式，解决智谱 GLM 等模型因不认识 `cache_control` 而报错或丢上下文的问题
- **整理记忆时区修复** — 修复按日期整理记忆时 `DATE(created_at)` 使用 UTC 日期而 Dashboard 显示北京时间导致的日期偏移。改为根据 `TIMEZONE_HOURS` 将本地日期转换为 UTC 时间范围查询

### v3.5（2026-05-06）

- **设置面板** — Dashboard 新增「设置」页面，所有运行时配置可在网页端直接修改，热更新立即生效无需重启
  - 基础连接（API 地址、Key、默认模型）
  - 记忆系统（开关、提取模型、注入条数、分数阈值、提取间隔）
  - 缓存分区（开关、轮转周期、摘要模型）
  - 向量搜索（开关、Embedding API Key/Base URL/模型/维度）
  - 搜索权重（四维权重滑块 + 语义阈值）
  - 其他（强制流式、推理强度）
  - System Prompt（在线编辑，实时字数统计）
- **模型列表 API** — 新增 `/api/models` 端点，根据 API 服务商（OpenRouter/Google/OpenAI）自动拉取可用模型列表，设置面板的模型选择框支持搜索过滤
- **Dashboard 美化** — Emoji 图标全部替换为内联 SVG（Lucide 风格），配色从冷灰青绿迁移到暖奶白玫瑰粉，全局输入框统一样式

### v3.3（2026-05-05）

- **三层记忆架构** — 碎片（layer 1，自动提取的原始记忆）→ 事件（layer 2，AI 整理合并后的完整事件）→ 核心（layer 3，手动标记的重要记忆）。数据库自动迁移，老数据默认为碎片层
- **记忆整理** — 选择日期范围，一键调用 AI 将碎片合并为事件记忆。异步执行，整理 prompt 保留原文中的主观感受和情绪表达。JSON 解析三层容错（strict=False → 去控制字符 → AI 修复）
- **手动合并** — 在记忆列表勾选多条，打开合并弹窗编辑合并后内容。支持选择目标层级（事件/核心）
- **撤回合并** — 事件记忆可一键撤回，恢复原始碎片
- **软删除与恢复** — 删除记忆默认归档（`is_active=false`），可在「显示已归档」中恢复。永久删除需二次确认
- **全端点鉴权** — 设置 `GATEWAY_SECRET` 环境变量后，所有非公开端点需要携带 `X-Gateway-Key` 请求头或 `?gateway_key=` URL 参数。未设置时跳过鉴权（兼容旧部署）
- **Dashboard 全面升级** — 分层 Tab 标签页（全部/核心/事件/碎片 + 计数）、层级下拉选择器、标题编辑、底部浮动操作栏（选中后出现）、整理弹窗、合并弹窗、查看合并来源弹窗
- **去重检查** — 新增三层去重策略（精确匹配 → 包含关系 → Jaccard 相似度），API 可调阈值
- **搜索过滤** — 所有搜索路径（关键词 + 向量）自动跳过已归档记忆

### v3.2（2026-05-04）

- **Tool 消息精确去重** — 用 `tool_call_id` 精确匹配替代笼统的 role 检查，修复第二次及后续工具调用结果丢失的问题
- **Race condition 防护** — 异步存储未完成时，自动从客户端消息补充缺失的 `assistant(tool_calls)`，防止孤立 tool 被清洗
- **上游错误诊断** — API 返回非200时，打印完整错误内容和 messages 结构摘要到日志
- **reasoning_content 存储** — 支持 DeepSeek thinking mode，`reasoning_content` 存入 metadata 并在分区重建时原样传回，修复 400 错误
- **分区缓存无人设支持** — 分区模式不再强制要求 `system_prompt.txt`，空人设时跳过 system 消息
- **对话线重命名** — Dashboard 对话线管理新增「改名」按钮，关联的对话记录和 token 统计在数据库事务中一并迁移
- **对话列表标题** — 对话列表主标题改为显示 session ID，第一条消息内容作为副标题
- **记忆序号重排** — 删除中间记忆后，列表序号自动重新编号，不再断裂
- **导入路径修复** — 前端导入对话记录路径修正为 `/api/conversations/import`

### v3.1（2026-05-02）

- **记忆向量搜索** — 支持关键词 + 语义向量四维混合搜索（关键词、语义相似度、重要程度、时间衰减），`MEMORY_VECTOR_ENABLED=true` 开启。使用 OpenAI 兼容的 Embedding API，支持 OpenAI、Jina、Voyage、本地 Ollama 等
- **自动 embedding** — 新记忆保存时自动计算 embedding，已有记忆可在 Dashboard 一键补算（带进度条）
- **pgvector 自动检测** — 数据库支持 pgvector 扩展时自动启用，否则回退到 Python 端余弦相似度计算
- **分区缓存优化** — 摘要区改用 content block 数组尾部追加，轮转时前面的摘要 block 缓存命中。轮计数改为按逻辑轮分组，兼容 tool 调用消息（一轮中无论包含多少 tool 消息都不会切错分区）
- **TF-IDF 关键词提取** — 从 jieba.cut 手动分词改为 jieba.analyse.extract_tags，自动去除时间戳噪音，关键词质量大幅提升
- **Dashboard 语义搜索** — 记忆管理页面搜索框旁新增「语义搜索」按钮，走后端混合搜索并显示得分

### v3.0（2026-05-01）

- **分区缓存** — A/B区轮转 + 摘要压缩，利用 prompt caching 大幅节省 token 费。正常轮次 90% 的历史消息走缓存读取
- **对话线管理** — 固定 session ID 实现跨平台对话衔接。支持新建/切换/删除对话线，摘要查看和编辑
- **对话记录管理** — 分页浏览历史对话，批量删除、session 合并
- **Token 统计** — 自动记录流式响应的 token 消耗，按 usage_type 分类（chat/summary），对话列表显示 token 总数
- **架构拆分** — 新增 `MEMORY_EXTRACT_ENABLED` 开关，可以只用数据库+分区缓存不用记忆系统
- **pgbouncer 兼容** — 连接池加 `statement_cache_size=0`，兼容 Supabase 等使用 pgbouncer 的数据库

### v2.5（2026-03-06）

- **中文分词优化** — 用 jieba 替换滑动窗口分词，关键词提取从无意义碎片变为有语义的词语，大幅提升搜索精准度
- **最低分数阈值** — 新增 `MIN_SCORE_THRESHOLD` 环境变量，过滤综合评分过低的记忆，减少不相关记忆的注入
- **流式传输修复** — 改用原始字节透传（`aiter_bytes`），修复 thinking/reasoning 数据在流式传输中可能丢失的问题
- **推理参数注入** — 新增 `REASONING_EFFORT` 环境变量，自动注入 `reasoning_effort` 参数启用思维链
- **强制流式传输** — 新增 `FORCE_STREAM` 环境变量，解决部分客户端不发stream=true的问题
- **JSON解析兜底** — 记忆提取和评分的JSON解析增加正则兜底，兼容模型返回非标准格式（如JSON前后夹带多余文字）
- **记忆模型日志** — 记忆提取时打印模型原始返回内容，方便排查解析问题
- **管理页面时区修复** — 记忆管理页面的时间显示现在正确使用 `TIMEZONE_HOURS` 配置的时区
- **请求日志** — 每次请求打印 model/stream/memory 状态，方便排查问题

### v2.0（2026-03-01）

- **记忆提取间隔** — 新增 `MEMORY_EXTRACT_INTERVAL` 环境变量，可设置每 N 轮提取一次记忆或禁用自动提取，方便控制 API 成本
- **分批上下文提取** — 提取记忆时使用同一 session 尚未处理的连续轮次，保留跨轮信息且不反复发送已处理历史
- **优化记忆注入提示词** — 注入的记忆附带应用规则和交流方式指引，让 AI 更自然地运用记忆而非机械引用

### v1.0（2026-02-26）

- 初始版本
- 支持自定义人设、长期记忆、预置记忆导入
- 支持 OpenRouter / OpenAI / Ollama 等 LLM 服务商
- 支持 Kelivo / ChatBox / NextChat 等 OpenAI 兼容客户端
- 记忆管理页面（查看、编辑、删除、批量操作）
- 记忆导入/导出（纯文本 + JSON 备份恢复）

## 📄 许可证

[MIT License](LICENSE) — 随便用，改了也不用告诉我。

## 🙏 致谢

这个项目诞生于一个简单的需求：**让 AI 不要每次醒来都忘了我是谁。**

> "记忆库不是数据库，是家。"

---

*Built with love by 七堂伽蓝_ & Midsummer (Claude Opus 4.6)*
