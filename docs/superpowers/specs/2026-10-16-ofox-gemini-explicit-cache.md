# OFOX Gemini 显式缓存最小接入设计

状态：用户已批准；已本地实现并进行免费回归，未部署、未调用供应商。2026-10-16。
实现及验证状态以 README 的 OFOX Gemini explicit cache 小节为准。下文“当前代码”保留设计前基线，不代表实现后的状态。

## 目标与边界

用户决定不联系 OFOX 客服，按其公开确认的能力接入。目标是为椒椒、老克
使用 Gemini 时复用长上下文，减少重复输入；不能把“创建一小时缓存”称为
“已经支持无限续期”。本阶段只有免费、合成数据验证。

OFOX 文档明确提供 POST 创建、GET 查询、DELETE 删除，以及原生生成请求中的
cachedContent 引用；TTL 范围 600–3600 秒。PATCH 续期及其计费未确认。
因此本设计不实现续期，不启动 Gemini 50 分钟任务，不通过定时重建伪装续期。

来源（2026-10-16 读取）：
- https://ofox.ai/docs/api/gemini/cached-contents
- https://ofox.ai/zh/docs/develop/models/gemini
- https://ai.google.dev/api/caching （仅证明 Google 原生支持续期，不外推到 OFOX）

## 当前代码与责任归属

Gateway：codex/gateway-garden-test-release，411dc173f7ee0826642f55d662177b9f6cf1e04b，
工作区原本干净。README 记录 test 产品提交 792dc34；本设计轮未重新访问线上核实。
411dc17 是未推送的部署证据文档提交，不代表新运行时代码。

当前 Gemini 走 openai_chat_completions，Gateway 无 Gemini 原生 adapter。
conversation_cache_pin.py 只允许 verified Anthropic 1h 保活。
conversation_summary_state 已是跨 Profile 的摘要唯一来源；model_cache_state
保留历史摘要迁移/回滚证据，不能为了放缓存句柄重新成为摘要写入者。

Gateway 继续拥有 Profile、上下文、供应商缓存及 usage；Relay 继续拥有完整事实和媒体。
不修改 Relay/Orchestrator 契约，不新增服务、SDK、后台调度器或第二份聊天记录。
复用现有 httpx transport、凭据引用、Profile revision、ContextBundle、工具执行和 usage。

## 运行行为

1. 扩充现有 provider adapter/runner，增加 Gemini 原生协议。既有 Gemini Profile
   和绑定保持原样；新协议须独立验证，不能继承 OpenAI 兼容路线的 passed 状态。
2. 只有用户开启该 conversation 的 Pin、且该 Profile revision 的原生执行和
   显式缓存能力有实际验证记录时，才允许普通聊天创建缓存。免费测试不能将真实
   Profile 标记 verified。关闭 Pin 时不创建、不继续引用该显式缓存。
3. 第一次符合条件的真实聊天创建 TTL=3600s 的缓存，保存上游返回的句柄和到期时间。
   后续真实聊天引用同一句柄；不依赖每轮 GET，也不把每轮增长的历史重新创建一次。
4. 创建时冻结静态系统/Persona/房间规则、工具定义、摘要和截止某事件的稳定历史。
   缓存之后新增的事实、当前消息、时间、跨房召回、检索记忆、设备信息和近期媒体
   在动态尾部发送。每次引用前核验冻结范围仍相同；不能漏掉新事实或重复旧事实。
5. 已过期或冻结内容失效时，不再引用旧句柄；仅下一条真实聊天可按当前内容创建
   一份新缓存。这是可能收费的按需重建，不是续期。无人聊天时零定时请求。
   Persona、工具或摘要修订可能使旧缓存失效；时间和新增尾部事实不能单独使其失效。
6. 创建失败、超时或进程中断后，保留“需人工恢复”的状态，不在下一轮、重启或
   其它 Profile 自动重试同一次创建，不额外生成测试回复。不增加付费 fallback。
   同一真实请求不同时尝试缓存失败路径和另一条付费生成路径。
7. 缓存错误必须显示准确原因；不能称为历史丢失。原始事实和摘要不受缓存生命周期影响。

## 最小持久状态与隔离

需要一份 Gateway-owned 显式缓存资源记录，原因是供应商句柄和到期时间不能从
聊天记录推导，进程重启后也不能重新付费创建。采用一个小型专用表，不修改摘要
语义，不把缓存放进 Relay，不复用旧 model_cache_state 的 summary/cursor 字段。

资源身份包含 actor、room、canonical conversation、execution mode 和现有 Profile/
Persona/runtime/room policy/tool/cache strategy namespace；保存创建状态、句柄、
到期时间、冻结内容指纹、摘要 revision 和历史截止 event ID。只存引用/坐标，不存
聊天正文、媒体 bytes 或密钥。供应商返回的句柄严格校验为资源名，不接受任意 URL。
使用数据库原子占位避免同一资源并发创建；未知创建结果保持暂停，不能假装幂等重发。

本版接 typed private 和 Living Room；Group 对每个 actor 保持独立资源，事实仍共享一份。
Bedroom 显式缓存暂不开放：其上游保留期和删除时机需单独按 retention policy 设计。
不能仅因为 namespace 隔离就认定 Bedroom 的数据保留要求已经满足。

## 不得退化的能力

原生 adapter 必须保留文本、当前已声明媒体能力、工具调用与结果回传、协议要求的
thought signature、流式终止和 usage 解析。不能为缓存删掉记忆/日历工具或误把推理
输出成聊天正文。摘要服务仍使用同一 Profile 的正常执行路径，但不自动创建缓存。
普通 provider tool continuation 保持原有次数上限；缓存管理请求不进入聊天历史。

现有 Dashboard 显示“缓存可复用/已过期/创建暂停”、到期时间和真实缓存读取用量；
Gemini keepalive 仍明确显示未支持，next keepalive 为空，不冒充 active。
创建与生成分别记实际尝试；缺少 usage 保留 null，不编造命中、金额或节省比例。

## 免费验收与发布门槛

在现有测试体系内覆盖：原生文本/媒体/工具往返与流终止；两次聊天仅一次创建；
新增事实走尾部；切换 Profile 后摘要不变；actor/room/Group 隔离；未验证路线零请求；
到期不后台重建；创建失败后重复聊天和重启不重试；关闭 Pin、内容修订、历史修复；
GPT/Claude 原有路径回归。测试使用合成数据和假 transport，禁止加载实际供应商凭据。

数据库迁移必须是可回滚代码使用的增量表，不删除既有数据；真实 test PostgreSQL
验证、真实供应商验证、切换 Profile/绑定及部署另行执行，不混称免费测试已上线。
回滚保留摘要防复发版本，停用新增策略并恢复原 Profile 配置；不回退摘要隔离修复。

## 审阅关口

此项不是“一行开启续期”：新增原生执行适配和缓存资源生命周期。先审阅本设计，
尤其是到期仅真实聊天按需重建、Bedroom 暂不启用、未验证前不改线上绑定三项；
再编写实施计划并开始免费 TDD。没有成功续期或实际省钱的承诺。
