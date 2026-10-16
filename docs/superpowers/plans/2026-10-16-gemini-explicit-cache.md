# Gemini 原生协议与显式缓存实施计划

Goal: 实施用户已批准的显式缓存设计，仅本地免费回归。
Architecture: Gateway 现有 runner 负责协议转换；小型资源表只保存供应商句柄与冻结范围，不保存聊天正文，不参与摘要写入。复用现有 Pin、Profile revision、transport、usage 和工具执行。
Tech Stack: Python / httpx / asyncpg / pytest；无新依赖。
Spec: ../specs/2026-10-16-ofox-gemini-explicit-cache.md
Global Constraints: 禁止真实供应商请求、线上绑定修改、部署和推送；Bedroom 不启用显式缓存；失败不重试或付费 fallback；原始事实和摘要不删除。

## Review Focus

1. Gemini thought signature 必须原样随工具往返，不能当正文展示：合成 SSE / tool continuation 回归。
2. 新增历史不能引发每轮重建或丢尾部：两次不同消息一次创建，检验请求正文边界。
3. 失败和进程中断不能导致重复付费：先原子占位；重建 service 后仍暂停；并发只准一人创建。
4. actor / room / Profile 与冻结范围必须隔离：同 conversation 不同 room/actor 不复用；历史修复/摘要变化使旧资源失效。
5. 免费验证不能授予真实能力：普通聊天要求当前 revision 验证证据；无 Pin / 未验证 / Bedroom 不创建；后台调度不调用 Gemini。

## Task 1 — 原生协议

Files: provider_adapters.py / gateway_provider_runner.py / model_profiles.py；tests/test_gemini_native.py。
Produces: gemini_native 协议，gemini_explicit_cache_v1 策略；原生渲染、SSE、usage 与工具续轮。
Consumes: ContextBundle、现有中性媒体 parts、现有工具库。
Steps: 先写合成文本/媒体/工具/签名/截断测试并观察 RED；实现最小转换；运行新测试与现有 runner/Profile 回归。
Contract: 无真实网络；旧协议输出不变；不输出 thought；必须检查流终止。

## Task 2 — 资源状态与成本门禁

Files: gemini_explicit_cache.py（唯一资源 owner）、database.py；tests/test_gemini_explicit_cache.py。
Produces: 内存/PG 同契约的原子占位、ready/paused 资源、冻结指纹与截止范围；TTL 固定 3600 秒。
Consumes: 已渲染原生请求，Pin 与 Profile probe 证据，现有 usage。
Steps: RED 覆盖复用/尾部/隔离/过期/失败/并发；实现持久占位后才 POST；错误保留暂停，绝不每轮 GET/PATCH；GREEN。真实 PostgreSQL 测试保持显式 opt-in，本轮不接线上 DB。
Contract: 缓存状态不写摘要或事实；未知创建结果不能自动重试；错误不能触发模型 fallback。

## Task 3 — 接入现有运行时与面板

Files: main.py、model_execution.py、conversation_cache_pin.py、templates/dashboard.html、static/js/dashboard.js；现有 API / Pin 回归。
Produces: 同一 Pin store 驱动显式缓存；摘要/探针不自动建缓存；现有 Dashboard 展示资源状态/expiry；Gemini next keepalive 为空。
Steps: RED 验证 runtime 接线与能力门禁；最小接入，不新增定时器/服务；GREEN，检查调用者和 fixture。
Contract: 不改任何线上 Profile/binding；当前兼容协议继续可用；创建/生成分别计量；显式错误停止当前路由链。

## Task 4 — 免费回归与交付

Files: README.md（更新真实状态）；以上测试。
Steps: 相关测试及全量 pytest；检查 diff / 秘密 / 临时文件；独立最终代码审阅，修复重要发现并重跑；本地提交。禁止 push/deploy。
Contract: 报告本地、提交、真实 DB、供应商和部署各自状态；未跑的项不标 PASS。

执行方式：用户明确要求连续施工，由当前 agent inline 执行；不再重复询问是否开工。任务结束统一汇报。

## 本地验证结果

四项任务已实现。最终全量免费回归：505 passed、62 skipped、2 subtests passed；真实 PostgreSQL 授权关闭。原生协议、缓存生命周期、HTTP 错误分类与恢复参数均经过 RED/GREEN；真实 DOM 验证费用提示、资源状态及无写入。独立审阅发现的旧费用/续期文案和错误原因丢失已修复并复核。

未推送、未部署、未付费调用、未改线上绑定。真实 PostgreSQL 迁移/重启和 OFOX 实际兼容/命中仍待另行验证；不将内存 store 重建等同于数据库重启。复用现有工作区，无新增依赖、服务或测试工程；完整状态见 README。
