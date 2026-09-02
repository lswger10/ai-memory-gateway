"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
让你的 AI 拥有长期记忆。

工作原理：
1. 接收客户端（Kelivo / ChatBox / 任何 OpenAI 兼容客户端）的消息
2. 自动搜索数据库中的相关记忆，注入 system prompt
3. 转发给 LLM API（支持 OpenRouter / OpenAI / 任何兼容接口）
4. 后台自动存储对话 + 用 AI 提取新记忆

环境变量 MEMORY_ENABLED=false 时退化为纯转发网关（第一阶段）。
"""

import os
import json
import re
import base64
import hashlib
import uuid
import asyncio
import secrets
import httpx
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import init_tables, close_pool, save_message, search_legacy_memories as search_memories, save_memory, get_all_memories_count, get_recent_memories, get_all_memories, get_pool, get_all_memories_detail, update_memory, delete_memory, delete_memories_batch, get_gateway_config, set_gateway_config, get_all_gateway_config, get_conversation_messages, get_session_cache_state, save_session_cache_state, delete_session_cache_state, save_token_usage, ensure_token_usage_table, get_conversations_paginated, delete_conversation, batch_delete_conversations, merge_sessions_to_target, list_all_session_cache_states, export_all_conversations, import_conversations, get_last_user_content, update_last_assistant_message, db_row_to_message, backfill_memory_embeddings, get_pending_memory_embedding_count, search_conversations, update_message_content, delete_single_message, rename_session_id, get_fragments_by_date, get_fragments_by_date_range, create_event_memory, deactivate_memories, promote_to_core, merge_memories, check_duplicate_memory, update_memory_with_layer, get_layer_statistics, cleanup_old_fragments, revert_merge, ensure_memory_extraction_cursor, get_memory_extraction_messages, save_memory_extraction_cursor, list_cold_archive_for_management, append_cold_archive_annotation
import database as _db_module  # 用于 memory settings 热更新 database.py 全局变量
from group_contracts import (
    CONTRACT_VERSION,
    ContractError,
    ClosedBurstExtractionRequest,
    ContextPackRequest,
    MemoryCandidateRequest,
)
from group_memory import (
    CandidateIngressService,
    ClosedBurstExtractionService,
    GroupBatchExtractionPipeline,
    GroupContextPackService,
    SensitiveCandidateError,
    StaleCandidateError,
    UnstableBurstError,
    build_synthetic_scoped_search,
)
from memory_extractor import (
    extract_group_memories,
    score_memories,
)
from memory_policy import group_memory_features_from_env
from relay_group_client import RelayGroupClient, RelayGroupError
from bedroom_memory import (
    BEDROOM_CONTRACT_VERSION,
    BedroomContractError,
    BedroomContextPackService,
    BedroomPackRequest,
    BedroomPostgresRepository,
    BedroomRetentionService,
)
from model_execution import GatewayModelExecutionService
from execution_context_builder import GatewayExecutionContextBuilder
from gateway_provider_runner import GatewayProviderRunner
from media_materialization import RelayMediaReader
from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
from cache_dashboard import build_cache_observability_summary, build_cache_usage_view
from anchored_history import InMemoryAnchoredHistoryStore, PostgresAnchoredHistoryStore
from conversation_partitions import (
    InMemoryConversationPartitionStore,
    PostgresConversationPartitionStore,
)
from cache_probe import GatewayCacheProbeService
from model_profile_store import InMemoryModelProfileStore
from model_usage_store import InMemoryModelUsageStore
from model_runtime_fixture import bootstrap_ephemeral_model_profiles
from model_execution_contracts import (
    CONTRACT_VERSION as MODEL_EXECUTION_CONTRACT_VERSION,
    ExecutionContractError,
    GatewayExecutionRequest,
)
from model_profiles import ModelProfile, ProfileContractError, resolve_feature_flags
from model_profile_store import ProfileStoreError
from memory_policy import room_members
from actor_prompt_profiles import load_actor_prompt_profiles
from actor_prompt_store import (
    ActiveActorPromptMapping,
    ActorPromptRevisionConflict,
    ActorPromptStoreError,
    InMemoryActorPromptVersionStore,
    PostgresActorPromptVersionStore,
)
from conversation_cache_pin import (
    CachePinError,
    CachePinService,
    InMemoryConversationCachePinStore,
    PostgresConversationCachePinStore,
)

# ============================================================
# 配置项 —— 全部从环境变量读取，部署时在云平台面板里设置
# ============================================================

# 网关端口
PORT = int(os.getenv("PORT", "8080"))

# 网关访问密钥（强烈建议设置！）
# 设置后所有非公开端点都需要鉴权，二选一：
#   - 请求头方式：X-Gateway-Key: 你的密钥（客户端/API 调用）
#   - URL参数方式：?gateway_key=你的密钥（方便浏览器访问 dashboard）
# 不设置则跳过鉴权（兼容旧部署，仅建议内网环境使用）
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")

# Narrow credential used only by Relay's actor Persona proxy. It is deliberately
# separate from GATEWAY_SECRET so possession never grants general management
# access.
ACTOR_PERSONA_PROXY_SECRET = os.getenv("ACTOR_PERSONA_PROXY_SECRET", "")

# 记忆系统开关（数据库出问题时可以临时关掉）
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"

# 每次注入的最大记忆条数
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))

# 自动记忆提取间隔（0 = 禁用自动批量，1 = 每轮，N = 每个 session 各自每 N 轮）
# 用户显式要求保存为长期记忆时不受这个间隔限制。
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))

# 记忆提取+注入总开关（false时数据库仍连接、消息仍存储，但不提取也不注入记忆）
MEMORY_EXTRACT_ENABLED = os.getenv("MEMORY_EXTRACT_ENABLED", "true").lower() == "true"

# 时区偏移（小时），用于记忆注入时的日期显示，默认 UTC+8
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))

# 记忆整理使用独立、显式的配置，不再回退到已退休的全局 AI 路线。
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")
MEMORY_API_BASE_URL = os.getenv("MEMORY_API_BASE_URL", "")
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "")

def get_memory_api_key() -> str:
    return MEMORY_API_KEY

# ============================================================
# 应用生命周期管理
# ============================================================

async def _group_extraction_worker() -> None:
    """Resume durable closed-burst work; failed units remain queued."""
    interval = max(1.0, float(os.environ.get("GROUP_EXTRACTION_POLL_SECONDS", "30")))
    while True:
        try:
            await _get_group_extraction_service().process_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[warning] Group memory extraction deferred: {type(exc).__name__}")
        await asyncio.sleep(interval)


async def _conversation_cache_pin_worker() -> None:
    poll_seconds = max(
        10.0, float(os.environ.get("CONVERSATION_CACHE_PIN_POLL_SECONDS", "60"))
    )
    while True:
        try:
            await (await _get_cache_pin_service()).run_due_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[warning] Conversation Cache Pin pass failed: {type(exc).__name__}")
        await asyncio.sleep(poll_seconds)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库，关闭时断开连接"""
    group_worker_task = None
    cache_pin_worker_task = None
    if MEMORY_ENABLED:
        try:
            await init_tables()
            await _ensure_actor_prompt_store()
            await ensure_token_usage_table()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
            
            # 从数据库恢复面板配置（重启后保持Dashboard修改过的值）
            try:
                db_cfg = await get_all_gateway_config()
                if db_cfg:
                    _RESTORE_MAIN = {
                        "MEMORY_ENABLED": lambda v: _parse_bool(v),
                        "MAX_MEMORIES_INJECT": int, "MEMORY_EXTRACT_INTERVAL": int,
                        "MEMORY_API_KEY": str,
                        "MEMORY_API_BASE_URL": str,
                        "MEMORY_MODEL": str,
                    }
                    _RESTORE_DB = {
                        "EMBEDDING_API_KEY": str, "EMBEDDING_BASE_URL": str,
                        "EMBEDDING_MODEL": str, "EMBEDDING_DIM": int,
                        "MIN_SCORE_THRESHOLD": float,
                        "MEMORY_VECTOR_ENABLED": lambda v: _parse_bool(v),
                        "MEMORY_HW_KEYWORD": float, "MEMORY_HW_SEMANTIC": float,
                        "MEMORY_HW_IMPORTANCE": float, "MEMORY_HW_RECENCY": float,
                        "MEMORY_SEMANTIC_THRESHOLD": float,
                    }
                    _ALLOW_EMPTY = {
                        "MEMORY_API_KEY", "MEMORY_API_BASE_URL", "MEMORY_MODEL"
                    }
                    restored = []
                    for key, val in db_cfg.items():
                        if not val:
                            if key in _ALLOW_EMPTY and key in _RESTORE_MAIN:
                                globals()[key] = _RESTORE_MAIN[key]("")
                                restored.append(key + "(显式空)")
                            continue
                        if key in _RESTORE_MAIN:
                            globals()[key] = _RESTORE_MAIN[key](val)
                            restored.append(key)
                        elif key in _RESTORE_DB:
                            setattr(_db_module, key, _RESTORE_DB[key](val))
                            restored.append(key)
                        if key in {"MEMORY_API_KEY", "MEMORY_API_BASE_URL", "MEMORY_MODEL"} and key in _RESTORE_MAIN:
                            import memory_extractor as _me_mod
                            setattr(_me_mod, key, str(val))
                    if restored:
                        print(f"🔄 从数据库恢复 {len(restored)} 项面板配置: {', '.join(restored)}")
            except Exception as e:
                print(f"[warning] 恢复面板配置失败: {e}")
            
            if not MEMORY_EXTRACT_ENABLED:
                print(f"ℹ️  记忆提取+注入已关闭（MEMORY_EXTRACT_ENABLED=false）")
            
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
            print("⚠️  记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️  记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")
    
    features = group_memory_features_from_env()
    if MEMORY_ENABLED and features["group_memory"] and features["burst_extraction"]:
        group_worker_task = asyncio.create_task(_group_extraction_worker())
    if MEMORY_ENABLED and resolve_feature_flags()["model_execution"]:
        cache_pin_worker_task = asyncio.create_task(_conversation_cache_pin_worker())

    yield

    if group_worker_task is not None:
        group_worker_task.cancel()
        try:
            await group_worker_task
        except asyncio.CancelledError:
            pass

    if cache_pin_worker_task is not None:
        cache_pin_worker_task.cancel()
        try:
            await cache_pin_worker_task
        except asyncio.CancelledError:
            pass

    if _model_provider_runner is not None:
        await _model_provider_runner.transport.close()
    
    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)
_group_context_service = None
_group_candidate_service = None
_group_extraction_service = None
_bedroom_context_service = None
_bedroom_retention_service = None
_model_execution_service: GatewayModelExecutionService | None = None
_model_profile_store = None
_model_usage_store = None
_model_provider_runner: GatewayProviderRunner | None = None
_model_context_builder: GatewayExecutionContextBuilder | None = None
_cache_probe_service: GatewayCacheProbeService | None = None
_cache_pin_service: CachePinService | None = None
_actor_prompt_store = None
_actor_prompt_mapping = None
_model_runtime_lock = asyncio.Lock()

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except Exception:
            pass
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v3",
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "model_execution_enabled": resolve_feature_flags()["model_execution"],
        "configuration_authority": "model_profiles_and_actor_personas",
    }


# ============================================================
# 网关鉴权中间件
# ============================================================

# 不需要鉴权的路径（根路径精确匹配，其余按前缀匹配）
PUBLIC_PATHS = ("/", "/static/", "/health", "/favicon.ico")


def actor_persona_proxy_path_allowed(path: str, method: str) -> bool:
    """Return whether the narrow Relay credential may access this request."""
    if method == "GET" and path == "/api/actor-prompts":
        return True
    if method == "POST" and re.fullmatch(
        r"/api/actor-prompts/(?:jiao|laoke)/(?:versions|activate)", path
    ):
        return True
    return bool(
        method == "GET"
        and re.fullmatch(
            r"/api/actor-prompts/(?:jiao|laoke)/versions/"
            r"[A-Za-z0-9:._-]+/export",
            path,
        )
    )

@app.middleware("http")
async def gateway_auth_middleware(request: Request, call_next):
    """检查 GATEWAY_SECRET，保护所有非公开端点"""
    path = request.url.path
    if (
        path.startswith("/internal/group/context-packs/")
        or path == "/internal/group/memory-candidates"
        or path == "/internal/group/extraction/closed-bursts"
        or path.startswith("/internal/bedroom/")
        or path.startswith("/internal/model-execution/")
    ):
        return await call_next(request)

    # 未设置任何密钥时跳过鉴权（兼容旧部署，但会打印警告）
    if not GATEWAY_SECRET and not ACTOR_PERSONA_PROXY_SECRET:
        if not hasattr(gateway_auth_middleware, "_warned"):
            print("⚠️  GATEWAY_SECRET 未设置！所有 API 端点不受保护！")
            print("⚠️  请在环境变量中设置 GATEWAY_SECRET 以启用鉴权")
            gateway_auth_middleware._warned = True
        return await call_next(request)

    # 公开路径不需要鉴权（根路径精确匹配）
    if path == "/":
        return await call_next(request)
    for prefix in PUBLIC_PATHS[1:]:
        if path.startswith(prefix):
            return await call_next(request)

    # OPTIONS 预检请求放行（CORS 需要）
    if request.method == "OPTIONS":
        return await call_next(request)

    # 从 header 或 query 参数获取密钥
    provided_key = (
        request.headers.get("X-Gateway-Key", "")
        or request.query_params.get("gateway_key", "")
    )
    persona_key = request.headers.get("X-Gateway-Persona-Key", "")

    # The full administrator credential preserves existing behavior. The
    # Persona credential is method-and-path scoped and never accepted through
    # the query string.
    if GATEWAY_SECRET and secrets.compare_digest(provided_key, GATEWAY_SECRET):
        return await call_next(request)
    if (
        ACTOR_PERSONA_PROXY_SECRET
        and secrets.compare_digest(persona_key, ACTOR_PERSONA_PROXY_SECRET)
        and actor_persona_proxy_path_allowed(path, request.method)
    ):
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized Gateway credential or scope."},
    )


def _group_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "contract_version": CONTRACT_VERSION,
            "error": {"code": code, "message": message},
        },
    )


def _group_bearer(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    prefix = "Bearer "
    return authorization[len(prefix):] if authorization.startswith(prefix) else ""


def _model_execution_error(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "contract_version": MODEL_EXECUTION_CONTRACT_VERSION,
            "error": {"code": code},
        },
    )


async def _stream_gateway_execution(request: Request, expected_kind: str):
    if not resolve_feature_flags()["model_execution"]:
        return _model_execution_error(404, "model_execution_disabled")
    if (
        request.headers.get("X-Gateway-Execution-Version")
        != MODEL_EXECUTION_CONTRACT_VERSION
    ):
        return _model_execution_error(409, "contract_version_mismatch")
    expected_key = os.environ.get("GROUP_ORCHESTRATOR_SERVICE_KEY", "")
    if not expected_key or not secrets.compare_digest(
        _group_bearer(request), expected_key
    ):
        return _model_execution_error(403, "principal_not_allowed")
    try:
        execution_request = GatewayExecutionRequest.from_dict(await request.json())
    except (ExecutionContractError, ValueError, TypeError, json.JSONDecodeError):
        return _model_execution_error(422, "invalid_execution_payload")
    if execution_request.execution_kind != expected_kind:
        return _model_execution_error(422, "execution_kind_mismatch")
    try:
        await _refresh_actor_prompt_store()
        service = await _get_model_execution_service()
    except Exception:
        return _model_execution_error(503, "model_execution_unavailable")

    async def events():
        async for event in service.stream(execution_request):
            yield f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def _get_model_execution_service() -> GatewayModelExecutionService:
    global _model_execution_service, _model_profile_store, _model_usage_store
    global _model_provider_runner, _model_context_builder
    if _model_execution_service is not None:
        return _model_execution_service
    async with _model_runtime_lock:
        if _model_execution_service is not None:
            return _model_execution_service
        ephemeral_fixture = os.environ.get(
            "MODEL_EXECUTION_EPHEMERAL_FIXTURE", ""
        ).strip()
        if ephemeral_fixture:
            _model_profile_store = InMemoryModelProfileStore()
            _model_usage_store = InMemoryModelUsageStore()
            await bootstrap_ephemeral_model_profiles(
                _model_profile_store, Path(ephemeral_fixture)
            )
            history_store = InMemoryAnchoredHistoryStore()
            conversation_store = InMemoryConversationPartitionStore()
        else:
            _model_profile_store = PostgresModelProfileStore(get_pool)
            _model_usage_store = PostgresModelUsageStore(get_pool)
            history_store = PostgresAnchoredHistoryStore(get_pool)
            conversation_store = PostgresConversationPartitionStore(get_pool)
        relay_url = os.environ.get("GROUP_RELAY_BASE_URL", "").strip()
        relay_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "").strip()
        _model_provider_runner = GatewayProviderRunner(
            media_reader=(
                RelayMediaReader(relay_url, relay_key)
                if relay_url and relay_key
                else None
            )
        )
        _model_context_builder = GatewayExecutionContextBuilder(
            group_context=_get_group_context_service(),
            bedroom_context=_get_bedroom_context_service(),
            history_store=history_store,
            conversation_store=conversation_store,
        )
        _model_execution_service = GatewayModelExecutionService(
            profiles=_model_profile_store,
            context_builder=_model_context_builder,
            provider_runner=_model_provider_runner,
            usage_store=_model_usage_store,
        )
        return _model_execution_service


async def _get_cache_pin_service() -> CachePinService:
    global _cache_pin_service
    if _cache_pin_service is not None:
        return _cache_pin_service
    await _get_model_execution_service()
    assert _model_profile_store is not None
    assert _model_provider_runner is not None
    assert _model_context_builder is not None
    fixture = bool(os.environ.get("MODEL_EXECUTION_EPHEMERAL_FIXTURE", "").strip())
    store = (
        InMemoryConversationCachePinStore()
        if fixture
        else PostgresConversationCachePinStore(get_pool)
    )
    interval = timedelta(
        seconds=max(
            60,
            int(os.environ.get("CONVERSATION_CACHE_PIN_INTERVAL_SECONDS", "3000")),
        )
    )
    _cache_pin_service = CachePinService(
        store=store,
        profiles=_model_profile_store,
        context_builder=_model_context_builder,
        provider_runner=_model_provider_runner,
        usage_store=_model_usage_store,
        interval=interval,
    )
    return _cache_pin_service


@app.post("/internal/model-execution/probe")
async def model_execution_probe(request: Request):
    return await _stream_gateway_execution(request, "probe")


@app.post("/internal/model-execution/stream")
async def model_execution_stream(request: Request):
    return await _stream_gateway_execution(request, "full")


def _safe_profile(profile: ModelProfile) -> dict:
    credential_env = (
        profile.credential_ref.removeprefix("env:")
        if profile.credential_ref.startswith("env:")
        else ""
    )
    return {
        "profile_id": profile.profile_id,
        "display_name": profile.display_name,
        "enabled": profile.enabled,
        "test_status": profile.test_status,
        "provider": profile.provider,
        "protocol": profile.protocol,
        "base_url": profile.base_url,
        "route_id": profile.route_id,
        "model": profile.model,
        "adapter_version": profile.adapter_version,
        "credential_configured": bool(credential_env and os.environ.get(credential_env)),
        "header_names": [name for name, _ in profile.headers],
        "capabilities": profile.capabilities.to_dict(),
        "cache_strategy": profile.cache_strategy,
        "requested_cache_ttl": profile.requested_cache_ttl,
        "revision": profile.revision,
    }


def _require_model_management() -> None:
    if not resolve_feature_flags()["model_profile_management"]:
        raise HTTPException(status_code=404, detail="model_profile_management_disabled")


@app.get("/api/model-profiles")
async def list_model_profiles():
    _require_model_management()
    await _get_model_execution_service()
    assert _model_profile_store is not None
    profiles = await _model_profile_store.list_profiles()
    return {"profiles": [_safe_profile(profile) for profile in profiles]}


@app.put("/api/model-profiles")
async def put_model_profile(request: Request):
    _require_model_management()
    try:
        profile = ModelProfile.from_dict(await request.json())
        # A management write declares a route; it is not evidence that the
        # route or its cache semantics were actually observed.  Only the
        # explicit bounded probe may promote test_status later.
        profile = replace(profile, test_status="unverified")
        await _get_model_execution_service()
        assert _model_profile_store is not None
        stored = await _model_profile_store.put_profile(profile)
    except (ProfileContractError, ProfileStoreError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid_model_profile") from exc
    return _safe_profile(stored)


@app.get("/api/model-bindings")
async def list_model_bindings():
    _require_model_management()
    await _get_model_execution_service()
    assert _model_profile_store is not None
    result = {}
    for actor_id in ("jiao", "laoke"):
        actor_rooms = [
            room_id for room_id in (
                "room_weiwei_jiao", "room_weiwei_laoke", "room_group_home"
            ) if actor_id in room_members(room_id)
        ]
        resolved = {}
        for room_id in actor_rooms:
            try:
                item = await _model_profile_store.resolve(actor_id, room_id)
            except ProfileStoreError:
                continue
            resolved[room_id] = {
                "profile_id": item.primary.profile_id,
                "source": item.source,
                "binding_revision": item.binding_revision,
                "approved_fallback_profile_ids": [p.profile_id for p in item.fallbacks],
            }
        result[actor_id] = resolved
    return {"bindings": result}


def _cache_pin_public(pin) -> dict:
    def timestamp(value):
        return value.isoformat() if value is not None else None

    return {
        "pin_id": pin.pin_id,
        "room_id": pin.room_id,
        "conversation_id": pin.conversation_id,
        "execution_mode": pin.execution_mode,
        "bedroom_session_id": pin.bedroom_session_id,
        "enabled": pin.enabled,
        "status": pin.status,
        "actors": {
            actor_id: {
                "status": state.status,
                "profile_id": state.profile_id,
                "last_keepalive": timestamp(state.last_keepalive_at),
                "next_keepalive": timestamp(state.next_keepalive_at),
                "call_count": state.call_count,
                "cache_read_input_tokens": state.cache_read_input_tokens,
                "last_error": state.last_error,
            }
            for actor_id, state in sorted(pin.actors.items())
        },
    }


@app.get("/api/cache-pins")
async def list_conversation_cache_pins():
    _require_model_management()
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="memory_storage_required")
    pins = await (await _get_cache_pin_service()).list_pins()
    return {"pins": [_cache_pin_public(pin) for pin in pins]}


@app.put("/api/cache-pins")
async def update_conversation_cache_pin(request: Request):
    _require_model_management()
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="memory_storage_required")
    try:
        payload = await request.json()
        allowed = {
            "room_id",
            "conversation_id",
            "execution_mode",
            "bedroom_session_id",
            "actor_id",
            "enabled",
        }
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise CachePinError("invalid cache pin payload")
        for field in ("room_id", "conversation_id", "execution_mode"):
            if not isinstance(payload.get(field), str) or not payload[field].strip():
                raise CachePinError(f"{field} is required")
        if not isinstance(payload.get("enabled"), bool):
            raise CachePinError("enabled must be boolean")
        pin = await (await _get_cache_pin_service()).set_pin(
            room_id=payload["room_id"],
            conversation_id=payload["conversation_id"],
            execution_mode=payload["execution_mode"],
            bedroom_session_id=payload.get("bedroom_session_id"),
            actor_id=payload.get("actor_id"),
            enabled=payload["enabled"],
        )
    except (CachePinError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="invalid_cache_pin") from exc
    return _cache_pin_public(pin)


@app.put("/api/model-bindings")
async def update_model_binding(request: Request):
    _require_model_management()
    await _get_model_execution_service()
    assert _model_profile_store is not None
    try:
        body = await request.json()
        action = body.get("action")
        actor_id = str(body.get("actor_id") or "")
        if actor_id not in {"jiao", "laoke"}:
            raise ValueError
        expected = body.get("expected_revision")
        if action == "set_actor_default":
            result = await _model_profile_store.set_actor_default(
                actor_id, str(body["profile_id"]), expected_revision=expected
            )
        elif action == "set_fallbacks":
            values = body.get("profile_ids")
            if not isinstance(values, list):
                raise ValueError
            result = await _model_profile_store.set_approved_fallbacks(
                actor_id, tuple(str(item) for item in values), expected_revision=expected
            )
        elif action == "set_room_override":
            room_id = str(body["room_id"])
            if actor_id not in room_members(room_id):
                raise ValueError
            result = await _model_profile_store.set_room_override(
                room_id, actor_id, str(body["profile_id"]), expected_revision=expected
            )
        else:
            raise ValueError
    except (KeyError, TypeError, ValueError, ProfileStoreError) as exc:
        raise HTTPException(status_code=409, detail="model_binding_rejected") from exc
    return {"accepted": True, "revision": result.revision}


@app.get("/api/model-usage/summary")
async def model_usage_summary():
    _require_model_management()
    await _get_model_execution_service()
    assert _model_usage_store is not None
    receipts = await _model_usage_store.list_receipts(limit=200)
    return {
        "cache_view": list(build_cache_usage_view(receipts)),
        "cache_observability": build_cache_observability_summary(receipts),
        "receipts": [
            {
                "generation_request_id": row.generation_request_id,
                "actor_id": row.actor_id,
                "room_id": row.room_id,
                "profile_id": row.profile_id,
                "provider": row.provider,
                "protocol": row.protocol,
                "route_id": row.route_id,
                "model": row.model,
                "cache_strategy": row.cache_strategy,
                "requested_cache_ttl": row.requested_cache_ttl,
                "observed_cache_support": row.observed_cache_support,
                "fallback_used": row.fallback_used,
                "fallback_from_profile_id": row.fallback_from_profile_id,
                "stable_prefix_hash": row.stable_prefix_hash,
                "prompt_cache_key": row.prompt_cache_key,
                "runtime_kernel_version": row.runtime_kernel_version,
                "persona_version": row.persona_version,
                "room_policy_version": row.room_policy_version,
                "tool_schema_hash": row.tool_schema_hash,
                "summary_version": row.summary_version,
                "compressed_up_to_event_id": row.compressed_up_to_event_id,
                "provider_usage_received": row.provider_usage_received,
                "usage": {
                    "input_tokens": row.usage.input_tokens,
                    "output_tokens": row.usage.output_tokens,
                    "cache_creation_input_tokens": row.usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": row.usage.cache_read_input_tokens,
                    "cached_tokens": row.usage.cached_tokens,
                },
            }
            for row in receipts
        ]
    }


def _usage_dict(usage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cached_tokens": usage.cached_tokens,
    }


def _actor_persona_management_enabled() -> bool:
    return os.environ.get("ACTOR_PERSONA_MANAGEMENT_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _require_actor_persona_management() -> None:
    if not _actor_persona_management_enabled():
        raise HTTPException(status_code=404, detail="actor_persona_management_disabled")
    if not GATEWAY_SECRET:
        raise HTTPException(status_code=503, detail="actor_persona_auth_not_configured")
    if not _db_module.DATABASE_URL:
        raise HTTPException(status_code=503, detail="actor_persona_storage_not_configured")


async def _ensure_actor_prompt_store():
    global _actor_prompt_store, _actor_prompt_mapping
    if _actor_prompt_store is None:
        builtins = load_actor_prompt_profiles()
        _actor_prompt_store = (
            PostgresActorPromptVersionStore(get_pool, builtins)
            if _db_module.DATABASE_URL
            else InMemoryActorPromptVersionStore(builtins)
        )
    await _actor_prompt_store.initialize()
    if (
        _actor_prompt_mapping is None
        or getattr(_actor_prompt_mapping, "store", None) is not _actor_prompt_store
    ):
        _actor_prompt_mapping = ActiveActorPromptMapping(_actor_prompt_store)
    return _actor_prompt_store


async def _refresh_actor_prompt_store():
    store = await _ensure_actor_prompt_store()
    await store.refresh_active()
    return store


def _active_actor_prompt_mapping():
    global _actor_prompt_store, _actor_prompt_mapping
    if _actor_prompt_store is None:
        builtins = load_actor_prompt_profiles()
        _actor_prompt_store = (
            PostgresActorPromptVersionStore(get_pool, builtins)
            if _db_module.DATABASE_URL
            else InMemoryActorPromptVersionStore(builtins)
        )
    if (
        _actor_prompt_mapping is None
        or getattr(_actor_prompt_mapping, "store", None) is not _actor_prompt_store
    ):
        _actor_prompt_mapping = ActiveActorPromptMapping(_actor_prompt_store)
    return _actor_prompt_mapping


def _actor_prompt_version_public(version, active_version_id: str) -> dict:
    return version.to_public_dict(active=version.version_id == active_version_id)


@app.get("/api/actor-prompts")
async def list_actor_prompts():
    _require_actor_persona_management()
    store = await _refresh_actor_prompt_store()
    actors = {}
    for actor_id in ("jiao", "laoke"):
        active = await store.get_active_state(actor_id)
        versions = await store.list_versions(actor_id)
        actors[actor_id] = {
            "actor_id": actor_id,
            "active_version_id": active.version_id,
            "revision": active.revision,
            "versions": [
                _actor_prompt_version_public(version, active.version_id)
                for version in versions
            ],
        }
    return {"actors": actors}


@app.post("/api/actor-prompts/{actor_id}/versions")
async def create_actor_prompt_version(actor_id: str, request: Request):
    _require_actor_persona_management()
    if actor_id not in {"jiao", "laoke"}:
        raise HTTPException(status_code=404, detail="actor_not_found")
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {
            "source_filename", "prompt_base64", "content_sha256"
        }:
            raise ValueError
        source_filename = body["source_filename"]
        prompt_base64 = body["prompt_base64"]
        expected_sha = body["content_sha256"]
        if (
            not isinstance(source_filename, str)
            or source_filename != Path(source_filename).name
            or "/" in source_filename
            or "\\" in source_filename
            or len(source_filename) > 160
            or not source_filename.lower().endswith(".md")
            or not isinstance(prompt_base64, str)
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ValueError
        maximum = max(1, int(os.environ.get("ACTOR_PROMPT_MAX_BYTES", "262144")))
        prompt_bytes = base64.b64decode(prompt_base64, validate=True)
        if len(prompt_bytes) > maximum:
            raise HTTPException(status_code=413, detail="actor_prompt_too_large")
        if hashlib.sha256(prompt_bytes).hexdigest() != expected_sha.lower():
            raise ValueError
        prompt_text = prompt_bytes.decode("utf-8", "strict")
        if not prompt_text.strip() or "\x00" in prompt_text:
            raise ValueError
        store = await _ensure_actor_prompt_store()
        version = await store.create_version(actor_id, source_filename, prompt_text)
    except HTTPException:
        raise
    except (ActorPromptStoreError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid_actor_prompt") from exc
    active = await store.get_active_state(actor_id)
    return {"version": _actor_prompt_version_public(version, active.version_id)}


@app.post("/api/actor-prompts/{actor_id}/activate")
async def activate_actor_prompt_version(actor_id: str, request: Request):
    _require_actor_persona_management()
    if actor_id not in {"jiao", "laoke"}:
        raise HTTPException(status_code=404, detail="actor_not_found")
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {"version_id", "expected_revision"}:
            raise ValueError
        version_id = body["version_id"]
        expected = body["expected_revision"]
        if (
            not isinstance(version_id, str)
            or not version_id
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 0
        ):
            raise ValueError
        store = await _ensure_actor_prompt_store()
        active = await store.activate(actor_id, version_id, expected_revision=expected)
    except ActorPromptRevisionConflict as exc:
        raise HTTPException(status_code=409, detail="actor_prompt_revision_conflict") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="actor_prompt_version_not_found") from exc
    except (ActorPromptStoreError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid_actor_prompt_activation") from exc
    return {"accepted": True, "active": active.to_public_dict()}


@app.get("/api/actor-prompts/{actor_id}/versions/{version_id}/export")
async def export_actor_prompt_version(actor_id: str, version_id: str):
    _require_actor_persona_management()
    if actor_id not in {"jiao", "laoke"}:
        raise HTTPException(status_code=404, detail="actor_not_found")
    try:
        store = await _ensure_actor_prompt_store()
        prompt_text = await store.export_text(actor_id, version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="actor_prompt_version_not_found") from exc
    filename = f"{actor_id}-{version_id.split(':')[-1][:16]}.md"
    return Response(
        content=prompt_text.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/cache-probes")
async def run_cache_probe(request: Request):
    global _cache_probe_service
    _require_model_management()
    try:
        body = await request.json()
        if body.get("confirm_provider_charges") is not True:
            raise HTTPException(status_code=409, detail="provider_charge_confirmation_required")
        actor_id = str(body["actor_id"])
        room_id = str(body["room_id"])
        if actor_id not in room_members(room_id):
            raise ValueError
        values = {
            "profile_id": str(body["profile_id"]),
            "actor_id": actor_id,
            "room_id": room_id,
            "conversation_id": str(body["conversation_id"]),
        }
        if _cache_probe_service is None:
            await _get_model_execution_service()
            assert _model_profile_store is not None
            assert _model_provider_runner is not None
            _cache_probe_service = GatewayCacheProbeService(
                profiles=_model_profile_store,
                provider_runner=_model_provider_runner,
            )
        result = await _cache_probe_service.run(**values)
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError, ProfileStoreError) as exc:
        raise HTTPException(status_code=422, detail="invalid_cache_probe") from exc
    return {
        "status": result.status,
        "first": _usage_dict(result.first),
        "second": _usage_dict(result.second),
    }


def _get_group_context_service() -> GroupContextPackService:
    global _group_context_service
    if _group_context_service is None:
        relay_url = os.environ.get("GROUP_RELAY_BASE_URL", "").strip()
        relay_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "").strip()
        if not relay_url or not relay_key:
            raise RelayGroupError(503, "dependency_unavailable")
        synthetic_fixture = os.environ.get(
            "GROUP_SYNTHETIC_MEMORY_FIXTURE", ""
        ).strip()
        search = None
        if synthetic_fixture:
            rows = json.loads(Path(synthetic_fixture).read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("synthetic memory fixture must be a JSON array")
            search = build_synthetic_scoped_search(rows)
        _group_context_service = GroupContextPackService(
            RelayGroupClient(relay_url, relay_key),
            prompt_profiles=_active_actor_prompt_mapping(),
            **({"search": search} if search is not None else {}),
        )
    return _group_context_service


def _get_bedroom_context_service() -> BedroomContextPackService:
    global _bedroom_context_service
    if _bedroom_context_service is None:
        relay_url = os.environ.get("GROUP_RELAY_BASE_URL", "").strip()
        relay_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "").strip()
        if not relay_url or not relay_key:
            raise RelayGroupError(503, "dependency_unavailable")
        _bedroom_context_service = BedroomContextPackService(
            RelayGroupClient(relay_url, relay_key),
            prompt_profiles=_active_actor_prompt_mapping(),
        )
    return _bedroom_context_service


def _get_bedroom_retention_service() -> BedroomRetentionService:
    global _bedroom_retention_service
    if _bedroom_retention_service is None:
        _bedroom_retention_service = BedroomRetentionService(
            BedroomPostgresRepository(),
            conversation_store=PostgresConversationPartitionStore(get_pool),
            history_store=PostgresAnchoredHistoryStore(get_pool),
        )
    return _bedroom_retention_service


def _get_group_candidate_service() -> CandidateIngressService:
    global _group_candidate_service
    if _group_candidate_service is None:
        relay_url = os.environ.get("GROUP_RELAY_BASE_URL", "").strip()
        relay_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "").strip()
        if not relay_url or not relay_key:
            raise RelayGroupError(503, "dependency_unavailable")
        _group_candidate_service = CandidateIngressService(
            RelayGroupClient(relay_url, relay_key)
        )
    return _group_candidate_service


def _get_group_extraction_service() -> ClosedBurstExtractionService:
    global _group_extraction_service
    if _group_extraction_service is None:
        relay_url = os.environ.get("GROUP_RELAY_BASE_URL", "").strip()
        relay_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "").strip()
        if not relay_url or not relay_key:
            raise RelayGroupError(503, "dependency_unavailable")
        _group_extraction_service = ClosedBurstExtractionService(
            RelayGroupClient(relay_url, relay_key),
            extractor=GroupBatchExtractionPipeline(extract_group_memories),
            burst_threshold=max(
                1, int(os.environ.get("GROUP_EXTRACTION_BURST_THRESHOLD", "3"))
            ),
            token_threshold=max(
                0, int(os.environ.get("GROUP_EXTRACTION_TOKEN_THRESHOLD", "4000"))
            ),
            max_wait_seconds=max(
                0.0, float(os.environ.get("GROUP_EXTRACTION_MAX_WAIT_SECONDS", "300"))
            ),
        )
    return _group_extraction_service


def _derive_candidate_actor(request: Request) -> str | None:
    provided = _group_bearer(request)
    matches = []
    for actor_id, env_name in (
        ("jiao", "GROUP_JIAO_MEMORY_CANDIDATE_KEY"),
        ("laoke", "GROUP_LAOKE_MEMORY_CANDIDATE_KEY"),
    ):
        expected = os.environ.get(env_name, "")
        if expected and secrets.compare_digest(provided, expected):
            matches.append(actor_id)
    return matches[0] if len(matches) == 1 else None


async def _build_group_context_pack(request: Request, pack_kind: str):
    if not group_memory_features_from_env()["group_memory"]:
        return _group_error(404, "group_feature_disabled", "Group memory is disabled")
    if request.headers.get("X-Group-Contract-Version") != CONTRACT_VERSION:
        return _group_error(
            409, "contract_version_mismatch", "Unsupported Group contract version"
        )
    expected_key = os.environ.get("GROUP_ORCHESTRATOR_SERVICE_KEY", "")
    if not expected_key or not secrets.compare_digest(_group_bearer(request), expected_key):
        return _group_error(401, "invalid_service_key", "Invalid service key")
    try:
        payload = await request.json()
        pack_request = ContextPackRequest.from_dict(payload)
    except (ValueError, json.JSONDecodeError, ContractError, TypeError):
        return _group_error(422, "invalid_group_payload", "Invalid Group payload")
    try:
        await _refresh_actor_prompt_store()
        pack = await _get_group_context_service().build(
            pack_request, pack_kind=pack_kind
        )
    except RelayGroupError as exc:
        return _group_error(exc.status_code, exc.code, "Group dependency rejected request")
    except ContractError:
        return _group_error(422, "invalid_group_payload", "Invalid Group payload")
    return JSONResponse(status_code=200, content=pack.to_dict())


@app.post("/internal/group/context-packs/probe")
async def group_context_pack_probe(request: Request):
    return await _build_group_context_pack(request, "probe")


@app.post("/internal/group/context-packs/full")
async def group_context_pack_full(request: Request):
    return await _build_group_context_pack(request, "full")


def _bedroom_enabled() -> bool:
    return os.environ.get("GATEWAY_BEDROOM_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


@app.post("/internal/bedroom/context-packs/full")
async def bedroom_context_pack_full(request: Request):
    if not _bedroom_enabled():
        return _group_error(404, "group_feature_disabled", "Bedroom memory is disabled")
    if request.headers.get("X-Bedroom-Contract-Version") != BEDROOM_CONTRACT_VERSION:
        return _group_error(409, "contract_version_mismatch", "Bedroom contract mismatch")
    expected = os.environ.get("GROUP_ORCHESTRATOR_SERVICE_KEY", "")
    if not expected or not secrets.compare_digest(_group_bearer(request), expected):
        return _group_error(401, "invalid_service_key", "Invalid service key")
    try:
        pack_request = BedroomPackRequest.from_dict(await request.json())
        await _refresh_actor_prompt_store()
        pack = await _get_bedroom_context_service().build(pack_request)
    except BedroomContractError:
        return _group_error(409, "stale_fence", "Bedroom generation is stale")
    except RelayGroupError as exc:
        return _group_error(exc.status_code, exc.code, "Bedroom facts unavailable")
    return JSONResponse(status_code=200, content=pack.to_dict())


@app.post("/internal/bedroom/retention")
async def bedroom_retention(request: Request):
    if not _bedroom_enabled():
        return _group_error(404, "group_feature_disabled", "Bedroom retention is disabled")
    if request.headers.get("X-Bedroom-Contract-Version") != BEDROOM_CONTRACT_VERSION:
        return _group_error(409, "contract_version_mismatch", "Bedroom contract mismatch")
    expected = os.environ.get("BEDROOM_GATEWAY_SERVICE_KEY", "")
    if not expected or not secrets.compare_digest(_group_bearer(request), expected):
        return _group_error(401, "invalid_service_key", "Invalid service key")
    try:
        payload = await request.json()
        receipt = await _get_bedroom_retention_service().persist(payload)
        session = payload.get("session", {}) if isinstance(payload, dict) else {}
        bedroom_session_id = session.get("bedroom_session_id")
        if isinstance(bedroom_session_id, str) and bedroom_session_id:
            try:
                await (await _get_cache_pin_service()).end_bedroom(bedroom_session_id)
            except Exception as exc:
                # Retention is already durable at this point. Cache Pin cleanup
                # is operational follow-up and must not turn that success into
                # a false failure response.
                print(f"[warning] Bedroom Cache Pin cleanup deferred: {type(exc).__name__}")
    except BedroomContractError:
        return _group_error(422, "invalid_group_payload", "Invalid Bedroom retention payload")
    return JSONResponse(status_code=200, content=receipt)


@app.post("/internal/group/memory-candidates")
async def group_memory_candidate(request: Request):
    features = group_memory_features_from_env()
    if not features["group_memory"] or not features["agent_candidates"]:
        return _group_error(404, "group_feature_disabled", "Group candidates are disabled")
    if request.headers.get("X-Group-Contract-Version") != CONTRACT_VERSION:
        return _group_error(
            409, "contract_version_mismatch", "Unsupported Group contract version"
        )
    actor_id = _derive_candidate_actor(request)
    if actor_id is None:
        return _group_error(403, "principal_not_allowed", "Principal is not allowed")
    try:
        candidate_request = MemoryCandidateRequest.from_dict(await request.json())
        receipt = await _get_group_candidate_service().accept(actor_id, candidate_request)
    except (ValueError, json.JSONDecodeError, ContractError, TypeError):
        return _group_error(422, "invalid_group_payload", "Invalid Group payload")
    except (StaleCandidateError,):
        return _group_error(409, "stale_fence", "Candidate final is stale")
    except SensitiveCandidateError:
        return _group_error(403, "principal_not_allowed", "Candidate is not permitted")
    except RelayGroupError as exc:
        code = "stale_fence" if exc.code in {"stale_fence", "fact_not_visible"} else exc.code
        return _group_error(exc.status_code, code, "Group dependency rejected request")
    return JSONResponse(status_code=200, content=receipt.to_dict())


@app.post("/internal/group/extraction/closed-bursts")
async def group_closed_burst_extraction(request: Request):
    features = group_memory_features_from_env()
    if not features["group_memory"] or not features["burst_extraction"]:
        return _group_error(404, "group_feature_disabled", "Group extraction is disabled")
    if request.headers.get("X-Group-Contract-Version") != CONTRACT_VERSION:
        return _group_error(
            409, "contract_version_mismatch", "Unsupported Group contract version"
        )
    expected_key = os.environ.get("GROUP_RELAY_SERVICE_KEY", "")
    if not expected_key or not secrets.compare_digest(_group_bearer(request), expected_key):
        return _group_error(403, "principal_not_allowed", "Principal is not allowed")
    try:
        body = ClosedBurstExtractionRequest.from_dict(await request.json())
        queued = await _get_group_extraction_service().enqueue(body)
    except (ValueError, json.JSONDecodeError, ContractError, TypeError):
        return _group_error(422, "invalid_group_payload", "Invalid Group payload")
    except UnstableBurstError as exc:
        return _group_error(409, exc.code, "Closed burst is not stable")
    except RelayGroupError as exc:
        code = (
            "burst_not_closed"
            if exc.code == "burst_not_closed"
            else ("stale_fence" if exc.code == "fact_not_visible" else exc.code)
        )
        return _group_error(exc.status_code, code, "Group dependency rejected request")
    ref = queued["closed_fence"]
    return JSONResponse(
        status_code=200,
        content={
            "contract_version": CONTRACT_VERSION,
            "accepted": True,
            "burst_id": ref["burst_id"],
            "fence_epoch": ref["fence_epoch"],
        },
    )


# ============================================================
# 记忆管理接口
# ============================================================


@app.get("/import/seed-memories")
async def import_seed_memories():
    """一次性导入预置记忆（从 seed_memories.py）"""
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    """
    导出所有记忆为 JSON（用于备份或迁移）
    浏览器访问这个地址就会返回所有记忆数据
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        memories = await get_all_memories()
        # 把 datetime 转成字符串
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        
        return {
            "total": len(memories),
            "exported_at": str(__import__("datetime").datetime.now()),
            "memories": memories,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard - 整合的记忆管理界面"""
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
    
    return templates.TemplateResponse(request, "dashboard.html")



# ============================================================
# 管理 API
# ============================================================

@app.get("/api/memories")
async def api_get_memories(
    layer: int = None,
    active_only: bool = None,
    scope: str = None,
    confidential: bool = None,
):
    """获取所有记忆（管理页面用）
    
    Query params:
        layer: 筛选层级（1=碎片, 2=事件, 3=核心）
        active_only: 是否只返回活跃记忆
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    allowed_scopes = {
        "legacy_unscoped", "weiwei-jiao", "weiwei-laoke", "jiao-laoke", "group"
    }
    if scope is not None and scope not in allowed_scopes:
        raise HTTPException(status_code=422, detail="invalid memory scope")
    memories = await get_all_memories_detail(
        layer=layer,
        active_only=active_only,
        scope=scope,
        confidential=confidential,
    )
    tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    # 获取层级统计
    try:
        layer_stats = await get_layer_statistics()
    except Exception:
        layer_stats = None
    
    result = {"memories": memories}
    if layer_stats:
        result["layer_stats"] = layer_stats
    return result


@app.get("/api/archive")
async def api_get_cold_archive(limit: int = 200):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="invalid archive limit")
    rows = await list_cold_archive_for_management(limit=limit)
    for row in rows:
        for key in ("raw_timestamp", "imported_at"):
            if row.get(key) is not None:
                row[key] = str(row[key])
        for annotation in row.get("annotations", []):
            if annotation.get("created_at") is not None:
                annotation["created_at"] = str(annotation["created_at"])
    return {"archive": rows}


@app.post("/api/archive/{archive_id}/annotations")
async def api_append_cold_archive_annotation(archive_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    body = await request.json()
    if not isinstance(body, dict) or set(body) != {"annotation_type", "payload"}:
        raise HTTPException(status_code=422, detail="invalid archive annotation")
    if body["annotation_type"] not in {
        "correction", "identity_mapping", "timestamp_fix", "redaction", "note"
    } or not isinstance(body["payload"], dict):
        raise HTTPException(status_code=422, detail="invalid archive annotation")
    try:
        annotation = await append_cold_archive_annotation(
            archive_id, body["annotation_type"], body["payload"]
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="invalid archive annotation") from exc
    if annotation.get("created_at") is not None:
        annotation["created_at"] = str(annotation["created_at"])
    return {"annotation": annotation}


@app.get("/api/memories/search")
async def api_search_memories(q: str = "", limit: int = 20):
    """语义搜索记忆（Dashboard用，走后端 search_memories）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if not q.strip():
        return {"error": "搜索关键词不能为空", "results": []}
    try:
        results = await search_memories(q.strip(), limit)
        tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
        out = []
        for r in results:
            item = dict(r)
            if item.get("created_at"):
                dt = item["created_at"]
                if hasattr(dt, 'tzinfo'):
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    item["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
            out.append(item)
        return {"results": out, "total": len(out)}
    except Exception as e:
        return {"error": str(e), "results": []}


@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    """更新单条记忆（支持 content / importance / title / layer）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory_with_layer(
        memory_id,
        content=data.get("content"),
        importance=data.get("importance"),
        title=data.get("title"),
        layer=data.get("layer"),
    )
    return {"status": "ok", "id": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int, soft: bool = False):
    """删除单条记忆
    
    Query params:
        soft: true=归档（is_active=false），false=永久删除
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if soft:
        await update_memory_with_layer(memory_id, is_active=False)
    else:
        await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}


@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    """批量更新记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}
    for item in updates:
        await update_memory_with_layer(
            item["id"],
            content=item.get("content"),
            importance=item.get("importance"),
            title=item.get("title"),
            layer=item.get("layer"),
        )
    return {"status": "ok", "updated": len(updates)}


@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    """批量删除记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


# ============================================================
# 三层记忆架构：整理 / 合并 / 升级 / 统计
# ============================================================

CONSOLIDATION_PROMPT = """
你是记忆整理助手。请将以下对话碎片整理成完整的事件记录。

要求：
1. 按主题/事件分组，相关的碎片合并到一起
2. 每个事件一条记录，不要太细碎也不要太笼统
3. 每条记录包含：标题（10字内）+ 完整描述
4. 合并重复内容，保留重要细节
5. 保留原文中的主观感受、情绪表达和个人化用语，不要改写为客观陈述或第三方总结
6. content字段中不要使用双引号，用单引号或书名号代替

碎片记忆：
{fragments}

请用 JSON 格式输出：
[
  {{
    "title": "事件标题（10字内）",
    "content": "完整的事件描述",
    "importance": 5,
    "merged_ids": [1, 2, 3]
  }}
]

只输出 JSON，不要其他内容。确保 JSON 语法正确。
"""

# 整理状态（异步执行，防重入）
_consolidate_status = {
    "running": False,
    "started_at": None,
    "result": None,
    "error": None,
}


async def consolidate_memories_for_date(event_date):
    """整理指定日期的碎片记忆"""
    return await consolidate_memories_for_date_range(event_date, event_date)


async def consolidate_memories_for_date_range(start_date, end_date):
    """整理指定时间段的碎片记忆"""
    from datetime import date
    import re
    
    # 获取该时间段的碎片
    fragments = await get_fragments_by_date_range(start_date, end_date)
    
    if not fragments:
        return {"status": "no_fragments", "start_date": str(start_date), "end_date": str(end_date)}
    
    # 构建碎片文本
    fragments_text = "\n".join([
        f"[ID={f['id']}] ({f['created_at'].strftime('%m-%d') if hasattr(f['created_at'], 'strftime') else str(f['created_at'])[:10]}) {f['content']}"
        for f in fragments
    ])
    
    # 调用 AI 进行整理
    prompt = CONSOLIDATION_PROMPT.format(fragments=fragments_text)
    
    # 使用环境变量配置的模型，默认 haiku 节省成本
    consolidation_model = MEMORY_MODEL

    if not MEMORY_API_BASE_URL or not get_memory_api_key() or not consolidation_model:
        return {
            "status": "error",
            "error": "memory provider configuration is incomplete",
        }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 最多重试2次（应对429限流）
            last_error = None
            for attempt in range(3):
                response = await client.post(
                    MEMORY_API_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {get_memory_api_key()}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": consolidation_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000
                    }
                )

                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"⚠️ 整理API 429限流，{wait_time}秒后重试（第{attempt+1}次）")
                    last_error = f"429 Too Many Requests (重试{attempt+1}次)"
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    print(f"⚠️ 整理API返回 {response.status_code}: {response.text[:200]}")
                    break

                last_error = None
                break

            if last_error:
                return {"status": "error", "error": f"API调用失败: {last_error}"}

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 解析 JSON（三层容错）
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                json_str = json_match.group()
                try:
                    events = json.loads(json_str)
                except json.JSONDecodeError:
                    # 方案1：用 strict=False
                    try:
                        events = json.loads(json_str, strict=False)
                    except json.JSONDecodeError:
                        # 方案2：去掉控制字符后重试
                        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
                        try:
                            events = json.loads(cleaned)
                        except json.JSONDecodeError as e:
                            # 方案3：让 AI 重新格式化
                            print(f"⚠️ JSON解析失败，尝试让AI修复: {e}")
                            fix_resp = await client.post(
                                MEMORY_API_BASE_URL,
                                headers={
                                    "Authorization": f"Bearer {get_memory_api_key()}",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "model": consolidation_model,
                                    "messages": [{"role": "user", "content": f"请修复以下JSON的语法错误，只输出修复后的JSON数组，不要其他内容：\n{json_str[:2000]}"}],
                                    "max_tokens": 2000
                                }
                            )
                            if fix_resp.status_code == 200:
                                fix_content = fix_resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                                fix_match = re.search(r'\[[\s\S]*\]', fix_content)
                                if fix_match:
                                    try:
                                        events = json.loads(fix_match.group())
                                        print(f"✅ AI修复JSON成功")
                                    except json.JSONDecodeError:
                                        return {"status": "error", "error": f"JSON解析失败（AI修复也失败）", "raw": content[:500]}
                                else:
                                    return {"status": "error", "error": "AI修复未返回有效JSON", "raw": content[:500]}
                            else:
                                return {"status": "error", "error": f"JSON解析失败，AI修复请求失败: HTTP {fix_resp.status_code}", "raw": content[:500]}
            else:
                return {"status": "error", "error": "无法解析 AI 返回的 JSON", "raw": content}
            
            # 创建事件记忆并停用碎片
            created_count = 0
            for event in events:
                merged_ids = event.get("merged_ids", [])
                if merged_ids:
                    await create_event_memory(
                        title=event.get("title", ""),
                        content=event.get("content", ""),
                        importance=event.get("importance", 5),
                        event_date=start_date,
                        merged_from=merged_ids
                    )
                    created_count += 1
            
            # 停用所有已处理的碎片
            all_fragment_ids = [f['id'] for f in fragments]
            await deactivate_memories(all_fragment_ids)
            
            return {
                "status": "ok",
                "start_date": str(start_date),
                "end_date": str(end_date),
                "fragments_processed": len(fragments),
                "events_created": created_count
            }
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/memories/consolidate")
async def api_manual_consolidate(request: Request):
    """手动触发整理（异步，立即返回）
    
    Body:
        start_date: 开始日期（YYYY-MM-DD 格式）
        end_date: 结束日期（YYYY-MM-DD 格式）
        或
        date: 单个日期（兼容旧版）
    """
    from datetime import date as date_type
    
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    if _consolidate_status.get("running"):
        return {"status": "already_running", "started_at": _consolidate_status.get("started_at")}
    
    data = await request.json()
    
    # 解析日期参数
    if "date" in data and "start_date" not in data:
        start_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        end_date = start_date
    else:
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        
        if not start_date_str or not end_date_str:
            return {"error": "请提供开始和结束日期"}
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if start_date > end_date:
            return {"error": "开始日期不能晚于结束日期"}
    
    async def _run():
        _consolidate_status.update({"running": True, "started_at": f"{start_date}~{end_date}", "result": None, "error": None})
        try:
            result = await consolidate_memories_for_date_range(start_date, end_date)
            _consolidate_status["result"] = result
            print(f"[manual/consolidate] 整理 {start_date}~{end_date}: {result}")
        except Exception as e:
            _consolidate_status["error"] = str(e)
            print(f"[manual/consolidate] 整理 {start_date}~{end_date} 失败: {e}")
        finally:
            _consolidate_status["running"] = False
    
    asyncio.create_task(_run())
    return {"status": "started", "start_date": str(start_date), "end_date": str(end_date)}


@app.get("/api/memories/consolidate/status")
async def api_consolidate_status():
    """查询整理任务状态"""
    return _consolidate_status


@app.post("/api/memories/{memory_id}/promote")
async def api_promote_to_core(memory_id: int, request: Request):
    """将记忆升级为核心记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    data = await request.json()
    title = data.get("title")
    
    await promote_to_core(memory_id, title=title)
    return {"status": "ok", "memory_id": memory_id, "layer": 3}


@app.post("/api/memories/merge")
async def api_merge_memories(request: Request):
    """手动合并多条记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    data = await request.json()
    memory_ids = data.get("ids", [])
    new_title = data.get("title", "")
    new_content = data.get("content", "")
    importance = data.get("importance", 5)
    layer = data.get("layer", 2)
    
    if not memory_ids or not new_content:
        return {"error": "请提供记忆ID列表和合并后内容"}
    
    new_id = await merge_memories(memory_ids, new_title, new_content, importance, layer)
    return {"status": "ok", "new_id": new_id, "merged": len(memory_ids)}


@app.post("/api/memories/check-duplicate")
async def api_check_duplicate(request: Request):
    """检查记忆是否重复"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    data = await request.json()
    content = data.get("content", "")
    threshold = data.get("threshold", 0.7)
    
    if not content:
        return {"error": "请提供记忆内容"}
    
    result = await check_duplicate_memory(content, threshold)
    return result


@app.post("/api/memories/cleanup-fragments")
async def api_cleanup_fragments(request: Request):
    """清理指定天数前的归档碎片
    
    Body:
        days: 清理多少天前的归档碎片（默认30天）
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    data = await request.json()
    days = data.get("days", 30)
    
    try:
        deleted = await cleanup_old_fragments(days)
        return {"status": "ok", "deleted": deleted, "days": days}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/memories/{memory_id}/revert-merge")
async def api_revert_merge(memory_id: int):
    """撤回合并操作：恢复原始碎片，删除合并后的事件记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    try:
        result = await revert_merge(memory_id)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/memories/{memory_id}/restore")
async def api_restore_memory(memory_id: int):
    """恢复已归档的记忆（将 is_active 设为 TRUE）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    try:
        await update_memory_with_layer(memory_id, is_active=True)
        return {"status": "ok", "id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/memories/layer-stats")
async def api_layer_statistics():
    """获取各层记忆统计数据"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    try:
        stats = await get_layer_statistics()
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/text")
async def import_text_memories(request: Request):
    """从纯文本导入记忆（每行一条），可选自动评分"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)
        
        if not lines:
            return {"error": "没有找到记忆条目"}
        
        if skip_scoring:
            scored = [{"content": t, "importance": 5} for t in lines]
        else:
            scored = await score_memories(lines)
        
        imported = 0
        skipped = 0
        
        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue
            
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )
            
            if existing > 0:
                skipped += 1
                continue
            
            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session="text-import",
            )
            imported += 1
        
        total = await get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/memories")
async def import_memories(request: Request):
    """从 JSON 导入记忆（用于迁移或恢复备份）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        data = await request.json()
        memories = data.get("memories", [])
        
        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}
        
        imported = 0
        skipped = 0
        
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )
            
            if existing > 0:
                skipped += 1
                continue
            
            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session=mem.get("source_session", "json-import"),
            )
            imported += 1
        
        total = await get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 对话记录管理 API
# ============================================================

@app.get("/api/conversations")
async def api_conversations(page: int = 1, per_page: int = 20):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        results, total = await get_conversations_paginated(page, per_page)
        total_pages = max(1, -(-total // per_page))  # 向上取整
        return {"conversations": results, "total": total, "page": page, "per_page": per_page, "total_pages": total_pages}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/conversations/{session_id}/messages")
async def api_conversation_messages(session_id: str, limit: int = 50, offset: int = 0):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE session_id = $1", session_id
            )
            rows = await conn.fetch("""
                SELECT id, role, content, created_at, room_id,
                       canonical_conversation_id, source_event_id, actor_id,
                       event_role, source_kind, bedroom_session_id,
                       retention_policy, attachments_json, provenance_json
                FROM conversations WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, session_id, limit, offset)
        def decoded(value, fallback):
            if value is None:
                return fallback
            return json.loads(value) if isinstance(value, str) else value
        msgs = [{
            "id": r["id"], "role": r["role"], "content": r["content"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "room_id": r.get("room_id"),
            "conversation_id": r.get("canonical_conversation_id") or session_id,
            "source_event_id": r.get("source_event_id"),
            "actor_id": r.get("actor_id"),
            "event_role": r.get("event_role"),
            "source_kind": r.get("source_kind") or "legacy",
            "bedroom_session_id": r.get("bedroom_session_id"),
            "retention_policy": r.get("retention_policy"),
            "attachments": decoded(r.get("attachments_json"), []),
            "provenance": decoded(r.get("provenance_json"), None),
        } for r in rows]
        return {"messages": msgs, "total": total}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/conversations/{session_id}")
async def api_delete_conversation(session_id: str):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        await delete_conversation(session_id)
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/conversations/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        body = await request.json()
        ids = body.get("session_ids", [])
        if ids:
            await batch_delete_conversations(ids)
        return {"status": "ok", "deleted": len(ids)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/admin/merge-sessions")
async def api_merge_sessions(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        body = await request.json()
        source_ids = [s for s in body.get("source_ids", []) if s != body.get("target_id", "")]
        target_id = body.get("target_id", "")
        if not source_ids or not target_id:
            return {"error": "source_ids 和 target_id 不能为空"}
        result = await merge_sessions_to_target(source_ids, target_id)
        return {"status": "ok", **result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/chat/search")
async def api_search_conversations(q: str = "", limit: int = 20, offset: int = 0):
    """搜索对话内容"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if not q.strip():
        return {"error": "搜索关键词不能为空", "results": [], "total": 0}
    try:
        results, total = await search_conversations(q.strip(), limit, offset)
        return {"results": results, "total": total}
    except Exception as e:
        return {"error": str(e), "results": [], "total": 0}


@app.patch("/api/chat/messages/{message_id}")
async def api_update_message(message_id: int, request: Request):
    """编辑单条消息内容"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return {"error": "内容不能为空"}
        updated = await update_message_content(message_id, content)
        if updated == 0:
            return {"error": "消息不存在"}
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/chat/messages/{message_id}")
async def api_delete_message(message_id: int):
    """删除单条消息"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        deleted = await delete_single_message(message_id)
        if deleted == 0:
            return {"error": "消息不存在"}
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/conversations/export")
async def api_export_conversations():
    """导出所有对话记录"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        data = await export_all_conversations()
        return JSONResponse(content=data)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/conversations/import")
async def api_import_conversations(request: Request):
    """导入对话记录（JSON格式，自动去重）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        records = await request.json()
        if not isinstance(records, list):
            return {"error": "格式错误：需要 JSON 数组"}
        imported, skipped = await import_conversations(records)
        return {"status": "ok", "imported": imported, "skipped": skipped, "total": imported + skipped}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 记忆向量补算（带进度追踪）
# ============================================================

_backfill_mem_status = {
    "running": False,
    "total": 0,
    "done": 0,
    "error": None,
    "finished_at": None,
}

@app.post("/api/admin/backfill-memory-embeddings")
async def api_backfill_memory_embeddings():
    """给已有记忆补算embedding（后台异步执行，前端轮询进度）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    
    if _backfill_mem_status["running"]:
        return {"error": "补算任务正在运行中，请等待完成"}
    
    try:
        total = await get_pending_memory_embedding_count()
    except Exception as e:
        return {"error": f"查询待处理数量失败: {e}"}
    
    if total == 0:
        return {"status": "done", "message": "所有记忆已有embedding，无需补算", "total": 0, "done": 0}
    
    _backfill_mem_status["running"] = True
    _backfill_mem_status["total"] = total
    _backfill_mem_status["done"] = 0
    _backfill_mem_status["error"] = None
    _backfill_mem_status["finished_at"] = None
    
    async def run_backfill():
        try:
            while _backfill_mem_status["running"]:
                updated = await backfill_memory_embeddings(batch_size=20)
                _backfill_mem_status["done"] += updated
                
                if updated == 0:
                    break
                
                await asyncio.sleep(1)
            
            _backfill_mem_status["finished_at"] = datetime.now(timezone.utc).isoformat()
            print(f"✅ 记忆embedding补算完成：{_backfill_mem_status['done']}/{_backfill_mem_status['total']}")
        except Exception as e:
            _backfill_mem_status["error"] = str(e)
            print(f"❌ 记忆embedding补算异常: {e}")
        finally:
            _backfill_mem_status["running"] = False
    
    asyncio.create_task(run_backfill())
    return {"status": "started", "total": total}

@app.get("/api/admin/backfill-memory-embeddings/status")
async def api_backfill_memory_embeddings_status():
    """查询记忆embedding补算进度"""
    return {
        "running": _backfill_mem_status["running"],
        "total": _backfill_mem_status["total"],
        "done": _backfill_mem_status["done"],
        "error": _backfill_mem_status["error"],
        "finished_at": _backfill_mem_status["finished_at"],
    }


# ============================================================
# Memory / embedding maintenance settings API. Model execution configuration
# belongs exclusively to Model Profiles and actor Persona versions.
# Dashboard 前端设置面板用，管理所有运行时可调配置
# ============================================================

def _mask_key(key_value: str) -> str:
    """API Key 打码：只露前5位和后4位"""
    if not key_value:
        return ""
    if len(key_value) < 10:
        return "****"
    return key_value[:5] + "****" + key_value[-4:]


def _is_masked(value: str) -> bool:
    """判断值是否是打码值（用户没改过）"""
    return "****" in str(value)


def _parse_bool(val, fallback=False) -> bool:
    """解析布尔值（兼容字符串/布尔/None）"""
    if val is None:
        return fallback
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


@app.get("/api/memory-settings")
async def get_settings():
    """获取高级设置（数据库优先，fallback 到环境变量/运行时默认值）"""
    try:
        db = await get_all_gateway_config()

        embedding_key_raw = db.get("EMBEDDING_API_KEY") or _db_module.EMBEDDING_API_KEY

        memory_key_raw = db.get("MEMORY_API_KEY") or MEMORY_API_KEY

        settings = {
            # 记忆系统
            "MEMORY_ENABLED":          _parse_bool(db.get("MEMORY_ENABLED"), MEMORY_ENABLED),
            "MEMORY_API_KEY":          _mask_key(memory_key_raw),
            "MEMORY_API_BASE_URL":     db.get("MEMORY_API_BASE_URL") or MEMORY_API_BASE_URL,
            "MEMORY_MODEL":            db.get("MEMORY_MODEL") or MEMORY_MODEL,
            "MAX_MEMORIES_INJECT":     int(db.get("MAX_MEMORIES_INJECT") or MAX_MEMORIES_INJECT),
            "MIN_SCORE_THRESHOLD":     float(db.get("MIN_SCORE_THRESHOLD") or _db_module.MIN_SCORE_THRESHOLD),
            "MEMORY_EXTRACT_INTERVAL": int(db.get("MEMORY_EXTRACT_INTERVAL") or MEMORY_EXTRACT_INTERVAL),

            # 向量搜索（开源版用 EMBEDDING_API_KEY + EMBEDDING_BASE_URL）
            "MEMORY_VECTOR_ENABLED":   _parse_bool(db.get("MEMORY_VECTOR_ENABLED"), _db_module.MEMORY_VECTOR_ENABLED),
            "EMBEDDING_API_KEY":       _mask_key(embedding_key_raw),
            "EMBEDDING_BASE_URL":      db.get("EMBEDDING_BASE_URL") or str(_db_module.EMBEDDING_BASE_URL),
            "EMBEDDING_MODEL":         db.get("EMBEDDING_MODEL") or str(_db_module.EMBEDDING_MODEL),
            "EMBEDDING_DIM":           int(db.get("EMBEDDING_DIM") or _db_module.EMBEDDING_DIM),

            # 搜索权重
            "MEMORY_HW_KEYWORD":        float(db.get("MEMORY_HW_KEYWORD") or _db_module.MEMORY_HW_KEYWORD),
            "MEMORY_HW_SEMANTIC":       float(db.get("MEMORY_HW_SEMANTIC") or _db_module.MEMORY_HW_SEMANTIC),
            "MEMORY_HW_IMPORTANCE":     float(db.get("MEMORY_HW_IMPORTANCE") or _db_module.MEMORY_HW_IMPORTANCE),
            "MEMORY_HW_RECENCY":        float(db.get("MEMORY_HW_RECENCY") or _db_module.MEMORY_HW_RECENCY),
            "MEMORY_SEMANTIC_THRESHOLD": float(db.get("MEMORY_SEMANTIC_THRESHOLD") or _db_module.MEMORY_SEMANTIC_THRESHOLD),

        }

        return {"status": "ok", "settings": settings}
    except Exception as e:
        print(f"[get_settings] 错误: {e}")
        return {"error": str(e)}


@app.put("/api/memory-settings")
async def save_settings(request: Request):
    """保存高级设置（写入数据库 + 热更新运行时变量，立即生效无需重启）"""
    try:
        data = await request.json()
        updated = []
        skipped = []

        # main.py 全局变量映射（key → 类型转换函数）
        _MAIN_VARS = {
            "MEMORY_API_KEY":        str,
            "MEMORY_API_BASE_URL":   str,
            "MEMORY_MODEL":          str,
            "MEMORY_ENABLED":        lambda v: _parse_bool(v),
            "MAX_MEMORIES_INJECT":   int,
            "MEMORY_EXTRACT_INTERVAL": int,
        }

        # database.py 全局变量映射（开源版用 EMBEDDING_API_KEY + EMBEDDING_BASE_URL）
        _DB_VARS = {
            "EMBEDDING_API_KEY":       str,
            "EMBEDDING_BASE_URL":      str,
            "EMBEDDING_MODEL":         str,
            "EMBEDDING_DIM":           int,
            "MIN_SCORE_THRESHOLD":     float,
            "MEMORY_VECTOR_ENABLED":   lambda v: _parse_bool(v),
            "MEMORY_HW_KEYWORD":       float,
            "MEMORY_HW_SEMANTIC":      float,
            "MEMORY_HW_IMPORTANCE":    float,
            "MEMORY_HW_RECENCY":       float,
            "MEMORY_SEMANTIC_THRESHOLD": float,
        }

        _ENV_ONLY = {}

        # 打码字段
        _MASKED_KEYS = {"EMBEDDING_API_KEY", "MEMORY_API_KEY"}

        for key, value in data.items():
            # --- 打码字段特殊处理 ---
            if key in _MASKED_KEYS:
                str_val = str(value).strip()
                if _is_masked(str_val):
                    skipped.append(key)
                    continue
                if not str_val:
                    await set_gateway_config(key, "")
                    if key in _MAIN_VARS:
                        globals()[key] = ""
                    elif key in _DB_VARS:
                        setattr(_db_module, key, "")
                    if key == "MEMORY_API_KEY":
                        import memory_extractor as _me_mod
                        _me_mod.MEMORY_API_KEY = ""
                    os.environ[key] = ""
                    updated.append(key)
                    continue

            # --- 常规字段 ---
            await set_gateway_config(key, str(value))

            if key in _MAIN_VARS:
                typed_value = _MAIN_VARS[key](value)
                globals()[key] = typed_value
                os.environ[key] = str(value)
                if key in {"MEMORY_API_KEY", "MEMORY_API_BASE_URL", "MEMORY_MODEL"}:
                    import memory_extractor as _me_mod
                    setattr(_me_mod, key, str(value))
                updated.append(key)
                print(f"[settings] {key} = {typed_value}")

            elif key in _DB_VARS:
                typed_value = _DB_VARS[key](value)
                setattr(_db_module, key, typed_value)
                os.environ[key] = str(value)
                updated.append(key)
                print(f"[settings] {key} = {typed_value} (database)")

            elif key in _ENV_ONLY:
                typed_value = _ENV_ONLY[key](value)
                os.environ[key] = str(typed_value)
                updated.append(key)
                print(f"[settings] {key} = {typed_value} (env)")

            else:
                skipped.append(key)

        return {
            "status": "ok",
            "updated": updated,
            "skipped": skipped,
            "message": f"已更新 {len(updated)} 项配置，立即生效"
        }
    except Exception as e:
        print(f"[save_settings] 错误: {e}")
        return {"error": str(e)}


# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print("🤖 模型执行：Model Profiles")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    if MEMORY_ENABLED:
        print(f"📝 记忆提取+注入：{'开启' if MEMORY_EXTRACT_ENABLED else '关闭'}")
    print(f"🔄 记忆提取间隔：{'禁用自动批量（显式请求仍处理）' if MEMORY_EXTRACT_INTERVAL == 0 else '每轮提取' if MEMORY_EXTRACT_INTERVAL == 1 else f'每个 session 每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
