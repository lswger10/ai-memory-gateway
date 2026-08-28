import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from group_memory import (
    ForbiddenSourceKind,
    ForbiddenMemoryWrite,
    InvalidSharedEvidence,
    MemoryAuthContext,
    ScopeAwareMemoryService,
    effective_confidence,
)
from memory_policy import (
    MemoryScope,
    MemoryStatus,
    MemoryType,
    MemoryWrite,
    Perspective,
    SourceKind,
)


def test_database_predicate_excludes_expired_inference_before_candidate_creation():
    import database
    from memory_policy import build_retrieval_policy, room_members

    policy = build_retrieval_policy(
        "jiao", "room_group_home", room_members("room_group_home")
    )
    fixed_now = datetime(2026, 8, 26, tzinfo=timezone.utc)
    sql, params = database._authorized_predicate(policy, 1, now=fixed_now)
    assert "POWER(0.5" in sql
    assert "last_supported_at" in sql
    assert "memory_type <> 'inference'" in sql
    assert params[2] == fixed_now
    assert params[3] > 0
    assert 0 <= params[4] <= 1


def test_agent_cannot_write_cross_pairwise_scope_from_group(service):
    write = MemoryWrite(
        content="不应由椒椒写入薇薇-老克私域",
        scope=MemoryScope.WEIWEI_LAOKE,
        memory_type=MemoryType.INFERENCE,
        perspective=Perspective.JIAO,
        confidential=False,
        source_kind=SourceKind.AGENT_CANDIDATE,
        confidence=0.5,
    )
    with pytest.raises(ForbiddenMemoryWrite):
        asyncio.run(
            service.persist_memory_write(
                write, MemoryAuthContext(actor_id="jiao", room_id="room_group_home")
            )
        )


@pytest.fixture
def service():
    return ScopeAwareMemoryService(
        identity_profiles={"jiao": {"voice": "椒椒-v1"}, "laoke": {"voice": "老克-v1"}}
    )


def test_user_view_of_actor_inner_state_is_not_shared(service):
    write = MemoryWrite(
        content="薇薇认为老克刚才在吃醋",
        scope=MemoryScope.WEIWEI_LAOKE,
        memory_type=MemoryType.INFERENCE,
        perspective=Perspective.SHARED,
        confidential=False,
        source_kind=SourceKind.EXPLICIT_USER_MEMORY,
        confidence=0.9,
        evidence_count=1,
        provenance={
            "asserts_inner_state_of": ["laoke"],
            "shared_evidence": "weiwei_shared_instruction",
        },
    )
    saved = asyncio.run(
        service.persist_memory_write(
            write, MemoryAuthContext(actor_id="weiwei", room_id="room_weiwei_laoke")
        )
    )
    assert saved.memory_type == "inference"
    assert saved.perspective == "weiwei"


def test_repeated_model_agreement_cannot_promote_shared(service):
    write = MemoryWrite(
        content="椒椒和老克都可能在逞强",
        scope=MemoryScope.JIAO_LAOKE,
        memory_type=MemoryType.INFERENCE,
        perspective=Perspective.SHARED,
        confidential=False,
        source_kind=SourceKind.CHAT_EXTRACTION,
        confidence=0.99,
        evidence_count=12,
        provenance={"shared_evidence": "repeated_model_agreement"},
    )
    with pytest.raises(InvalidSharedEvidence):
        asyncio.run(
            service.persist_memory_write(
                write, MemoryAuthContext(actor_id="jiao", room_id="room_group_home")
            )
        )


def test_shared_requires_common_fact_or_all_relevant_confirmations(service):
    common = MemoryWrite(
        content="薇薇明确要求你们都记住客厅叫小树屋",
        scope=MemoryScope.GROUP,
        memory_type=MemoryType.FACT,
        perspective=Perspective.SHARED,
        confidential=False,
        source_kind=SourceKind.EXPLICIT_USER_MEMORY,
        provenance={"shared_evidence": "weiwei_shared_instruction"},
    )
    confirmed = MemoryWrite(
        content="椒椒和老克都明确确认彼此信任",
        scope=MemoryScope.JIAO_LAOKE,
        memory_type=MemoryType.INFERENCE,
        perspective=Perspective.SHARED,
        confidential=False,
        source_kind=SourceKind.CHAT_EXTRACTION,
        confidence=0.8,
        evidence_count=2,
        provenance={
            "shared_evidence": "all_relevant_confirmed",
            "asserts_inner_state_of": ["jiao", "laoke"],
            "confirmed_actor_ids": ["jiao", "laoke"],
        },
    )
    assert asyncio.run(
        service.persist_memory_write(
            common, MemoryAuthContext(actor_id="weiwei", room_id="room_group_home")
        )
    ).perspective == "shared"
    assert asyncio.run(
        service.persist_memory_write(
            confirmed, MemoryAuthContext(actor_id="jiao", room_id="room_group_home")
        )
    ).perspective == "shared"


def test_scope_disclosure_creates_derived_group_record_without_mutating_source(service):
    private = asyncio.run(service.persist_memory_write(
        MemoryWrite(
            content="私聊中确认的偏好",
            scope=MemoryScope.WEIWEI_JIAO,
            memory_type=MemoryType.FACT,
            perspective=Perspective.WEIWEI,
            confidential=False,
            source_kind=SourceKind.EXPLICIT_USER_MEMORY,
        ),
        MemoryAuthContext(actor_id="weiwei", room_id="room_weiwei_jiao"),
    ))
    disclosed = asyncio.run(service.disclose_to_group(
        private.id,
        source_event_id=101,
        auth_context=MemoryAuthContext(actor_id="weiwei", room_id="room_group_home"),
    ))
    assert disclosed.scope == "group"
    assert disclosed.derived_from == private.id
    assert asyncio.run(service.get(private.id)).scope == "weiwei-jiao"


def test_relationship_write_never_overwrites_actor_identity(service):
    before = asyncio.run(service.identity_profile("jiao"))
    asyncio.run(service.persist_memory_write(
        MemoryWrite(
            content="椒椒在薇薇面前更爱撒娇",
            scope=MemoryScope.WEIWEI_JIAO,
            memory_type=MemoryType.INFERENCE,
            perspective=Perspective.WEIWEI,
            confidential=False,
            source_kind=SourceKind.CHAT_EXTRACTION,
            confidence=0.7,
        ),
        MemoryAuthContext(actor_id="weiwei", room_id="room_weiwei_jiao"),
    ))
    assert asyncio.run(service.identity_profile("jiao")) == before


def test_only_weiwei_can_create_user_attested_memory(service):
    write = MemoryWrite(
        content="没有 raw archive 的更早事实",
        scope=MemoryScope.WEIWEI_JIAO,
        memory_type=MemoryType.FACT,
        perspective=Perspective.WEIWEI,
        confidential=False,
        source_kind=SourceKind.USER_ATTESTED_MEMORY,
        provenance={
            "attested_by": "weiwei",
            "source": "weiwei_manual_attestation",
            "approximate_period": "2025",
        },
    )
    with pytest.raises(ForbiddenSourceKind):
        asyncio.run(
            service.persist_memory_write(
                write, MemoryAuthContext(actor_id="jiao", room_id="room_weiwei_jiao")
            )
        )
    saved = asyncio.run(
        service.persist_memory_write(
            write, MemoryAuthContext(actor_id="weiwei", room_id="room_weiwei_jiao")
        )
    )
    assert saved.provenance["attested_by"] == "weiwei"
    assert saved.provenance["source"] == "weiwei_manual_attestation"


def test_effective_confidence_is_deterministic_and_does_not_mutate_status():
    now = datetime(2026, 8, 26, tzinfo=timezone.utc)
    supported = now - timedelta(days=30)
    value = effective_confidence(
        0.8, supported, now, half_life_seconds=30 * 86400
    )
    assert value == pytest.approx(0.4)
    assert MemoryStatus.ACTIVE.value == "active"


def test_below_threshold_inference_never_enters_candidate_set(service):
    now = datetime(2026, 8, 26, tzinfo=timezone.utc)
    expired = asyncio.run(service.persist_memory_write(
        MemoryWrite(
            content="old inference",
            scope=MemoryScope.GROUP,
            memory_type=MemoryType.INFERENCE,
            perspective=Perspective.JIAO,
            confidential=False,
            source_kind=SourceKind.CHAT_EXTRACTION,
            confidence=0.8,
            provenance={"last_supported_at": "2025-01-01T00:00:00Z"},
        ),
        MemoryAuthContext(actor_id="jiao", room_id="room_group_home"),
    ))
    result = asyncio.run(service.search_authorized(
        "old inference",
        actor_id="jiao",
        room_id="room_group_home",
        now=now,
        half_life_seconds=30 * 86400,
        expiry_threshold=0.1,
    ))
    assert expired.id not in result.candidate_ids
    assert expired.status == "active"
