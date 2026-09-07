from types import SimpleNamespace
import pytest
from model_execution import ProviderChunk
from model_execution_contracts import ProviderUsage
from model_usage_store import InMemoryModelUsageStore
from tests.test_execution_context_builder import _profile


@pytest.mark.anyio
async def test_game_uses_bound_profile_and_only_table_context():
    from doudizhu_execution import decide
    class Profiles:
        async def resolve(self, actor, room):
            assert actor == 'jiao' and room == 'room_weiwei_jiao'
            return SimpleNamespace(primary=_profile())
    class Group:
        def build_stable_execution_components(self, actor, room):
            return dict(static_system=('kernel', 'persona', 'policy'), actor_prompt_version='v1', runtime_kernel_version='v1')
    class Runner:
        async def run(self, **kw):
            c = kw['context']
            assert c.stable_history == () and c.actor_memory_context is None
            assert c.tool_schema_hash == 'doudizhu.v1'
            await kw['on_attempt']('synthetic-attempt', ProviderUsage.from_provider_values(input_tokens=10,output_tokens=5), 'succeeded', True, 'unverified')
            yield ProviderChunk('final', {'text':'{"action":{"type":"bid","value":1},"say":"叫一分"}'})
    usage = InMemoryModelUsageStore()
    result = await decide(actor_id='jiao', payload={'phase':'bid','hand':['S3']}, profiles=Profiles(), group=Group(), runner=Runner(), usage_store=usage)
    assert result['action']['type'] == 'bid'
    assert (await usage.list_receipts())[0].execution_purpose == 'doudizhu'
    with pytest.raises(ValueError):
        await decide(actor_id='other', payload={}, profiles=None, group=None, runner=None, usage_store=None)


def test_game_endpoint_requires_its_own_service_key(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setenv('DOUDIZHU_SERVICE_KEY','synthetic-game')
    c=TestClient(main.app)
    assert c.post('/internal/doudizhu/decide',json={}).status_code == 403
    assert c.post('/internal/doudizhu/decide',headers={'Authorization':'Bearer synthetic-game'},json={}).status_code == 422
