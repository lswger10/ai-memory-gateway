import asyncio

import httpx
import pytest

from provider_transport import PooledHttpTransport


def _request(**changes):
    return {
        "pool_key": "route-a",
        "base_url": "https://old.invalid/v1",
        "headers": {"Authorization": "Bearer old", "X-Route": "old"},
        "method": "POST",
        "path": "/v1/responses",
        "json_body": {},
        **changes,
    }


def _transport(handler):
    clients = []

    def factory(**kwargs):
        client = httpx.AsyncClient(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    return PooledHttpTransport(client_factory=factory), clients


def _echo(request):
    return httpx.Response(200, json={
        "url": str(request.url),
        "authorization": request.headers["Authorization"],
        "route": request.headers["X-Route"],
    })


@pytest.mark.anyio
@pytest.mark.parametrize(("changes", "expected"), [
    ({"base_url": "https://new.invalid/api"},
     {"url": "https://new.invalid/api/v1/responses", "authorization": "Bearer old", "route": "old"}),
    ({"headers": {"Authorization": "Bearer new", "X-Route": "old"}},
     {"url": "https://old.invalid/v1/responses", "authorization": "Bearer new", "route": "old"}),
    ({"headers": {"Authorization": "Bearer old", "X-Route": "new"}},
     {"url": "https://old.invalid/v1/responses", "authorization": "Bearer old", "route": "new"}),
])
async def test_route_setting_changes_reach_new_requests_and_close_idle_client(changes, expected):
    transport, clients = _transport(_echo)
    try:
        await transport.request(**_request())
        response = await transport.request(**_request(**changes))
        assert response.json() == expected
        assert clients[0].is_closed
        assert not clients[1].is_closed
        await transport.request(**_request(**changes))
        assert len(clients) == 2
    finally:
        await transport.close()
    assert all(client.is_closed for client in clients)


class _Chunks(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"first"
        yield b"last"


@pytest.mark.anyio
async def test_rotation_keeps_old_stream_alive_until_its_context_exits():
    def handler(request):
        if request.headers["Authorization"] == "Bearer old":
            return httpx.Response(200, stream=_Chunks())
        return _echo(request)

    transport, clients = _transport(handler)
    try:
        async with await transport.open_stream(**_request()) as response:
            chunks = response.aiter_bytes()
            assert await anext(chunks) == b"first"
            for number in range(5):
                result = await transport.request(**_request(headers={
                    "Authorization": f"Bearer new-{number}", "X-Route": "new"}))
                assert result.json()["authorization"] == f"Bearer new-{number}"
                assert not clients[0].is_closed
                assert sum(not client.is_closed for client in clients) == 2
            assert await anext(chunks) == b"last"
        assert clients[0].is_closed
        assert not clients[-1].is_closed
    finally:
        await transport.close()
    assert all(client.is_closed for client in clients)


@pytest.mark.anyio
async def test_rotation_keeps_inflight_request_alive_and_releases_it_on_failure():
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if request.headers["Authorization"] == "Bearer old":
            entered.set()
            await release.wait()
            raise httpx.ReadError("upstream disconnected")
        return _echo(request)

    transport, clients = _transport(handler)
    pending = asyncio.create_task(transport.request(**_request()))
    try:
        await entered.wait()
        await transport.request(**_request(headers={"Authorization": "Bearer new", "X-Route": "new"}))
        assert not clients[0].is_closed
        release.set()
        with pytest.raises(httpx.ReadError):
            await pending
        assert clients[0].is_closed
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await transport.close()


@pytest.mark.anyio
async def test_close_rejects_new_requests_and_drains_active_stream():
    transport, clients = _transport(lambda request: httpx.Response(200, stream=_Chunks()))
    async with await transport.open_stream(**_request()) as response:
        await transport.close()
        await transport.close()
        assert not clients[0].is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await transport.request(**_request())
        assert await response.aread() == b"firstlast"
    assert clients[0].is_closed
