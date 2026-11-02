from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx


def _endpoint_path_for_base_url(base_url: str, path: str) -> str:
    base_segments = tuple(
        segment for segment in urlsplit(base_url).path.split("/") if segment
    )
    path_segments = tuple(segment for segment in path.split("/") if segment)
    maximum_overlap = min(len(base_segments), len(path_segments))
    for size in range(maximum_overlap, 0, -1):
        if base_segments[-size:] == path_segments[:size]:
            remaining = path_segments[size:]
            return "/" + "/".join(remaining)
    return "/" + "/".join(path_segments)


class EnvironmentCredentialResolver:
    def resolve(self, credential_ref: str) -> str:
        if not isinstance(credential_ref, str) or not credential_ref.startswith("env:"):
            raise ValueError("credential_ref must use the env: scheme")
        name = credential_ref.removeprefix("env:")
        if not name:
            raise ValueError("credential environment variable is missing")
        value = os.getenv(name)
        if not value:
            raise ValueError(f"credential environment variable is not configured: {name}")
        return value


@dataclass(slots=True)
class _ClientSlot:
    signature: tuple[str, tuple[tuple[str, str], ...]]
    client: Any
    active: int = 0


class PooledHttpTransport:
    def __init__(
        self,
        *,
        client_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._client_factory = client_factory or httpx.AsyncClient
        self._timeout_seconds = timeout_seconds
        self._clients: dict[str, _ClientSlot] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @asynccontextmanager
    async def _client(
        self, pool_key: str, base_url: str, headers: dict[str, str]
    ):
        signature = (base_url.rstrip("/"), tuple(sorted(headers.items())))
        retired = None
        async with self._lock:
            if self._closed:
                raise RuntimeError("provider transport is closed")
            slot = self._clients.get(pool_key)
            if slot is None or slot.signature != signature:
                client = self._client_factory(
                    base_url=signature[0],
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                if slot is not None and slot.active == 0:
                    retired = slot.client
                slot = _ClientSlot(signature=signature, client=client)
                self._clients[pool_key] = slot
            slot.active += 1
        try:
            if retired is not None:
                await retired.aclose()
            yield slot.client
        finally:
            async with self._lock:
                slot.active -= 1
                # Replaced clients stay alive until their last request or stream exits.
                should_close = slot.active == 0 and self._clients.get(pool_key) is not slot
            if should_close:
                await slot.client.aclose()

    async def request(
        self,
        *,
        pool_key: str,
        base_url: str,
        headers: dict[str, str],
        method: str,
        path: str,
        json_body: dict[str, Any],
    ) -> Any:
        async with self._client(pool_key, base_url, headers) as client:
            endpoint_path = _endpoint_path_for_base_url(base_url, path)
            return await client.request(method, endpoint_path, json=json_body)

    async def open_stream(
        self,
        *,
        pool_key: str,
        base_url: str,
        headers: dict[str, str],
        method: str,
        path: str,
        json_body: dict[str, Any],
    ):
        @asynccontextmanager
        async def stream():
            async with self._client(pool_key, base_url, headers) as client:
                endpoint_path = _endpoint_path_for_base_url(base_url, path)
                async with client.stream(method, endpoint_path, json=json_body) as response:
                    yield response

        return stream()

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            clients = [slot.client for slot in self._clients.values() if slot.active == 0]
            self._clients.clear()
            self._closed = True
        for client in clients:
            await client.aclose()
