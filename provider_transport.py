from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx


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

    async def _client(
        self, pool_key: str, base_url: str, headers: dict[str, str]
    ) -> Any:
        signature = (base_url.rstrip("/"), tuple(sorted(headers.items())))
        async with self._lock:
            if self._closed:
                raise RuntimeError("provider transport is closed")
            slot = self._clients.get(pool_key)
            if slot is not None:
                if slot.signature != signature:
                    raise RuntimeError("pool_key cannot be reused with different route settings")
                return slot.client
            client = self._client_factory(
                base_url=signature[0],
                headers=headers,
                timeout=self._timeout_seconds,
            )
            self._clients[pool_key] = _ClientSlot(signature=signature, client=client)
            return client

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
        client = await self._client(pool_key, base_url, headers)
        return await client.request(method, path, json=json_body)

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
        client = await self._client(pool_key, base_url, headers)
        return client.stream(method, path, json=json_body)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            clients = [slot.client for slot in self._clients.values()]
            self._clients.clear()
            self._closed = True
        for client in clients:
            await client.aclose()
