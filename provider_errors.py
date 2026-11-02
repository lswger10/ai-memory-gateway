"""Bounded, sanitized provider diagnostics for backend logs only."""

import json

import anyio


def http_reason_code(status: int) -> str:
    return {
        400: "provider_bad_request", 401: "provider_auth_failed",
        402: "provider_payment_required", 403: "provider_auth_failed",
        410: "provider_model_retired", 429: "provider_rate_limited",
    }.get(status, "provider_unavailable" if 500 <= status < 600 else "generation_unavailable")


def _strings(value):
    if isinstance(value, str):
        yield value
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except ValueError:
                pass
            else:
                yield from _strings(decoded)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _strings(item)


async def provider_error_summary(response, *, credential, headers, body, context) -> str:
    # Never drain an unbounded/slow error stream just to enrich a diagnostic.
    if not hasattr(response, "aiter_bytes"):
        return ""
    data = bytearray()
    message = None
    try:
        with anyio.move_on_after(2):
            async for chunk in response.aiter_bytes(chunk_size=1024):
                data.extend(chunk)
                if len(data) >= 8192:
                    return ""
            message = data.decode("utf-8", errors="replace")
            try:
                payload = json.loads(message)
            except ValueError:
                # Malformed JSON may contain a request dump; only accept plain errors.
                if message.lstrip().startswith(("{", "[", "<")):
                    return ""
            else:
                error = payload.get("error", payload) if isinstance(payload, dict) else None
                message = error.get("message") if isinstance(error, dict) else error
            if not isinstance(message, str):
                return ""
        if message is None:
            return ""
    except Exception:
        # A diagnostic read must not replace the already-known HTTP failure.
        return ""
    sensitive = {credential, *headers.values(), *_strings(body),
                 *_strings(context.static_system), *_strings(context.stable_summary),
                 *_strings(context.stable_history),
                 *_strings([part.content for part in context.dynamic_tail])}
    for name, value in headers.items():
        if name.lower() in {"authorization", "proxy-authorization"}:
            sensitive.add(value.split(" ", 1)[-1])
    # Redact before truncation so a secret crossing the cutoff is not leaked.
    for value in sorted(filter(None, sensitive), key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    return " ".join("".join(char if char.isprintable() else " " for char in message).split())[:200]
