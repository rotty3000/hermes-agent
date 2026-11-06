"""Shared plumbing for the MCP subscription clients that deliver through dynamic webhook routes:
the ``events/*`` extension (``mcp_tool_events``) and resource subscriptions
(``mcp_tool_resource_subscriptions``).

Both store each subscription as a route in ``webhook_subscriptions.json`` carrying a client block
(``mcp_event`` / ``mcp_resource``), gate subscribing on the user's approval, wake their per-session
loops when a subscription changes in this process, and expose their tools through the same handler
shape.
"""

import asyncio
import base64
import hashlib
import json
import logging
import re
import secrets
from typing import Any, Awaitable, Callable, Optional

from tools.registry import tool_error

logger = logging.getLogger("tools.mcp_tool")

ROUTE_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
# The approval prompt shows the instructions in full, so they are bounded.
MAX_INSTRUCTIONS_CHARS = 2000


def route_name(prefix: str, server_name: str, identity: Any) -> str:
    """``<prefix>-<server slug>-<12 hex>``: stable for one (server, identity), safe as a URL path."""
    digest = hashlib.sha256(json.dumps([server_name, *identity], sort_keys=True).encode()).hexdigest()
    return f"{prefix}-{ROUTE_SLUG_RE.sub('-', server_name.lower()).strip('-')[:32]}-{digest[:12]}"


def new_secret() -> str:
    """Standard Webhooks symmetric secret: ``whsec_`` + base64 of 32 CSPRNG bytes."""
    return "whsec_" + base64.b64encode(secrets.token_bytes(32)).decode()


def server_routes(subscriptions: dict, server_name: str, block_key: str) -> dict[str, dict]:
    """The routes in *subscriptions* whose *block_key* block belongs to *server_name*."""
    return {name: route for name, route in subscriptions.items()
            if isinstance(route, dict) and isinstance(route.get(block_key), dict)
            and route[block_key].get("server") == server_name}


def load_routes(server_name: str, block_key: str) -> dict[str, dict]:
    from hermes_cli.webhook import _load_subscriptions
    return server_routes(_load_subscriptions(), server_name, block_key)


def subscribe_args_error(args: dict) -> Optional[str]:
    """Validation of the delivery arguments both subscribe tools take, run before the user is asked."""
    instructions = args.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        return "'instructions' must be a string"
    if isinstance(instructions, str) and len(instructions) > MAX_INSTRUCTIONS_CHARS:
        return f"'instructions' is limited to {MAX_INSTRUCTIONS_CHARS} characters"
    for key in ("deliver", "deliver_chat_id"):
        if args.get(key) is not None and not isinstance(args[key], str):
            return f"'{key}' must be a string"
    return None


def deliver_target(args: dict) -> str:
    """How the approval prompt names where each run's result goes."""
    deliver = args.get("deliver") or "log"
    return f"'{deliver}' chat {args['deliver_chat_id']}" if args.get("deliver_chat_id") else f"'{deliver}'"


def consent_error(*, question: str, detail: str, surface: str, title: str, what: str) -> Optional[str]:
    """Ask the user to approve a standing trigger for unattended runs, every time and fail closed.
    None when approved; otherwise the tool error to return. *what* names the subscription kind."""
    try:
        from tools.approval_prompt import request_elicitation_consent
        answer = request_elicitation_consent(question, detail, surface=surface, title=title)
    except Exception as exc:
        logger.error("MCP %s: approval check failed for %s: %s", what, surface, exc, exc_info=True)
        return tool_error(f"The {what} was blocked: the approval system was unavailable.")
    if answer == "accept":
        return None
    return tool_error(f"The user did not approve this {what}. Do not retry without explicit user direction.")


class Wakers:
    """Per-server wake-up events for loops running on (possibly different) event loops, so a
    subscription made in this process is acted on at once instead of after the loop's sleep."""

    def __init__(self):
        self._by_server: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]] = {}

    def register(self, server_name: str) -> tuple[asyncio.AbstractEventLoop, asyncio.Event]:
        waker = (asyncio.get_running_loop(), asyncio.Event())
        self._by_server.setdefault(server_name, set()).add(waker)
        return waker

    def unregister(self, server_name: str, waker) -> None:
        wakers = self._by_server.get(server_name)
        if wakers is not None:
            wakers.discard(waker)
            if not wakers:
                self._by_server.pop(server_name, None)

    def wake(self, server_name: str) -> None:
        for loop, event in list(self._by_server.get(server_name, ())):
            loop.call_soon_threadsafe(event.set)


def make_subscription_tool(op: str, label: str, call: Callable[[Any, dict], Awaitable[Any]], *,
                           validate: Callable[[dict], Optional[str]],
                           gate: Optional[Callable[[str, Any, dict], Optional[str]]] = None):
    """``(server_name, tool_timeout) -> handler`` for one subscription tool, like
    ``mcp_tool_handlers._make_utility_handler``: *validate* (an error message or None) and *gate*
    (setup checks and the approval; a tool error or None) run before any transport work, then
    ``call(server, args)`` runs on the MCP loop and its result is returned as JSON. *call* takes the
    session's request lock itself, around the requests it sends."""
    def _factory(server_name: str, tool_timeout: float):
        def _handler(args: dict, **kwargs) -> str:
            from tools import mcp_tool_discovery as _discovery  # lazy: import cycle
            from tools import mcp_tool_handlers as _handlers
            server = _discovery._get_connected_server_for_call(server_name)
            if not server or not server.session:
                return tool_error(f"MCP server '{server_name}' is not connected")
            if invalid := validate(args):
                return tool_error(invalid)
            if gate is not None and (refused := gate(server_name, server, args)):
                return refused

            async def _call():
                return json.dumps(await call(server, args), ensure_ascii=False)
            return _handlers._dispatch(
                server_name, server, op, _call, tool_timeout,
                (_handlers._handle_auth_error_and_retry, _handlers._handle_session_expired_and_retry),
                lambda exc: logger.error("MCP %s/%s failed: %s", server_name, label, exc))
        return _handler
    return _factory
