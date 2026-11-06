#!/usr/bin/env python3
"""The 2026-07-28 counterpart of ``mock_mcp_resource_server.py``, built on the MCP SDK's
``MCPServer``: resource updates are published to ``subscriptions/listen`` streams (SEP-2575),
which replaced ``resources/subscribe`` in that revision. Shares the JSON store and its CLI with the
stdlib mock (``MOCK_MCP_STORE``), so ``mock_mcp_resource_server.py set URI TEXT`` drives both.

The SDK server also answers the 2025 ``initialize`` handshake; a client reaches the 2026 era with
``server/discover`` (Hermes: ``protocol: modern`` on the server's config).
"""

import sys
from pathlib import Path

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus, ResourcesListChanged, ResourceUpdated

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_mcp_resource_server as store  # shared store + CLI

bus = InMemorySubscriptionBus()
server = MCPServer("mock-mcp-resources-modern", subscriptions=bus)


@server.resource("mock://docs/{name}", mime_type="text/markdown")
def doc(name: str) -> str:
    resource = store.load_store().get(f"mock://docs/{name}")
    if resource is None:
        raise ValueError(f"Resource not found: mock://docs/{name}")
    return resource.get("text", "")


@server.tool()
def set_resource(uri: str, text: str) -> str:
    """Create or update a mock resource"""
    store.set_resource(uri, text)
    return f"set {uri}"


async def watch(closed: anyio.Event) -> None:
    previous = store.load_store()
    while True:
        with anyio.move_on_after(store.POLL_S):  # the store is a file other processes write: poll it
            await closed.wait()
        if closed.is_set():
            return
        current = store.load_store()
        if current == previous:
            continue
        # Unlike the stdlib mock this publishes before any handshake: with no listen stream open yet,
        # the bus has nobody to deliver to.
        for uri in store.changed_uris(previous, current):
            await bus.publish(ResourceUpdated(uri=uri))
        if set(previous) != set(current):
            await bus.publish(ResourcesListChanged())
        previous = current


async def main() -> None:
    closed = anyio.Event()  # set when the stdio session ends: the watcher stops with it
    async with anyio.create_task_group() as tg:
        tg.start_soon(watch, closed)
        try:
            await server.run_stdio_async()
        finally:
            closed.set()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(store.main(sys.argv[1:]))
    anyio.run(main)
