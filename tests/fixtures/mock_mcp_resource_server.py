#!/usr/bin/env python3
"""A small, dependency-free stdio MCP server that mocks the MCP resource model, for testing
resource subscriptions end to end (stdlib only, so it does not depend on an SDK version).

Implements (spec 2025-11-25): initialize, ping, resources/list, resources/read,
resources/templates/list, resources/subscribe, resources/unsubscribe, tools/list, tools/call,
prompts/list; and sends notifications/resources/updated (to subscribers of a changed URI) and
notifications/resources/list_changed (when URIs are added or removed).

The resources live in a JSON file shared by every server process, because each MCP client
process spawns its own stdio child. Change it from outside with the CLI below (or the server's
``set_resource`` / ``delete_resource`` tools) and every running server notices within ~0.5 s:

    mock_mcp_resource_server.py                    # serve on stdio (what an MCP client runs)
    mock_mcp_resource_server.py set URI TEXT       # create or update a resource
    mock_mcp_resource_server.py delete URI
    mock_mcp_resource_server.py list

Environment: MOCK_MCP_STORE (the JSON file; default ./mock_mcp_resources.json),
MOCK_MCP_LOG (optional JSONL file recording every message received and sent).
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
POLL_S = 0.5
TEMPLATES = [{"uriTemplate": "mock://docs/{name}", "name": "doc", "description": "A mock document by name",
              "mimeType": "text/markdown"}]
TOOLS = [
    {"name": "set_resource", "description": "Create or update a mock resource",
     "inputSchema": {"type": "object", "properties": {"uri": {"type": "string"}, "text": {"type": "string"}},
                     "required": ["uri", "text"]}},
    {"name": "delete_resource", "description": "Delete a mock resource",
     "inputSchema": {"type": "object", "properties": {"uri": {"type": "string"}}, "required": ["uri"]}},
]


def store_path() -> Path:
    return Path(os.environ.get("MOCK_MCP_STORE") or "mock_mcp_resources.json").expanduser()


def load_store() -> dict:
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_store(data: dict) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)  # atomic: a polling server never reads half a file


def set_resource(uri: str, text: str) -> None:
    data = load_store()
    name = uri.rstrip("/").rsplit("/", 1)[-1] or uri
    data[uri] = {"name": name, "mimeType": "text/markdown", "text": text, "version": data.get(uri, {}).get("version", 0) + 1}
    save_store(data)


def delete_resource(uri: str) -> bool:
    data = load_store()
    if data.pop(uri, None) is None:
        return False
    save_store(data)
    return True


def changed_uris(previous: dict, current: dict) -> list[str]:
    """URIs added, removed or changed between two snapshots of the store."""
    return sorted(uri for uri in set(previous) | set(current) if previous.get(uri) != current.get(uri))


class Server:
    def __init__(self):
        self.out_lock = threading.Lock()
        self.subscriptions: set[str] = set()
        self.sub_lock = threading.Lock()
        self.initialized = threading.Event()
        self.closed = threading.Event()  # set when the client closes stdin: the watcher stops with it
        log = os.environ.get("MOCK_MCP_LOG")
        self.log = open(log, "a", encoding="utf-8", buffering=1) if log else None

    def record(self, direction: str, message: dict) -> None:
        if self.log:
            self.log.write(json.dumps({"t": time.time(), "pid": os.getpid(), "dir": direction, "msg": message}) + "\n")

    def send(self, message: dict) -> None:
        self.record("out", message)
        with self.out_lock:
            sys.stdout.write(json.dumps(message) + "\n")
            sys.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    # ---- requests ----

    def handle(self, method: str, params: dict):
        if method == "initialize":
            requested = params.get("protocolVersion")
            return {"protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                    "capabilities": {"resources": {"subscribe": True, "listChanged": True},
                                     "tools": {}, "prompts": {"listChanged": True}},
                    "serverInfo": {"name": "mock-mcp-resources", "version": "0.1.0"}}
        if method == "ping":
            return {}
        if method == "resources/list":
            return {"resources": [{"uri": uri, "name": r.get("name", uri), "mimeType": r.get("mimeType", "text/plain")}
                                  for uri, r in sorted(load_store().items())]}
        if method == "resources/templates/list":
            return {"resourceTemplates": TEMPLATES}
        if method == "resources/read":
            uri = params.get("uri")
            resource = load_store().get(uri)
            if resource is None:
                raise RpcError(-32002, f"Resource not found: {uri}", {"uri": uri})
            return {"contents": [{"uri": uri, "mimeType": resource.get("mimeType", "text/plain"),
                                  "text": resource.get("text", "")}]}
        if method == "resources/subscribe":
            uri = self._uri(params)
            with self.sub_lock:
                self.subscriptions.add(uri)
            return {}
        if method == "resources/unsubscribe":
            uri = self._uri(params)
            with self.sub_lock:
                self.subscriptions.discard(uri)
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name == "set_resource":
                set_resource(str(args["uri"]), str(args["text"]))
                return {"content": [{"type": "text", "text": f"set {args['uri']}"}]}
            if name == "delete_resource":
                found = delete_resource(str(args["uri"]))
                return {"content": [{"type": "text", "text": f"deleted {args['uri']}" if found else "not found"}],
                        "isError": not found}
            raise RpcError(-32602, f"Unknown tool: {name}")
        if method == "prompts/list":
            return {"prompts": []}
        raise RpcError(-32601, f"Method not found: {method}")

    @staticmethod
    def _uri(params: dict) -> str:
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            raise RpcError(-32602, "'uri' is required")
        return uri

    # ---- change detection ----

    def watch(self) -> None:
        previous = load_store()
        while not self.closed.wait(POLL_S):  # the store is a file other processes write: poll it
            current = load_store()
            if current == previous:
                continue
            if not self.initialized.is_set():
                previous = current
                continue
            with self.sub_lock:
                targets = [uri for uri in changed_uris(previous, current) if uri in self.subscriptions]
            for uri in targets:
                self.notify("notifications/resources/updated", {"uri": uri})
            if set(previous) != set(current):
                self.notify("notifications/resources/list_changed", {})
            previous = current

    def serve(self) -> None:
        threading.Thread(target=self.watch, daemon=True).start()
        try:
            self._read_requests()
        finally:
            self.closed.set()

    def _read_requests(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            self.record("in", message)
            method, msg_id = message.get("method"), message.get("id")
            if method is None:
                continue  # a response to something we never send
            if msg_id is None:  # notification
                if method == "notifications/initialized":
                    self.initialized.set()
                continue
            try:
                result = self.handle(method, message.get("params") or {})
                self.send({"jsonrpc": "2.0", "id": msg_id, "result": result})
            except RpcError as exc:
                self.send({"jsonrpc": "2.0", "id": msg_id, "error": exc.as_dict()})
            except Exception as exc:  # a bug in the mock: report it, keep serving
                self.send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": repr(exc)}})


class RpcError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **({"data": self.data} if self.data is not None else {})}


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "serve":
        Server().serve()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "set" and len(rest) == 2:
        set_resource(rest[0], rest[1])
        return 0
    if command == "delete" and len(rest) == 1:
        return 0 if delete_resource(rest[0]) else 1
    if command == "list" and not rest:
        print(json.dumps(load_store(), indent=2, sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
