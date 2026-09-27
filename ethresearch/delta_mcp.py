"""Read-only stdio client for Delta Exchange's local MCP server.

Only the tools in READ_TOOLS can be called through this bridge. Credentials are
owned by the MCP process and are never sent to the browser or logged here.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from typing import Any


READ_TOOLS = frozenset({
    "get_connection_status", "get_ticker", "get_recent_trades", "get_candles",
    "get_product", "get_wallet_balances",
    "get_margined_positions", "get_wallet_transactions", "get_fills",
    "get_open_orders",
})

TRADE_TOOLS = frozenset({
    "place_order", "cancel_order", "batch_place_orders", "close_all_positions",
})


class DeltaMcpError(RuntimeError):
    pass


def unpack_tool_result(result: dict[str, Any]) -> Any:
    """Return the server payload while preserving response metadata/cursors."""
    if result.get("isError"):
        detail = next((c.get("text") for c in result.get("content", [])
                       if isinstance(c, dict) and c.get("type") == "text"), None)
        raise DeltaMcpError(str(detail or "Delta MCP tool failed"))
    value = result.get("structuredContent")
    if value is None:
        texts = [c.get("text", "") for c in result.get("content", [])
                 if isinstance(c, dict) and c.get("type") == "text"]
        if not texts:
            raise DeltaMcpError("Delta MCP returned no data")
        try:
            value = json.loads(texts[0])
        except json.JSONDecodeError as exc:
            raise DeltaMcpError("Delta MCP returned a non-JSON response") from exc
    if isinstance(value, dict) and value.get("success") is False:
        raise DeltaMcpError("Delta rejected the account request")
    return value


class DeltaMcpClient:
    def __init__(self, environment: str = "india_prod", timeout: float = 25.0,
                 command: tuple[str, ...] | None = None, allow_trading: bool = False):
        if environment not in {"india_prod", "india_testnet"}:
            raise ValueError("Unsupported Delta MCP environment")
        self.environment = environment
        self.timeout = timeout
        self.allow_trading = allow_trading
        self.command = command or (("delta-exchange-mcp",) if shutil.which("delta-exchange-mcp")
                                   else ("uvx", "delta-exchange-mcp==0.7.0"))
        self._lock = threading.RLock()
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._tools: set[str] = set()

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._responses = queue.Queue()
        self._tools.clear()
        env = os.environ.copy()
        env["DELTA_MCP_ENV"] = self.environment
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1, env=env,
            )
        except OSError as exc:
            raise DeltaMcpError(f"Cannot start Delta MCP: {type(exc).__name__}") from exc
        threading.Thread(target=self._read_stdout,
                         args=(self._proc, self._responses), daemon=True).start()
        self._exchange("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "ethresearch-account-dashboard", "version": "1.0"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = self._exchange("tools/list", {})
        self._tools = {item.get("name") for item in listed.get("tools", [])
                       if isinstance(item, dict) and isinstance(item.get("name"), str)}

    @staticmethod
    def _read_stdout(proc: subprocess.Popen[str], responses: queue.Queue[dict[str, Any]]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    responses.put(payload)
            except json.JSONDecodeError:
                continue
        responses.put({"_closed": True})

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise DeltaMcpError("Delta MCP server is not running")
        try:
            self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise DeltaMcpError("Delta MCP connection closed") from exc

    def _exchange(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            try:
                response = self._responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise DeltaMcpError(f"Delta MCP timed out during {method}") from exc
            if response.get("_closed"):
                raise DeltaMcpError("Delta MCP server closed its connection")
            if response.get("id") != request_id:
                continue  # Notifications or server-initiated requests.
            if "error" in response:
                error = response["error"]
                message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
                raise DeltaMcpError(f"Delta MCP {method}: {message}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise DeltaMcpError(f"Delta MCP {method} returned no result")
            return result

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        allowed = READ_TOOLS | (TRADE_TOOLS if self.allow_trading else frozenset())
        if name not in allowed:
            raise DeltaMcpError(f"Tool {name} is not allowed by the read-only dashboard")
        with self._lock:
            self._start()
            if name not in self._tools:
                raise DeltaMcpError(f"Delta MCP tool {name} is unavailable; verify your Delta API key permissions")
            return unpack_tool_result(self._exchange("tools/call", {
                "name": name, "arguments": arguments or {},
            }))

    def available_tools(self) -> set[str]:
        with self._lock:
            self._start()
            return set(self._tools)

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
            self._tools.clear()
