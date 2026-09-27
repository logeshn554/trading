"""Read-only Delta MCP account dashboard, with Google sign-in when public."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import base64
import binascii
import hmac
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request as UrlRequest, urlopen

from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
STRATEGY = json.loads((ROOT / "config/production_strategy.json").read_text(encoding="utf-8"))
MCP_ENV = os.environ.get("DELTA_MCP_ENV", "india_prod")
PUBLIC_MODE = os.environ.get("DASHBOARD_PUBLIC") == "1"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_ALLOWED_EMAIL = os.environ.get("GOOGLE_ALLOWED_EMAIL", "")
SESSION_SECRET = os.environ.get("DASHBOARD_SESSION_SECRET", "")
_login_lock = threading.Lock()
_pending_logins: dict[str, tuple[str, float]] = {}
CLIENT = DeltaMcpClient(environment=MCP_ENV)
_cache_lock = threading.Lock()
_cache: dict[str, object] = {"at": 0.0, "data": None}


def _rows(payload: object) -> tuple[list[dict], str | None]:
    """Extract Delta's result rows and pagination cursor without guessing values."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], None
    if not isinstance(payload, dict):
        raise DeltaMcpError("Delta returned an unexpected account response")
    value = payload.get("result", payload.get("data", payload))
    if isinstance(value, dict):
        value = value.get("result", value.get("data", value.get("rows")))
    if not isinstance(value, list):
        raise DeltaMcpError("Delta returned no account rows")
    meta = payload.get("meta", {})
    after = meta.get("after") if isinstance(meta, dict) else None
    return [row for row in value if isinstance(row, dict)], after if isinstance(after, str) else None


def _number(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value)) if value is not None else None
        return number if number is not None and number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _settlement_asset(position: dict) -> str | None:
    direct = position.get("settling_asset_symbol")
    if isinstance(direct, str):
        return direct
    product = position.get("product")
    if isinstance(product, dict):
        asset = product.get("settling_asset")
        if isinstance(asset, dict) and isinstance(asset.get("symbol"), str):
            return asset["symbol"]
    return None


def _pnl_by_asset(positions: list[dict], field: str) -> dict[str, str]:
    totals: dict[str, Decimal] = {}
    for row in positions:
        asset = _settlement_asset(row)
        number = _number(row.get(field))
        if asset and number is not None:
            totals[asset] = totals.get(asset, Decimal(0)) + number
    return {asset: str(value) for asset, value in sorted(totals.items())}


def trading_readiness(strategy: dict) -> dict:
    """Fail closed until the research and execution prerequisites are real."""
    blockers = list(strategy.get("blockers", []))
    limits = strategy.get("risk_limits", {})
    needed = (
        "max_trades_per_day", "daily_net_profit_target", "daily_max_loss",
        "per_trade_stop_loss", "per_trade_take_profit",
    )
    if any(limits.get(name) is None for name in needed):
        blockers.append("INR daily and per-trade risk limits are not configured")
    if not strategy.get("backtest", {}).get("selection_pass"):
        blockers.append("the saved strategy failed its research selection gate")
    blockers.append("no validated continuous Delta signal feed or live order lifecycle is implemented")
    return {
        "requested_enabled": False,
        "effective_enabled": False,
        "can_enable": False,
        "blockers": list(dict.fromkeys(blockers)),
    }


def build_snapshot(client: DeltaMcpClient) -> dict:
    now = datetime.now(timezone.utc)
    result: dict = {
        "as_of": now.isoformat(), "environment": client.environment, "public_mode": PUBLIC_MODE,
        "strategy": {"id": STRATEGY["strategy_id"], "live_orders_enabled": False,
                     "validation": "BLOCKED", "reason": STRATEGY["blockers"][0],
                     "risk_limits": STRATEGY["risk_limits"],
                     "trading_readiness": trading_readiness(STRATEGY)},
        "connection": "unavailable", "wallets": [], "positions": [],
        "fills": [], "fills_after": None, "transactions": [],
        "transactions_after": None, "open_orders": [],
        "realized_pnl_open_positions": {}, "unrealized_pnl_open_positions": {},
        "ticker": None, "errors": {},
    }
    tools = client.available_tools()
    if "get_connection_status" in tools:
        try:
            result["connection_status"] = client.call("get_connection_status")
        except DeltaMcpError as exc:
            result["errors"]["connection"] = str(exc)
    if "get_ticker" in tools:
        try:
            result["ticker"] = client.call("get_ticker", {"symbol": "ETHUSD"})
        except DeltaMcpError as exc:
            result["errors"]["ticker"] = str(exc)
    if "get_wallet_balances" not in tools:
        result["connection"] = "needs_read_data_key"
        return result
    result["connection"] = "connected"

    since_us = int((now - timedelta(days=30)).timestamp() * 1_000_000)
    calls = (
        ("wallets", "get_wallet_balances", {}),
        ("positions", "get_margined_positions", {}),
        ("fills", "get_fills", {"start_time_us": since_us, "page_size": 100}),
        ("transactions", "get_wallet_transactions", {"start_time_us": since_us, "page_size": 100}),
        ("open_orders", "get_open_orders", {"page_size": 100}),
    )
    for key, tool, arguments in calls:
        if tool not in tools:
            result["errors"][key] = f"Delta MCP tool {tool} is unavailable"
            continue
        try:
            payload = client.call(tool, arguments)
            rows, cursor = _rows(payload)
            result[key] = rows
            if key in {"fills", "transactions"}:
                result[f"{key}_after"] = cursor
        except DeltaMcpError as exc:
            result["errors"][key] = str(exc)
    result["realized_pnl_open_positions"] = _pnl_by_asset(result["positions"], "realized_pnl")
    result["unrealized_pnl_open_positions"] = _pnl_by_asset(result["positions"], "unrealized_pnl")
    return result


def snapshot(*, fresh: bool = False) -> dict:
    with _cache_lock:
        if not fresh and _cache["data"] is not None and time.monotonic() - float(_cache["at"]) < 30:
            return _cache["data"]  # type: ignore[return-value]
        previous = _cache["data"]
        if fresh or (isinstance(previous, dict) and previous.get("connection") == "needs_read_data_key"):
            CLIENT.close()  # Pick up credentials added with the MCP login command.
        data = build_snapshot(CLIENT)
        _cache.update(at=time.monotonic(), data=data)
        return data


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        if self.path.startswith("/auth/"):
            return  # OAuth callback URLs contain a short-lived authorization code.
        super().log_message(format, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        host = self.headers.get("Host", "")
        allowed_hosts = {"127.0.0.1", "localhost"}
        if PUBLIC_MODE:
            allowed_hosts.update(filter(None, os.environ.get("DASHBOARD_ALLOWED_HOSTS", "").split(",")))
            allowed_hosts.add(os.environ.get("RENDER_EXTERNAL_HOSTNAME", ""))
        if host.split(":", 1)[0].lower() not in {name.strip().lower() for name in allowed_hosts}:
            self.send_error(403, "Local dashboard only")
            return
        if PUBLIC_MODE and parsed.path == "/auth/login":
            self._start_google_login(host)
            return
        if PUBLIC_MODE and parsed.path == "/auth/callback":
            self._finish_google_login(parsed, host)
            return
        if PUBLIC_MODE and parsed.path == "/auth/logout":
            self._redirect("/auth/login", "dashboard_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax")
            return
        if PUBLIC_MODE and not self._authenticated():
            if parsed.path.startswith("/api/"):
                self._send_json({"error": "Google sign-in required"}, 401)
            else:
                self._redirect("/auth/login")
            return
        if parsed.path == "/api/snapshot":
            try:
                fresh = parse_qs(parsed.query).get("fresh", ["0"])[0] == "1"
                self._send_json(snapshot(fresh=fresh))
            except (DeltaMcpError, ValueError) as exc:
                self._send_json({"error": str(exc), "connection": "unavailable"}, 503)
        elif parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif parsed.path in {"/", "/index.html", "/app.js", "/styles.css"}:
            self.path = parsed.path
            super().do_GET()
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        self.send_error(405, "Dashboard is read-only")

    def _cookie(self, name: str) -> str | None:
        for item in self.headers.get("Cookie", "").split(";"):
            key, separator, value = item.strip().partition("=")
            if separator and key == name:
                return value
        return None

    def _authenticated(self) -> bool:
        token = self._cookie("dashboard_session")
        if not token or len(token) > 2048:
            return False
        encoded, separator, signature = token.partition(".")
        if not separator:
            return False
        expected = hmac.new(SESSION_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        try:
            body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
            return (body.get("email") == GOOGLE_ALLOWED_EMAIL and
                    int(body.get("expires", 0)) > time.time())
        except (ValueError, TypeError, binascii.Error):
            return False

    def _redirect(self, location: str, cookie: str | None = None):
        self.send_response(302)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    @staticmethod
    def _redirect_uri(host: str) -> str:
        return f"https://{host}/auth/callback"

    def _start_google_login(self, host: str):
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with _login_lock:
            now = time.time()
            for old_state, (_, expiry) in list(_pending_logins.items()):
                if expiry < now:
                    del _pending_logins[old_state]
            if len(_pending_logins) >= 1000:
                del _pending_logins[next(iter(_pending_logins))]
            _pending_logins[state] = (nonce, now + 600)
        query = urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": self._redirect_uri(host),
            "response_type": "code", "scope": "openid email",
            "state": state, "nonce": nonce,
            "login_hint": GOOGLE_ALLOWED_EMAIL,
        })
        self._redirect("https://accounts.google.com/o/oauth2/v2/auth?" + query,
                       f"oauth_state={state}; Path=/auth/callback; Max-Age=600; HttpOnly; Secure; SameSite=Lax")

    def _finish_google_login(self, parsed, host: str):
        params = parse_qs(parsed.query)
        state = params.get("state", [""])[0]
        code = params.get("code", [""])[0]
        if not state or not code or not hmac.compare_digest(state, self._cookie("oauth_state") or ""):
            self.send_error(403, "Google login state mismatch")
            return
        with _login_lock:
            pending = _pending_logins.pop(state, None)
        if not pending or pending[1] < time.time():
            self.send_error(403, "Google login expired")
            return
        try:
            body = urlencode({
                "code": code, "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": self._redirect_uri(host),
                "grant_type": "authorization_code",
            }).encode()
            request = UrlRequest("https://oauth2.googleapis.com/token", data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(request, timeout=15) as response:
                token = json.load(response).get("id_token")
            if not isinstance(token, str):
                raise ValueError("Google returned no ID token")
            from google.auth.transport.requests import Request as GoogleRequest
            from google.oauth2 import id_token
            claims = id_token.verify_oauth2_token(token, GoogleRequest(), GOOGLE_CLIENT_ID)
            if (claims.get("iss") not in {"https://accounts.google.com", "accounts.google.com"} or
                claims.get("nonce") != pending[0] or
                claims.get("email_verified") is not True or
                claims.get("email") != GOOGLE_ALLOWED_EMAIL):
                raise ValueError("Google identity is not allowed")
        except Exception:
            self.send_error(403, "Google sign-in failed")
            return
        payload = {"email": GOOGLE_ALLOWED_EMAIL, "expires": int(time.time()) + 43200}
        encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
        signature = hmac.new(SESSION_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        self._redirect("/", f"dashboard_session={encoded}.{signature}; Path=/; Max-Age=43200; HttpOnly; Secure; SameSite=Lax")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()

    def _send_json(self, value: object, status: int = 200):
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8000):
    if PUBLIC_MODE and (not all((GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_ALLOWED_EMAIL)) or
                        len(SESSION_SECRET) < 32 or
                        not os.environ.get("DELTA_API_KEY") or not os.environ.get("DELTA_API_SECRET")):
        raise SystemExit("Public dashboard requires Google OAuth, a session secret, and both Delta API credential variables")
    bind = "0.0.0.0" if PUBLIC_MODE else "127.0.0.1"
    server = ThreadingHTTPServer((bind, port), DashboardHandler)
    print(f"Delta account dashboard listening on {bind}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        CLIENT.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    run_server(port)
