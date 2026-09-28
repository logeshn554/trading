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
from ethresearch.trader import GTrXLAutomatedTrader


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
STRATEGY = json.loads((ROOT / "config/production_strategy.json").read_text(encoding="utf-8"))
MCP_ENV = os.environ.get("DELTA_MCP_ENV", "india_prod")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_ALLOWED_EMAIL = os.environ.get("GOOGLE_ALLOWED_EMAIL", "")
PUBLIC_MODE = (os.environ.get("DASHBOARD_PUBLIC") == "1") and bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
SESSION_SECRET = os.environ.get("DASHBOARD_SESSION_SECRET", "")
OTP_SECRET = os.environ.get("OTP_SECRET") or SESSION_SECRET or "delta_india_otp_secret_477554"
CORRECT_OTP = os.environ.get("DASHBOARD_OTP", "477554")
OTP_LOCKOUT_SECONDS = 30.0
_otp_lockouts: dict[str, float] = {}
_lockout_lock = threading.Lock()
_login_lock = threading.Lock()
_pending_logins: dict[str, tuple[str, float]] = {}
CLIENT = DeltaMcpClient(environment=MCP_ENV, allow_trading=STRATEGY.get("live_order_submission_enabled", False))
TRADER = GTrXLAutomatedTrader(client=CLIENT, strategy_config=STRATEGY)
_cache_lock = threading.Lock()
_cache: dict[str, object] = {"at": 0.0, "data": None}
_cached_ip: str | None = None


def _create_otp_token() -> str:
    expires = int(time.time() + 86400 * 7)  # 7 days session
    payload = json.dumps({"otp_verified": True, "expires": expires})
    encoded = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    signature = hmac.new(OTP_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _is_otp_token_valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    encoded, separator, signature = token.partition(".")
    if not separator:
        return False
    expected = hmac.new(OTP_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False
    try:
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        return bool(data.get("otp_verified") and int(data.get("expires", 0)) > time.time())
    except Exception:
        return False


def _get_lockout_remaining(client_ip: str) -> int:
    with _lockout_lock:
        until = _otp_lockouts.get(client_ip, 0.0)
        remaining = until - time.time()
        if remaining <= 0:
            _otp_lockouts.pop(client_ip, None)
            return 0
        return int(remaining) + 1


def get_outbound_ip() -> str:
    global _cached_ip
    if _cached_ip:
        return _cached_ip
    endpoints = (
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://icanhazip.com",
        "https://ifconfig.me/ip",
    )
    try:
        import requests
        for endpoint in endpoints:
            try:
                resp = requests.get(endpoint, timeout=6, headers={"User-Agent": "curl/7.68.0"})
                if resp.status_code == 200:
                    ip = resp.text.strip()
                    if ip and len(ip) <= 45 and "<" not in ip:
                        _cached_ip = ip
                        print(f"Detected outbound IP: {_cached_ip}", flush=True)
                        return _cached_ip
            except Exception:
                continue
    except Exception:
        pass

    for endpoint in endpoints:
        try:
            req = UrlRequest(endpoint, headers={"User-Agent": "curl/7.68.0"})
            with urlopen(req, timeout=6) as resp:
                ip = resp.read().decode("utf-8").strip()
                if ip and len(ip) <= 45 and "<" not in ip:
                    _cached_ip = ip
                    print(f"Detected outbound IP: {_cached_ip}", flush=True)
                    return _cached_ip
        except Exception:
            continue
    return "Unavailable"


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
    """Evaluate if strategy passes risk and readiness gates."""
    blockers = [b for b in strategy.get("blockers", []) if b not in (
        "INR daily and per-trade risk limits are not configured",
        "the saved strategy failed its research selection gate",
    )]
    limits = strategy.setdefault("risk_limits", {})
    defaults = {
        "max_trades_per_day": 50,
        "contract_size": 1,
        "daily_net_profit_target": 1200.0,
        "daily_max_loss": 1000.0,
        "per_trade_stop_loss": 300.0,
        "per_trade_take_profit": 600.0,
        "confidence_threshold": 0.55,
        "min_trend_spread": 0.0005,
    }
    for k, v in defaults.items():
        if limits.get(k) is None:
            limits[k] = v

    needed = (
        "max_trades_per_day", "daily_net_profit_target", "daily_max_loss",
        "per_trade_stop_loss", "per_trade_take_profit",
    )
    if any(limits.get(name) is None for name in needed):
        blockers.append("INR daily and per-trade risk limits are not configured")
    if not strategy.get("backtest", {}).get("selection_pass"):
        blockers.append("the saved strategy failed its research selection gate")
    can_enable = len(blockers) == 0
    enabled = bool(strategy.get("live_order_submission_enabled", False)) and can_enable
    return {
        "requested_enabled": bool(strategy.get("live_order_submission_enabled", False)),
        "effective_enabled": enabled,
        "can_enable": can_enable,
        "blockers": list(dict.fromkeys(blockers)),
    }


def build_snapshot(client: DeltaMcpClient) -> dict:
    now = datetime.now(timezone.utc)
    readiness = trading_readiness(STRATEGY)
    cur_lots = STRATEGY.get("risk_limits", {}).get("contract_size", 1)
    lot_str = f"{cur_lots} lot{'s' if cur_lots != 1 else ''}"
    reason = f"Algorithmic execution active ({lot_str}) with INR risk caps enforced" if readiness["effective_enabled"] else (STRATEGY["blockers"][0] if STRATEGY.get("blockers") else "Live Trading Enabled")
    result: dict = {
        "as_of": now.isoformat(), "environment": client.environment, "public_mode": PUBLIC_MODE,
        "outbound_ip": get_outbound_ip(),
        "strategy": {"id": STRATEGY["strategy_id"],
                     "live_orders_enabled": readiness["effective_enabled"],
                     "validation": "ACTIVE · LIVE" if readiness["effective_enabled"] else "BLOCKED",
                     "reason": reason,
                     "risk_limits": STRATEGY["risk_limits"],
                     "trading_readiness": readiness},
        "connection": "unavailable", "wallets": [], "positions": [],
        "fills": [], "fills_after": None, "transactions": [],
        "transactions_after": None, "open_orders": [],
        "realized_pnl_open_positions": {}, "unrealized_pnl_open_positions": {},
        "ticker": None, "errors": {},
        "gtrxl_trader": TRADER.get_status(),
        "paper_trading": TRADER.paper_account.get_summary() if hasattr(TRADER, "paper_account") else {},
    }
    tools = set()
    try:
        tools = client.available_tools()
    except Exception as exc:
        result["errors"]["connection"] = str(exc)
        result["connection"] = "unavailable"
        return result
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

    def _client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if hasattr(self, "client_address") and self.client_address:
            return str(self.client_address[0])
        return "127.0.0.1"

    def _otp_authenticated(self) -> bool:
        token = self._cookie("otp_session")
        return _is_otp_token_valid(token)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        host = self.headers.get("Host", "")
        host_name = host.split(":", 1)[0].lower()
        if PUBLIC_MODE:
            allowed_hosts = {"127.0.0.1", "localhost"}
            allowed_hosts.update(filter(None, os.environ.get("DASHBOARD_ALLOWED_HOSTS", "").split(",")))
            allowed_hosts.add(os.environ.get("RENDER_EXTERNAL_HOSTNAME", ""))
            if host_name not in {name.strip().lower() for name in allowed_hosts} and not host_name.endswith(".onrender.com"):
                self.send_error(403, "Invalid host")
                return

        # OTP status check endpoint
        if parsed.path == "/api/auth/otp-status":
            client_ip = self._client_ip()
            lockout = _get_lockout_remaining(client_ip)
            self._send_json({
                "authenticated": self._otp_authenticated(),
                "lockout_remaining": lockout,
                "otp_required": True,
            })
            return

        if PUBLIC_MODE and parsed.path == "/auth/login":
            self._start_google_login(host)
            return
        if PUBLIC_MODE and parsed.path == "/auth/callback":
            self._finish_google_login(parsed, host)
            return
        if parsed.path in {"/auth/logout", "/api/auth/logout"}:
            self.send_response(200 if parsed.path.startswith("/api/") else 302)
            if not parsed.path.startswith("/api/"):
                self.send_header("Location", "/auth/login" if PUBLIC_MODE else "/")
            self.send_header("Set-Cookie", "otp_session=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")
            self.send_header("Set-Cookie", "dashboard_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax")
            self.end_headers()
            if parsed.path.startswith("/api/"):
                self.wfile.write(b'{"success": true, "logged_out": true}')
            return

        if PUBLIC_MODE and not self._authenticated():
            if parsed.path.startswith("/api/"):
                self._send_json({"error": "Google sign-in required"}, 401)
            else:
                self._redirect("/auth/login")
            return

        if parsed.path == "/api/snapshot":
            has_auth = self._otp_authenticated() or (PUBLIC_MODE and self._authenticated())
            if PUBLIC_MODE and not has_auth:
                self._send_json({"error": "OTP authentication required", "requires_otp": True}, 401)
                return
            try:
                fresh = parse_qs(parsed.query).get("fresh", ["0"])[0] == "1"
                self._send_json(snapshot(fresh=fresh))
            except Exception as exc:
                self._send_json({"error": str(exc), "connection": "unavailable"}, 200)
            return
        elif parsed.path == "/api/gtrxl/status":
            try:
                from ethresearch.gtrxl import GTrXLActorCritic
                info = {
                    "status": "ready",
                    "architecture": "GTrXL (Gated Transformer-XL)",
                    "framework": "PyTorch",
                    "components": ["GRUGate", "RelMultiHeadAttention", "GTrXLBlock", "ActorCriticHead", "AuxHorizonHead"],
                    "horizons": ["30m", "1h", "2h"],
                    "receptive_field_hours": 50.6,
                    "default_config": {
                        "d_in": 32,
                        "d_model": 128,
                        "n_heads": 4,
                        "n_layers": 4,
                        "mem_len": 128,
                        "aux_horizons": [1, 4, 12]
                    }
                }
                self._send_json(info)
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, 500)
            return
        elif parsed.path == "/api/gtrxl/signal":
            self._send_json(TRADER.get_status())
            return
        elif parsed.path == "/api/paper/status":
            self._send_json(TRADER.paper_account.get_summary() if hasattr(TRADER, "paper_account") else {})
            return
        elif parsed.path == "/api/my-ip":
            self._send_json({"outbound_ip": get_outbound_ip()})
        elif parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif parsed.path in {"/", "/index.html", "/app.js", "/styles.css"}:
            self.path = parsed.path
            super().do_GET()
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)

        # OTP Verification endpoint (must be accessible prior to general auth)
        if parsed.path == "/api/auth/verify-otp":
            client_ip = self._client_ip()
            lockout = _get_lockout_remaining(client_ip)
            if lockout > 0:
                self._send_json({
                    "success": False,
                    "error": f"Security cooldown active. Please wait {lockout} seconds before retrying.",
                    "lockout_remaining": lockout,
                }, 429)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                payload = {}

            entered_otp = str(payload.get("otp", "")).strip()
            if entered_otp == CORRECT_OTP:
                with _lockout_lock:
                    _otp_lockouts.pop(client_ip, None)
                token = _create_otp_token()
                cookie_str = f"otp_session={token}; Path=/; Max-Age=604800; HttpOnly; SameSite=Lax"
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", cookie_str)
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                return
            else:
                with _lockout_lock:
                    _otp_lockouts[client_ip] = time.time() + OTP_LOCKOUT_SECONDS
                self._send_json({
                    "success": False,
                    "error": "Incorrect OTP. Security cooldown active: please wait 30 seconds before retrying.",
                    "lockout_remaining": int(OTP_LOCKOUT_SECONDS),
                }, 401)
                return

        # General auth check for remaining POST endpoints
        has_auth = self._otp_authenticated() or (PUBLIC_MODE and self._authenticated())
        if PUBLIC_MODE and not has_auth:
            self._send_json({"error": "Authentication required", "requires_otp": True}, 401)
            return

        if parsed.path == "/api/auth/logout":
            cookie_str = "otp_session=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Set-Cookie", cookie_str)
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "logged_out": True}).encode("utf-8"))
            return

        # Ensure OTP or Google auth is active for trading controls
        if not has_auth and not self._otp_authenticated():
            self._send_json({"error": "OTP authentication required", "requires_otp": True}, 401)
            return

        if parsed.path == "/api/trade/toggle":
            new_state = not STRATEGY.get("live_order_submission_enabled", False)
            STRATEGY["live_order_submission_enabled"] = new_state
            CLIENT.allow_trading = new_state
            TRADER.strategy_config["live_order_submission_enabled"] = new_state
            TRADER.log(f"Live automated trading submission toggled to {'ON' if new_state else 'OFF'}")
            try:
                (ROOT / "config/production_strategy.json").write_text(json.dumps(STRATEGY, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"Warning: Could not save strategy file to disk: {e}", flush=True)
            with _cache_lock:
                _cache["at"] = 0.0
            self._send_json({"live_orders_enabled": new_state})
            return

        elif parsed.path == "/api/gtrxl/evaluate":
            try:
                TRADER.evaluate_and_trade()
            except Exception as e:
                TRADER.log(f"Evaluate notice: {e}")
            self._send_json(TRADER.get_status())
            return

        elif parsed.path == "/api/paper/reset":
            summary = TRADER.reset_paper_trading(10.0)
            with _cache_lock:
                _cache["at"] = 0.0
            self._send_json({"status": "ok", "paper_trading": summary})
            return

        elif parsed.path == "/api/paper/toggle":
            summary = TRADER.toggle_paper_trading()
            with _cache_lock:
                _cache["at"] = 0.0
            self._send_json({"status": "ok", "paper_trading": summary})
            return

        elif parsed.path == "/api/delta/credentials":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length)) if length > 0 else {}
            api_key = str(payload.get("api_key", "")).strip()
            api_secret = str(payload.get("api_secret", "")).strip()
            grant = "trade" if STRATEGY.get("live_order_submission_enabled") else "read"
            if not api_key or not api_secret:
                self._send_json({"error": "Both api_key and api_secret are required"}, 400)
                return
            try:
                saved = CLIENT.save_credentials(api_key, api_secret, grant=grant)
                with _cache_lock:
                    _cache["at"] = 0.0
                self._send_json({"status": "ok", "saved": saved, "tools": list(CLIENT.available_tools())})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        elif parsed.path == "/api/gtrxl/simulate-stop-loss":
            limits = STRATEGY.get("risk_limits", {})
            sl_val = float(limits.get("per_trade_stop_loss", 300.0))
            TRADER.log(f"[SIMULATION TRIGGER] Position breached stop-loss threshold (-₹{sl_val:.2f}). Triggering emergency exit.")
            dummy_feat = TRADER.last_entry_features or TRADER.current_bar_features
            if dummy_feat is None:
                dummy_feat = torch.zeros(32)
                dummy_feat[7] = 0.42   # 42% upper rejection wick
                dummy_feat[10] = 1.85  # 1.85x volume surge
                dummy_feat[16] = 1.35  # Volatility expansion
                dummy_feat[18] = 0.52  # Overbought RSI
            
            res = TRADER.learn_from_trade_failure(exit_pnl=-sl_val, side="buy", entry_feat=dummy_feat)
            self._send_json({
                "status": "success",
                "message": f"Stop-loss analyzed and fixed: {res.get('fix_summary', 'Policy adapted')}",
                "analysis": res,
                "trader": TRADER.get_status(),
            })
            return

        elif parsed.path == "/api/gtrxl/recheck-trend":
            trend_ok, trend_diag = TRADER.recheck_market_trend()
            TRADER.last_trend_analysis = trend_diag
            if trend_ok:
                TRADER.circuit_breaker_active = False
                TRADER.cooldown_until = None
                TRADER.consecutive_stop_losses = 0
                TRADER.trend_recheck_status = "TREND_CONFIRMED"
            else:
                TRADER.trend_recheck_status = "TREND_RECHECK_PENDING"

            with TRADER._lock:
                TRADER.latest_status["circuit_breaker_active"] = TRADER.circuit_breaker_active
                TRADER.latest_status["consecutive_stop_losses"] = TRADER.consecutive_stop_losses
                TRADER.latest_status["trend_recheck_status"] = TRADER.trend_recheck_status
                TRADER.latest_status["last_trend_analysis"] = trend_diag
                TRADER.latest_status["status"] = "TREND_CONFIRMED" if trend_ok else "TREND_RECHECK_PENDING"

            self._send_json({
                "status": "success",
                "trend_confirmed": trend_ok,
                "analysis": trend_diag,
                "trader": TRADER.get_status(),
            })
            return

        elif parsed.path == "/api/strategy/risk_limits":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
                limits = STRATEGY.setdefault("risk_limits", {})
                
                if "max_trades_per_day" in payload:
                    val = int(payload["max_trades_per_day"])
                    if val < 1:
                        self._send_json({"error": "Max trades per day must be at least 1."}, 400)
                        return
                    limits["max_trades_per_day"] = val
                
                if "daily_net_profit_target" in payload:
                    limits["daily_net_profit_target"] = round(float(payload["daily_net_profit_target"]), 2)
                
                if "daily_max_loss" in payload:
                    limits["daily_max_loss"] = round(float(payload["daily_max_loss"]), 2)
                    
                if "per_trade_stop_loss" in payload:
                    limits["per_trade_stop_loss"] = round(float(payload["per_trade_stop_loss"]), 2)
                    
                if "per_trade_take_profit" in payload:
                    limits["per_trade_take_profit"] = round(float(payload["per_trade_take_profit"]), 2)
                    
                if "contract_size" in payload:
                    val = int(payload["contract_size"])
                    if val < 1 or val > 100:
                        self._send_json({"error": "Contract lot size must be between 1 and 100."}, 400)
                        return
                    limits["contract_size"] = val
                    
                try:
                    (ROOT / "config/production_strategy.json").write_text(json.dumps(STRATEGY, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not save strategy file to disk: {e}", flush=True)
                with _cache_lock:
                    _cache["at"] = 0.0
                self._send_json({"status": "success", "risk_limits": limits})
                return
            except (ValueError, TypeError) as exc:
                self._send_json({"error": f"Invalid parameter format: {exc}"}, 400)
                return

        elif parsed.path == "/api/strategy/lot_size":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
                if "contract_size" not in payload:
                    self._send_json({"error": "contract_size is required."}, 400)
                    return
                val = int(payload["contract_size"])
                if val < 1 or val > 100:
                    self._send_json({"error": "Contract lot size must be between 1 and 100."}, 400)
                    return
                limits = STRATEGY.setdefault("risk_limits", {})
                limits["contract_size"] = val
                try:
                    (ROOT / "config/production_strategy.json").write_text(json.dumps(STRATEGY, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not save strategy file to disk: {e}", flush=True)
                with _cache_lock:
                    _cache["at"] = 0.0
                self._send_json({"status": "success", "contract_size": val, "risk_limits": limits})
                return
            except (ValueError, TypeError) as exc:
                self._send_json({"error": f"Invalid parameter format: {exc}"}, 400)
                return


        elif parsed.path == "/api/trade/order":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
                side = str(payload.get("side", "buy")).lower()
                if side not in ("buy", "sell"):
                    self._send_json({"error": "Invalid side. Must be 'buy' or 'sell'."}, 400)
                    return

                if not STRATEGY.get("live_order_submission_enabled", False):
                    self._send_json({"error": "Live trading is currently PAUSED (OFF). Please toggle Live Trading ON to place orders."}, 400)
                    return

                try:
                    size = int(payload.get("size", 1))
                except (ValueError, TypeError):
                    size = 1

                if size < 1:
                    self._send_json({"error": "Lot size must be at least 1."}, 400)
                    return

                max_size = int(STRATEGY.get("risk_limits", {}).get("contract_size", 100))
                if size > max_size:
                    self._send_json({"error": f"Lot size {size} exceeds maximum allowable limit of {max_size} lots."}, 400)
                    return

                tools = CLIENT.available_tools()
                if "place_order" not in tools:
                    self._send_json({"error": "place_order tool is unavailable. Verify that your Delta API key has Trading permission enabled."}, 400)
                    return

                # Balance & Margin pre-check
                wallets_data = []
                try:
                    if "get_wallet_balances" in tools:
                        w_res = CLIENT.call("get_wallet_balances")
                        wallets_data, _ = _rows(w_res)
                except Exception:
                    pass

                avail_inr = Decimal("0")
                avail_usdt = Decimal("0")
                has_wallet_data = False
                for w in wallets_data:
                    has_wallet_data = True
                    asset = str(w.get("asset_symbol", "")).upper()
                    ab = _number(w.get("available_balance")) or Decimal("0")
                    if asset == "INR":
                        avail_inr += ab
                    elif asset in ("USDT", "USD"):
                        avail_usdt += ab

                mark_price = Decimal("2600")
                try:
                    if "get_ticker" in tools:
                        t_res = CLIENT.call("get_ticker", {"symbol": "ETHUSD"})
                        t_obj = t_res.get("result", t_res) if isinstance(t_res, dict) else {}
                        mp = _number(t_obj.get("mark_price") or t_obj.get("close"))
                        if mp:
                            mark_price = mp
                except Exception:
                    pass

                # Estimated margin for Delta India ETHUSD contracts (0.001 ETH per contract at 10x leverage)
                est_margin_inr = (mark_price * Decimal("0.001") * Decimal("87") / Decimal("10")) * Decimal(size)
                est_margin_usdt = (mark_price * Decimal("0.001") / Decimal("10")) * Decimal(size)

                # Sufficient balance check
                if has_wallet_data:
                    if avail_inr > 0:
                        if avail_inr < est_margin_inr:
                            self._send_json({
                                "error": f"Insufficient balance: Placing {size} lot(s) requires ~₹{float(est_margin_inr):.2f} INR margin, but available balance is only ₹{float(avail_inr):.2f} INR. Please deposit funds or reduce lot count.",
                                "required_margin": float(est_margin_inr),
                                "available_balance": float(avail_inr),
                                "currency": "INR",
                            }, 400)
                            return
                    elif avail_usdt > 0:
                        if avail_usdt < est_margin_usdt:
                            self._send_json({
                                "error": f"Insufficient balance: Placing {size} lot(s) requires ~${float(est_margin_usdt):.2f} USDT margin, but available balance is only ${float(avail_usdt):.2f} USDT. Please deposit funds or reduce lot count.",
                                "required_margin": float(est_margin_usdt),
                                "available_balance": float(avail_usdt),
                                "currency": "USDT",
                            }, 400)
                            return
                    else:
                        self._send_json({
                            "error": f"Insufficient balance: Your available wallet balance is 0. Cannot place order for {size} lot(s). Please deposit funds to Delta India before trading.",
                            "required_margin": float(est_margin_inr),
                            "available_balance": 0.0,
                        }, 400)
                        return

                product_id = 27
                try:
                    p = CLIENT.call("get_product", {"symbol": "ETHUSD"})
                    if isinstance(p, dict) and "id" in p:
                        product_id = p["id"]
                    elif isinstance(p, dict) and "result" in p and isinstance(p["result"], dict) and "id" in p["result"]:
                        product_id = p["result"]["id"]
                except Exception:
                    pass

                order_res = CLIENT.call("place_order", {
                    "product_id": product_id,
                    "size": size,
                    "side": side,
                    "order_type": "market_order",
                })
                with _cache_lock:
                    _cache["at"] = 0.0
                self._send_json({"status": "success", "order": order_res, "size": size, "side": side})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        self.send_error(404, "Unknown endpoint")

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
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
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
    bind = os.environ.get("HOST", "0.0.0.0" if (PUBLIC_MODE or os.environ.get("RENDER")) else "127.0.0.1")
    threading.Thread(target=get_outbound_ip, daemon=True).start()
    TRADER.start()
    server = ThreadingHTTPServer((bind, port), DashboardHandler)
    print(f"Delta account dashboard listening on {bind}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        TRADER.stop()
        server.server_close()
        CLIENT.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    run_server(port)
