import unittest
import base64
import hashlib
import hmac
import http.client
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError, unpack_tool_result
import serve_dashboard
from serve_dashboard import build_snapshot, trading_readiness, STRATEGY


class FakeDeltaMcp:
    environment = "india_prod"

    def __init__(self, account=True):
        self.account = account

    def available_tools(self):
        public = {"get_connection_status", "get_ticker"}
        account = {"get_wallet_balances", "get_margined_positions", "get_fills",
                   "get_wallet_transactions", "get_open_orders"}
        return public | account if self.account else public

    def call(self, name, arguments=None):
        cases = {
            "get_connection_status": {"credentials_configured": self.account},
            "get_ticker": {"result": {"mark_price": "2700"}},
            "get_wallet_balances": {"result": [
                {"asset_symbol": "INR", "balance": "10000", "available_balance": "8000"},
                {"asset_symbol": "USDT", "balance": "5", "available_balance": "5"},
            ]},
            "get_margined_positions": {"result": [
                {"product_symbol": "ETHUSD", "settling_asset_symbol": "INR", "realized_pnl": "50", "unrealized_pnl": "-10"},
                {"product_symbol": "BTCUSD", "settling_asset_symbol": "USDT", "realized_pnl": "2", "unrealized_pnl": "3"},
            ]},
            "get_fills": {"result": [{"id": "f1"}], "meta": {"after": "next-page"}},
            "get_wallet_transactions": {"result": [{"transaction_type": "cashflow", "amount": "10"}]},
            "get_open_orders": {"result": []},
        }
        return cases[name]


class DeltaDashboardTests(unittest.TestCase):
    def test_read_only_allowlist_refuses_order_tool(self):
        client = DeltaMcpClient()
        with self.assertRaises(DeltaMcpError):
            client.call("place_order", {"size": 1})

    def test_mcp_text_result_decodes_json(self):
        decoded = unpack_tool_result({"content": [{"type": "text", "text": '{"result":[]}'}]})
        self.assertEqual(decoded, {"result": []})

    def test_snapshot_preserves_asset_units_and_cursor(self):
        data = build_snapshot(FakeDeltaMcp())
        self.assertEqual(data["connection"], "connected")
        self.assertEqual(data["realized_pnl_open_positions"], {"INR": "50", "USDT": "2"})
        self.assertEqual(data["unrealized_pnl_open_positions"], {"INR": "-10", "USDT": "3"})
        self.assertEqual(data["fills_after"], "next-page")
        self.assertEqual(len(data["wallets"]), 2)
        self.assertEqual(data["strategy"]["live_orders_enabled"], STRATEGY.get("live_order_submission_enabled", False))

    def test_missing_account_key_returns_no_fake_balance(self):
        data = build_snapshot(FakeDeltaMcp(account=False))
        self.assertEqual(data["connection"], "needs_read_data_key")
        self.assertEqual(data["wallets"], [])
        self.assertEqual(data["realized_pnl_open_positions"], {})

    def test_live_trading_fails_closed_with_unset_limits_and_unvalidated_signal(self):
        unvalidated = {
            "strategy_id": "test",
            "backtest": {"selection_pass": False},
            "risk_limits": {},
            "blockers": ["no validated continuous Delta signal feed or live order lifecycle is implemented"],
        }
        status = trading_readiness(unvalidated)
        self.assertFalse(status["can_enable"])
        self.assertFalse(status["effective_enabled"])
        self.assertTrue(any("risk limits" in reason for reason in status["blockers"]))
        self.assertTrue(any("signal feed" in reason for reason in status["blockers"]))

    def test_public_dashboard_requires_google_session_for_account_api_and_files(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_dashboard.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(path, cookie=None, host="dashboard.example"):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            headers = {"Host": host}
            if cookie:
                headers["Cookie"] = cookie
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            status = response.status
            location = response.getheader("Location")
            response.read()
            connection.close()
            return status, location

        try:
            with patch.object(serve_dashboard, "PUBLIC_MODE", True), \
                 patch.object(serve_dashboard, "GOOGLE_ALLOWED_EMAIL", "owner@example.com"), \
                 patch.object(serve_dashboard, "SESSION_SECRET", "s" * 32), \
                 patch.object(serve_dashboard, "GOOGLE_CLIENT_ID", "client-id"), \
                 patch.dict(os.environ, {"DASHBOARD_ALLOWED_HOSTS": "dashboard.example"}):
                self.assertEqual(request("/health", host="internal.render")[0], 200)
                self.assertEqual(request("/api/snapshot")[0], 401)
                self.assertEqual(request("/")[1], "/auth/login")
                self.assertEqual(request("/", "dashboard_session=invalid")[0], 302)
                payload = {"email": "owner@example.com", "expires": int(time.time()) + 60}
                encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
                signature = hmac.new(b"s" * 32, encoded.encode(), hashlib.sha256).hexdigest()
                self.assertEqual(request("/", f"dashboard_session={encoded}.{signature}")[0], 200)
                self.assertEqual(request("/auth/callback?code=unused&state=bad")[0], 403)
                other = base64.urlsafe_b64encode(json.dumps({"email": "other@example.com", "expires": int(time.time()) + 60}).encode()).rstrip(b"=").decode()
                other_signature = hmac.new(b"s" * 32, other.encode(), hashlib.sha256).hexdigest()
                self.assertEqual(request("/", f"dashboard_session={other}.{other_signature}")[0], 302)
                self.assertEqual(request("/auth/login")[0], 302)
                self.assertEqual(request("/", host="other.example")[0], 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_otp_authentication_and_lockout(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_dashboard.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post(path, payload, cookie=None):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            headers = {"Content-Type": "application/json", "Connection": "close"}
            if cookie:
                headers["Cookie"] = cookie
            body = json.dumps(payload)
            conn.request("POST", path, body=body, headers=headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            status = res.status
            set_cookie = res.getheader("Set-Cookie")
            conn.close()
            return status, data, set_cookie

        try:
            # 1. Wrong OTP should return 401 and trigger 30s lockout
            status, data, _ = post("/api/auth/verify-otp", {"otp": "000000"})
            self.assertEqual(status, 401)
            self.assertFalse(data.get("success"))
            self.assertEqual(data.get("lockout_remaining"), 30)

            # 2. Immediate subsequent attempt should return 429 cooldown active
            status_blocked, data_blocked, _ = post("/api/auth/verify-otp", {"otp": "477554"})
            self.assertEqual(status_blocked, 429)
            self.assertFalse(data_blocked.get("success"))

            # 3. Clear lockout and test correct OTP 477554
            with serve_dashboard._lockout_lock:
                serve_dashboard._otp_lockouts.clear()

            status_ok, data_ok, cookie = post("/api/auth/verify-otp", {"otp": "477554"})
            self.assertEqual(status_ok, 200)
            self.assertTrue(data_ok.get("success"))
            self.assertIn("otp_session=", cookie or "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_order_balance_safeguards(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_dashboard.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post_order(payload, cookie):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            headers = {"Content-Type": "application/json", "Connection": "close", "Cookie": cookie}
            conn.request("POST", "/api/trade/order", body=json.dumps(payload), headers=headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            status = res.status
            conn.close()
            return status, data

        try:
            # Login with OTP
            token = serve_dashboard._create_otp_token()
            auth_cookie = f"otp_session={token}"

            # Mock Delta client with 0 wallet balance
            fake_delta = FakeDeltaMcp()
            fake_delta.available_tools = lambda: {"place_order", "get_wallet_balances", "get_ticker"}
            # Set balance to 0
            fake_delta.call = lambda name, args=None: {
                "get_wallet_balances": {"result": [{"asset_symbol": "INR", "balance": "0", "available_balance": "0"}]},
                "get_ticker": {"result": {"mark_price": "2700"}},
                "place_order": {"id": "ord-123", "status": "filled"}
            }.get(name, {})

            with patch.object(serve_dashboard, "CLIENT", fake_delta), \
                 patch.dict(serve_dashboard.STRATEGY, {"live_order_submission_enabled": True}):
                # Attempt to place order when available balance is 0
                status, data = post_order({"side": "buy", "size": 2}, auth_cookie)
                self.assertEqual(status, 400)
                self.assertIn("Insufficient balance", data.get("error", ""))

                # Now provide sufficient balance: ₹50,000 INR
                fake_delta.call = lambda name, args=None: {
                    "get_wallet_balances": {"result": [{"asset_symbol": "INR", "balance": "50000", "available_balance": "50000"}]},
                    "get_ticker": {"result": {"mark_price": "2700"}},
                    "place_order": {"id": "ord-123", "status": "filled"}
                }.get(name, {})

                status_ok, data_ok = post_order({"side": "buy", "size": 2}, auth_cookie)
                self.assertEqual(status_ok, 200)
                self.assertEqual(data_ok.get("status"), "success")
                self.assertEqual(data_ok.get("size"), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()


