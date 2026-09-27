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
        self.assertFalse(data["strategy"]["live_orders_enabled"])

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


if __name__ == "__main__":
    unittest.main()
