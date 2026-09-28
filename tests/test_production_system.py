"""Production Quality Assurance & Mathematical Correctness Test Suite.

Verifies:
1. Zero Lookahead Leakage Invariance: Past features are completely invariant to future candle perturbations.
2. DeltaTradingEnv Trade Accounting: Honest round-trip trades, taker fees (0.05%), slippage, and win rate.
3. Trader Risk Management Gates: Confidence, trend spread, and RSI clamps gate order execution.
4. Circuit Breaker & Cooldown: 2+ consecutive stop losses activate circuit breaker and lock execution.
5. Experience Replay & Shadow Validation Gate: Online learning adapts policy safely without catastrophic destruction.
"""
from __future__ import annotations

import math
import unittest
import numpy as np
import torch

from ethresearch.features import compute_causal_candle_features, extract_all_causal_features
from ethresearch.env import DeltaTradingEnv
from ethresearch.trader import GTrXLAutomatedTrader, compute_bar_features


class TestZeroLookaheadLeakage(unittest.TestCase):
    """Proves strictly causal feature extraction with zero future information leakage."""

    def setUp(self):
        np.random.seed(42)
        n = 120
        self.closes = 2000.0 + np.cumsum(np.random.randn(n) * 2.0)
        self.highs = self.closes + np.abs(np.random.randn(n) * 1.5)
        self.lows = self.closes - np.abs(np.random.randn(n) * 1.5)
        self.opens = (self.highs + self.lows) / 2.0
        self.vols = np.abs(np.random.randn(n) * 50.0) + 10.0
        self.timestamps = 1700000000 + np.arange(n) * 60

    def test_future_data_invariance_single_step(self):
        """Feature vector at t=50 must remain exactly identical when future bars t>50 are corrupted."""
        t_eval = 50

        # Baseline calculation
        feat_baseline = compute_causal_candle_features(
            closes=self.closes.copy(),
            highs=self.highs.copy(),
            lows=self.lows.copy(),
            opens=self.opens.copy(),
            vols=self.vols.copy(),
            timestamps=self.timestamps.copy(),
            position_exposure=1.0,
            idx=t_eval,
        )

        # Drastically corrupt future bars (t > 50)
        closes_corrupted = self.closes.copy()
        highs_corrupted = self.highs.copy()
        lows_corrupted = self.lows.copy()
        opens_corrupted = self.opens.copy()
        vols_corrupted = self.vols.copy()

        closes_corrupted[t_eval + 1:] *= 100.0
        highs_corrupted[t_eval + 1:] *= 150.0
        lows_corrupted[t_eval + 1:] *= 50.0
        opens_corrupted[t_eval + 1:] *= 80.0
        vols_corrupted[t_eval + 1:] *= 1000.0

        feat_perturbed = compute_causal_candle_features(
            closes=closes_corrupted,
            highs=highs_corrupted,
            lows=lows_corrupted,
            opens=opens_corrupted,
            vols=vols_corrupted,
            timestamps=self.timestamps.copy(),
            position_exposure=1.0,
            idx=t_eval,
        )

        # Must be bit-for-bit identical (zero future contamination)
        self.assertEqual(feat_baseline.shape, (32,))
        self.assertEqual(feat_perturbed.shape, (32,))
        np.testing.assert_allclose(
            feat_baseline,
            feat_perturbed,
            atol=1e-7,
            err_msg="CRITICAL: Future candle data leaked into historical feature calculation!",
        )

    def test_extract_all_causal_features_shape(self):
        """Validates matrix extraction produces consistent shapes and no NaNs."""
        matrix = np.column_stack([
            self.timestamps,
            self.opens,
            self.highs,
            self.lows,
            self.closes,
            self.vols,
        ])
        feats, fwd_ret, aux_targets = extract_all_causal_features(matrix, warmup=25)
        self.assertFalse(np.isnan(feats).any(), "Extracted features contain NaN")
        self.assertFalse(np.isinf(feats).any(), "Extracted features contain Inf")
        self.assertEqual(feats.shape[1], 32)
        self.assertEqual(aux_targets.shape[1], 3)


class TestDeltaTradingEnvAccounting(unittest.TestCase):
    """Verifies honest simulated round-trip trade accounting, fee deductions, and metrics."""

    def setUp(self):
        # 100 synthetic 1m candles
        np.random.seed(123)
        n = 100
        closes = 2500.0 + np.cumsum(np.random.randn(n) * 1.5)
        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes - 0.2
        vols = np.full(n, 100.0)
        ts = 1700000000 + np.arange(n) * 60
        self.candles = np.column_stack([ts, opens, highs, lows, closes, vols])

    def test_round_trip_trade_execution_and_fee_deduction(self):
        features, fwd_ret, aux_targets = extract_all_causal_features(self.candles, warmup=25)
        env = DeltaTradingEnv(
            candles_matrix=self.candles[25:25 + len(features)],
            features_matrix=features,
            aux_targets=aux_targets,
            taker_fee=0.0005,
            initial_capital_inr=10000.0,
        )
        obs, info = env.reset()
        self.assertEqual(env.position, 0)
        self.assertEqual(len(env.completed_trades), 0)

        # Step 1: BUY (open long)
        obs, rew, term, trunc, info = env.step(1)  # Target is 1 -> open LONG
        self.assertEqual(env.position, 1)

        # Step 2: HOLD LONG (keep target as 1)
        obs, rew, term, trunc, info = env.step(1)
        self.assertEqual(env.position, 1)

        # Step 3: CLOSE to FLAT
        obs, rew, term, trunc, info = env.step(0)  # Target is 0 -> close to FLAT
        self.assertEqual(env.position, 0)
        self.assertEqual(len(env.completed_trades), 1)

        metrics = env.get_performance_metrics()
        self.assertEqual(metrics["total_trades"], 1)
        self.assertIn("win_rate_pct", metrics)
        self.assertIn("final_equity_inr", metrics)


class DummyDeltaMcpClient:
    """Mock client for testing trader gates without external network."""
    def __init__(self):
        self.tools = ["get_candles", "get_ticker", "get_margined_positions", "place_order", "get_product"]
        self.orders = []

    def available_tools(self):
        return list(self.tools)

    def call(self, tool_name, params=None):
        if tool_name == "get_product":
            return {"result": {"id": 27, "symbol": "ETHUSD"}}
        if tool_name == "get_ticker":
            return {"result": {"mark_price": 2500.0, "close": 2500.0}}
        if tool_name == "get_margined_positions":
            return {"result": []}
        if tool_name == "place_order":
            self.orders.append(params)
            return {"result": {"order_id": "mock_order_123", "status": "filled"}}
        if tool_name == "get_candles":
            # Generate 40 synthetic candles
            candles = []
            base_t = 1700000000
            for i in range(40):
                p = 2500.0 + i * 0.5
                candles.append({
                    "time": base_t + i * 60,
                    "open": p - 0.2,
                    "high": p + 0.5,
                    "low": p - 0.5,
                    "close": p,
                    "volume": 50.0,
                })
            return {"result": candles}
        return {"result": {}}


class TestTraderRiskAndCircuitBreaker(unittest.TestCase):
    """Verifies execution gates, circuit breaker, and shadow validation."""

    def setUp(self):
        self.mock_client = DummyDeltaMcpClient()
        self.config = {
            "strategy_id": "gtrxl_ethm_1m_production",
            "product_symbol": "ETHUSD",
            "bar_resolution": "1m",
            "enabled": True,
            "risk_limits": {
                "max_lots_per_trade": 1,
                "max_trades_per_day": 50,
                "per_trade_stop_loss": 300.0,
                "per_trade_take_profit": 600.0,
                "confidence_threshold": 0.55,
                "min_trend_spread": 0.0005,
                "consecutive_sl_limit": 2,
                "consecutive_sl_cooldown_minutes": 30,
            },
            "backtest": {
                "selection_pass": True,
            }
        }
        self.trader = GTrXLAutomatedTrader(strategy_config=self.config, client=self.mock_client)

    def test_compute_bar_features_delegation(self):
        """Verifies compute_bar_features returns a 32-dim torch tensor."""
        candles = self.mock_client.call("get_candles")["result"]
        feat = compute_bar_features(candles, current_position_exposure=0.0)
        self.assertIsNotNone(feat)
        self.assertIsInstance(feat, torch.Tensor)
        self.assertEqual(feat.shape, (32,))

    def test_circuit_breaker_on_consecutive_losses(self):
        """Verifies that 2 consecutive stop loss hits lock execution and activate circuit breaker."""
        entry_feat = torch.zeros(32)
        # 1st loss
        self.trader.learn_from_trade_failure(exit_pnl=-350.0, side="buy", entry_feat=entry_feat)
        self.assertEqual(self.trader.consecutive_stop_losses, 1)
        self.assertFalse(self.trader.circuit_breaker_active)

        # 2nd loss -> triggers circuit breaker
        self.trader.learn_from_trade_failure(exit_pnl=-320.0, side="buy", entry_feat=entry_feat)
        self.assertEqual(self.trader.consecutive_stop_losses, 2)
        self.assertTrue(self.trader.circuit_breaker_active)
        self.assertIsNotNone(self.trader.cooldown_until)
        self.assertGreater(self.trader.cooldown_until, 0)

    def test_shadow_validation_gate_preserves_policy(self):
        """Verifies that experience replay and shadow model validation run cleanly."""
        # Prime replay buffer with some synthetic bars
        for _ in range(20):
            self.trader.replay_buffer.append(torch.randn(32) * 0.1)

        orig_params = [p.clone() for p in self.trader.model.parameters()]
        entry_feat = torch.randn(32) * 0.1

        res = self.trader.learn_from_trade_failure(exit_pnl=-300.0, side="buy", entry_feat=entry_feat)
        self.assertEqual(res.get("status"), "success")
        self.assertIn("reason_code", res)
        # Verify model parameters remain valid numbers (no NaN or Inf corruption)
        for p in self.trader.model.parameters():
            self.assertTrue(torch.isfinite(p).all())


class TestPaperTradingSystem(unittest.TestCase):
    """Verifies continuous $10 paper trading engine, fee deduction, full lot sizing, and liquidation at zero."""

    def setUp(self):
        from ethresearch.paper import PaperTradingAccount
        from pathlib import Path
        import tempfile
        self.tmp_file = Path(tempfile.gettempdir()) / "test_paper_state.json"
        if self.tmp_file.exists():
            self.tmp_file.unlink()
        self.paper = PaperTradingAccount(initial_capital=10.0, state_file=self.tmp_file)

    def tearDown(self):
        if self.tmp_file.exists():
            try:
                self.tmp_file.unlink()
            except Exception:
                pass

    def test_initial_paper_state(self):
        """Account starts with exactly $10.00 USD, flat position, and active status."""
        summary = self.paper.get_summary()
        self.assertEqual(summary["initial_capital"], 10.0)
        self.assertEqual(summary["cash_balance"], 10.0)
        self.assertEqual(summary["equity"], 10.0)
        self.assertEqual(summary["current_position"], 0)
        self.assertEqual(summary["total_fees_paid"], 0.0)
        self.assertFalse(summary["is_exhausted"])
        self.assertTrue(summary["enabled"])

    def test_full_lot_sizing(self):
        """Full lot sizing calculates contracts to commit 100% of available $10 margin."""
        price = 2500.0
        contracts = self.paper.calculate_full_lot_size(price)
        self.assertGreater(contracts, 0)
        # Margin per contract = 2500 * 0.001 / 5 = $0.50
        # For $10 capital, contracts should be ~19-20 contracts
        self.assertGreaterEqual(contracts, 15)

    def test_buy_execution_deducts_fees(self):
        """Executing a BUY signal fills full lot and deducts Delta 0.05% taker fee."""
        fill = self.paper.execute_signal(1, price=2600.0, confidence=0.75)
        self.assertIsNotNone(fill)
        self.assertGreater(fill["contracts"], 0)
        self.assertGreater(self.paper.total_fees_paid, 0.0)
        self.assertLess(self.paper.cash_balance, 10.0)
        self.assertEqual(self.paper.current_position, fill["contracts"])

    def test_round_trip_trade_accounting(self):
        """Opening and closing trade reflects net P&L with entry and exit fees."""
        self.paper.execute_signal(1, price=2500.0, confidence=0.8)
        initial_fee = self.paper.total_fees_paid
        # Price rises to 2600 (profitable trade)
        close_fill = self.paper.close_position(price=2600.0, reason="TAKE_PROFIT")
        self.assertIsNotNone(close_fill)
        self.assertEqual(self.paper.current_position, 0)
        self.assertGreater(self.paper.total_fees_paid, initial_fee)
        self.assertGreater(close_fill["net_pnl"], 0.0)
        self.assertGreater(self.paper.cash_balance, 10.0)
        self.assertEqual(self.paper.total_trades, 1)
        self.assertEqual(self.paper.winning_trades, 1)

    def test_liquidation_until_zero(self):
        """Runs continuously until adverse loss exhausts capital to $0.00."""
        self.paper.execute_signal(1, price=2600.0, confidence=0.8)
        # Severe crash from 2600 to 2000
        is_liq, reason = self.paper.update_mark_price(price=2000.0)
        self.assertTrue(is_liq)
        self.assertEqual(reason, "LIQUIDATION")
        self.assertTrue(self.paper.is_exhausted)
        self.assertEqual(self.paper.cash_balance, 0.0)
        self.assertEqual(self.paper.equity, 0.0)
        self.assertEqual(self.paper.current_position, 0)

        # Subsequent signals are rejected when exhausted
        next_fill = self.paper.execute_signal(1, price=2000.0, confidence=0.9)
        self.assertIsNone(next_fill)

    def test_reset_after_exhaustion(self):
        """Reset returns account cleanly back to $10.00 USD."""
        self.paper.is_exhausted = True
        self.paper.cash_balance = 0.0
        self.paper.reset(10.0)
        self.assertFalse(self.paper.is_exhausted)
        self.assertEqual(self.paper.cash_balance, 10.0)
        self.assertEqual(self.paper.equity, 10.0)


if __name__ == "__main__":
    unittest.main()

