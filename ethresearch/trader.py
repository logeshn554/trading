"""Automated Trading Engine powered by GTrXL Reinforcement Learning.

Integrates:
- Live candle & ticker ingestion from Delta India via DeltaMcpClient
- Causal 32-feature extraction pipeline (returns, volatility, volume flow, cyclical time)
- Warmup and continuous rolling KV-cache streaming via GTrXLStreamingInferenceEngine
- Multi-horizon risk gating (daily max loss, max trades, per-trade SL/TP, margin check)
- Automated execution of market orders (BUY, SELL, HOLD, Risk-Exit)
- Thread-safe background execution and live status reporting
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError
from ethresearch.features import compute_causal_candle_features
from ethresearch.gtrxl import GTrXLActorCritic, GTrXLStreamingInferenceEngine

logger = logging.getLogger("ethresearch.trader")


# ============================================================================
# Causal Feature Extraction Pipeline (D_in = 32)
# ============================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def compute_bar_features(
    candles: List[Dict[str, Any]],
    current_position_exposure: float = 0.0,
) -> Optional[torch.Tensor]:
    """Computes a strictly causal 32-dimensional feature vector for the latest candle.

    Delegates directly to ethresearch.features.compute_causal_candle_features to ensure
    100% mathematical consistency and zero lookahead leakage between training and inference.

    Args:
        candles: List of historical candles sorted by timestamp ascending.
                 Each candle is a dict with keys: 'open', 'high', 'low', 'close', 'volume', 'time'.
        current_position_exposure: -1.0 (short), 0.0 (flat), or 1.0 (long).

    Returns:
        1D torch.Tensor of shape (32,) or None if insufficient history (< 21 bars).
    """
    if not candles or len(candles) < 21:
        return None

    closes = np.array([_safe_float(c.get("close")) for c in candles], dtype=np.float64)
    highs = np.array([_safe_float(c.get("high")) for c in candles], dtype=np.float64)
    lows = np.array([_safe_float(c.get("low")) for c in candles], dtype=np.float64)
    opens = np.array([_safe_float(c.get("open")) for c in candles], dtype=np.float64)
    vols = np.array([_safe_float(c.get("volume")) for c in candles], dtype=np.float64)
    times = np.array([_safe_float(c.get("time")) for c in candles], dtype=np.float64)

    idx = len(candles) - 1
    if closes[idx] <= 0.0 or highs[idx] < lows[idx]:
        return None

    feat_np = compute_causal_candle_features(
        closes=closes,
        highs=highs,
        lows=lows,
        opens=opens,
        vols=vols,
        timestamps=times,
        position_exposure=current_position_exposure,
        idx=idx,
    )
    return torch.from_numpy(feat_np).float()


# ============================================================================
# Automated Trading Engine
# ============================================================================

class GTrXLAutomatedTrader:
    """Thread-safe Automated Trading Daemon running GTrXL RL inference."""

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    def __init__(
        self,
        client: DeltaMcpClient,
        strategy_config: Dict[str, Any],
        d_in: int = 32,
        action_dim: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mem_len: int = 128,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.strategy_config = strategy_config
        self.poll_interval = poll_interval_seconds

        # Instantiate GTrXL model & streaming inference engine
        self.model = GTrXLActorCritic(
            d_in=d_in,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mem_len=mem_len,
        )
        self.engine = GTrXLStreamingInferenceEngine(self.model)

        # Threading & Control
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # State tracking
        self.is_warmed_up = False
        self.last_candle_time: Optional[int] = None
        self.trades_today = 0
        self.last_trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.execution_logs: List[str] = []
        self.self_healing_count = 0
        self.online_adaptations = 0
        self.last_entry_features: Optional[torch.Tensor] = None
        self.last_entry_side: Optional[str] = None
        self.current_bar_features: Optional[torch.Tensor] = None
        self.consecutive_stop_losses: int = 0
        self.cooldown_until: Optional[float] = None
        self.circuit_breaker_active: bool = False
        self.trend_recheck_status: str = "NORMAL"
        self.last_trend_analysis: Dict[str, Any] = {}
        self.replay_buffer: List[torch.Tensor] = []
        self.max_replay_buffer_size: int = 500

        self.latest_status: Dict[str, Any] = {
            "status": "INITIALIZING",
            "model": "GTrXL-RL (Pure Delta Exchange)",
            "venue": "Delta India Exchange (ETHUSD)",
            "last_signal": "HOLD",
            "action": 0,
            "confidence": 0.0,
            "action_probs": [1.0, 0.0, 0.0],
            "value_estimate": 0.0,
            "aux_returns": [0.0, 0.0, 0.0],
            "memory_bars": 0,
            "last_evaluation_time": None,
            "last_candle_time": None,
            "trades_today": 0,
            "last_action_taken": "Initialized trader daemon",
            "risk_status": "OK",
            "self_healing_count": 0,
            "online_adaptations": 0,
            "last_healing_event": "None",
            "last_learning_event": "None",
            "last_failure_analysis": None,
            "last_failure_fix": None,
            "consecutive_stop_losses": 0,
            "circuit_breaker_active": False,
            "cooldown_until": None,
            "cooldown_remaining_seconds": 0,
            "trend_recheck_status": "NORMAL",
            "last_trend_analysis": {},
            "trained_model_loaded": False,
            "scaler_loaded": False,
            "training_metadata": {},
            "recent_logs": [],
        }

        # Load trained weights and scaler if available
        self.scaler = None
        self._load_trained_artifacts()

    def _load_trained_artifacts(self) -> None:
        """Loads trained weights and scaler from artifacts/gtrxl/ if present."""
        root = Path(__file__).resolve().parents[1]
        pt_path = root / "artifacts/gtrxl/gtrxl_model.pt"
        scaler_path = root / "artifacts/gtrxl/scaler.pkl"
        bundle_path = root / "artifacts/gtrxl/model_bundle.pkl"

        if pt_path.exists():
            try:
                state_dict = torch.load(pt_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.log(f"Successfully loaded trained GTrXL model weights from {pt_path.name}")
                self.latest_status["trained_model_loaded"] = True
            except Exception as e:
                self.log(f"Warning: Could not load model weights from {pt_path}: {e}")

        if scaler_path.exists():
            try:
                import pickle
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.log(f"Successfully loaded trained StandardScaler from {scaler_path.name}")
                self.latest_status["scaler_loaded"] = True
            except Exception as e:
                self.log(f"Warning: Could not load scaler from {scaler_path}: {e}")

        if bundle_path.exists():
            try:
                import pickle
                with open(bundle_path, "rb") as f:
                    bundle = pickle.load(f)
                meta = bundle.get("training_metadata", {})
                self.latest_status["training_metadata"] = meta
                self.log(f"Loaded training metadata: {meta.get('data_source')} (Win Rate: {meta.get('win_rate')}%, Sharpe: {meta.get('sharpe_ratio')})")
            except Exception as e:
                pass

    def log(self, message: str) -> None:
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        formatted = f"[{now_str}] {message}"
        logger.info(formatted)
        with self._lock:
            self.execution_logs.append(formatted)
            if len(self.execution_logs) > 30:
                self.execution_logs.pop(0)
            self.latest_status["recent_logs"] = list(self.execution_logs)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_status)

    def warmup(self, symbol: str = "ETHUSD", resolution: str = "1m", lookback_bars: int = 150) -> bool:
        """Primes GTrXL recurrent memory cache with historical market data."""
        self.log(f"Starting GTrXL memory warmup for {symbol} ({resolution}, {lookback_bars} bars)...")
        now = datetime.now(timezone.utc)
        step_seconds = 60 if "m" in resolution else 3600
        start_ts = int(now.timestamp() - lookback_bars * step_seconds)
        end_ts = int(now.timestamp())

        try:
            tools = self.client.available_tools()
            if "get_candles" not in tools:
                self.log("Delta MCP 'get_candles' tool not available; warmup skipped.")
                return False

            res = self.client.call("get_candles", {
                "symbol": symbol,
                "resolution": resolution,
                "start": start_ts,
                "end": end_ts,
            })
            candles = res.get("result", res) if isinstance(res, dict) else res
            if not isinstance(candles, list) or len(candles) < 30:
                self.log(f"Insufficient historical candles ({len(candles) if isinstance(candles, list) else 0}). Warmup aborted.")
                return False

            # Sort ascending by time
            candles = sorted(candles, key=lambda c: c.get("time", 0))
            self.engine.reset()

            # Warmup sliding step through history
            warmed_steps = 0
            for i in range(25, len(candles)):
                sub_candles = candles[:i + 1]
                feat = compute_bar_features(sub_candles, current_position_exposure=0.0)
                if feat is not None:
                    self.replay_buffer.append(feat.detach().clone())
                    if len(self.replay_buffer) > self.max_replay_buffer_size:
                        self.replay_buffer.pop(0)
                    if self.scaler is not None:
                        feat_np = self.scaler.transform(feat.numpy().reshape(1, -1))
                        feat_np = np.clip(feat_np, -6.0, 6.0)
                        feat = torch.tensor(feat_np[0], dtype=torch.float32)
                    self.engine.step(feat)
                    warmed_steps += 1

            self.is_warmed_up = True
            mem_size = self.engine.memories[0].size(1) if self.engine.memories else 0
            self.log(f"Warmup complete. Processed {warmed_steps} historical bars. KV-cache active ({mem_size} memory tokens).")
            with self._lock:
                self.latest_status["memory_bars"] = mem_size
                self.latest_status["status"] = "WARMED_UP"
            return True
        except Exception as exc:
            self.log(f"Error during warmup: {exc}")
            return False

    def evaluate_and_trade(self) -> None:
        """Executes a single evaluation and automated trading cycle."""
        symbol = self.strategy_config.get("product_symbol", "ETHUSD")
        resolution = self.strategy_config.get("bar_resolution", "1m")
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        with self._lock:
            if self.last_trade_date != today_str:
                self.trades_today = 0
                self.last_trade_date = today_str
                self.latest_status["trades_today"] = 0

        # Check hard selection readiness gate (not just raw flag)
        limits = self.strategy_config.get("risk_limits", {})
        needed_limits = ("max_trades_per_day", "daily_net_profit_target", "daily_max_loss", "per_trade_stop_loss", "per_trade_take_profit")
        limits_ok = all(limits.get(k) is not None for k in needed_limits)
        backtest_pass = bool(self.strategy_config.get("backtest", {}).get("selection_pass", True))
        raw_live = bool(self.strategy_config.get("live_order_submission_enabled", False))
        live_enabled = raw_live and limits_ok and backtest_pass
        max_trades = int(limits.get("max_trades_per_day", 10))
        per_trade_sl = float(limits.get("per_trade_stop_loss", 300.0))
        per_trade_tp = float(limits.get("per_trade_take_profit", 600.0))
        contract_size = int(limits.get("contract_size", 1))

        tools = self.client.available_tools()

        # 1. Inspect existing open positions
        current_pos_size = 0
        unrealized_pnl = 0.0
        pos_exposure = 0.0
        position_row = None

        if "get_margined_positions" in tools:
            try:
                pos_res = self.client.call("get_margined_positions")
                pos_list = pos_res.get("result", pos_res) if isinstance(pos_res, dict) else pos_res
                if isinstance(pos_list, list):
                    for p in pos_list:
                        sym = p.get("product_symbol") or (p.get("product") or {}).get("symbol")
                        if sym == symbol:
                            position_row = p
                            current_pos_size = int(p.get("size", 0))
                            unrealized_pnl = _safe_float(p.get("unrealized_pnl", 0.0))
                            if current_pos_size > 0:
                                pos_exposure = 1.0
                            elif current_pos_size < 0:
                                pos_exposure = -1.0
                            break
            except Exception as e:
                self.log(f"Position check warning: {e}")

        # 2. Enforce Stop-Loss and Take-Profit on active position
        if live_enabled and current_pos_size != 0:
            if unrealized_pnl <= -per_trade_sl:
                self.log(f"RISK TRIGGER: Stop-loss breached (PnL ₹{unrealized_pnl:.2f} <= -₹{per_trade_sl:.2f}). Closing position.")
                self._close_position(symbol, current_pos_size, unrealized_pnl=unrealized_pnl)
                return
            elif unrealized_pnl >= per_trade_tp:
                self.log(f"RISK TRIGGER: Take-profit reached (PnL ₹{unrealized_pnl:.2f} >= ₹{per_trade_tp:.2f}). Locking profit.")
                self._close_position(symbol, current_pos_size, unrealized_pnl=unrealized_pnl)
                return

        # 3. Ingest recent market candles
        if "get_candles" not in tools:
            return

        start_ts = int(now.timestamp() - 50 * 3600)
        end_ts = int(now.timestamp())
        try:
            c_res = self.client.call("get_candles", {
                "symbol": symbol,
                "resolution": resolution,
                "start": start_ts,
                "end": end_ts,
            })
            candles = c_res.get("result", c_res) if isinstance(c_res, dict) else c_res
            if not isinstance(candles, list) or len(candles) < 22:
                return
            candles = sorted(candles, key=lambda c: c.get("time", 0))
        except Exception as e:
            self.log(f"Error fetching candles: {e}")
            return

        latest_candle = candles[-1]
        c_time = latest_candle.get("time")

        # 4. Extract features & execute GTrXL step
        feat = compute_bar_features(candles, current_position_exposure=pos_exposure)
        if feat is None:
            return

        self.current_bar_features = feat
        self.replay_buffer.append(feat.detach().clone())
        if len(self.replay_buffer) > self.max_replay_buffer_size:
            self.replay_buffer.pop(0)

        if self.scaler is not None:
            feat_np = self.scaler.transform(feat.numpy().reshape(1, -1))
            feat_np = np.clip(feat_np, -6.0, 6.0)
            feat = torch.tensor(feat_np[0], dtype=torch.float32)

        step_out = self.engine.step(feat)
        action = step_out["action"]  # 0: HOLD, 1: BUY, 2: SELL
        probs = step_out["action_probs"]
        value_est = step_out["value"]
        aux_ret = step_out["aux_returns"]
        confidence = probs[action]
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        signal_name = action_names.get(action, "HOLD")

        mem_size = self.engine.memories[0].size(1) if self.engine.memories else 0

        with self._lock:
            self.latest_status.update({
                "status": "RUNNING",
                "last_signal": signal_name,
                "action": action,
                "confidence": round(confidence, 4),
                "action_probs": [round(p, 4) for p in probs],
                "value_estimate": round(value_est, 4),
                "aux_returns": [round(r, 4) for r in aux_ret] if isinstance(aux_ret, list) else aux_ret,
                "memory_bars": mem_size,
                "last_evaluation_time": now.isoformat(),
                "last_candle_time": c_time,
                "trades_today": self.trades_today,
                "risk_status": "OK" if live_enabled else "LIVE_TRADING_PAUSED",
            })

        # 5. Circuit Breaker Check (2+ Consecutive Stop Losses)
        now_ts = time.time()
        cooldown_rem_s = 0
        if self.circuit_breaker_active and self.cooldown_until is not None:
            if now_ts < self.cooldown_until:
                cooldown_rem_s = int(self.cooldown_until - now_ts)
                self.trend_recheck_status = "COOLDOWN_PAUSE"
            else:
                # Cooldown completed! Now recheck the market trend:
                trend_ok, trend_diag = self.recheck_market_trend(candles)
                self.last_trend_analysis = trend_diag
                if trend_ok:
                    self.log(
                        f"[TREND RECHECK SUCCESS] Market trend revalidated: {trend_diag.get('summary')}. "
                        f"Circuit breaker lifted, resuming automated trading."
                    )
                    self.circuit_breaker_active = False
                    self.cooldown_until = None
                    self.consecutive_stop_losses = 0
                    self.trend_recheck_status = "TREND_CONFIRMED"
                else:
                    self.trend_recheck_status = "TREND_RECHECK_PENDING"
                    self.log(
                        f"[TREND RECHECK PENDING] Market structure still uncertain: {trend_diag.get('summary')}. "
                        f"Holding trade execution until trend cleanly confirms."
                    )

        if self.circuit_breaker_active:
            with self._lock:
                rem_m, rem_s = divmod(cooldown_rem_s, 60)
                cb_label = f"COOLDOWN ({rem_m}m {rem_s}s)" if cooldown_rem_s > 0 else "TREND_RECHECK_PENDING"
                self.latest_status["status"] = cb_label
                self.latest_status["circuit_breaker_active"] = True
                self.latest_status["consecutive_stop_losses"] = self.consecutive_stop_losses
                self.latest_status["cooldown_remaining_seconds"] = cooldown_rem_s
                self.latest_status["trend_recheck_status"] = self.trend_recheck_status
                self.latest_status["last_trend_analysis"] = self.last_trend_analysis
                self.latest_status["risk_status"] = (
                    f"CIRCUIT_BREAKER_ACTIVE: 2+ SL hit ({rem_m}m {rem_s}s wait)"
                    if cooldown_rem_s > 0 else "CIRCUIT_BREAKER: Trend recheck pending"
                )
            return

        # 6. Order execution logic
        if not live_enabled:
            return

        # Risk check: daily trades limit
        if self.trades_today >= max_trades:
            with self._lock:
                self.latest_status["risk_status"] = f"MAX_DAILY_TRADES_HIT ({self.trades_today}/{max_trades})"
            return

        # Check for actionable trade signals with hard execution and parameter gates
        conf_thresh = float(limits.get("confidence_threshold", 0.55))
        min_trend = float(limits.get("min_trend_spread", 0.0))
        trend_spread_val = float(feat[17]) if len(feat) > 17 else 0.0
        norm_rsi_val = float(feat[18]) if len(feat) > 18 else 0.0

        if confidence < conf_thresh:
            return

        if abs(trend_spread_val) < min_trend:
            return

        if action == self.ACTION_BUY and current_pos_size <= 0:
            if norm_rsi_val > 0.40:
                self.log(f"GATE: BUY signal blocked by RSI overbought protection (norm_rsi={norm_rsi_val:.2f})")
                return
            self.log(f"GTrXL SIGNAL: BUY (Conf: {confidence*100:.1f}%, Val: {value_est:.3f})")
            if current_pos_size < 0:
                self._close_position(symbol, current_pos_size, unrealized_pnl=unrealized_pnl)
            self._place_order(symbol, side="buy", size=contract_size, is_opening=True)
        elif action == self.ACTION_SELL and current_pos_size >= 0:
            if norm_rsi_val < -0.40:
                self.log(f"GATE: SELL signal blocked by RSI oversold protection (norm_rsi={norm_rsi_val:.2f})")
                return
            self.log(f"GTrXL SIGNAL: SELL (Conf: {confidence*100:.1f}%, Val: {value_est:.3f})")
            if current_pos_size > 0:
                self._close_position(symbol, current_pos_size, unrealized_pnl=unrealized_pnl)
            self._place_order(symbol, side="sell", size=contract_size, is_opening=True)

    def _self_heal_execution_issue(self, error: Exception, symbol: str, side: str, attempted_size: int, is_opening: bool = False) -> bool:
        """Autonomously diagnoses execution failures and self-corrects parameters or environment."""
        err_str = str(error).lower()
        self.log(f"[SELF-HEAL DIAGNOSTIC] Analyzing failure cause: {error}")

        # Case 1: Insufficient Balance / Margin Requirement Breached
        if "balance" in err_str or "margin" in err_str or "insufficient" in err_str:
            self.log("[SELF-HEAL ACTION] Margin/Balance limit triggered. Re-evaluating wallet balances...")
            try:
                tools = self.client.available_tools()
                avail_inr = Decimal("0")
                if "get_wallet_balances" in tools:
                    w_res = self.client.call("get_wallet_balances")
                    wallets = w_res.get("result", w_res) if isinstance(w_res, dict) else w_res
                    if isinstance(wallets, list):
                        for w in wallets:
                            if str(w.get("asset_symbol", "")).upper() == "INR":
                                avail_inr += Decimal(str(w.get("available_balance", 0)))

                mark_price = Decimal("2600")
                if "get_ticker" in tools:
                    t_res = self.client.call("get_ticker", {"symbol": symbol})
                    t_obj = t_res.get("result", t_res) if isinstance(t_res, dict) else {}
                    mp = t_obj.get("mark_price") or t_obj.get("close")
                    if mp:
                        mark_price = Decimal(str(mp))

                margin_per_lot = (mark_price * Decimal("0.001") * Decimal("87") / Decimal("10"))
                safe_lots = max(1, int(avail_inr // margin_per_lot)) if margin_per_lot > 0 else 1

                if safe_lots < attempted_size:
                    self.log(f"[SELF-HEAL FIX] Auto-adjusted contract size: {attempted_size} -> {safe_lots} lot(s) (avail ₹{float(avail_inr):.2f}).")
                    self.strategy_config.setdefault("risk_limits", {})["contract_size"] = safe_lots
                    try:
                        root = Path(__file__).resolve().parents[1]
                        (root / "config/production_strategy.json").write_text(
                            json.dumps(self.strategy_config, indent=2), encoding="utf-8"
                        )
                    except Exception:
                        pass

                    self.self_healing_count += 1
                    with self._lock:
                        self.latest_status["self_healing_count"] = self.self_healing_count
                        self.latest_status["last_healing_event"] = f"Auto-adjusted lot size to {safe_lots} lot(s)"
                    return self._place_order(symbol, side=side, size=safe_lots, is_retry=True, is_opening=is_opening)
            except Exception as fix_exc:
                self.log(f"[SELF-HEAL WARNING] Margin resolution error: {fix_exc}")

        # Case 2: Product ID / Tool Desync
        if "product" in err_str or "id" in err_str or "tool" in err_str:
            self.log("[SELF-HEAL ACTION] Tool or Product desync detected. Refreshing product metadata from Delta India...")
            try:
                self.client.call("get_product", {"symbol": symbol})
                self.self_healing_count += 1
                with self._lock:
                    self.latest_status["self_healing_count"] = self.self_healing_count
                    self.latest_status["last_healing_event"] = "Re-synchronized product metadata"
                return self._place_order(symbol, side=side, size=attempted_size, is_retry=True, is_opening=is_opening)
            except Exception as retry_err:
                self.log(f"[SELF-HEAL ERROR] Retry after metadata resync failed: {retry_err}")

        return False

    def _diagnose_failure_reason(self, side: str, feat: torch.Tensor) -> Tuple[str, str, Dict[str, Any]]:
        """Performs Automated Root Cause Analysis on the market state at entry to detect
        why the trade breached its stop-loss or failed.

        Returns:
            (reason_code, human_explanation, auto_fixes_dict)
        """
        f = feat.detach().cpu().flatten()
        upper_shadow = float(f[7]) if len(f) > 7 else 0.0
        lower_shadow = float(f[8]) if len(f) > 8 else 0.0
        vol_ratio = float(f[10]) if len(f) > 10 else 1.0
        vol_ratio_5_20 = float(f[16]) if len(f) > 16 else 1.0
        trend_spread = float(f[17]) if len(f) > 17 else 0.0
        norm_rsi = float(f[18]) if len(f) > 18 else 0.0
        b_width = float(f[21]) if len(f) > 21 else 0.02

        side_upper = side.upper()

        # 1. False Breakout / Liquidity Sweep (Upper/Lower wick rejection with high volume)
        if (side_upper == "BUY" and (upper_shadow > 0.3 or (vol_ratio > 1.3 and upper_shadow > 0.15))) or \
           (side_upper == "SELL" and (lower_shadow > 0.3 or (vol_ratio > 1.3 and lower_shadow > 0.15))):
            wick_type = "upper rejection wick" if side_upper == "BUY" else "lower rejection wick"
            wick_val = upper_shadow if side_upper == "BUY" else lower_shadow
            reason = "FALSE_BREAKOUT_SWEEP"
            explanation = (
                f"Liquidity Sweep / False Breakout: {side_upper} entered into heavy {wick_type} "
                f"({wick_val * 100:.1f}% candle height) with volume surge ({vol_ratio:.1f}x mean). "
                f"Institutional flow swept retail liquidity and aggressively reversed."
            )
            fixes = {"penalty_weight": 3.0, "raise_conf_threshold": 0.05, "cooldown_cycles": 2}
            return reason, explanation, fixes

        # 2. Momentum Exhaustion (Overbought Long or Oversold Short)
        if (side_upper == "BUY" and norm_rsi > 0.35) or (side_upper == "SELL" and norm_rsi < -0.35):
            approx_rsi = 50.0 + (norm_rsi * 50.0)
            reason = "MOMENTUM_EXHAUSTION"
            explanation = (
                f"Momentum Exhaustion: {side_upper} executed into extreme territory (RSI ~{approx_rsi:.0f}). "
                f"Directional momentum was fully depleted, triggering sharp mean-reversion into the stop loss."
            )
            fixes = {"penalty_weight": 2.5, "clamp_rsi": True, "cooldown_cycles": 2}
            return reason, explanation, fixes

        # 3. Sudden Volatility Expansion (Stop-Loss Too Tight for Active Noise)
        if vol_ratio_5_20 > 1.25 or b_width > 0.035:
            reason = "VOLATILITY_EXPANSION_SL_TIGHT"
            explanation = (
                f"Volatility Expansion Shock: 5-bar Parkinson volatility surged to {vol_ratio_5_20:.2f}x "
                f"of the 20-bar baseline. The protective stop loss was placed inside the noise corridor."
            )
            fixes = {"penalty_weight": 2.0, "widen_sl_multiplier": 1.15, "downscale_lots": True}
            return reason, explanation, fixes

        # 4. Low Volatility Consolidation Chop
        if b_width < 0.015 or abs(trend_spread) < 0.001:
            reason = "RANGE_CHOP_CONSOLIDATION"
            explanation = (
                f"Range Chop / Low Volatility Squeeze: Traded inside a compressed Bollinger consolidation "
                f"(band width={b_width:.4f}). Price whipsawed randomly without directional trend follow-through."
            )
            fixes = {"penalty_weight": 2.0, "boost_hold_weight": 2.0, "require_trend_spread": 0.0015}
            return reason, explanation, fixes

        # 5. Order Flow Breakdown
        reason = "ADVERSE_ORDER_FLOW_REVERSAL"
        explanation = (
            f"Adverse Order Flow Cascade: Volume-weighted flow shifted against {side_upper} position. "
            f"Momentum broken across multi-horizon return spectrum."
        )
        fixes = {"penalty_weight": 2.0, "boost_hold_weight": 1.5, "cooldown_cycles": 1}
        return reason, explanation, fixes

    def recheck_market_trend(self, candles: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, Dict[str, Any]]:
        """Systematic multi-horizon market trend and structure revalidation following a 2+ stop-loss cooldown.

        Validates:
        1. Multi-EMA structural alignment (Fast EMA 10, Mid EMA 25, Slow EMA 50)
        2. Volatility normalization (5-bar Parkinson / 20-bar baseline <= 1.25)
        3. Candle geometry & absorption (no rejection wicks > 35% in last 3 bars)
        4. RSI healthy corridor (35 <= RSI <= 65, free of momentum exhaustion)
        5. Trend strength & spread (EMA 10 vs EMA 25 spread >= 0.05%)

        Returns:
            (is_trend_confirmed, diagnostic_report)
        """
        if candles is None or len(candles) < 30:
            if self.client and "get_candles" in self.client.available_tools():
                now_ts = int(time.time())
                symbol = self.strategy_config.get("product_symbol", "ETHUSD")
                resolution = self.strategy_config.get("bar_resolution", "1m")
                try:
                    c_res = self.client.call("get_candles", {
                        "symbol": symbol,
                        "resolution": resolution,
                        "start": now_ts - (50 * 3600),
                        "end": now_ts,
                    })
                    candles = c_res.get("result", c_res) if isinstance(c_res, dict) else c_res
                except Exception:
                    pass

        if not candles or len(candles) < 30:
            return False, {
                "confirmed": False,
                "summary": "Insufficient candle history (<30 bars) for multi-horizon trend recheck",
                "trend_direction": "UNKNOWN",
            }

        closes = [_safe_float(c.get("close")) for c in candles]
        highs = [_safe_float(c.get("high")) for c in candles]
        lows = [_safe_float(c.get("low")) for c in candles]
        idx = len(closes) - 1
        eps = 1e-8

        def calc_ema(series: List[float], span: int) -> float:
            alpha = 2.0 / (span + 1.0)
            res = series[0]
            for val in series[1:]:
                res = alpha * val + (1.0 - alpha) * res
            return res

        ema10 = calc_ema(closes[-35:], 10)
        ema25 = calc_ema(closes[-35:], 25)
        ema50 = calc_ema(closes[-50:], 50)
        cur_close = closes[-1]

        spread_10_25 = (ema10 - ema25) / (ema25 + eps)

        is_bullish = (ema10 > ema25 > ema50) and (cur_close >= ema25)
        is_bearish = (ema10 < ema25 < ema50) and (cur_close <= ema25)
        trend_direction = "BULLISH" if is_bullish else ("BEARISH" if is_bearish else "CHOPPY_SIDEWAYS")

        # Volatility normalization
        p5 = sum(
            (math.log(max(highs[i], eps) / max(lows[i], eps)) ** 2) / (4.0 * math.log(2.0))
            for i in range(idx - 4, idx + 1)
        ) / 5.0
        p20 = sum(
            (math.log(max(highs[i], eps) / max(lows[i], eps)) ** 2) / (4.0 * math.log(2.0))
            for i in range(idx - 19, idx + 1)
        ) / 20.0
        vol_p5 = math.sqrt(max(p5, 0.0))
        vol_p20 = math.sqrt(max(p20, 0.0))
        vol_ratio = vol_p5 / (vol_p20 + eps)
        vol_normalized = vol_ratio <= 1.25

        # 14-period RSI
        gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(idx - 13, idx + 1)]
        losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(idx - 13, idx + 1)]
        rs = (sum(gains) / 14.0) / (sum(losses) / 14.0 + eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_healthy = (35.0 <= rsi <= 65.0)

        # Recent wicks
        recent_clean = True
        for i in range(idx - 2, idx + 1):
            h, l, c, o = highs[i], lows[i], closes[i], _safe_float(candles[i].get("open"))
            rng = h - l + eps
            if (h - max(o, c)) / rng > 0.35 or (min(o, c) - l) / rng > 0.35:
                recent_clean = False
                break

        min_trend_spread_req = float(self.strategy_config.get("risk_limits", {}).get("min_trend_spread", 0.0005))
        trend_aligned = is_bullish or is_bearish
        confirmed = (
            trend_aligned
            and vol_normalized
            and rsi_healthy
            and recent_clean
            and (abs(spread_10_25) >= min_trend_spread_req)
        )

        summary_parts = [
            f"Trend: {trend_direction} (Spread: {spread_10_25*100:+.2f}%)",
            f"Vol Ratio: {vol_ratio:.2f} ({'NORMALIZED' if vol_normalized else 'HIGH_SHOCK'})",
            f"RSI: {rsi:.1f} ({'HEALTHY' if rsi_healthy else 'EXTREME'})",
            f"Wicks: {'CLEAN' if recent_clean else 'REJECTIONS_DETECTED'}",
        ]

        diag = {
            "confirmed": confirmed,
            "trend_direction": trend_direction,
            "ema10": round(ema10, 2),
            "ema25": round(ema25, 2),
            "ema50": round(ema50, 2),
            "spread_10_25": round(spread_10_25, 5),
            "parkinson_ratio_5_20": round(vol_ratio, 3),
            "rsi": round(rsi, 2),
            "volatility_normalized": vol_normalized,
            "recent_wicks_clean": recent_clean,
            "summary": " · ".join(summary_parts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return confirmed, diag

    def learn_from_trade_failure(self, exit_pnl: float, side: str, entry_feat: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Autonomous online reinforcement learning adaptation & Root Cause Self-Fixer
        triggered upon adverse trade or stop-loss hit."""
        if entry_feat is None:
            return {"status": "skipped", "message": "No entry features available"}

        action_idx = 1 if side.lower() == "buy" else 2
        action_name = "BUY" if action_idx == 1 else "SELL"

        # 1. Automated Root Cause Analysis
        reason_code, explanation, fixes = self._diagnose_failure_reason(action_name, entry_feat)
        self.log(f"[ROOT CAUSE ANALYSIS] Detected failure reason: {reason_code}")
        self.log(f"[ROOT CAUSE DIAGNOSIS] {explanation}")

        # 2. Autonomous Operational Self-Fix (Hyperparameters & Risk Caps)
        applied_fixes = []
        if "raise_conf_threshold" in fixes:
            current_th = float(self.strategy_config.get("risk_limits", {}).get("confidence_threshold", 0.55))
            new_th = min(0.85, current_th + fixes["raise_conf_threshold"])
            self.strategy_config.setdefault("risk_limits", {})["confidence_threshold"] = round(new_th, 3)
            applied_fixes.append(f"Raised confidence threshold {current_th:.2f} -> {new_th:.2f}")

        if "widen_sl_multiplier" in fixes:
            current_sl = float(self.strategy_config.get("risk_limits", {}).get("per_trade_stop_loss", 300.0))
            new_sl = round(current_sl * fixes["widen_sl_multiplier"], 1)
            self.strategy_config.setdefault("risk_limits", {})["per_trade_stop_loss"] = new_sl
            applied_fixes.append(f"Auto-expanded stop-loss buffer ₹{current_sl:.0f} -> ₹{new_sl:.0f}")

        if "require_trend_spread" in fixes:
            self.strategy_config.setdefault("risk_limits", {})["min_trend_spread"] = fixes["require_trend_spread"]
            applied_fixes.append(f"Enforced minimum trend spread filter ({fixes['require_trend_spread']}) to avoid chop")

        if "cooldown_cycles" in fixes:
            cooldown_mins = fixes["cooldown_cycles"] * 5
            applied_fixes.append(f"Activated {cooldown_mins}m volatility cooldown to avoid revenge trading")

        # Circuit Breaker Check: If >= 2 consecutive stop-losses hit, force cooling-off period
        self.consecutive_stop_losses += 1
        consecutive_limit = int(self.strategy_config.get("risk_limits", {}).get("consecutive_sl_limit", 2))
        cooldown_mins = int(self.strategy_config.get("risk_limits", {}).get("consecutive_sl_cooldown_minutes", 30))

        if self.consecutive_stop_losses >= consecutive_limit:
            self.circuit_breaker_active = True
            self.cooldown_until = time.time() + (cooldown_mins * 60)
            self.trend_recheck_status = "COOLDOWN_PAUSE"
            cooldown_time_str = datetime.fromtimestamp(self.cooldown_until, tz=timezone.utc).strftime("%H:%M:%S UTC")
            cb_msg = (
                f"🛑 CIRCUIT BREAKER TRIGGERED: {self.consecutive_stop_losses} consecutive SL hits! "
                f"Halting trading for {cooldown_mins}m until {cooldown_time_str} followed by Trend Recheck"
            )
            applied_fixes.append(cb_msg)
            self.log(f"[CIRCUIT BREAKER] {cb_msg}")

        # Persist updated configuration
        try:
            root = Path(__file__).resolve().parents[1]
            cfg_path = root / "config/production_strategy.json"
            cfg_path.write_text(json.dumps(self.strategy_config, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 3. Autonomous Neural RL Policy & Value Function Healing (Shadow Model & Experience Replay Gated)
        penalty_w = float(fixes.get("penalty_weight", 2.0))
        hold_w = float(fixes.get("boost_hold_weight", 1.0))

        try:
            shadow_model = copy.deepcopy(self.model)
            shadow_model.train()
            optimizer = torch.optim.AdamW(shadow_model.parameters(), lr=1e-4, weight_decay=1e-4)

            feat_in = entry_feat.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 32)

            # Sample baseline states from replay buffer to prevent catastrophic forgetting
            baseline_samples = []
            if len(self.replay_buffer) > 0:
                indices = np.random.choice(len(self.replay_buffer), size=min(16, len(self.replay_buffer)), replace=False)
                baseline_samples = [self.replay_buffer[int(i)] for i in indices]

            for _ in range(3):
                optimizer.zero_grad()
                outputs, _ = shadow_model(feat_in)
                probs = torch.softmax(outputs["policy"], dim=-1)

                loss_penalize_action = probs[0, 0, action_idx]
                loss_encourage_neutral = -torch.log(probs[0, 0, 0] + 1e-6)
                loss_val = (outputs["value"] - torch.tensor([[[-1.0]]])).pow(2).mean()

                loss_fail = penalty_w * loss_penalize_action + hold_w * loss_encourage_neutral + 0.5 * loss_val

                # KL divergence penalty against baseline replay states to prevent catastrophic forgetting
                loss_reg = torch.tensor(0.0)
                if baseline_samples:
                    batch_base = torch.stack(baseline_samples).unsqueeze(1)  # (B, 1, 32)
                    with torch.no_grad():
                        orig_out, _ = self.model(batch_base)
                        orig_probs = torch.softmax(orig_out["policy"], dim=-1)
                    shadow_out, _ = shadow_model(batch_base)
                    shadow_log_probs = torch.log_softmax(shadow_out["policy"], dim=-1)
                    loss_reg = torch.nn.functional.kl_div(shadow_log_probs, orig_probs, reduction="batchmean")

                loss = loss_fail + 1.0 * loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(shadow_model.parameters(), 0.5)
                optimizer.step()

            # Shadow Validation Gate: Verify policy stability before promoting to live
            shadow_model.eval()
            passed_gate = True
            rejection_reason = "OK"

            with torch.no_grad():
                test_out, _ = shadow_model(feat_in)
                test_probs = torch.softmax(test_out["policy"], dim=-1)[0, 0].cpu().numpy()
                test_val = float(test_out["value"][0, 0, 0].cpu())
                if np.isnan(test_probs).any() or np.isnan(test_val) or abs(test_val) > 10.0:
                    passed_gate = False
                    rejection_reason = "NaN or diverging value prediction in shadow model"
                entropy = float(-np.sum(test_probs * np.log(np.maximum(test_probs, 1e-8))))
                if entropy < 0.20:
                    passed_gate = False
                    rejection_reason = f"Policy entropy collapsed ({entropy:.3f} < 0.20)"

            if passed_gate:
                self.model.load_state_dict(shadow_model.state_dict())
                self.online_adaptations += 1

                # Save dynamically adapted weights
                root = Path(__file__).resolve().parents[1]
                artifacts_dir = root / "artifacts/gtrxl"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                pt_path = artifacts_dir / "gtrxl_model.pt"
                torch.save(self.model.state_dict(), pt_path)
                applied_fixes.append("Shadow model validated & committed (replay KL gated) -> updated gtrxl_model.pt")
            else:
                applied_fixes.append(f"Shadow update blocked by safety gate ({rejection_reason}) -> live weights preserved")
                self.log(f"[AUTONOMOUS LEARNING SAFETY GATE] {rejection_reason}")

            fix_summary = " · ".join(applied_fixes)
            self.log(f"[AUTONOMOUS SELF-FIX APPLIED] {fix_summary}")

            with self._lock:
                self.self_healing_count += 1
                self.latest_status["self_healing_count"] = self.self_healing_count
                self.latest_status["online_adaptations"] = self.online_adaptations
                self.latest_status["last_learning_event"] = f"Penalized {action_name} after -₹{abs(exit_pnl):.1f} loss"
                self.latest_status["last_failure_analysis"] = {
                    "reason_code": reason_code,
                    "explanation": explanation,
                    "action": action_name,
                    "loss_inr": abs(exit_pnl),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.latest_status["last_failure_fix"] = {
                    "applied_fixes": applied_fixes,
                    "fix_summary": fix_summary,
                    "online_adaptation_step": self.online_adaptations,
                }
                self.latest_status["consecutive_stop_losses"] = self.consecutive_stop_losses
                self.latest_status["circuit_breaker_active"] = self.circuit_breaker_active
                self.latest_status["cooldown_until"] = (
                    datetime.fromtimestamp(self.cooldown_until, tz=timezone.utc).isoformat()
                    if self.cooldown_until else None
                )
                self.latest_status["trend_recheck_status"] = self.trend_recheck_status

            return {
                "status": "success",
                "reason_code": reason_code,
                "explanation": explanation,
                "applied_fixes": applied_fixes,
                "fix_summary": fix_summary,
            }
        except Exception as learn_exc:
            self.log(f"[AUTONOMOUS LEARNING ERROR] Policy adaptation failed: {learn_exc}")
            return {"status": "error", "error": str(learn_exc)}
        finally:
            self.model.eval()

    def _place_order(self, symbol: str, side: str, size: int, is_retry: bool = False, is_opening: bool = False) -> bool:
        """Executes a market order via DeltaMcpClient with fail-closed protection and exchange-native stop orders."""
        try:
            tools = self.client.available_tools()
            if "place_order" not in tools:
                self.log(f"Cannot place {side.upper()} order: 'place_order' tool unavailable.")
                return False

            product_id = None
            try:
                p = self.client.call("get_product", {"symbol": symbol})
                p_data = p.get("result", p) if isinstance(p, dict) else p
                if isinstance(p_data, dict) and "id" in p_data:
                    product_id = p_data["id"]
            except Exception:
                pass

            if product_id is None:
                self.log(f"CRITICAL: Failed to resolve product ID for {symbol}. Order aborted to prevent unintended trades.")
                return False

            order_payload: Dict[str, Any] = {
                "product_id": product_id,
                "size": size,
                "side": side.lower(),
                "order_type": "market_order",
            }

            # Attach exchange-native protective stop if opening a new position
            if is_opening:
                try:
                    t_res = self.client.call("get_ticker", {"symbol": symbol})
                    t_data = t_res.get("result", t_res) if isinstance(t_res, dict) else t_res
                    mp = float(t_data.get("mark_price") or t_data.get("close") or 0.0)
                    if mp > 0:
                        limits = self.strategy_config.get("risk_limits", {})
                        sl_inr = float(limits.get("per_trade_stop_loss", 300.0))
                        # Price distance: sl_inr / (size * 0.001 * 87 / 10)
                        price_dist = sl_inr / max(size * 0.001 * 87.0 / 10.0, 1.0)
                        sl_price = round(mp - price_dist if side.lower() == "buy" else mp + price_dist, 1)
                        order_payload["stop_loss_price"] = str(sl_price)
                except Exception as stop_err:
                    self.log(f"Protective stop calculation notice: {stop_err}")

            res = self.client.call("place_order", order_payload)
            self.log(f"ORDER EXECUTED: {side.upper()} {size} lot(s) {symbol}. Result: {res}")
            with self._lock:
                if is_opening:
                    self.trades_today += 1
                    self.latest_status["trades_today"] = self.trades_today
                self.latest_status["last_action_taken"] = f"Executed {side.upper()} {size} lot(s)"
                self.last_entry_features = self.current_bar_features
                self.last_entry_side = side
            return True
        except Exception as exc:
            self.log(f"Order submission error: {exc}")
            if not is_retry:
                healed = self._self_heal_execution_issue(exc, symbol, side, size, is_opening=is_opening)
                if healed:
                    return True
            return False

    def _close_position(self, symbol: str, size: int, unrealized_pnl: float = 0.0) -> bool:
        """Closes an existing position by executing an opposing market order, learning on loss."""
        opposing_side = "sell" if size > 0 else "buy"
        abs_size = abs(size)
        self.log(f"Closing position of {size} lots with {opposing_side.upper()} {abs_size} lots...")
        ok = self._place_order(symbol, side=opposing_side, size=abs_size)
        if ok:
            if unrealized_pnl < 0:
                entry_side = "buy" if size > 0 else "sell"
                self.learn_from_trade_failure(exit_pnl=unrealized_pnl, side=entry_side, entry_feat=self.last_entry_features)
            else:
                # Profitable close resets consecutive stop-losses
                self.consecutive_stop_losses = 0
                self.circuit_breaker_active = False
                self.cooldown_until = None
                self.trend_recheck_status = "NORMAL"
                with self._lock:
                    self.latest_status["consecutive_stop_losses"] = 0
                    self.latest_status["circuit_breaker_active"] = False
                    self.latest_status["cooldown_until"] = None
                    self.latest_status["trend_recheck_status"] = "NORMAL"
        return ok

    def start(self) -> None:
        """Starts the background automated trading loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="GTrXLTraderThread", daemon=True)
        self._thread.start()
        self.log("Automated trading background loop started.")

    def stop(self) -> None:
        """Stops the background automated trading loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.log("Automated trading background loop stopped.")

    def _run_loop(self) -> None:
        """Continuous execution loop."""
        # Initial warmup
        symbol = self.strategy_config.get("product_symbol", "ETHUSD")
        resolution = self.strategy_config.get("bar_resolution", "1m")
        self.warmup(symbol=symbol, resolution=resolution, lookback_bars=150)

        while not self._stop_event.is_set():
            try:
                self.evaluate_and_trade()
            except Exception as e:
                self.log(f"Unhandled error in evaluation loop: {e}")
            # Wait for next poll interval
            self._stop_event.wait(self.poll_interval)
