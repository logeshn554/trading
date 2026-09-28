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

from datetime import datetime, timezone
from decimal import Decimal
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError
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
    """Computes a causal 32-dimensional feature vector for the latest candle.

    Args:
        candles: List of historical candles sorted by timestamp ascending.
                 Each candle is a dict with keys: 'open', 'high', 'low', 'close', 'volume', 'time'.
        current_position_exposure: -1.0 (short), 0.0 (flat), or 1.0 (long).

    Returns:
        1D torch.Tensor of shape (32,) or None if insufficient history (< 21 bars).
    """
    if len(candles) < 21:
        return None

    closes = [_safe_float(c.get("close")) for c in candles]
    highs = [_safe_float(c.get("high")) for c in candles]
    lows = [_safe_float(c.get("low")) for c in candles]
    opens = [_safe_float(c.get("open")) for c in candles]
    vols = [_safe_float(c.get("volume")) for c in candles]
    times = [c.get("time", 0) for c in candles]

    idx = len(candles) - 1
    c_t = closes[idx]
    h_t = highs[idx]
    l_t = lows[idx]
    o_t = opens[idx]
    v_t = vols[idx]

    if c_t <= 0.0 or h_t < l_t:
        return None

    eps = 1e-6

    # 1-5: Multi-horizon log returns
    r1 = math.log(max(c_t, eps) / max(closes[idx - 1], eps))
    r2 = math.log(max(c_t, eps) / max(closes[idx - 2], eps))
    r4 = math.log(max(c_t, eps) / max(closes[idx - 4], eps))
    r8 = math.log(max(c_t, eps) / max(closes[idx - 8], eps))
    r16 = math.log(max(c_t, eps) / max(closes[idx - 16], eps))

    # 6-10: Candle geometry
    hl_range = (h_t - l_t) / (c_t + eps)
    close_loc = (c_t - l_t) / (h_t - l_t + eps)
    upper_shadow = (h_t - max(o_t, c_t)) / (h_t - l_t + eps)
    lower_shadow = (min(o_t, c_t) - l_t) / (h_t - l_t + eps)
    body_ratio = (c_t - o_t) / (h_t - l_t + eps)

    # 11-13: Volume dynamics
    recent_vols = vols[max(0, idx - 10):idx + 1]
    mean_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 1.0
    vol_ratio = v_t / (mean_vol + eps)
    vol_log_ret = math.log(max(v_t, eps) / max(vols[idx - 1], eps))
    vol_price_trend = (1.0 if c_t >= closes[idx - 1] else -1.0) * math.log(v_t + 1.0)

    # 14-17: Volatility metrics
    # 5-bar Parkinson volatility
    p5 = sum(
        (math.log(max(highs[i], eps) / max(lows[i], eps)) ** 2) / (4.0 * math.log(2.0))
        for i in range(idx - 4, idx + 1)
    ) / 5.0
    vol_parkinson_5 = math.sqrt(max(p5, 0.0))

    # 20-bar Parkinson volatility
    p20 = sum(
        (math.log(max(highs[i], eps) / max(lows[i], eps)) ** 2) / (4.0 * math.log(2.0))
        for i in range(idx - 19, idx + 1)
    ) / 20.0
    vol_parkinson_20 = math.sqrt(max(p20, 0.0))

    tr = max(h_t - l_t, abs(h_t - closes[idx - 1]), abs(l_t - closes[idx - 1])) / (c_t + eps)
    vol_ratio_5_20 = vol_parkinson_5 / (vol_parkinson_20 + eps)

    # 18-20: Trend & Momentum
    # EMA 5 vs EMA 20 proxy
    ema5 = sum(closes[idx - 4:idx + 1]) / 5.0
    ema20 = sum(closes[idx - 19:idx + 1]) / 20.0
    trend_spread = (ema5 - ema20) / (ema20 + eps)

    # 14-period RSI
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(idx - 13, idx + 1)]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(idx - 13, idx + 1)]
    avg_gain = sum(gains) / 14.0
    avg_loss = sum(losses) / 14.0
    rs = avg_gain / (avg_loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    norm_rsi = (rsi - 50.0) / 50.0  # Range [-1, 1]

    # MACD momentum proxy
    macd_proxy = (r1 + r2 * 0.5) - (r8 * 0.25)

    # 21-22: Bollinger Bands (%B and width)
    sma20 = sum(closes[idx - 19:idx + 1]) / 20.0
    variance = sum((p - sma20) ** 2 for p in closes[idx - 19:idx + 1]) / 20.0
    std20 = math.sqrt(variance)
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    b_percent = (c_t - lower_b) / (upper_b - lower_b + eps)
    b_width = (upper_b - lower_b) / (sma20 + eps)

    # 23-25: Flow dynamics
    flow_proxy = ((c_t - l_t) - (h_t - c_t)) / (h_t - l_t + eps) * math.log(v_t + 1.0)
    flow_6 = sum(
        ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i] + eps) * math.log(vols[i] + 1.0)
        for i in range(max(0, idx - 5), idx + 1)
    ) / 6.0
    flow_12 = sum(
        ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i] + eps) * math.log(vols[i] + 1.0)
        for i in range(max(0, idx - 11), idx + 1)
    ) / 12.0

    # 26-27: Return acceleration & 20-bar Breakout position
    r1_prev = math.log(max(closes[idx - 1], eps) / max(closes[idx - 2], eps))
    return_accel = r1 - r1_prev

    min20 = min(lows[idx - 19:idx + 1])
    max20 = max(highs[idx - 19:idx + 1])
    breakout_pos = (c_t - min20) / (max20 - min20 + eps)

    # 28-31: Cyclical time features (hour of day, day of week)
    try:
        ts = int(times[idx])
        # If timestamp is in microseconds or milliseconds
        if ts > 1e12:
            ts = ts / 1e6 if ts > 1e15 else ts / 1e3
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)

    hour_angle = 2.0 * math.pi * dt.hour / 24.0
    sin_hour = math.sin(hour_angle)
    cos_hour = math.cos(hour_angle)

    dow_angle = 2.0 * math.pi * dt.weekday() / 7.0
    sin_dow = math.sin(dow_angle)
    cos_dow = math.cos(dow_angle)

    # 32: Current portfolio exposure (-1.0, 0.0, +1.0)
    pos_feat = float(current_position_exposure)

    features = [
        r1, r2, r4, r8, r16,
        hl_range, close_loc, upper_shadow, lower_shadow, body_ratio,
        vol_ratio, vol_log_ret, vol_price_trend,
        vol_parkinson_5, vol_parkinson_20, tr, vol_ratio_5_20,
        trend_spread, norm_rsi, macd_proxy,
        b_percent, b_width,
        flow_proxy, flow_6, flow_12,
        return_accel, breakout_pos,
        sin_hour, cos_hour, sin_dow, cos_dow,
        pos_feat,
    ]

    return torch.tensor(features, dtype=torch.float32)


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

        self.latest_status: Dict[str, Any] = {
            "status": "INITIALIZING",
            "model": "GTrXL-RL",
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
            "recent_logs": [],
        }

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

    def warmup(self, symbol: str = "ETHUSD", resolution: str = "1h", lookback_bars: int = 150) -> bool:
        """Primes GTrXL recurrent memory cache with historical market data."""
        self.log(f"Starting GTrXL memory warmup for {symbol} ({resolution}, {lookback_bars} bars)...")
        now = datetime.now(timezone.utc)
        start_ts = int((now.timestamp() - lookback_bars * 3600))
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
        resolution = self.strategy_config.get("bar_resolution", "1h")
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        with self._lock:
            if self.last_trade_date != today_str:
                self.trades_today = 0
                self.last_trade_date = today_str
                self.latest_status["trades_today"] = 0

        # Check live submission switch
        live_enabled = bool(self.strategy_config.get("live_order_submission_enabled", False))
        limits = self.strategy_config.get("risk_limits", {})
        max_trades = int(limits.get("max_trades_per_day", 5))
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
                self._close_position(symbol, current_pos_size)
                return
            elif unrealized_pnl >= per_trade_tp:
                self.log(f"RISK TRIGGER: Take-profit reached (PnL ₹{unrealized_pnl:.2f} >= ₹{per_trade_tp:.2f}). Locking profit.")
                self._close_position(symbol, current_pos_size)
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

        # 5. Order execution logic
        if not live_enabled:
            return

        # Risk check: daily trades limit
        if self.trades_today >= max_trades:
            with self._lock:
                self.latest_status["risk_status"] = f"MAX_DAILY_TRADES_HIT ({self.trades_today}/{max_trades})"
            return

        # Check for actionable trade signals
        if action == self.ACTION_BUY and current_pos_size <= 0:
            self.log(f"GTrXL SIGNAL: BUY (Conf: {confidence*100:.1f}%, Val: {value_est:.3f})")
            if current_pos_size < 0:
                self._close_position(symbol, current_pos_size)
            self._place_order(symbol, side="buy", size=contract_size)
        elif action == self.ACTION_SELL and current_pos_size >= 0:
            self.log(f"GTrXL SIGNAL: SELL (Conf: {confidence*100:.1f}%, Val: {value_est:.3f})")
            if current_pos_size > 0:
                self._close_position(symbol, current_pos_size)
            self._place_order(symbol, side="sell", size=contract_size)

    def _place_order(self, symbol: str, side: str, size: int) -> bool:
        """Executes a market order via DeltaMcpClient."""
        try:
            tools = self.client.available_tools()
            if "place_order" not in tools:
                self.log(f"Cannot place {side.upper()} order: 'place_order' tool unavailable.")
                return False

            product_id = 27
            try:
                p = self.client.call("get_product", {"symbol": symbol})
                p_data = p.get("result", p) if isinstance(p, dict) else p
                if isinstance(p_data, dict) and "id" in p_data:
                    product_id = p_data["id"]
            except Exception:
                pass

            res = self.client.call("place_order", {
                "product_id": product_id,
                "size": size,
                "side": side.lower(),
                "order_type": "market_order",
            })
            self.log(f"ORDER EXECUTED: {side.upper()} {size} lot(s) {symbol}. Result: {res}")
            with self._lock:
                self.trades_today += 1
                self.latest_status["trades_today"] = self.trades_today
                self.latest_status["last_action_taken"] = f"Executed {side.upper()} {size} lot(s)"
            return True
        except DeltaMcpError as exc:
            self.log(f"Order submission failed: {exc}")
            return False

    def _close_position(self, symbol: str, size: int) -> bool:
        """Closes an existing position by executing an opposing market order."""
        opposing_side = "sell" if size > 0 else "buy"
        abs_size = abs(size)
        self.log(f"Closing position of {size} lots with {opposing_side.upper()} {abs_size} lots...")
        return self._place_order(symbol, side=opposing_side, size=abs_size)

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
        resolution = self.strategy_config.get("bar_resolution", "1h")
        self.warmup(symbol=symbol, resolution=resolution, lookback_bars=150)

        while not self._stop_event.is_set():
            try:
                self.evaluate_and_trade()
            except Exception as e:
                self.log(f"Unhandled error in evaluation loop: {e}")
            # Wait for next poll interval
            self._stop_event.wait(self.poll_interval)
