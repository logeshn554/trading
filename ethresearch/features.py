"""Unified Strictly Causal 32-Feature Extraction Engine.

Guarantees 100% mathematical equivalence between Training and Real-Time Inference:
- ZERO LOOKAHEAD LEAKAGE: All indicators use trailing windows [t-k : t] exclusively.
- No centered convolutions, no future-peeking modes.
- Shared between offline RL training and real-time streaming trader.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


EPS = 1e-8


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def compute_causal_candle_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    vols: np.ndarray,
    timestamps: np.ndarray,
    position_exposure: float = 0.0,
    idx: Optional[int] = None,
) -> np.ndarray:
    """Computes a strictly causal 32-dimensional feature vector for candle at `idx`.
    
    Uses strictly past and current observations [0 : idx + 1]. Absolutely NO lookahead.
    
    Feature Layout:
    1-5:   Multi-horizon log returns (r1, r2, r4, r8, r16)
    6-10:  Candle geometry (hl_range, close_loc, upper_shadow, lower_shadow, body_ratio)
    11-13: Volume dynamics (vol_ratio, vol_log_ret, vol_price_trend)
    14-17: Volatility metrics (vol_p5, vol_p20, tr, vol_ratio_5_20)
    18-20: Trend & Momentum (trend_spread, norm_rsi, macd_proxy)
    21-22: Bollinger Bands (b_percent, b_width)
    23-25: Flow dynamics (flow_raw, flow_6, flow_12)
    26-27: Acceleration & 20-bar Breakout location
    28-31: Cyclical time (sin/cos hour, sin/cos day-of-week)
    32:    Current portfolio exposure (-1.0, 0.0, 1.0)
    """
    if idx is None:
        idx = len(closes) - 1

    if idx < 20:
        return np.zeros(32, dtype=np.float32)

    c_t = float(closes[idx])
    h_t = float(highs[idx])
    l_t = float(lows[idx])
    o_t = float(opens[idx])
    v_t = float(vols[idx])
    ts_t = float(timestamps[idx])

    # 1-5: Multi-horizon log returns
    r1 = math.log(max(c_t, EPS) / max(float(closes[idx - 1]), EPS))
    r2 = math.log(max(c_t, EPS) / max(float(closes[idx - 2]), EPS))
    r4 = math.log(max(c_t, EPS) / max(float(closes[idx - 4]), EPS))
    r8 = math.log(max(c_t, EPS) / max(float(closes[idx - 8]), EPS))
    r16 = math.log(max(c_t, EPS) / max(float(closes[idx - 16]), EPS))

    # 6-10: Candle geometry
    hl = max(h_t - l_t, EPS)
    hl_range = hl / max(c_t, EPS)
    close_loc = (c_t - l_t) / hl
    upper_shadow = (h_t - max(o_t, c_t)) / hl
    lower_shadow = (min(o_t, c_t) - l_t) / hl
    body_ratio = (c_t - o_t) / hl

    # 11-13: Volume dynamics (causal trailing 10-bar mean)
    vol_start = max(0, idx - 9)
    v_slice = vols[vol_start:idx + 1]
    vol_mean_10 = float(np.mean(v_slice)) if len(v_slice) > 0 else v_t
    vol_ratio = v_t / max(vol_mean_10, EPS)
    vol_log_ret = math.log(max(v_t, EPS) / max(float(vols[idx - 1]), EPS))
    vol_price_trend = (1.0 if c_t >= float(closes[idx - 1]) else -1.0) * math.log(v_t + 1.0)

    # 14-17: Causal Parkinson Volatility (strictly trailing 5-bar and 20-bar)
    h_5 = highs[idx - 4:idx + 1]
    l_5 = lows[idx - 4:idx + 1]
    p5_sq = np.sum((np.log(np.maximum(h_5, EPS) / np.maximum(l_5, EPS)) ** 2) / (4.0 * math.log(2.0))) / 5.0
    vol_p5 = math.sqrt(max(float(p5_sq), 0.0))

    h_20 = highs[idx - 19:idx + 1]
    l_20 = lows[idx - 19:idx + 1]
    p20_sq = np.sum((np.log(np.maximum(h_20, EPS) / np.maximum(l_20, EPS)) ** 2) / (4.0 * math.log(2.0))) / 20.0
    vol_p20 = math.sqrt(max(float(p20_sq), 0.0))

    prev_c = float(closes[idx - 1])
    tr = max(h_t - l_t, abs(h_t - prev_c), abs(l_t - prev_c)) / max(c_t, EPS)
    vol_ratio_5_20 = vol_p5 / max(vol_p20, EPS)

    # 18-20: Trend & Momentum (strictly trailing EMA 5 vs 20)
    ema5 = float(np.mean(closes[idx - 4:idx + 1]))
    ema20 = float(np.mean(closes[idx - 19:idx + 1]))
    trend_spread = (ema5 - ema20) / max(ema20, EPS)

    # Causal 14-period RSI
    diffs = np.diff(closes[idx - 14:idx + 1])
    gains = np.maximum(diffs, 0.0)
    losses = np.maximum(-diffs, 0.0)
    avg_g = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_l = float(np.mean(losses)) if len(losses) > 0 else 0.0
    rs = avg_g / max(avg_l, EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    norm_rsi = (rsi - 50.0) / 50.0

    macd_proxy = (r1 + r2 * 0.5) - (r16 * 0.25)

    # 21-22: Bollinger Bands (causal trailing 20-bar)
    c_20 = closes[idx - 19:idx + 1]
    sma20 = float(np.mean(c_20))
    std20 = float(np.std(c_20))
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    b_percent = (c_t - lower_b) / max(upper_b - lower_b, EPS)
    b_width = (upper_b - lower_b) / max(sma20, EPS)

    # 23-25: Flow dynamics (strictly trailing 6-bar and 12-bar)
    flow_raw = ((c_t - l_t) - (h_t - c_t)) / hl * math.log(v_t + 1.0)
    
    # Causal flow slices
    f6_c = closes[idx - 5:idx + 1]
    f6_h = highs[idx - 5:idx + 1]
    f6_l = lows[idx - 5:idx + 1]
    f6_v = vols[idx - 5:idx + 1]
    flow_6 = float(np.mean(((f6_c - f6_l) - (f6_h - f6_c)) / np.maximum(f6_h - f6_l, EPS) * np.log(f6_v + 1.0)))

    f12_c = closes[idx - 11:idx + 1]
    f12_h = highs[idx - 11:idx + 1]
    f12_l = lows[idx - 11:idx + 1]
    f12_v = vols[idx - 11:idx + 1]
    flow_12 = float(np.mean(((f12_c - f12_l) - (f12_h - f12_c)) / np.maximum(f12_h - f12_l, EPS) * np.log(f12_v + 1.0)))

    # 26-27: Acceleration & 20-bar Breakout
    r1_prev = math.log(max(float(closes[idx - 1]), EPS) / max(float(closes[idx - 2]), EPS))
    return_accel = r1 - r1_prev

    low_20 = float(np.min(l_20))
    high_20 = float(np.max(h_20))
    breakout_pos = (c_t - low_20) / max(high_20 - low_20, EPS)

    # 28-31: Cyclical time encoding
    ts_sec = int(ts_t)
    if ts_sec > 10**12:
        ts_sec = ts_sec // 1000000 if ts_sec > 10**15 else ts_sec // 1000
    hour = (ts_sec // 3600) % 24
    dow = ((ts_sec // 86400) + 4) % 7
    sin_hour = math.sin(2.0 * math.pi * hour / 24.0)
    cos_hour = math.cos(2.0 * math.pi * hour / 24.0)
    sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
    cos_dow = math.cos(2.0 * math.pi * dow / 7.0)

    # 32: Current Portfolio Exposure
    exposure = float(np.clip(position_exposure, -1.0, 1.0))

    return np.array([
        r1, r2, r4, r8, r16,
        hl_range, close_loc, upper_shadow, lower_shadow, body_ratio,
        vol_ratio, vol_log_ret, vol_price_trend,
        vol_p5, vol_p20, tr, vol_ratio_5_20,
        trend_spread, norm_rsi, macd_proxy,
        b_percent, b_width, flow_raw, flow_6, flow_12,
        return_accel, breakout_pos,
        sin_hour, cos_hour, sin_dow, cos_dow,
        exposure,
    ], dtype=np.float32)


def extract_all_causal_features(
    candles_matrix: np.ndarray,
    position_exposures: Optional[np.ndarray] = None,
    warmup: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates strictly causal features for all candles starting from `warmup`.
    
    Args:
        candles_matrix: Array of shape (N, 6) with columns [time, open, high, low, close, volume]
        position_exposures: Array of shape (N,) indicating simulated/active position
        warmup: Number of bars used for initial trailing windows (minimum 25)
    
    Returns:
        features: Array of shape (N - warmup - 12, 32)
        fwd_returns: Array of shape (N - warmup - 12,) next-step return
        aux_targets: Array of shape (N - warmup - 12, 3) 1, 4, 12-step returns
    """
    n = len(candles_matrix)
    timestamps = candles_matrix[:, 0]
    opens = candles_matrix[:, 1]
    highs = candles_matrix[:, 2]
    lows = candles_matrix[:, 3]
    closes = candles_matrix[:, 4]
    vols = candles_matrix[:, 5]

    if position_exposures is None:
        position_exposures = np.zeros(n, dtype=np.float32)

    valid_len = n - warmup - 12
    if valid_len <= 0:
        raise ValueError(f"Insufficient candles ({n}) for warmup ({warmup}) and horizon (12)")

    feat_list = []
    fwd_ret1 = []
    fwd_ret4 = []
    fwd_ret12 = []

    for i in range(warmup, warmup + valid_len):
        f = compute_causal_candle_features(
            closes=closes,
            highs=highs,
            lows=lows,
            opens=opens,
            vols=vols,
            timestamps=timestamps,
            position_exposure=float(position_exposures[i]),
            idx=i,
        )
        feat_list.append(f)

        c_now = closes[i]
        fwd_ret1.append((closes[i + 1] - c_now) / max(c_now, EPS))
        fwd_ret4.append((closes[i + 4] - c_now) / max(c_now, EPS))
        fwd_ret12.append((closes[i + 12] - c_now) / max(c_now, EPS))

    features = np.stack(feat_list, axis=0)
    returns = np.array(fwd_ret1, dtype=np.float32)
    aux_targets = np.column_stack([fwd_ret1, fwd_ret4, fwd_ret12]).astype(np.float32)

    return features, returns, aux_targets
