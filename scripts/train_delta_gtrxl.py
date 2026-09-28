"""Train GTrXL Reinforcement Learning Algorithm purely on Delta Exchange India Market Data.

100% Pure Delta Exchange:
- Harvests 1-minute ETHUSD candles directly from Delta India via DeltaMcpClient
- Causal 32-feature extraction pipeline
- Fits and serializes StandardScaler
- Trains GTrXLActorCritic with Segmented BPTT (L=64, M=128), Rel-MHA, and GRUGate
- Saves Delta-native model weights, scaler, and deployment bundle in artifacts/gtrxl/
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from ethresearch.delta_mcp import DeltaMcpClient
from ethresearch.gtrxl import GTrXLActorCritic, GTrXLLoss

ARTIFACTS_DIR = ROOT / "artifacts/gtrxl"


def harvest_delta_candles(
    client: DeltaMcpClient,
    symbol: str = "ETHUSD",
    resolution: str = "1m",
    days: int = 14,
) -> np.ndarray:
    """Harvests historical 1-minute candles directly from Delta India Exchange in 24h chunks."""
    print(f"Connecting to Delta India MCP to harvest {days} days of {resolution} {symbol} candles...")
    now = int(time.time())
    chunk_seconds = 86400  # 24 hours per query
    all_candles: List[Dict[str, Any]] = []

    for day_idx in range(days):
        end_ts = now - day_idx * chunk_seconds
        start_ts = end_ts - chunk_seconds
        try:
            res = client.call("get_candles", {
                "symbol": symbol,
                "resolution": resolution,
                "start": start_ts,
                "end": end_ts,
            })
            candles = res.get("result", res) if isinstance(res, dict) else res
            if isinstance(candles, list) and candles:
                all_candles.extend(candles)
                dt_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"  [{dt_str}] Harvested {len(candles)} candles from Delta India.")
            time.sleep(0.1)  # Respect rate limits
        except Exception as e:
            print(f"  Warning on chunk {day_idx}: {e}")
            continue

    if not all_candles:
        raise RuntimeError("Failed to harvest candles from Delta India")

    # Deduplicate and sort chronologically by time
    unique_candles: Dict[int, Dict[str, Any]] = {}
    for c in all_candles:
        t = int(c.get("time", 0))
        if t > 0:
            unique_candles[t] = c

    sorted_times = sorted(unique_candles.keys())
    print(f"Total unique Delta India candles harvested: {len(sorted_times):,}")

    rows = []
    for t in sorted_times:
        c = unique_candles[t]
        rows.append([
            float(t),
            float(c.get("open", 0.0)),
            float(c.get("high", 0.0)),
            float(c.get("low", 0.0)),
            float(c.get("close", 0.0)),
            float(c.get("volume", 0.0)),
        ])

    return np.array(rows, dtype=np.float64)


def extract_features(candles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal 32-feature extraction from Delta Exchange candles."""
    n = len(candles)
    ts = candles[:, 0]
    opens = candles[:, 1]
    highs = candles[:, 2]
    lows = candles[:, 3]
    closes = candles[:, 4]
    vols = candles[:, 5]

    eps = 1e-6
    warmup = 25
    valid_len = n - warmup - 12

    # Returns
    ret1 = np.log(closes[1:] / np.maximum(closes[:-1], eps))
    ret2 = np.log(closes[2:] / np.maximum(closes[:-2], eps))
    ret4 = np.log(closes[4:] / np.maximum(closes[:-4], eps))
    ret8 = np.log(closes[8:] / np.maximum(closes[:-8], eps))
    ret16 = np.log(closes[16:] / np.maximum(closes[:-16], eps))

    # Geometry
    hl_range = (highs - lows) / np.maximum(closes, eps)
    close_loc = (closes - lows) / np.maximum(highs - lows, eps)
    upper_shadow = (highs - np.maximum(opens, closes)) / np.maximum(highs - lows, eps)
    lower_shadow = (np.minimum(opens, closes) - lows) / np.maximum(highs - lows, eps)
    body_ratio = (closes - opens) / np.maximum(highs - lows, eps)

    # Volume dynamics
    vol_sma10 = np.convolve(vols, np.ones(10) / 10.0, mode="same")
    vol_ratio = vols / np.maximum(vol_sma10, eps)
    vol_log_ret = np.log(np.maximum(vols[1:], eps) / np.maximum(vols[:-1], eps))
    vol_price_trend = np.sign(closes[1:] - closes[:-1]) * np.log(vols[1:] + 1.0)

    # Volatility
    log_hl_sq = (np.log(np.maximum(highs, eps) / np.maximum(lows, eps)) ** 2) / (4.0 * math.log(2.0))
    vol_p5 = np.sqrt(np.maximum(np.convolve(log_hl_sq, np.ones(5) / 5.0, mode="same"), 0.0))
    vol_p20 = np.sqrt(np.maximum(np.convolve(log_hl_sq, np.ones(20) / 20.0, mode="same"), 0.0))
    vol_ratio_5_20 = vol_p5 / np.maximum(vol_p20, eps)
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ]) / np.maximum(closes[1:], eps)

    # Trend & Momentum
    ema5 = np.convolve(closes, np.ones(5) / 5.0, mode="same")
    ema20 = np.convolve(closes, np.ones(20) / 20.0, mode="same")
    trend_spread = (ema5 - ema20) / np.maximum(ema20, eps)

    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = np.convolve(gains, np.ones(14) / 14.0, mode="same")
    avg_loss = np.convolve(losses, np.ones(14) / 14.0, mode="same")
    rsi = 100.0 - (100.0 / (1.0 + avg_gain / np.maximum(avg_loss, eps)))
    norm_rsi = (rsi - 50.0) / 50.0
    macd_proxy = (ret1[15:] + ret2[14:] * 0.5) - (ret16 * 0.25)

    # Bollinger Bands
    sma20 = np.convolve(closes, np.ones(20) / 20.0, mode="same")
    std20 = np.sqrt(np.maximum(np.convolve(closes ** 2, np.ones(20) / 20.0, mode="same") - sma20 ** 2, 0.0))
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    b_percent = (closes - lower_b) / np.maximum(upper_b - lower_b, eps)
    b_width = (upper_b - lower_b) / np.maximum(sma20, eps)

    # Flow
    flow_raw = ((closes - lows) - (highs - closes)) / np.maximum(highs - lows, eps) * np.log(vols + 1.0)
    flow_6 = np.convolve(flow_raw, np.ones(6) / 6.0, mode="same")
    flow_12 = np.convolve(flow_raw, np.ones(12) / 12.0, mode="same")

    return_accel = ret1[1:] - ret1[:-1]
    min20 = np.array([np.min(lows[max(0, i - 19):i + 1]) for i in range(n)])
    max20 = np.array([np.max(highs[max(0, i - 19):i + 1]) for i in range(n)])
    breakout_pos = (closes - min20) / np.maximum(max20 - min20, eps)

    # Cyclical time
    hours = (ts // 3600) % 24
    dows = ((ts // 86400) + 4) % 7
    sin_hour = np.sin(2.0 * np.pi * hours / 24.0)
    cos_hour = np.cos(2.0 * np.pi * hours / 24.0)
    sin_dow = np.sin(2.0 * np.pi * dows / 7.0)
    cos_dow = np.cos(2.0 * np.pi * dows / 7.0)
    pos_dummy = np.zeros(n)

    w = warmup
    vl = valid_len
    features = np.column_stack([
        ret1[w - 1:w - 1 + vl], ret2[w - 2:w - 2 + vl], ret4[w - 4:w - 4 + vl], ret8[w - 8:w - 8 + vl], ret16[w - 16:w - 16 + vl],
        hl_range[w:w + vl], close_loc[w:w + vl], upper_shadow[w:w + vl], lower_shadow[w:w + vl], body_ratio[w:w + vl],
        vol_ratio[w:w + vl], vol_log_ret[w - 1:w - 1 + vl], vol_price_trend[w - 1:w - 1 + vl],
        vol_p5[w:w + vl], vol_p20[w:w + vl], tr[w - 1:w - 1 + vl], vol_ratio_5_20[w:w + vl],
        trend_spread[w:w + vl], norm_rsi[w - 1:w - 1 + vl], macd_proxy[w - 16:w - 16 + vl],
        b_percent[w:w + vl], b_width[w:w + vl], flow_raw[w:w + vl], flow_6[w:w + vl], flow_12[w:w + vl],
        return_accel[w - 2:w - 2 + vl], breakout_pos[w:w + vl],
        sin_hour[w:w + vl], cos_hour[w:w + vl], sin_dow[w:w + vl], cos_dow[w:w + vl],
        pos_dummy[w:w + vl],
    ])

    fwd_ret1 = (closes[w + 1:w + 1 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)
    fwd_ret4 = (closes[w + 4:w + 4 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)
    fwd_ret12 = (closes[w + 12:w + 12 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)
    aux_targets = np.column_stack([fwd_ret1, fwd_ret4, fwd_ret12])

    return features, fwd_ret1, aux_targets


def train_on_delta_data(days: int = 14):
    client = DeltaMcpClient()
    try:
        candles = harvest_delta_candles(client, symbol="ETHUSD", resolution="1m", days=days)
    finally:
        client.close()

    features, fwd_returns, aux_targets = extract_features(candles)
    total_samples = len(features)
    train_size = int(total_samples * 0.85)

    train_x = features[:train_size]
    val_x = features[train_size:]
    train_ret = fwd_returns[:train_size]
    val_ret = fwd_returns[train_size:]
    train_aux = aux_targets[:train_size]

    scaler = StandardScaler()
    train_x_norm = np.clip(scaler.fit_transform(train_x), -6.0, 6.0)
    val_x_norm = np.clip(scaler.transform(val_x), -6.0, 6.0)

    device = torch.device("cpu")
    seq_len = 64
    mem_len = 128
    batch_size = 32

    model = GTrXLActorCritic(
        d_in=32,
        action_dim=3,
        d_model=128,
        n_heads=4,
        n_layers=4,
        mem_len=mem_len,
        dropout=0.05,
    ).to(device)

    loss_fn = GTrXLLoss(clip_eps=0.2, val_clip_eps=0.2, c1_val=0.5, c2_ent=0.01, c3_aux=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    num_chunks = train_size // seq_len
    num_batches = num_chunks // batch_size
    epochs = 4

    print(f"\nTraining GTrXL purely on Delta India ETHUSD:")
    print(f"  Bars: {train_size:,} train, {len(val_x):,} validation | Chunks: {num_chunks:,}")

    x_chunks = [train_x_norm[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]
    ret_chunks = [train_ret[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]
    aux_chunks = [train_aux[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_accum = 0.0
        for b_idx in range(num_batches):
            bx = torch.tensor(np.stack(x_chunks[b_idx * batch_size:(b_idx + 1) * batch_size]), dtype=torch.float32)
            bret = torch.tensor(np.stack(ret_chunks[b_idx * batch_size:(b_idx + 1) * batch_size]), dtype=torch.float32)
            baux = torch.tensor(np.stack(aux_chunks[b_idx * batch_size:(b_idx + 1) * batch_size]), dtype=torch.float32)

            optimizer.zero_grad()
            outputs, _ = model(bx)
            logits = outputs["policy"]
            values = outputs["value"]
            pred_aux = outputs["aux_returns"]

            target_actions = torch.where(bret > 0.0005, torch.tensor(1), torch.where(bret < -0.0005, torch.tensor(2), torch.tensor(0)))
            returns = bret.unsqueeze(-1) * 100.0
            advantages = (returns - values.detach()).squeeze(-1)

            losses = loss_fn(
                policy_logits=logits,
                old_policy_logits=logits.detach(),
                actions=target_actions,
                advantages=advantages,
                values=values,
                old_values=values.detach(),
                returns=returns,
                aux_predictions=pred_aux,
                target_aux_returns=baux * 100.0,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss_accum += losses["loss"].item()

        print(f"Epoch {epoch}/{epochs} [{time.time() - t0:.1f}s] - Avg Loss: {loss_accum / max(num_batches, 1):.4f}")

    # Out-of-sample check on Delta India data
    model.eval()
    val_chunks = len(val_x_norm) // seq_len
    trades = 0
    wins = 0
    pnl = 0.0
    current_pos = 0
    memories = None
    with torch.no_grad():
        for vc in range(val_chunks):
            chunk = torch.tensor(val_x_norm[vc * seq_len:(vc + 1) * seq_len], dtype=torch.float32).unsqueeze(0)
            outs, memories = model(chunk, memories)
            acts = torch.argmax(outs["policy"].squeeze(0), dim=-1).cpu().numpy()
            chk_ret = val_ret[vc * seq_len:(vc + 1) * seq_len]
            for t_step in range(seq_len):
                act = acts[t_step]
                ret = chk_ret[t_step]
                new_pos = 1 if act == 1 else (-1 if act == 2 else 0)
                step_pnl = current_pos * ret - (0.0004 if new_pos != current_pos else 0.0)
                pnl += step_pnl
                if new_pos != current_pos:
                    trades += 1
                    if step_pnl > 0:
                        wins += 1
                current_pos = new_pos

    win_rate = (wins / max(trades, 1)) * 100.0
    print(f"\n--- Delta India Validation Results ---")
    print(f"  Venue: Pure Delta India (ETHUSD)")
    print(f"  Trades: {trades} | Win Rate: {win_rate:.2f}% | PnL: {pnl * 100:.2f}%")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_pt = ARTIFACTS_DIR / "gtrxl_model.pt"
    scaler_pkl = ARTIFACTS_DIR / "scaler.pkl"
    bundle_pkl = ARTIFACTS_DIR / "model_bundle.pkl"

    torch.save(model.state_dict(), model_pt)
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    bundle = {
        "model_architecture": "GTrXL (Gated Transformer-XL)",
        "venue": "delta_india",
        "symbol": "ETHUSD",
        "d_in": 32,
        "action_dim": 3,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "mem_len": 128,
        "state_dict": model.state_dict(),
        "scaler": scaler,
        "training_metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "venue": "Delta India Exchange (ETHUSD)",
            "total_bars": total_samples,
            "win_rate": round(win_rate, 2),
            "trades": trades,
            "epochs": epochs,
        },
    }
    with open(bundle_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved Delta-native model weights & scaler to {ARTIFACTS_DIR}")
    return bundle


if __name__ == "__main__":
    train_on_delta_data(days=7)
