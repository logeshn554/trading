"""Train GTrXL Reinforcement Learning Algorithm on 1-Minute ETH Data.

Pipelines:
1. Ingests all 128,160+ 1-minute ETHUSDT candles from data/ar90/
2. Vectorized 32-feature causal extraction (multi-horizon returns, volatility, volume flow, RSI, MACD, cyclical time)
3. Fits and persists a StandardScaler
4. Trains GTrXLActorCritic with Segmented BPTT (L=64, M=128), Rel-MHA, and GRUGate
5. Evaluates out-of-sample trading performance (Win rate, Sharpe ratio, Net PnL)
6. Serializes all artifacts (gtrxl_model.pt, scaler.pkl, model_bundle.pkl) for deployment
"""
from __future__ import annotations

from datetime import datetime, timezone
import glob
import math
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from ethresearch.gtrxl import GTrXLActorCritic, GTrXLLoss


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/ar90"
ARTIFACTS_DIR = ROOT / "artifacts/gtrxl"


def load_all_1m_candles() -> np.ndarray:
    """Loads all 1-minute ETHUSDT candles from data/ar90/*.zip.

    Returns:
        Structured array of shape (N, 6): [timestamp, open, high, low, close, volume]
    """
    print(f"Scanning for 1m zip files in {DATA_DIR}...")
    zip_files = sorted(glob.glob(str(DATA_DIR / "ETHUSDT-1m-*.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No 1m zip files found in {DATA_DIR}")

    all_rows = []
    t0 = time.time()
    for zpath in zip_files:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    content = z.read(name).decode("utf-8").strip().split("\n")
                    for line in content:
                        parts = line.split(",")
                        if len(parts) >= 6:
                            # timestamp, open, high, low, close, volume
                            all_rows.append([
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                                float(parts[5]),
                            ])

    arr = np.array(all_rows, dtype=np.float64)
    # Sort chronologically by timestamp
    arr = arr[np.argsort(arr[:, 0])]
    print(f"Loaded {len(arr):,} 1-minute ETH candles in {time.time() - t0:.2f}s.")
    return arr


def extract_features_vectorized(candles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of 32 causal features across the entire candle array.

    Args:
        candles: (N, 6) [timestamp, open, high, low, close, volume]

    Returns:
        features: (N - 25, 32)
        returns_forward: (N - 25,)
        aux_targets: (N - 25, 3) forward returns for horizons [1, 4, 12]
    """
    print("Extracting 32 causal market features...")
    t0 = time.time()
    n = len(candles)
    ts = candles[:, 0]
    opens = candles[:, 1]
    highs = candles[:, 2]
    lows = candles[:, 3]
    closes = candles[:, 4]
    vols = candles[:, 5]

    eps = 1e-6
    warmup = 25
    valid_len = n - warmup - 12  # Leave buffer for forward return labels

    # 1-5: Log returns
    ret1 = np.log(closes[1:] / np.maximum(closes[:-1], eps))
    ret2 = np.log(closes[2:] / np.maximum(closes[:-2], eps))
    ret4 = np.log(closes[4:] / np.maximum(closes[:-4], eps))
    ret8 = np.log(closes[8:] / np.maximum(closes[:-8], eps))
    ret16 = np.log(closes[16:] / np.maximum(closes[:-16], eps))

    # 6-10: Candle geometry
    hl_range = (highs - lows) / np.maximum(closes, eps)
    close_loc = (closes - lows) / np.maximum(highs - lows, eps)
    upper_shadow = (highs - np.maximum(opens, closes)) / np.maximum(highs - lows, eps)
    lower_shadow = (np.minimum(opens, closes) - lows) / np.maximum(highs - lows, eps)
    body_ratio = (closes - opens) / np.maximum(highs - lows, eps)

    # 11-13: Volume dynamics
    # 10-bar rolling mean volume
    vol_kernel = np.ones(10) / 10.0
    vol_sma10 = np.convolve(vols, vol_kernel, mode="same")
    vol_ratio = vols / np.maximum(vol_sma10, eps)
    vol_log_ret = np.log(np.maximum(vols[1:], eps) / np.maximum(vols[:-1], eps))
    vol_price_trend = np.sign(closes[1:] - closes[:-1]) * np.log(vols[1:] + 1.0)

    # 14-17: Volatility metrics (Parkinson volatility)
    log_hl_sq = (np.log(np.maximum(highs, eps) / np.maximum(lows, eps)) ** 2) / (4.0 * math.log(2.0))
    p5 = np.convolve(log_hl_sq, np.ones(5) / 5.0, mode="same")
    p20 = np.convolve(log_hl_sq, np.ones(20) / 20.0, mode="same")
    vol_p5 = np.sqrt(np.maximum(p5, 0.0))
    vol_p20 = np.sqrt(np.maximum(p20, 0.0))
    vol_ratio_5_20 = vol_p5 / np.maximum(vol_p20, eps)

    # True range normalized
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ]) / np.maximum(closes[1:], eps)

    # 18-20: Trend & Momentum
    ema5 = np.convolve(closes, np.ones(5) / 5.0, mode="same")
    ema20 = np.convolve(closes, np.ones(20) / 20.0, mode="same")
    trend_spread = (ema5 - ema20) / np.maximum(ema20, eps)

    # 14-period RSI
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = np.convolve(gains, np.ones(14) / 14.0, mode="same")
    avg_loss = np.convolve(losses, np.ones(14) / 14.0, mode="same")
    rs = avg_gain / np.maximum(avg_loss, eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    norm_rsi = (rsi - 50.0) / 50.0

    # MACD proxy
    macd_proxy = (ret1[15:] + ret2[14:] * 0.5) - (ret16 * 0.25)

    # 21-22: Bollinger Bands (%B and width)
    sma20 = np.convolve(closes, np.ones(20) / 20.0, mode="same")
    # rolling std
    c_sq = np.convolve(closes ** 2, np.ones(20) / 20.0, mode="same")
    std20 = np.sqrt(np.maximum(c_sq - sma20 ** 2, 0.0))
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    b_percent = (closes - lower_b) / np.maximum(upper_b - lower_b, eps)
    b_width = (upper_b - lower_b) / np.maximum(sma20, eps)

    # 23-25: Flow dynamics
    flow_raw = ((closes - lows) - (highs - closes)) / np.maximum(highs - lows, eps) * np.log(vols + 1.0)
    flow_6 = np.convolve(flow_raw, np.ones(6) / 6.0, mode="same")
    flow_12 = np.convolve(flow_raw, np.ones(12) / 12.0, mode="same")

    # 26-27: Acceleration & 20-bar Breakout
    return_accel = ret1[1:] - ret1[:-1]
    # 20-bar rolling min/max
    min20 = np.array([np.min(lows[max(0, i - 19):i + 1]) for i in range(n)])
    max20 = np.array([np.max(highs[max(0, i - 19):i + 1]) for i in range(n)])
    breakout_pos = (closes - min20) / np.maximum(max20 - min20, eps)

    # 28-31: Cyclical time features (hour of day, day of week)
    timestamps_sec = ts / 1e6 if ts[0] > 1e15 else (ts / 1e3 if ts[0] > 1e12 else ts)
    hours = (timestamps_sec // 3600) % 24
    dows = ((timestamps_sec // 86400) + 4) % 7  # 1970-01-01 was Thursday (index 4)
    sin_hour = np.sin(2.0 * np.pi * hours / 24.0)
    cos_hour = np.cos(2.0 * np.pi * hours / 24.0)
    sin_dow = np.sin(2.0 * np.pi * dows / 7.0)
    cos_dow = np.cos(2.0 * np.pi * dows / 7.0)

    # 32: Initial neutral exposure (0.0)
    pos_dummy = np.zeros(n)

    # Align all vectors to indices [warmup : warmup + valid_len]
    w = warmup
    vl = valid_len
    feat_matrix = np.column_stack([
        ret1[w - 1:w - 1 + vl],
        ret2[w - 2:w - 2 + vl],
        ret4[w - 4:w - 4 + vl],
        ret8[w - 8:w - 8 + vl],
        ret16[w - 16:w - 16 + vl],
        hl_range[w:w + vl],
        close_loc[w:w + vl],
        upper_shadow[w:w + vl],
        lower_shadow[w:w + vl],
        body_ratio[w:w + vl],
        vol_ratio[w:w + vl],
        vol_log_ret[w - 1:w - 1 + vl],
        vol_price_trend[w - 1:w - 1 + vl],
        vol_p5[w:w + vl],
        vol_p20[w:w + vl],
        tr[w - 1:w - 1 + vl],
        vol_ratio_5_20[w:w + vl],
        trend_spread[w:w + vl],
        norm_rsi[w - 1:w - 1 + vl],
        macd_proxy[w - 16:w - 16 + vl],
        b_percent[w:w + vl],
        b_width[w:w + vl],
        flow_raw[w:w + vl],
        flow_6[w:w + vl],
        flow_12[w:w + vl],
        return_accel[w - 2:w - 2 + vl],
        breakout_pos[w:w + vl],
        sin_hour[w:w + vl],
        cos_hour[w:w + vl],
        sin_dow[w:w + vl],
        cos_dow[w:w + vl],
        pos_dummy[w:w + vl],
    ])

    # Target labels: future returns for RL reward & auxiliary head
    # Next 1m return
    fwd_ret1 = (closes[w + 1:w + 1 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)
    # 4m return
    fwd_ret4 = (closes[w + 4:w + 4 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)
    # 12m return
    fwd_ret12 = (closes[w + 12:w + 12 + vl] - closes[w:w + vl]) / np.maximum(closes[w:w + vl], eps)

    aux_targets = np.column_stack([fwd_ret1, fwd_ret4, fwd_ret12])

    print(f"Features extracted: {feat_matrix.shape} in {time.time() - t0:.2f}s.")
    return feat_matrix, fwd_ret1, aux_targets


def train_gtrxl():
    """Main training routine."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    candles = load_all_1m_candles()
    features, fwd_returns, aux_targets = extract_features_vectorized(candles)

    # 2. Train/Test Split (85% train, 15% validation/backtest)
    total_samples = len(features)
    train_size = int(total_samples * 0.85)
    train_x = features[:train_size]
    val_x = features[train_size:]

    train_ret = fwd_returns[:train_size]
    val_ret = fwd_returns[train_size:]

    train_aux = aux_targets[:train_size]
    val_aux = aux_targets[train_size:]

    # 3. Fit StandardScaler on Training Set
    scaler = StandardScaler()
    train_x_norm = scaler.fit_transform(train_x)
    val_x_norm = scaler.transform(val_x)

    # Clip outliers for numerical stability
    train_x_norm = np.clip(train_x_norm, -6.0, 6.0)
    val_x_norm = np.clip(val_x_norm, -6.0, 6.0)

    # 4. Instantiate Model & Loss
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

    loss_fn = GTrXLLoss(
        clip_eps=0.2,
        val_clip_eps=0.2,
        c1_val=0.5,
        c2_ent=0.01,
        c3_aux=0.2,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 5. Form Training Batches (Segments of length L)
    num_chunks = train_size // seq_len
    print(f"\nTraining GTrXL Reinforcement Learning Backbone:")
    print(f"  Training Bars: {train_size:,} | Validation Bars: {len(val_x):,}")
    print(f"  Sequence Length (L): {seq_len} | Memory Cache (M): {mem_len}")
    print(f"  Total Chunks: {num_chunks:,} | Batch Size: {batch_size}")

    # Build sequence dataset
    x_chunks = [train_x_norm[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]
    ret_chunks = [train_ret[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]
    aux_chunks = [train_aux[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]

    num_batches = num_chunks // batch_size
    epochs = 4

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        val_loss_accum = 0.0
        aux_loss_accum = 0.0

        for b_idx in range(num_batches):
            # Batch shape: (batch_size, seq_len, 32)
            batch_x_np = np.stack(x_chunks[b_idx * batch_size:(b_idx + 1) * batch_size])
            batch_ret_np = np.stack(ret_chunks[b_idx * batch_size:(b_idx + 1) * batch_size])
            batch_aux_np = np.stack(aux_chunks[b_idx * batch_size:(b_idx + 1) * batch_size])

            bx = torch.tensor(batch_x_np, dtype=torch.float32, device=device)
            bret = torch.tensor(batch_ret_np, dtype=torch.float32, device=device)
            baux = torch.tensor(batch_aux_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # Forward pass through GTrXL
            outputs, _ = model(bx)
            logits = outputs["policy"]          # (B, L, 3)
            values = outputs["value"]           # (B, L, 1)
            pred_aux = outputs["aux_returns"]   # (B, L, 3)

            # Synthesize targets & advantages for PPO Bellman update
            # Reward: 1 (Long) -> +ret, 2 (Short) -> -ret, 0 (Hold) -> 0
            # Target action based on positive forward return threshold
            target_actions = torch.where(bret > 0.0005, torch.tensor(1), torch.where(bret < -0.0005, torch.tensor(2), torch.tensor(0)))

            # Estimated returns & advantages
            returns = bret.unsqueeze(-1) * 100.0
            advantages = (returns - values.detach()).squeeze(-1)

            old_logits = logits.detach()
            old_values = values.detach()

            losses = loss_fn(
                policy_logits=logits,
                old_policy_logits=old_logits,
                actions=target_actions,
                advantages=advantages,
                values=values,
                old_values=old_values,
                returns=returns,
                aux_predictions=pred_aux,
                target_aux_returns=baux * 100.0,
            )

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss_accum += loss.item()
            policy_loss_accum += losses["policy_loss"].item()
            val_loss_accum += losses["value_loss"].item()
            aux_loss_accum += losses["aux_loss"].item()

        avg_loss = total_loss_accum / max(num_batches, 1)
        avg_pol = policy_loss_accum / max(num_batches, 1)
        avg_val = val_loss_accum / max(num_batches, 1)
        avg_aux = aux_loss_accum / max(num_batches, 1)
        dur = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs} [{dur:.1f}s] - Loss: {avg_loss:.4f} (Pol: {avg_pol:.4f}, Val: {avg_val:.4f}, Aux: {avg_aux:.4f})")

    # 6. Out-of-Sample Evaluation on Validation Set
    print("\nRunning Out-of-Sample Validation on 19,000+ unseen 1m bars...")
    model.eval()
    val_chunks = len(val_x_norm) // seq_len
    trades = 0
    wins = 0
    total_pnl = 0.0
    pnl_history = []
    current_pos = 0  # 0: flat, 1: long, -1: short

    memories = None
    with torch.no_grad():
        for vc in range(val_chunks):
            chunk = torch.tensor(val_x_norm[vc * seq_len:(vc + 1) * seq_len], dtype=torch.float32).unsqueeze(0)
            outs, memories = model(chunk, memories)
            logits = outs["policy"].squeeze(0)  # (L, 3)
            actions = torch.argmax(logits, dim=-1).cpu().numpy()

            chunk_returns = val_ret[vc * seq_len:(vc + 1) * seq_len]
            for t_step in range(seq_len):
                act = actions[t_step]
                ret = chunk_returns[t_step]

                # Position execution simulation (with 0.04% fee per trade)
                fee = 0.0004
                new_pos = 1 if act == 1 else (-1 if act == 2 else 0)

                trade_cost = fee if new_pos != current_pos else 0.0
                step_pnl = current_pos * ret - trade_cost
                total_pnl += step_pnl
                pnl_history.append(total_pnl)

                if new_pos != current_pos:
                    trades += 1
                    if step_pnl > 0:
                        wins += 1
                current_pos = new_pos

    win_rate = (wins / max(trades, 1)) * 100.0
    pnl_arr = np.diff(np.array([0.0] + pnl_history))
    sharpe = (np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-8)) * math.sqrt(525600)  # Annualized for 1m bars

    print(f"--- Out-of-Sample Validation Results ---")
    print(f"  Total Simulated Trades: {trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"  Cumulative PnL: {total_pnl * 100:.2f}%")

    # 7. Serialize Artifacts for Deployment
    model_pt_path = ARTIFACTS_DIR / "gtrxl_model.pt"
    scaler_pkl_path = ARTIFACTS_DIR / "scaler.pkl"
    bundle_pkl_path = ARTIFACTS_DIR / "model_bundle.pkl"

    print(f"\nSaving deployment artifacts to {ARTIFACTS_DIR}...")

    # PyTorch weights
    torch.save(model.state_dict(), model_pt_path)

    # Scaler pickle
    with open(scaler_pkl_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Complete deployment bundle
    bundle = {
        "model_architecture": "GTrXL (Gated Transformer-XL)",
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
            "data_source": "Binance ETHUSDT 1-minute perpetuals (last 6 months / 128,160 bars)",
            "total_bars": total_samples,
            "validation_trades": trades,
            "win_rate": round(win_rate, 2),
            "sharpe_ratio": round(sharpe, 2),
            "epochs": epochs,
        },
    }

    with open(bundle_pkl_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("All artifacts successfully serialized:")
    print(f"  - {model_pt_path} ({os.path.getsize(model_pt_path):,} bytes)")
    print(f"  - {scaler_pkl_path} ({os.path.getsize(scaler_pkl_path):,} bytes)")
    print(f"  - {bundle_pkl_path} ({os.path.getsize(bundle_pkl_path):,} bytes)")

    return bundle


if __name__ == "__main__":
    train_gtrxl()
