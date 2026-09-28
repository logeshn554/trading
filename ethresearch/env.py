"""Delta India ETHUSD Trading Environment with True Completed Trade Accounting.

Features:
- Gym-style Step/Reset interface
- Realistic transaction fees (0.05% taker fee, 0.02% slippage on Delta India)
- Funding rate friction
- Complete round-trip trade accounting (entry price -> exit price, net of fees)
- Accurate win-rate, profit factor, Sharpe, and drawdown metrics
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DeltaTradingEnv:
    """Trading Environment simulating Delta India ETHUSD inverse/perpetual contracts."""

    def __init__(
        self,
        candles_matrix: np.ndarray,
        features_matrix: np.ndarray,
        aux_targets: np.ndarray,
        taker_fee: float = 0.0005,      # 0.05% Delta India taker fee
        slippage: float = 0.0002,       # 0.02% market order slippage
        funding_rate: float = 0.00001,   # ~0.001% per 8h amortized per bar
        initial_capital_inr: float = 10000.0,
        contract_size_eth: float = 0.001,
        leverage: float = 10.0,
    ) -> None:
        self.candles = candles_matrix
        self.features = features_matrix
        self.aux_targets = aux_targets
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.total_cost_per_trade = taker_fee + slippage
        self.funding_rate = funding_rate
        self.initial_capital = initial_capital_inr
        self.contract_size = contract_size_eth
        self.leverage = leverage

        self.num_steps = len(self.features)
        self.current_idx = 0
        self.position = 0  # -1: SHORT, 0: FLAT, 1: LONG
        self.entry_price = 0.0
        self.entry_step = 0
        self.equity_inr = initial_capital_inr
        self.equity_curve = [initial_capital_inr]
        self.completed_trades: List[Dict[str, Any]] = []

    def reset(self, start_step: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_idx = start_step
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = 0
        self.equity_inr = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.completed_trades = []

        obs = self.features[self.current_idx].copy()
        obs[31] = 0.0  # Current position exposure
        return obs, {"step": self.current_idx, "position": 0}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single step in the environment.
        
        Args:
            action: 0: HOLD / FLAT, 1: LONG, 2: SHORT
        
        Returns:
            next_obs, reward, terminated, truncated, info
        """
        target_pos = 1 if action == 1 else (-1 if action == 2 else 0)
        curr_price = float(self.candles[self.current_idx, 4])  # close
        next_price = float(self.candles[self.current_idx + 1, 4]) if self.current_idx + 1 < len(self.candles) else curr_price
        step_ret = (next_price - curr_price) / max(curr_price, 1e-8)

        gross_step_pnl = self.position * step_ret
        cost = 0.0
        trade_closed = False
        closed_trade_info = None

        # Position transitions
        if target_pos != self.position:
            # 1. Close existing position if active
            if self.position != 0:
                exit_price = curr_price * (1.0 - self.slippage if self.position > 0 else 1.0 + self.slippage)
                raw_return = (exit_price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - exit_price) / self.entry_price
                net_return = raw_return - self.total_cost_per_trade
                trade_pnl_inr = self.equity_inr * net_return * self.leverage

                closed_trade_info = {
                    "entry_idx": self.entry_step,
                    "exit_idx": self.current_idx,
                    "bars_held": self.current_idx - self.entry_step,
                    "side": "LONG" if self.position > 0 else "SHORT",
                    "entry_price": self.entry_price,
                    "exit_price": exit_price,
                    "gross_return": raw_return,
                    "net_return": net_return,
                    "pnl_inr": trade_pnl_inr,
                    "won": net_return > 0.0,
                }
                self.completed_trades.append(closed_trade_info)
                self.equity_inr += trade_pnl_inr
                cost += self.total_cost_per_trade
                trade_closed = True

            # 2. Open new position if target != 0
            if target_pos != 0:
                self.position = target_pos
                self.entry_price = curr_price * (1.0 + self.slippage if target_pos > 0 else 1.0 - self.slippage)
                self.entry_step = self.current_idx
                cost += self.total_cost_per_trade
            else:
                self.position = 0
                self.entry_price = 0.0

        # Step net reward calculation
        # If continuing to hold position:
        funding_cost = self.funding_rate if self.position != 0 else 0.0
        step_net_ret = (self.position * step_ret) - cost - funding_cost
        
        # Reward scaled for RL agent
        reward = float(step_net_ret * 100.0)

        # Advance step
        self.current_idx += 1
        self.equity_curve.append(self.equity_inr)
        done = self.current_idx >= (self.num_steps - 1)

        # Build next observation with genuine current position exposure
        next_obs = self.features[min(self.current_idx, self.num_steps - 1)].copy()
        next_obs[31] = float(self.position)

        info = {
            "step": self.current_idx,
            "position": self.position,
            "equity_inr": self.equity_inr,
            "completed_trades_count": len(self.completed_trades),
            "last_closed_trade": closed_trade_info,
        }

        return next_obs, reward, done, False, info

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculates true completed-trade metrics."""
        trades = self.completed_trades
        n_trades = len(trades)
        if n_trades == 0:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
            }

        wins = [t for t in trades if t["won"]]
        losses = [t for t in trades if not t["won"]]
        n_wins = len(wins)
        win_rate = (n_wins / n_trades) * 100.0

        gross_gains = sum(t["pnl_inr"] for t in wins) if wins else 0.0
        gross_losses = abs(sum(t["pnl_inr"] for t in losses)) if losses else 1e-6
        profit_factor = gross_gains / gross_losses if gross_losses > 0 else 0.0

        total_return = ((self.equity_inr - self.initial_capital) / self.initial_capital) * 100.0

        # Drawdown calculation
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / np.maximum(peak, 1.0)
        max_dd = float(np.max(drawdowns)) * 100.0

        # Trade returns Sharpe
        trade_rets = [t["net_return"] for t in trades]
        std_ret = float(np.std(trade_rets)) if len(trade_rets) > 1 else 1.0
        mean_ret = float(np.mean(trade_rets)) if trade_rets else 0.0
        sharpe = (mean_ret / max(std_ret, 1e-6)) * math.sqrt(252 * 1440)  # Annualized for 1m bars

        return {
            "total_trades": n_trades,
            "win_trades": n_wins,
            "loss_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "final_equity_inr": round(self.equity_inr, 2),
        }
