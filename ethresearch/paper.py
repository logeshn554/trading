"""Continuous Automated Paper Trading Engine with $10 Starting Capital & Delta India Fees.

Enforces:
- Starting capital: exactly $10.00 USD
- Full lot sizing: commits the entire available equity/margin into each trade
- Exact Delta Exchange India perpetual fee model:
  - 0.05% taker fee
  - 0.02% execution slippage friction
- Continuous algorithmic execution driven by GTrXL model signals
- Real-time mark-to-market and liquidation monitoring until balance reaches $0.00
- Persistent state management in config/paper_trading_state.json
- One-click reset back to $10.00
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ethresearch.paper")


class PaperTradingAccount:
    """Simulated Delta India ETHUSD perpetual trading account."""

    def __init__(
        self,
        initial_capital: float = 10.0,
        contract_unit_eth: float = 0.001,
        taker_fee_rate: float = 0.0005,      # 0.05% Delta Exchange taker fee
        slippage_rate: float = 0.0002,       # 0.02% execution slippage
        leverage: float = 5.0,               # 5x leverage for $10 full lot margin
        state_file: Optional[Path] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.contract_unit_eth = float(contract_unit_eth)
        self.taker_fee_rate = float(taker_fee_rate)
        self.slippage_rate = float(slippage_rate)
        self.leverage = float(leverage)
        self.state_file = state_file or (Path(__file__).resolve().parents[1] / "config/paper_trading_state.json")

        # Account Balances & Metrics
        self.cash_balance: float = self.initial_capital
        self.equity: float = self.initial_capital
        self.current_position: int = 0       # Positive = Long, Negative = Short, 0 = Flat
        self.entry_price: float = 0.0
        self.entry_time: Optional[str] = None
        self.current_price: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

        # Lifespan & Exhaustion State
        self.is_exhausted: bool = False
        self.exhaustion_reason: Optional[str] = None
        self.exhausted_at: Optional[str] = None
        self.enabled: bool = True
        self.trades_history: List[Dict[str, Any]] = []

        # Load saved state if present
        self.load_state()

    def reset(self, initial_capital: float = 10.0) -> None:
        """Resets paper account balance back to initial capital."""
        self.initial_capital = float(initial_capital)
        self.cash_balance = self.initial_capital
        self.equity = self.initial_capital
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.is_exhausted = False
        self.exhaustion_reason = None
        self.exhausted_at = None
        self.enabled = True
        self.trades_history = []
        self.save_state()
        logger.info(f"[PAPER TRADING] Account reset back to ${self.initial_capital:.2f} USD.")

    def calculate_full_lot_size(self, price: float) -> int:
        """Calculates contracts to fully deploy all available equity into the trade.

        1 contract of ETHUSD = 0.001 ETH.
        At price P, 1 contract notional = P * 0.001.
        Required margin per contract = (P * 0.001) / leverage.
        Target contracts = floor(available_equity / margin_per_contract).
        """
        if price <= 0.0 or self.cash_balance <= 0.05:
            return 0
        contract_notional = price * self.contract_unit_eth
        margin_per_contract = contract_notional / max(self.leverage, 1.0)
        
        # Reserve for estimated taker fee (entry + exit)
        available_margin = max(0.0, self.cash_balance * 0.98)
        contracts = int(available_margin // margin_per_contract) if margin_per_contract > 0 else 0
        return max(1, contracts) if available_margin >= margin_per_contract else 0

    def update_mark_price(self, price: float, high: Optional[float] = None, low: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Updates mark price and evaluates live mark-to-market & liquidation."""
        if price <= 0.0:
            return False, None
        self.current_price = price

        if self.current_position == 0:
            self.unrealized_pnl = 0.0
            self.equity = self.cash_balance
            return False, None

        # Position is open: compute unrealized P&L
        pos_size = abs(self.current_position)
        contract_val = self.contract_unit_eth * pos_size

        if self.current_position > 0:  # LONG
            self.unrealized_pnl = (price - self.entry_price) * contract_val
            # Check extreme low if provided
            worst_pnl = (low - self.entry_price) * contract_val if low else self.unrealized_pnl
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * contract_val
            # Check extreme high if provided
            worst_pnl = (self.entry_price - high) * contract_val if high else self.unrealized_pnl

        self.equity = self.cash_balance + self.unrealized_pnl

        # Liquidation Check: equity drops to 0 or below
        if self.equity <= 0.0 or (self.cash_balance + worst_pnl) <= 0.0:
            liquidation_price = price
            logger.warning(f"[PAPER TRADING] LIQUIDATION: Equity dropped to ${self.equity:.4f} <= $0.00 at price ${liquidation_price:.2f}!")
            self._record_liquidation(liquidation_price)
            return True, "LIQUIDATION"

        return False, None

    def _record_liquidation(self, price: float) -> None:
        """Closes position on liquidation and exhausts paper account."""
        exit_fee = abs(self.current_position) * self.contract_unit_eth * price * self.taker_fee_rate
        self.total_fees_paid += exit_fee
        gross_loss = self.unrealized_pnl
        net_loss = gross_loss - exit_fee
        self.realized_pnl += net_loss
        self.cash_balance = 0.0
        self.equity = 0.0
        self.is_exhausted = True
        self.exhausted_at = datetime.now(timezone.utc).isoformat()
        self.exhaustion_reason = f"Account liquidated: Capital fully exhausted to $0.00 from position loss & trading fees."

        trade_record = {
            "id": len(self.trades_history) + 1,
            "timestamp": self.exhausted_at,
            "type": "LIQUIDATION_CLOSE",
            "side": "SELL" if self.current_position > 0 else "BUY",
            "contracts": abs(self.current_position),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(price, 2),
            "fee_paid": round(exit_fee, 4),
            "net_pnl": round(net_loss, 4),
            "balance_after": 0.0,
            "reason": "LIQUIDATION_AT_ZERO",
        }
        self.trades_history.insert(0, trade_record)
        if len(self.trades_history) > 100:
            self.trades_history.pop()

        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.total_trades += 1
        self.losing_trades += 1
        self.save_state()

    def close_position(self, price: float, reason: str = "SIGNAL_EXIT") -> Optional[Dict[str, Any]]:
        """Closes open paper position at current price with slippage and taker fee."""
        if self.current_position == 0:
            return None

        # Apply slippage on exit
        exit_price = price * (1.0 - self.slippage_rate if self.current_position > 0 else 1.0 + self.slippage_rate)
        pos_size = abs(self.current_position)
        contract_val = self.contract_unit_eth * pos_size
        exit_notional = contract_val * exit_price
        exit_fee = exit_notional * self.taker_fee_rate

        if self.current_position > 0:
            gross_pnl = (exit_price - self.entry_price) * contract_val
        else:
            gross_pnl = (self.entry_price - exit_price) * contract_val

        net_pnl = gross_pnl - exit_fee
        self.cash_balance += net_pnl
        self.total_fees_paid += exit_fee
        self.realized_pnl += net_pnl
        self.unrealized_pnl = 0.0
        self.equity = max(0.0, self.cash_balance)

        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Check if balance exhausted after close
        if self.cash_balance <= 0.05:
            self.cash_balance = 0.0
            self.equity = 0.0
            self.is_exhausted = True
            self.exhausted_at = datetime.now(timezone.utc).isoformat()
            self.exhaustion_reason = f"Balance reached $0.00 after trade fees and P&L."

        trade_record = {
            "id": len(self.trades_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "POSITION_CLOSE",
            "side": "SELL" if self.current_position > 0 else "BUY",
            "contracts": pos_size,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "fee_paid": round(exit_fee, 4),
            "net_pnl": round(net_pnl, 4),
            "balance_after": round(max(0.0, self.cash_balance), 4),
            "reason": reason,
        }
        self.trades_history.insert(0, trade_record)
        if len(self.trades_history) > 100:
            self.trades_history.pop()

        logger.info(
            f"[PAPER TRADING] Closed {'LONG' if self.current_position > 0 else 'SHORT'} {pos_size} contracts @ ${exit_price:.2f} | "
            f"Net PnL: ${net_pnl:+.4f} (Fee: ${exit_fee:.4f}) | Balance: ${self.cash_balance:.4f}"
        )

        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.save_state()
        return trade_record

    def execute_signal(
        self,
        action: int,
        price: float,
        confidence: float = 0.0,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
    ) -> Optional[Dict[str, Any]]:
        """Executes GTrXL model signal (1=BUY, 2=SELL) allocating the full $10 lot."""
        if not self.enabled:
            return None
        if self.is_exhausted:
            logger.info("[PAPER TRADING] Algorithmic execution skipped: Paper balance exhausted ($0.00).")
            return None
        if price <= 0.0:
            return None

        # 1. Action = BUY (Long)
        if action == 1:
            if self.current_position > 0:
                return None  # Already in Long position
            if self.current_position < 0:
                # Close existing Short first
                self.close_position(price, reason="SIGNAL_FLIP_TO_BUY")
                if self.is_exhausted:
                    return None

            # Open new LONG with full available capital
            contracts = self.calculate_full_lot_size(price)
            if contracts <= 0:
                self.is_exhausted = True
                self.exhausted_at = datetime.now(timezone.utc).isoformat()
                self.exhaustion_reason = "Insufficient capital to open 1 contract with fees."
                self.save_state()
                return None

            fill_price = price * (1.0 + self.slippage_rate)
            notional = contracts * self.contract_unit_eth * fill_price
            entry_fee = notional * self.taker_fee_rate
            self.cash_balance -= entry_fee
            self.total_fees_paid += entry_fee
            self.current_position = contracts
            self.entry_price = fill_price
            self.entry_time = datetime.now(timezone.utc).isoformat()
            self.equity = self.cash_balance

            fill_record = {
                "id": len(self.trades_history) + 1,
                "timestamp": self.entry_time,
                "type": "POSITION_OPEN",
                "side": "BUY",
                "contracts": contracts,
                "entry_price": round(fill_price, 2),
                "exit_price": None,
                "fee_paid": round(entry_fee, 4),
                "net_pnl": 0.0,
                "balance_after": round(self.cash_balance, 4),
                "reason": f"GTrXL_BUY_SIGNAL (Conf: {confidence*100:.1f}%)",
            }
            self.trades_history.insert(0, fill_record)
            logger.info(
                f"[PAPER TRADING] FULL LOT BUY {contracts} contracts @ ${fill_price:.2f} "
                f"(Notional: ${notional:.2f}, Fee: ${entry_fee:.4f}) | Balance: ${self.cash_balance:.4f}"
            )
            self.save_state()
            return fill_record

        # 2. Action = SELL (Short)
        elif action == 2:
            if self.current_position < 0:
                return None  # Already in Short position
            if self.current_position > 0:
                # Close existing Long first
                self.close_position(price, reason="SIGNAL_FLIP_TO_SELL")
                if self.is_exhausted:
                    return None

            # Open new SHORT with full available capital
            contracts = self.calculate_full_lot_size(price)
            if contracts <= 0:
                self.is_exhausted = True
                self.exhausted_at = datetime.now(timezone.utc).isoformat()
                self.exhaustion_reason = "Insufficient capital to open 1 contract with fees."
                self.save_state()
                return None

            fill_price = price * (1.0 - self.slippage_rate)
            notional = contracts * self.contract_unit_eth * fill_price
            entry_fee = notional * self.taker_fee_rate
            self.cash_balance -= entry_fee
            self.total_fees_paid += entry_fee
            self.current_position = -contracts
            self.entry_price = fill_price
            self.entry_time = datetime.now(timezone.utc).isoformat()
            self.equity = self.cash_balance

            fill_record = {
                "id": len(self.trades_history) + 1,
                "timestamp": self.entry_time,
                "type": "POSITION_OPEN",
                "side": "SELL",
                "contracts": contracts,
                "entry_price": round(fill_price, 2),
                "exit_price": None,
                "fee_paid": round(entry_fee, 4),
                "net_pnl": 0.0,
                "balance_after": round(self.cash_balance, 4),
                "reason": f"GTrXL_SELL_SIGNAL (Conf: {confidence*100:.1f}%)",
            }
            self.trades_history.insert(0, fill_record)
            logger.info(
                f"[PAPER TRADING] FULL LOT SELL {contracts} contracts @ ${fill_price:.2f} "
                f"(Notional: ${notional:.2f}, Fee: ${entry_fee:.4f}) | Balance: ${self.cash_balance:.4f}"
            )
            self.save_state()
            return fill_record

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured metrics for the web dashboard."""
        win_rate = (self.winning_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0
        pos_desc = "FLAT"
        if self.current_position > 0:
            pos_desc = f"LONG {self.current_position} lot(s)"
        elif self.current_position < 0:
            pos_desc = f"SHORT {abs(self.current_position)} lot(s)"

        return {
            "initial_capital": round(self.initial_capital, 2),
            "cash_balance": round(max(0.0, self.cash_balance), 4),
            "equity": round(max(0.0, self.equity), 4),
            "current_position": self.current_position,
            "position_desc": pos_desc,
            "entry_price": round(self.entry_price, 2) if self.entry_price > 0 else None,
            "current_price": round(self.current_price, 2) if self.current_price > 0 else None,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "total_fees_paid": round(self.total_fees_paid, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(win_rate, 1),
            "is_exhausted": self.is_exhausted,
            "exhaustion_reason": self.exhaustion_reason,
            "exhausted_at": self.exhausted_at,
            "enabled": self.enabled,
            "leverage": self.leverage,
            "contract_unit_eth": self.contract_unit_eth,
            "taker_fee_rate": self.taker_fee_rate,
            "recent_trades": self.trades_history[:25],
        }

    def save_state(self) -> None:
        """Persists paper account state to JSON file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "initial_capital": self.initial_capital,
                "cash_balance": self.cash_balance,
                "equity": self.equity,
                "current_position": self.current_position,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl": self.realized_pnl,
                "total_fees_paid": self.total_fees_paid,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "is_exhausted": self.is_exhausted,
                "exhaustion_reason": self.exhaustion_reason,
                "exhausted_at": self.exhausted_at,
                "enabled": self.enabled,
                "trades_history": self.trades_history[:100],
            }
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not persist paper trading state: {e}")

    def load_state(self) -> None:
        """Loads persisted state from JSON file if available."""
        if not self.state_file.exists():
            return
        try:
            content = self.state_file.read_text(encoding="utf-8")
            data = json.loads(content)
            self.initial_capital = float(data.get("initial_capital", 10.0))
            self.cash_balance = float(data.get("cash_balance", 10.0))
            self.equity = float(data.get("equity", 10.0))
            self.current_position = int(data.get("current_position", 0))
            self.entry_price = float(data.get("entry_price", 0.0))
            self.entry_time = data.get("entry_time")
            self.unrealized_pnl = float(data.get("unrealized_pnl", 0.0))
            self.realized_pnl = float(data.get("realized_pnl", 0.0))
            self.total_fees_paid = float(data.get("total_fees_paid", 0.0))
            self.total_trades = int(data.get("total_trades", 0))
            self.winning_trades = int(data.get("winning_trades", 0))
            self.losing_trades = int(data.get("losing_trades", 0))
            self.is_exhausted = bool(data.get("is_exhausted", False))
            self.exhaustion_reason = data.get("exhaustion_reason")
            self.exhausted_at = data.get("exhausted_at")
            self.enabled = bool(data.get("enabled", True))
            self.trades_history = data.get("trades_history", [])
        except Exception as e:
            logger.warning(f"Could not load paper trading state: {e}")
