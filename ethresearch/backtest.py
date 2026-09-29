"""Deterministic Candle Range Theory (CRT) Backtesting Engine.

Uses the EXACT same CRT signal logic from ethresearch.crt.signal as live trading.
Supports historical Delta ETHUSD 15-minute candles, configurable risk parameters,
fee and slippage modeling, and closed-trade performance metrics.

Can be run directly via:
    python -m ethresearch.backtest --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from ethresearch.crt import INTERVAL, signal, closed_candles

IST = timezone(timedelta(hours=5, minutes=30))


def wilson_interval(wins: int, count: int) -> Optional[List[float]]:
    if not count:
        return None
    z = 1.959963984540054  # 95% confidence
    p = wins / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def calculate_trade_statistics(trades: List[Dict[str, Any]], fx_inr: float = 85.0) -> Dict[str, Any]:
    """Calculate performance metrics from executed, closed trades only."""
    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'win_rate': 0.0,
            'win_percentage': "0.00%",
            'wilson_95': None,
            'gross_profit_usd': 0.0,
            'gross_loss_usd': 0.0,
            'net_profit_usd': 0.0,
            'net_profit_inr': 0.0,
            'profit_factor': 0.0,
            'expectancy_usd': 0.0,
            'expectancy_inr': 0.0,
            'average_r': 0.0,
            'max_drawdown_usd': 0.0,
            'max_drawdown_inr': 0.0,
            'max_consecutive_losses': 0,
            'average_holding_period_minutes': 0.0,
            'long_trades': 0,
            'short_trades': 0,
        }

    net_usd = [t['net_pnl_usd'] for t in trades]
    wins = sum(1 for x in net_usd if x > 1e-8)
    losses = sum(1 for x in net_usd if x < -1e-8)
    breakeven = sum(1 for x in net_usd if abs(x) <= 1e-8)
    total = len(trades)
    win_rate = wins / total if total > 0 else 0.0

    gross_profit = sum(t['gross_pnl_usd'] for t in trades if t['gross_pnl_usd'] > 0)
    gross_loss = abs(sum(t['gross_pnl_usd'] for t in trades if t['gross_pnl_usd'] < 0))
    total_net_usd = sum(net_usd)
    total_net_inr = total_net_usd * fx_inr

    equity = 0.0
    peak = 0.0
    max_dd_usd = 0.0
    streak = 0
    max_streak = 0
    for val in net_usd:
        equity += val
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd_usd:
            max_dd_usd = dd
        if val < -1e-8:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else (float('inf') if gross_profit > 0 else 0.0)
    expectancy_usd = total_net_usd / total if total > 0 else 0.0
    expectancy_inr = expectancy_usd * fx_inr
    r_multiples = [t['r_multiple'] for t in trades if t.get('r_multiple') is not None]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

    durations = [t.get('duration_seconds', 0) / 60.0 for t in trades]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    longs = [t for t in trades if t['direction'] == 'buy']
    shorts = [t for t in trades if t['direction'] == 'sell']

    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'breakeven': breakeven,
        'win_rate': round(win_rate, 4),
        'win_percentage': f"{win_rate * 100:.2f}%",
        'wilson_95': wilson_interval(wins, total),
        'gross_profit_usd': round(gross_profit, 4),
        'gross_loss_usd': round(gross_loss, 4),
        'net_profit_usd': round(total_net_usd, 4),
        'net_profit_inr': round(total_net_inr, 2),
        'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else 'Infinity',
        'expectancy_usd': round(expectancy_usd, 4),
        'expectancy_inr': round(expectancy_inr, 2),
        'average_r': round(avg_r, 4),
        'max_drawdown_usd': round(max_dd_usd, 4),
        'max_drawdown_inr': round(max_dd_usd * fx_inr, 2),
        'max_consecutive_losses': max_streak,
        'average_holding_period_minutes': round(avg_duration, 1),
        'long_trades': len(longs),
        'short_trades': len(shorts),
    }


class CRTBacktestEngine:
    """Backtesting engine that enforces single-source-of-truth CRT signal logic."""

    def __init__(
        self,
        candles: List[Dict[str, float]],
        *,
        tick: float = 0.05,
        contract_value: float = 0.01,
        fee_bps: float = 6.0,
        slippage_bps: float = 10.0,
        spread_bps: float = 10.0,
        usd_inr: float = 85.0,
        risk_per_trade_inr: float = 100.0,
        max_contracts: int = 1,
        daily_loss_inr: float = 500.0,
        daily_profit_inr: float = 500.0,
        max_trades_per_day: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.raw_candles = candles
        self.tick = tick
        self.contract_value = contract_value
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.usd_inr = usd_inr
        self.risk_per_trade_inr = risk_per_trade_inr
        self.max_contracts = max_contracts
        self.daily_loss_inr = daily_loss_inr
        self.daily_profit_inr = daily_profit_inr
        self.max_trades_per_day = max_trades_per_day
        self.start_date = start_date
        self.end_date = end_date

    def run(self) -> Dict[str, Any]:
        """Execute backtest over historical candles using live signal logic."""
        if len(self.raw_candles) < 2:
            raise ValueError("At least 2 complete candles are required for backtesting")

        bars = sorted(self.raw_candles, key=lambda c: c['time'])

        all_setups = []
        executed_trades = []
        skipped_counts = {
            'signal_hold': 0,
            'date_filter': 0,
            'daily_trade_limit': 0,
            'daily_stop': 0,
            'risk_budget': 0,
            'reward_to_costs': 0,
            'slippage_cap': 0,
        }

        daily_counts: Dict[str, int] = {}
        daily_pnl_inr: Dict[str, float] = {}

        i = 1
        while i < len(bars):
            ref_bar = bars[i - 1]
            sweep_bar = bars[i]

            # Signal evaluation at the exact moment the sweep candle closes + 1 second
            eval_time = sweep_bar['time'] + INTERVAL + 1
            sig = signal([ref_bar, sweep_bar], eval_time, self.tick)

            if sig['side'] == 'hold':
                skipped_counts['signal_hold'] += 1
                i += 1
                continue

            setup_id = sig['id']
            sig_time = sig['signal_time']
            sig_dt_utc = datetime.fromtimestamp(sig_time, timezone.utc)
            sig_date_str = sig_dt_utc.date().isoformat()

            # Date range filtering
            if self.start_date and sig_date_str < self.start_date:
                skipped_counts['date_filter'] += 1
                i += 1
                continue
            if self.end_date and sig_date_str > self.end_date:
                skipped_counts['date_filter'] += 1
                i += 1
                continue

            buy = sig['side'] == 'buy'
            direction_str = sig['side']
            all_setups.append(setup_id)

            # Check daily limits (IST midnight boundary)
            ist_dt = sig_dt_utc.astimezone(IST)
            ist_day = ist_dt.date().isoformat()
            today_trades = daily_counts.get(ist_day, 0)
            today_pnl = daily_pnl_inr.get(ist_day, 0.0)

            if today_trades >= self.max_trades_per_day:
                skipped_counts['daily_trade_limit'] += 1
                i += 1
                continue

            if today_pnl <= -self.daily_loss_inr or today_pnl >= self.daily_profit_inr:
                skipped_counts['daily_stop'] += 1
                i += 1
                continue

            # Model slippage-adjusted entry cap (exact same formula as live engine)
            slippage_factor = 1 + (1 if buy else -1) * self.slippage_bps / 10000.0
            raw_entry = sig['entry'] * slippage_factor
            rounding = ROUND_FLOOR if buy else ROUND_CEILING
            cap = float((Decimal(str(raw_entry)) / Decimal(str(self.tick))).to_integral_value(rounding=rounding) * Decimal(str(self.tick)))

            # Stop / target bounds validation
            if buy and not (sig['stop'] < cap < sig['target']):
                skipped_counts['slippage_cap'] += 1
                i += 1
                continue
            if not buy and not (sig['target'] < cap < sig['stop']):
                skipped_counts['slippage_cap'] += 1
                i += 1
                continue

            unit = self.contract_value * self.usd_inr
            costs_per_contract = (cap + sig['stop']) * self.fee_bps / 10000.0 * unit
            risk_per_contract = abs(cap - sig['stop']) * unit + costs_per_contract
            reward_per_contract = abs(sig['target'] - cap) * unit - (cap + sig['target']) * self.fee_bps / 10000.0 * unit

            if reward_per_contract <= 0:
                skipped_counts['reward_to_costs'] += 1
                i += 1
                continue

            remaining_daily_budget = self.daily_loss_inr + today_pnl
            budget = min(self.risk_per_trade_inr, remaining_daily_budget)
            if risk_per_contract <= 0:
                i += 1
                continue

            size = min(self.max_contracts, int(budget / risk_per_contract))
            if size < 1:
                skipped_counts['risk_budget'] += 1
                i += 1
                continue

            # Forward simulation: check next bars until stop or target is hit
            entry_price = cap
            exit_price = None
            exit_reason = None
            exit_bar_idx = None

            adverse_exit_factor = (self.slippage_bps + self.spread_bps / 2.0) / 10000.0

            j = i + 1
            while j < len(bars):
                b = bars[j]
                stop_hit = b['low'] <= sig['stop'] if buy else b['high'] >= sig['stop']
                target_hit = b['high'] >= sig['target'] if buy else b['low'] <= sig['target']

                # Pessimistic conservative rule: if both stop and target touched on same bar, stop first
                if stop_hit and target_hit:
                    exit_reason = 'stop'
                    raw_exit = min(sig['stop'], b['open']) if buy else max(sig['stop'], b['open'])
                    exit_price = raw_exit * (1 - adverse_exit_factor) if buy else raw_exit * (1 + adverse_exit_factor)
                    exit_bar_idx = j
                    break
                elif stop_hit:
                    exit_reason = 'stop'
                    raw_exit = min(sig['stop'], b['open']) if buy else max(sig['stop'], b['open'])
                    exit_price = raw_exit * (1 - adverse_exit_factor) if buy else raw_exit * (1 + adverse_exit_factor)
                    exit_bar_idx = j
                    break
                elif target_hit:
                    exit_reason = 'target'
                    raw_exit = sig['target']
                    exit_price = raw_exit * (1 - adverse_exit_factor) if buy else raw_exit * (1 + adverse_exit_factor)
                    exit_bar_idx = j
                    break

                j += 1

            if exit_bar_idx is None:
                # End of dataset
                exit_bar_idx = len(bars) - 1
                exit_reason = 'end_of_data'
                raw_exit = bars[exit_bar_idx]['close']
                exit_price = raw_exit * (1 - adverse_exit_factor) if buy else raw_exit * (1 + adverse_exit_factor)

            # Trade outcome calculations
            exit_time = bars[exit_bar_idx]['time'] + INTERVAL
            duration_sec = exit_time - sig_time
            side_mult = 1 if buy else -1

            gross_pnl_usd = (exit_price - entry_price) * side_mult * self.contract_value * size
            entry_fee_usd = entry_price * self.contract_value * size * self.fee_bps / 10000.0
            exit_fee_usd = exit_price * self.contract_value * size * self.fee_bps / 10000.0
            total_fees_usd = entry_fee_usd + exit_fee_usd
            slippage_usd = abs(entry_price - sig['entry']) * self.contract_value * size + abs(exit_price - (sig['target'] if exit_reason == 'target' else sig['stop'])) * self.contract_value * size
            net_pnl_usd = gross_pnl_usd - total_fees_usd
            net_pnl_inr = net_pnl_usd * self.usd_inr

            initial_risk_usd = risk_per_contract * size / self.usd_inr
            r_multiple = net_pnl_usd / initial_risk_usd if initial_risk_usd > 1e-8 else 0.0

            # Session classification
            ist_hour = ist_dt.hour
            session_ist = '00-06' if ist_hour < 6 else '06-12' if ist_hour < 12 else '12-18' if ist_hour < 18 else '18-24'
            month_str = ist_dt.strftime('%Y-%m')
            day_of_week = ist_dt.strftime('%A')

            trade_record = {
                'setup_id': setup_id,
                'signal_time_utc': sig_dt_utc.isoformat(),
                'signal_time_ist': ist_dt.isoformat(),
                'month': month_str,
                'day_of_week': day_of_week,
                'session_ist': session_ist,
                'direction': direction_str,
                'reference_high': sig['reference']['high'],
                'reference_low': sig['reference']['low'],
                'sweep_high': sig['sweep']['high'],
                'sweep_low': sig['sweep']['low'],
                'signal_close': sig['sweep']['close'],
                'intended_entry': sig['entry'],
                'entry_price': round(entry_price, 4),
                'stop_price': round(sig['stop'], 4),
                'target_price': round(sig['target'], 4),
                'contracts': size,
                'exit_time_utc': datetime.fromtimestamp(exit_time, timezone.utc).isoformat(),
                'exit_price': round(exit_price, 4),
                'exit_reason': exit_reason,
                'duration_seconds': duration_sec,
                'duration_bars': exit_bar_idx - i,
                'gross_pnl_usd': round(gross_pnl_usd, 4),
                'fees_usd': round(total_fees_usd, 4),
                'slippage_usd': round(slippage_usd, 4),
                'funding_usd': 0.0,
                'net_pnl_usd': round(net_pnl_usd, 4),
                'net_pnl_inr': round(net_pnl_inr, 2),
                'initial_risk_usd': round(initial_risk_usd, 4),
                'r_multiple': round(r_multiple, 4),
            }

            executed_trades.append(trade_record)
            daily_counts[ist_day] = today_trades + 1
            daily_pnl_inr[ist_day] = today_pnl + net_pnl_inr

            # No concurrent positions; fast forward to the exit bar
            i = exit_bar_idx + 1

        overall_stats = calculate_trade_statistics(executed_trades, self.usd_inr)

        # Partitions
        long_trades = [t for t in executed_trades if t['direction'] == 'buy']
        short_trades = [t for t in executed_trades if t['direction'] == 'sell']

        # Monthly breakdown
        months = sorted({t['month'] for t in executed_trades})
        by_month = {m: calculate_trade_statistics([t for t in executed_trades if t['month'] == m], self.usd_inr) for m in months}

        # Day of week breakdown
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day = {d: calculate_trade_statistics([t for t in executed_trades if t['day_of_week'] == d], self.usd_inr) for d in days}

        # Session breakdown
        sessions = ['00-06', '06-12', '12-18', '18-24']
        by_session = {s: calculate_trade_statistics([t for t in executed_trades if t['session_ist'] == s], self.usd_inr) for s in sessions}

        return {
            'strategy': 'CRT_ETHUSD_15M',
            'summary': {
                'total_setups': len(all_setups),
                'executed_trades': len(executed_trades),
                'skipped_setups': skipped_counts,
                **overall_stats,
            },
            'long_statistics': calculate_trade_statistics(long_trades, self.usd_inr),
            'short_statistics': calculate_trade_statistics(short_trades, self.usd_inr),
            'by_month': by_month,
            'by_day_of_week': by_day,
            'by_session_ist': by_session,
            'trades': executed_trades,
            'parameters': {
                'tick': self.tick,
                'contract_value': self.contract_value,
                'fee_bps': self.fee_bps,
                'slippage_bps': self.slippage_bps,
                'spread_bps': self.spread_bps,
                'usd_inr': self.usd_inr,
                'risk_per_trade_inr': self.risk_per_trade_inr,
                'max_contracts': self.max_contracts,
                'daily_loss_inr': self.daily_loss_inr,
                'daily_profit_inr': self.daily_profit_inr,
                'max_trades_per_day': self.max_trades_per_day,
                'start_date': self.start_date,
                'end_date': self.end_date,
            }
        }

    def run_sensitivity_analysis(self) -> Dict[str, Any]:
        """Perform fee and slippage sensitivity analysis."""
        fee_grid = [2.0, 6.0, 10.0, 15.0]
        slippage_grid = [0.0, 5.0, 10.0, 20.0]
        results = {}

        original_fee = self.fee_bps
        original_slippage = self.slippage_bps

        try:
            for fee in fee_grid:
                for slip in slippage_grid:
                    key = f"fee_{fee}bps_slip_{slip}bps"
                    self.fee_bps = fee
                    self.slippage_bps = slip
                    res = self.run()
                    s = res['summary']
                    results[key] = {
                        'fee_bps': fee,
                        'slippage_bps': slip,
                        'trades': s['total_trades'],
                        'win_rate': s['win_rate'],
                        'net_profit_usd': s['net_profit_usd'],
                        'net_profit_inr': s['net_profit_inr'],
                        'profit_factor': s['profit_factor'],
                        'average_r': s['average_r'],
                        'max_drawdown_usd': s['max_drawdown_usd'],
                    }
        finally:
            self.fee_bps = original_fee
            self.slippage_bps = original_slippage

        return results


def load_candles_from_json(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'result' in data:
        data = data['result']
    if not isinstance(data, list):
        raise ValueError(f"Expected list of candles in {path}")
    candles = []
    for item in data:
        candles.append({
            'time': float(item['time']),
            'open': float(item['open']),
            'high': float(item['high']),
            'low': float(item['low']),
            'close': float(item['close']),
        })
    return candles


def load_candles_from_csv(path: Path) -> List[Dict[str, float]]:
    import csv
    candles = []
    with path.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get('time') or row.get('timestamp') or row.get('open_time') or 0)
            candles.append({
                'time': t,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            })
    return candles


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic CRT Backtesting Engine for Delta ETHUSD 15m")
    parser.add_argument('--candles-file', type=Path, help="Path to JSON or CSV file containing 15m candles")
    parser.add_argument('--start', type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument('--tick', type=float, default=0.05, help="Delta exchange tick size (default: 0.05)")
    parser.add_argument('--contract-value', type=float, default=0.01, help="ETH contract value in ETH (default: 0.01)")
    parser.add_argument('--fee-bps', type=float, default=6.0, help="Fee per side in bps (default: 6.0)")
    parser.add_argument('--slippage-bps', type=float, default=10.0, help="Max slippage in bps (default: 10.0)")
    parser.add_argument('--spread-bps', type=float, default=10.0, help="Max spread in bps (default: 10.0)")
    parser.add_argument('--usd-inr', type=float, default=85.0, help="USD to INR rate (default: 85.0)")
    parser.add_argument('--risk-inr', type=float, default=100.0, help="Risk per trade in INR (default: 100.0)")
    parser.add_argument('--max-contracts', type=int, default=1, help="Max contracts per trade (default: 1)")
    parser.add_argument('--daily-loss-inr', type=float, default=500.0, help="Daily loss stop in INR (default: 500.0)")
    parser.add_argument('--daily-profit-inr', type=float, default=500.0, help="Daily profit stop in INR (default: 500.0)")
    parser.add_argument('--max-trades-per-day', type=int, default=5, help="Max entries per day (default: 5)")
    parser.add_argument('--output-json', type=Path, default=None, help="Save backtest results to JSON file")
    parser.add_argument('--sensitivity', action='store_true', help="Run fee/slippage sensitivity matrix")

    args = parser.parse_args(argv)

    candles = []
    if args.candles_file:
        if str(args.candles_file).endswith('.csv'):
            candles = load_candles_from_csv(args.candles_file)
        else:
            candles = load_candles_from_json(args.candles_file)
    else:
        # Default sample or check artifacts
        sample_path = Path(__file__).resolve().parents[1] / 'artifacts' / 'crt' / 'sample_candles.json'
        if sample_path.is_file():
            candles = load_candles_from_json(sample_path)
        else:
            print("No candle file provided. Generating diagnostic run info...", file=sys.stderr)
            # Create synthetic test bars for smoke test
            base_time = 1704067200  # 2024-01-01 00:00:00 UTC
            candles = [
                {'time': base_time, 'open': 2000.0, 'high': 2020.0, 'low': 1990.0, 'close': 2010.0},
                {'time': base_time + 900, 'open': 2010.0, 'high': 2015.0, 'low': 1980.0, 'close': 2005.0},  # Low sweep -> Buy
                {'time': base_time + 1800, 'open': 2005.0, 'high': 2025.0, 'low': 2000.0, 'close': 2020.0}, # Target hit
            ]

    engine = CRTBacktestEngine(
        candles=candles,
        tick=args.tick,
        contract_value=args.contract_value,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        usd_inr=args.usd_inr,
        risk_per_trade_inr=args.risk_inr,
        max_contracts=args.max_contracts,
        daily_loss_inr=args.daily_loss_inr,
        daily_profit_inr=args.daily_profit_inr,
        max_trades_per_day=args.max_trades_per_day,
        start_date=args.start,
        end_date=args.end,
    )

    results = engine.run()
    if args.sensitivity:
        results['sensitivity_analysis'] = engine.run_sensitivity_analysis()

    summary = results['summary']
    print("=" * 60)
    print("CRT 15M BACKTEST SUMMARY (Deterministic Single-Source-of-Truth)")
    print("=" * 60)
    print(f"Total Setups Detected:  {summary['total_setups']}")
    print(f"Executed Trades:        {summary['executed_trades']}")
    print(f"Wins / Losses / BE:     {summary['wins']} / {summary['losses']} / {summary['breakeven']}")
    print(f"Win Rate:               {summary['win_percentage']}")
    print(f"Gross Profit:           ${summary['gross_profit_usd']:.2f}")
    print(f"Gross Loss:             ${summary['gross_loss_usd']:.2f}")
    print(f"Net Profit (USD):       ${summary['net_profit_usd']:.2f}")
    print(f"Net Profit (INR):       INR {summary['net_profit_inr']:.2f}")
    print(f"Profit Factor:          {summary['profit_factor']}")
    print(f"Expectancy:             ${summary['expectancy_usd']:.2f} / trade")
    print(f"Average R:              {summary['average_r']:.2f}R")
    print(f"Max Drawdown:           ${summary['max_drawdown_usd']:.2f} (INR {summary['max_drawdown_inr']:.2f})")
    print(f"Max Consecutive Losses: {summary['max_consecutive_losses']}")
    print(f"Avg Holding Period:     {summary['average_holding_period_minutes']} mins")
    print("=" * 60)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f"Full backtest results saved to {args.output_json}")

    return results


if __name__ == '__main__':
    main()
