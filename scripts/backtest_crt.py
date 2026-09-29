"""Reproducible CRT OHLC backtest; uses the exact production signal() function.

Input: checksum-verified Binance ETHUSDT spot 1m archives from data/ar90/.
These are a venue/instrument proxy, not Delta ETHUSD data or fill evidence.
Ambiguous same-bar stop/target touches are always resolved stop-first.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ethresearch.crt import INTERVAL, signal

DEFAULT_MANIFEST = ROOT / 'data/ar90/manifest.json'
DEFAULT_JSON = ROOT / 'artifacts/crt/backtest.json'
DEFAULT_CSV = ROOT / 'artifacts/crt/trades.csv'
FIELDS = ('entry_time_utc', 'entry_time_ist', 'session_ist', 'atr14_pct', 'volatility_regime', 'side', 'entry', 'stop', 'target', 'exit_time_utc',
          'exit_price', 'exit_reason', 'gross_pnl_usd', 'fees_usd', 'net_pnl_usd', 'initial_risk_usd', 'r_multiple')


def load_minute_archives(manifest_path: Path, start_date: str, end_date: str):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('venue') != 'binance_spot' or manifest.get('symbol') != 'ETHUSDT':
        raise ValueError('This runner accepts only the declared Binance ETHUSDT spot archive')
    rows = {}
    digests = []
    for record in manifest['records']:
        name = record['file']
        dates = name.replace('ETHUSDT-1m-', '').replace('.zip', '')
        if dates < start_date[:7] or dates > end_date:
            continue
        path = manifest_path.parent / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing immutable input: {name}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
        digests.append({'file': name, 'sha256': digest})
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m.endswith('.csv')]
            if len(members) != 1:
                raise ValueError(f'Unexpected CSV layout in {name}')
            with archive.open(members[0]) as stream:
                for line in stream:
                    parts = line.decode('utf-8').strip().split(',')
                    if len(parts) < 5:
                        raise ValueError(f'Malformed Binance candle in {name}')
                    micros = int(parts[0])
                    timestamp = micros // 1_000_000
                    day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
                    if not start_date <= day <= end_date:
                        continue
                    minute = timestamp // 60 * 60
                    bucket = minute // INTERVAL * INTERVAL
                    o, h, l, c = map(float, parts[1:5])
                    if not all(map(math.isfinite, (o, h, l, c))) or min(l, o, c) <= 0 or h < max(o, c) or l > min(o, c):
                        raise ValueError(f'Invalid OHLC row in {name}')
                    item = rows.get(bucket)
                    if item is None:
                        rows[bucket] = {'time': bucket, 'open': o, 'high': h, 'low': l,
                                        'close': c, '_last': minute, '_count': 1}
                    else:
                        if minute != item['_last'] + 60:
                            raise ValueError(f'Duplicate/out-of-order minute in {name}')
                        item['high'] = max(item['high'], h)
                        item['low'] = min(item['low'], l)
                        item['close'] = c
                        item['_last'] = minute
                        item['_count'] += 1
    bars = []
    for timestamp, row in sorted(rows.items()):
        expected = list(range(timestamp, timestamp + INTERVAL, 60))
        if row['_count'] == 15 and row['_last'] == expected[-1]:
            bars.append({k: row[k] for k in ('time', 'open', 'high', 'low', 'close')})
    if len(bars) < 200:
        raise ValueError(f'Insufficient complete 15m bars: {len(bars)}')
    # A hash over the ordered archive list and aggregated bars makes the run traceable.
    source_hash = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return bars, digests, source_hash


def wilson_interval(wins: int, count: int):
    if not count:
        return None
    z = 1.959963984540054
    p = wins / count
    denom = 1 + z*z/count
    center = (p + z*z/(2*count))/denom
    margin = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denom
    return [max(0, center-margin), min(1, center+margin)]


def stats(trades):
    net = [t['net_pnl_usd'] for t in trades]
    wins = sum(x > 1e-12 for x in net)
    equity = peak = drawdown = 0.0
    max_losses = streak = 0
    for value in net:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
        streak = streak + 1 if value < -1e-12 else 0
        max_losses = max(max_losses, streak)
    gross_profit = sum(x for x in net if x > 0)
    gross_loss = -sum(x for x in net if x < 0)
    return {'closed_trades': len(net), 'wins': wins, 'losses': sum(x < -1e-12 for x in net),
            'breakeven': sum(abs(x) <= 1e-12 for x in net),
            'win_rate': wins/len(net) if net else None,
            'wilson_95': wilson_interval(wins, len(net)),
            'net_pnl_usd_per_contract': sum(net),
            'gross_profit_usd': gross_profit, 'gross_loss_usd': gross_loss,
            'profit_factor': gross_profit/gross_loss if gross_loss else (None if not gross_profit else 'infinity'),
            'expectancy_usd_per_contract': sum(net)/len(net) if net else None,
            'mean_r': sum(t['r_multiple'] for t in trades)/len(trades) if trades else None,
            'max_drawdown_usd_per_contract': drawdown, 'max_consecutive_losses': max_losses}


def simulate(bars, *, tick, contract_value, fee_bps, spread_bps, slippage_bps,
             daily_attempts, risk_inr, fx_inr_per_usd, daily_loss_inr, daily_profit_inr,
             available_usd=None):
    trades, skipped = [], {'signal': 0, 'spread_not_observable': 0, 'slippage': 0,
                           'risk_budget': 0, 'collateral': 0, 'daily_trade_limit': 0, 'daily_stop': 0}
    i, daily_counts, daily_pnl = 1, {}, {}
    while i < len(bars)-1:
        now = bars[i]['time'] + INTERVAL + 1
        s = signal(bars[max(0, i-1):i+1], now, tick)
        if s['side'] == 'hold':
            skipped['signal'] += 1; i += 1; continue
        day = datetime.fromtimestamp(s['signal_time'], timezone.utc).date().isoformat()
        if daily_counts.get(day, 0) >= daily_attempts:
            skipped['daily_trade_limit'] += 1; i += 1; continue
        if daily_pnl.get(day, 0.0) >= daily_profit_inr or daily_pnl.get(day, 0.0) <= -daily_loss_inr:
            skipped['daily_stop'] += 1; i += 1; continue
        side_sign = 1 if s['side'] == 'buy' else -1
        entry = s['entry'] * (1 + side_sign*slippage_bps/10_000)
        if (side_sign > 0 and entry < s['stop']) or (side_sign < 0 and entry > s['stop']):
            skipped['risk_budget'] += 1; i += 1; continue
        # One contract risk in INR includes estimated round-trip fees.
        unit = contract_value
        entry_fee = entry*unit*fee_bps/10_000
        exit_adverse = slippage_bps + spread_bps/2
        modeled_stop_exit = s['stop']*(1-exit_adverse/10_000) if side_sign > 0 else s['stop']*(1+exit_adverse/10_000)
        stop_fee = modeled_stop_exit*unit*fee_bps/10_000
        modeled_risk_inr = (max(0, (entry-modeled_stop_exit)*side_sign)*unit + entry_fee + stop_fee)*fx_inr_per_usd
        if modeled_risk_inr > risk_inr:
            skipped['risk_budget'] += 1; i += 1; continue
        if available_usd is not None and entry*unit > available_usd:
            skipped['collateral'] += 1; i += 1; continue

        daily_counts[day] = daily_counts.get(day, 0) + 1
        exit_index, exit_price, reason = None, None, None
        j = i+1
        while j < len(bars):
            b = bars[j]
            stop_hit = b['low'] <= s['stop'] if side_sign > 0 else b['high'] >= s['stop']
            target_hit = b['high'] >= s['target'] if side_sign > 0 else b['low'] <= s['target']
            if stop_hit:
                reason, exit_index = 'stop', j
                raw = min(s['stop'], b['open']) if side_sign > 0 else max(s['stop'], b['open'])
                exit_price = raw*(1-exit_adverse/10_000) if side_sign > 0 else raw*(1+exit_adverse/10_000)
                break
            if target_hit:
                reason, exit_index = 'target', j
                exit_price = s['target']*(1-exit_adverse/10_000) if side_sign > 0 else s['target']*(1+exit_adverse/10_000)
                break
            j += 1
        if exit_index is None:
            exit_index = len(bars)-1; reason = 'end_of_data'
            raw = bars[exit_index]['close']
            exit_price = raw*(1-exit_adverse/10_000) if side_sign > 0 else raw*(1+exit_adverse/10_000)
        exit_fee = exit_price*unit*fee_bps/10_000
        gross = (exit_price-entry)*side_sign*unit
        fees = entry_fee+exit_fee
        net = gross-fees
        initial_risk = modeled_risk_inr/fx_inr_per_usd
        ist = datetime.fromtimestamp(s['signal_time'], timezone.utc).astimezone(ZoneInfo('Asia/Kolkata'))
        session = '00-06' if ist.hour < 6 else '06-12' if ist.hour < 12 else '12-18' if ist.hour < 18 else '18-24'
        begin = max(1, i-13)
        recent = bars[begin:i+1]
        true_ranges = [max(b['high']-b['low'], abs(b['high']-bars[k-1]['close']),
                           abs(b['low']-bars[k-1]['close'])) for k, b in enumerate(recent, start=begin)]
        atr14_pct = sum(true_ranges)/len(true_ranges)/s['sweep']['close']
        trade = {'entry_time_utc': datetime.fromtimestamp(s['signal_time'], timezone.utc).isoformat(),
                 'entry_time_ist': ist.isoformat(), 'session_ist': session, 'atr14_pct': atr14_pct,
                 'volatility_regime': None, 'side': s['side'], 'entry': entry,
                 'stop': s['stop'], 'target': s['target'],
                 'exit_time_utc': datetime.fromtimestamp(bars[exit_index]['time']+INTERVAL, timezone.utc).isoformat(),
                 'exit_price': exit_price, 'exit_reason': reason, 'gross_pnl_usd': gross,
                 'fees_usd': fees, 'net_pnl_usd': net, 'initial_risk_usd': initial_risk,
                 'r_multiple': net/initial_risk if initial_risk else 0.0,
                 'reference_time_utc': datetime.fromtimestamp(s['reference']['time'], timezone.utc).isoformat(),
                 'sweep_time_utc': datetime.fromtimestamp(s['sweep']['time'], timezone.utc).isoformat()}
        trades.append(trade)
        daily_pnl[day] = daily_pnl.get(day, 0.0) + net*fx_inr_per_usd
        i = exit_index+1  # No concurrent trades; resume after this position closes.
    return trades, skipped, daily_pnl


def run(args):
    bars, inputs, source_hash = load_minute_archives(args.manifest, args.start, args.end)
    trades, skipped, _daily = simulate(bars, tick=args.tick, contract_value=args.contract_value,
        fee_bps=args.fee_bps, spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
        daily_attempts=args.max_trades_per_day, risk_inr=args.risk_inr,
        fx_inr_per_usd=args.usd_inr, daily_loss_inr=args.daily_loss_inr,
        daily_profit_inr=args.daily_profit_inr, available_usd=args.available_usd)
    diagnostic_trades, diagnostic_skips, _ = simulate(
        bars, tick=args.tick, contract_value=args.contract_value,
        fee_bps=args.fee_bps, spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
        daily_attempts=args.max_trades_per_day, risk_inr=1e12, fx_inr_per_usd=args.usd_inr,
        daily_loss_inr=1e12, daily_profit_inr=1e12)
    cut = datetime.fromisoformat(args.test_start).replace(tzinfo=timezone.utc).timestamp()
    atrs = sorted(t['atr14_pct'] for t in diagnostic_trades)
    cuts = [atrs[min(len(atrs)-1, int((len(atrs)-1)*q))] for q in (.25,.5,.75)] if atrs else [0,0,0]
    for t in diagnostic_trades:
        t['volatility_regime'] = ('low' if t['atr14_pct'] <= cuts[0] else
                                  'medium_low' if t['atr14_pct'] <= cuts[1] else
                                  'medium_high' if t['atr14_pct'] <= cuts[2] else 'high')
    partitions = {
        'configured_risk_gates': {'statistics': stats(trades), 'skipped_setups': skipped},
        'diagnostic_one_contract_unconstrained_by_inr_risk': {
            'statistics': stats(diagnostic_trades), 'skipped_setups': diagnostic_skips,
            'note': 'Analytical only; ignores configured per-trade and daily INR stops. Not executable account performance.'},
        'all_diagnostic_one_contract': stats(diagnostic_trades),
        'before_test_start_exploratory': stats([t for t in diagnostic_trades if datetime.fromisoformat(t['entry_time_utc']).timestamp() < cut]),
        'test_period_exploratory': stats([t for t in diagnostic_trades if datetime.fromisoformat(t['entry_time_utc']).timestamp() >= cut]),
        'by_side': {side: stats([t for t in diagnostic_trades if t['side'] == side]) for side in ('buy','sell')},
        'by_ist_session': {session: stats([t for t in diagnostic_trades if t['session_ist'] == session]) for session in ('00-06','06-12','12-18','18-24')},
        'by_volatility_quartile': {regime: stats([t for t in diagnostic_trades if t['volatility_regime'] == regime]) for regime in ('low','medium_low','medium_high','high')},
    }
    signal_hash = hashlib.sha256((ROOT/'ethresearch/crt.py').read_bytes()).hexdigest()
    report = {'strategy': 'CRT_ETHUSD_15M_V1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'rules_source': 'ethresearch/crt.py::signal',
        'signal_source_sha256': signal_hash,
        'data': {'symbol': 'ETHUSDT', 'venue': 'Binance spot', 'source_type': 'proxy; not Delta ETHUSD',
                 'start': args.start, 'end': args.end, 'complete_15m_bars': len(bars),
                 'input_files': inputs, 'manifest_ordered_sha256': source_hash,
                 'final_input_bar_open_utc': datetime.fromtimestamp(bars[-1]['time'], timezone.utc).isoformat()},
        'protocol': {'test_start': args.test_start, 'partition_note': 'Exploratory chronological partition, not an untouched holdout; no parameter search was run.',
                     'tick': args.tick, 'contract_value_eth': args.contract_value, 'contracts': 1,
                     'taker_fee_bps_per_side': args.fee_bps, 'spread_bps_round_trip': args.spread_bps,
                     'slippage_bps_per_side': args.slippage_bps, 'max_trades_per_day': args.max_trades_per_day,
                     'max_entry_delay_seconds': 90, 'entry_model': 'signal close adjusted adversely by configured slippage cap',
                     'exit_model': '15m OHLC; stop wins if stop and target both touch in the same candle; gaps at stop fill at open when worse; full opposite-edge target; end of data closes open positions adversely. Entry slippage uses the live cap; exits include configured slippage plus half the round-trip spread estimate.',
                     'spread_validation': 'Historical 1m OHLC has no bid/ask. The configured spread cannot be checked against historical quotes and is only included as a modeled exit cost.',
                     'account_inr_conversion': args.usd_inr, 'risk_per_trade_inr': args.risk_inr,
                     'daily_loss_stop_inr': args.daily_loss_inr, 'daily_profit_stop_inr': args.daily_profit_inr},
        'results': partitions, 'skipped_setups': skipped,
        'capital_feasibility': {'available_usd_assumption': args.available_usd,
            'latest_close_usd': bars[-1]['close'], 'one_contract_notional_usd_approx': bars[-1]['close']*args.contract_value,
            'one_contract_notional_inr_approx': bars[-1]['close']*args.contract_value*args.usd_inr,
            'available_collateral_covers_one_contract': args.available_usd is None or args.available_usd >= bars[-1]['close']*args.contract_value*args.usd_inr/args.usd_inr,
            'note': 'Indicative notional check only; no exchange margin, liquidation, balance movement, funding or actual fills modeled.'},
        'validation_gaps': ['Binance ETHUSDT spot proxy, not Delta ETHUSD historical prices',
            'OHLC simulation does not validate IOC fills, latency, exchange bracket/OCO or partial-fill protection',
            'Fees, spread, slippage and USD/INR are user-configurable approximations',
            'No live orders were placed; results do not establish profitable expectancy or a 90% win rate.']}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows([{k:t[k] for k in FIELDS} for t in diagnostic_trades])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--start', default='2026-07-01')
    parser.add_argument('--end', default='2026-09-27')
    parser.add_argument('--test-start', default='2026-09-01')
    parser.add_argument('--tick', type=float, default=.05)
    parser.add_argument('--contract-value', type=float, default=.01)
    parser.add_argument('--fee-bps', type=float, default=15, help='per side, includes a conservative allowance')
    parser.add_argument('--spread-bps', type=float, default=10, help='round trip')
    parser.add_argument('--slippage-bps', type=float, default=10, help='per side')
    parser.add_argument('--max-trades-per-day', type=int, default=1)
    parser.add_argument('--risk-inr', type=float, default=1)
    parser.add_argument('--daily-loss-inr', type=float, default=1)
    parser.add_argument('--daily-profit-inr', type=float, default=1)
    parser.add_argument('--usd-inr', type=float, default=96.5)
    parser.add_argument('--available-usd', type=float, default=None)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--output-csv', type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    report = run(args)
    for label, item in report['results'].items():
        if 'closed_trades' not in item:
            continue
        rate = item['win_rate']
        print(f"{label}: trades={item['closed_trades']} win_rate={'N/A' if rate is None else f'{rate:.1%}'} net_usd={item['net_pnl_usd_per_contract']:.6f}")
    print(f"Report: {args.output_json}")
    print(f"Trades: {args.output_csv}")


if __name__ == '__main__':
    main()
