"""Deterministic 15-minute Candle Range Theory. No model or adaptive parameters."""
from __future__ import annotations

import math

INTERVAL = 900


def closed_candles(rows, now):
    result = []
    seen = set()
    for row in rows:
        candle = {key: float(row[key]) for key in ('time', 'open', 'high', 'low', 'close')}
        if not all(math.isfinite(v) for v in candle.values()):
            raise ValueError('Non-finite candle')
        t = candle['time']
        if t % INTERVAL or t in seen:
            raise ValueError('Duplicate or misaligned candle')
        seen.add(t)
        if not 0 < candle['low'] <= min(candle['open'], candle['close']) <= max(candle['open'], candle['close']) <= candle['high']:
            raise ValueError('Invalid OHLC candle')
        if t + INTERVAL <= now:
            result.append(candle)
    return sorted(result, key=lambda x: x['time'])


def signal(rows, now, tick):
    bars = closed_candles(rows, now)
    if len(bars) < 2:
        return {'side': 'hold', 'reason': 'Waiting for two closed candles'}
    reference, sweep = bars[-2:]
    base = {'reference': reference, 'sweep': sweep, 'midpoint': (reference['high'] + reference['low']) / 2,
            'signal_time': int(sweep['time'] + INTERVAL)}
    if sweep['time'] - reference['time'] != INTERVAL or now - base['signal_time'] > 90:
        return dict(base, side='hold', reason='Missing candle or entry window expired (90 seconds)')
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError('Invalid exchange tick size')
    low = sweep['low'] < reference['low']
    high = sweep['high'] > reference['high']
    inside = reference['low'] < sweep['close'] < reference['high']
    if low == high or not inside:
        return dict(base, side='hold', reason='Requires exactly one sweep and a close strictly inside the range')
    side = 'buy' if low else 'sell'
    stop = (math.floor(sweep['low'] / tick) - 1) * tick if low else (math.ceil(sweep['high'] / tick) + 1) * tick
    stop = round(stop, 8)
    target = reference['high'] if low else reference['low']
    return dict(base, side=side, reason='Confirmed low sweep' if low else 'Confirmed high sweep',
                entry=sweep['close'], stop=stop, target=target,
                id=f'crt15-{base["signal_time"]}-{side}')
