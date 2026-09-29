"""Property and fuzz tests for CRT signal and risk engine.

Ensures that no malformed OHLC, extreme prices, NaN/infinite values, or
corrupted transaction streams can cause crashes or unintended order placement.
"""
from __future__ import annotations

import math
import random
import unittest

from ethresearch.crt import signal, closed_candles, INTERVAL
from ethresearch.crt_live import number


class PropertyFuzzTests(unittest.TestCase):
    def test_fuzz_malformed_ohlc(self):
        """Fuzz OHLC candle validation with randomized distorted data."""
        random.seed(42)
        base_time = 1800000

        for _ in range(500):
            # Generate random variations
            o = random.uniform(-1000, 10000)
            h = random.uniform(-1000, 10000)
            l = random.uniform(-1000, 10000)
            c = random.uniform(-1000, 10000)
            t = base_time + random.choice([0, 900, 1800, 123, -900])

            candle = {'time': t, 'open': o, 'high': h, 'low': l, 'close': c}

            # If candle violates physical OHLC constraints, closed_candles must reject it
            is_valid_time = (t % INTERVAL == 0)
            is_valid_ohlc = (0 < l <= min(o, c) <= max(o, c) <= h)
            is_finite = all(math.isfinite(x) for x in (t, o, h, l, c))

            if not (is_valid_time and is_valid_ohlc and is_finite):
                with self.assertRaises(ValueError):
                    closed_candles([candle], t + INTERVAL + 1)
            else:
                bars = closed_candles([candle], t + INTERVAL + 1)
                self.assertEqual(len(bars), 1)

    def test_fuzz_extreme_prices_and_ticks(self):
        """Fuzz signal generation across extreme tick and price regimes."""
        random.seed(1337)
        for _ in range(200):
            ref_low = random.uniform(10, 50000)
            ref_high = ref_low + random.uniform(1, 1000)
            ref_open = random.uniform(ref_low, ref_high)
            ref_close = random.uniform(ref_low, ref_high)

            tick = random.choice([0.0001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])

            ref_candle = {'time': 0, 'open': ref_open, 'high': ref_high, 'low': ref_low, 'close': ref_close}

            # Generate sweep candle
            sweep_type = random.choice(['low_sweep', 'high_sweep', 'double_sweep', 'none'])
            if sweep_type == 'low_sweep':
                sw_low = ref_low - random.uniform(1, 50)
                sw_high = ref_high - random.uniform(1, 10)
                sw_close = random.uniform(ref_low + 0.01, sw_high)
            elif sweep_type == 'high_sweep':
                sw_high = ref_high + random.uniform(1, 50)
                sw_low = ref_low + random.uniform(1, 10)
                sw_close = random.uniform(sw_low, ref_high - 0.01)
            elif sweep_type == 'double_sweep':
                sw_low = ref_low - random.uniform(1, 50)
                sw_high = ref_high + random.uniform(1, 50)
                sw_close = random.uniform(ref_low + 0.01, ref_high - 0.01)
            else:
                sw_low = ref_low + 0.1
                sw_high = ref_high - 0.1
                sw_close = (sw_low + sw_high) / 2.0

            sw_open = (sw_low + sw_high) / 2.0
            sweep_candle = {'time': 900, 'open': sw_open, 'high': sw_high, 'low': sw_low, 'close': sw_close}

            res = signal([ref_candle, sweep_candle], 1801, tick)

            if sweep_type == 'low_sweep' and ref_low < sw_close < ref_high:
                self.assertEqual(res['side'], 'buy')
                self.assertLess(res['stop'], res['entry'])
                self.assertGreater(res['target'], res['entry'])
            elif sweep_type == 'high_sweep' and ref_low < sw_close < ref_high:
                self.assertEqual(res['side'], 'sell')
                self.assertGreater(res['stop'], res['entry'])
                self.assertLess(res['target'], res['entry'])
            else:
                self.assertEqual(res['side'], 'hold')

    def test_number_validation_fuzz(self):
        """Fuzz `number()` helper with bad types and edge cases."""
        bad_inputs = [None, True, False, float('nan'), float('inf'), float('-inf'),
                      "abc", {}, [], object(), "NaN", "infinity"]
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                number(bad)

        valid_inputs = [1, 0, -1, 3.1415, "123.45", "-50.2", 1e-5]
        for val in valid_inputs:
            res = number(val)
            self.assertTrue(math.isfinite(res))


if __name__ == '__main__':
    unittest.main()
