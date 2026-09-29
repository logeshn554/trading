"""Single-process, durable CRT execution with exchange-hosted bracket exits."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from ethresearch.crt import signal, closed_candles
from ethresearch.alerts import ALERTS, logger

IST = timezone(timedelta(hours=5, minutes=30))
LIMITS = ('max_trades_per_day', 'max_contracts', 'risk_per_trade_inr', 'daily_profit_inr',
          'daily_loss_inr', 'quote_to_inr', 'fee_bps_per_side', 'max_spread_bps', 'max_slippage_bps')


def number(value):
    if value is None or isinstance(value, bool):
        raise ValueError('A required numeric value is missing or invalid')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('A required numeric value is missing or invalid') from None
    if not math.isfinite(value):
        raise ValueError('Invalid numeric exchange value')
    return value


def unwrap(value):
    if isinstance(value, dict) and value.get('success') is False:
        raise ValueError('Exchange rejected request')
    return value.get('result', value) if isinstance(value, dict) else value


class CRTTrader:
    def __init__(self, client, config, state_dir=None, emergency_close_unprotected: bool = False):
        self.client = client
        self.config = copy.deepcopy(config)
        self.root = Path(state_dir or os.environ.get('CRT_STATE_DIR', 'runtime/crt'))
        self.lock = threading.RLock()
        self.quit = threading.Event()
        self.thread = None
        self.enabled = False  # Every process restart requires explicit arming.
        self.status = {'state': 'OFF', 'reason': 'Live entries disabled', 'candles': [], 'signal': {}}
        self.db = None
        self._bracket_recovery_attempts = 0
        self._max_bracket_recovery_attempts = 3
        self.emergency_close_unprotected = emergency_close_unprotected

    def market(self, tool, args):
        """Public Delta feed works even when the MCP session still needs account login."""
        try:
            return unwrap(self.client.call(tool, args))
        except Exception:
            endpoints = {'get_product': '/v2/products/ETHUSD', 'get_ticker': '/v2/tickers/ETHUSD',
                         'get_candles': '/v2/history/candles?' + urlencode(args)}
            if tool not in endpoints:
                raise
            base = 'https://api.india.delta.exchange' if self.client.environment == 'india_prod' else 'https://cdn-ind.testnet.deltaex.org'
            try:
                with urlopen(base + endpoints[tool], timeout=10) as response:
                    return unwrap(json.load(response))
            except HTTPError as exc:
                raise ConnectionError(f'Delta public API returned HTTP {exc.code}. Retrying automatically; entries blocked.') from None
            except (URLError, TimeoutError) as exc:
                raise ConnectionError(
                    'Cannot connect securely to Delta API on this network. '
                    'Check your internet connection, VPN or firewall. '
                    'Retrying automatically; entries blocked until fresh data returns.'
                ) from None

    def initialize(self):
        if self.db is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        # An OS file lock prevents two workers sharing this state directory.
        self.lease = (self.root / 'engine.lock').open('a+b')
        if os.name == 'nt':
            import msvcrt
            self.lease.seek(0); self.lease.write(b'0'); self.lease.flush(); self.lease.seek(0)
            try:
                msvcrt.locking(self.lease.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError(f'Another CRT bot instance is already active in {self.root}')
        else:
            import fcntl
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError(f'Another CRT bot instance is already active in {self.root}')
        self.db = sqlite3.connect(self.root / 'state.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS intents (id TEXT PRIMARY KEY, day TEXT, payload TEXT, response TEXT, state TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, payload TEXT)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS trades (
            setup_id TEXT PRIMARY KEY,
            signal_timestamp INTEGER,
            direction TEXT,
            intended_entry REAL,
            actual_fills INTEGER,
            actual_average_entry REAL,
            actual_exit REAL,
            actual_average_exit REAL,
            stop REAL,
            target REAL,
            actual_fees REAL,
            funding REAL,
            taxes REAL,
            slippage REAL,
            realized_pnl REAL,
            net_pnl REAL,
            r_multiple REAL,
            exit_reason TEXT,
            duration REAL,
            created_at TEXT
        )''')
        saved = self.db.execute('SELECT payload FROM settings WHERE id=1').fetchone()
        if saved:
            self.config['risk_limits'] = json.loads(saved[0])
        self.db.commit()

    def configure(self, values):
        with self.lock:
            self.initialize()
            if self.enabled:
                raise ValueError('Switch entries OFF before changing limits')
            if set(values) != set(LIMITS):
                raise ValueError('Provide every displayed risk setting')
            parsed = {k: number(v) for k, v in values.items()}
            if any(v <= 0 for v in parsed.values()):
                raise ValueError('All limits must be positive')
            for k in ('max_trades_per_day', 'max_contracts'):
                if parsed[k] != int(parsed[k]) or parsed[k] > 1000:
                    raise ValueError('Trade count and contracts must be integers from 1 to 1000')
                parsed[k] = int(parsed[k])
            if parsed['max_slippage_bps'] > 100 or parsed['max_spread_bps'] > 100:
                raise ValueError('Spread and slippage limits cannot exceed 100 bps')
            self.db.execute('INSERT OR REPLACE INTO settings VALUES (1,?)', (json.dumps(parsed),))
            self.db.commit()
            self.config['risk_limits'] = parsed

    def arm(self, enabled):
        with self.lock:
            self.initialize()
            if type(enabled) is not bool:
                raise ValueError('enabled must be a boolean')
            if enabled:
                if os.environ.get('CRT_LIVE_ENABLED') != '1':
                    raise ValueError('Set CRT_LIVE_ENABLED=1 on the server to permit live orders')
                if not self.client.allow_trading:
                    raise ValueError('Exchange client is read-only')
                if any(not self.config['risk_limits'].get(k) for k in LIMITS):
                    raise ValueError('Save all risk limits first')
                if self.db.execute("SELECT 1 FROM intents WHERE state IN ('SUBMITTING','UNKNOWN')").fetchone():
                    raise ValueError('Uncertain order needs reconciliation; inspect Delta and the durable intent ledger')
                self.client.available_tools()
                schema = self.client.tool_schema('place_order').get('properties', {})
                if not {'bracket_stop_loss_price', 'bracket_take_profit_price', 'client_order_id', 'time_in_force'} <= set(schema):
                    raise ValueError('Installed MCP lacks required bracket-order fields')
                self.step(allow_entry=False)  # Refresh data and verify connectivity before arming.
                if self.status.get('error'):
                    raise ValueError(self.status['error'])
            self.enabled = enabled
            self.status['state'] = 'ARMED' if enabled else 'OFF'

    def pages(self, tool, args):
        result, seen = [], set()
        args = dict(args, page_size=100)
        for _ in range(100):
            raw = self.client.call(tool, args)
            rows = unwrap(raw)
            if not isinstance(rows, list):
                raise ValueError(f'Invalid {tool} response')
            result.extend(rows)
            cursor = raw.get('meta', {}).get('after') if isinstance(raw, dict) else None
            if not cursor:
                return result
            if cursor in seen:
                break
            seen.add(cursor); args['after'] = cursor
        raise ValueError('Account history incomplete; new entries blocked')

    def reconcile(self):
        """Read uncertain identities; never resend an entry or infer absent orders are cancelled."""
        pending = self.db.execute("SELECT id,payload FROM intents WHERE state IN ('SUBMITTING','UNKNOWN')").fetchall()
        for identity, payload in pending:
            try:
                raw = self.client.call('get_order_by_id', {'client_order_id': identity})
                order, sent = unwrap(raw), json.loads(payload)
                if order.get('client_order_id') != identity:
                    continue
                terminal_empty = order.get('state') == 'cancelled' and number(order['unfilled_size']) == number(order['size'])
                protected = (number(order.get('bracket_stop_loss_price')) == number(sent['bracket_stop_loss_price']) and
                             number(order.get('bracket_take_profit_price')) == number(sent['bracket_take_profit_price'])) if not terminal_empty else False
                # Reconcile partial fills: record actual filled size for downstream risk accounting.
                filled_size = number(order['size']) - number(order['unfilled_size']) if not terminal_empty else 0
                new_state = 'CANCELLED_UNFILLED' if terminal_empty else 'ACKNOWLEDGED'
                response_record = dict(raw if isinstance(raw, dict) else {}, _reconciled_filled_size=filled_size)
                if terminal_empty or protected:
                    self.db.execute('UPDATE intents SET state=?,response=? WHERE id=?',
                                    (new_state, json.dumps(response_record), identity))
                    self.db.commit()
            except Exception:
                continue  # A missing/failed lookup is not proof that no order was placed.

    def _verify_position_protection(self, positions, orders):
        """Check that every active position has matching SL and TP bracket orders with sufficient quantity.

        Delta represents protective orders with:
          order_type: "market_order" or "limit_order"
          stop_order_type: "stop_loss_order" or "take_profit_order"
          stop_price: the trigger price
          size: contracts to close

        Returns (protected: bool, unprotected_positions: list, details: str).
        """
        active = [p for p in positions if number(p.get('size', 0)) != 0]
        if not active:
            return True, [], 'No active positions'

        unprotected = []
        for pos in active:
            product_id = pos.get('product_id')
            pos_qty = abs(number(pos.get('size', 0)))
            pos_side = 'buy' if number(pos.get('size', 0)) > 0 else 'sell'

            sl_orders = []
            tp_orders = []

            for o in orders:
                if o.get('product_id') != product_id:
                    continue
                # A protective order closes the position: opposite side.
                order_side = o.get('side', '')
                if order_side == pos_side:
                    continue  # Same side doesn't protect.

                # Delta uses stop_order_type to classify bracket orders.
                stop_type = o.get('stop_order_type', '')
                if stop_type == 'stop_loss_order':
                    sl_orders.append(o)
                elif stop_type == 'take_profit_order':
                    tp_orders.append(o)

            has_stop = len(sl_orders) > 0
            has_tp = len(tp_orders) > 0
            sl_qty = sum(number(o.get('size', pos_qty)) for o in sl_orders)
            tp_qty = sum(number(o.get('size', pos_qty)) for o in tp_orders)
            qty_protected = (sl_qty >= pos_qty) and (tp_qty >= pos_qty)

            if not has_stop or not has_tp or not qty_protected:
                unprotected.append(pos)

        if unprotected:
            details = f'{len(unprotected)} position(s) missing SL/TP bracket protection or insufficient protected size'
            return False, unprotected, details
        return True, [], 'All positions have bracket protection'

    def _emergency_close_position(self, pos):
        """Optional emergency close when protective orders are missing and recovery fails."""
        if not self.emergency_close_unprotected:
            return False
        product_id = pos.get('product_id')
        pos_size = abs(number(pos.get('size', 0)))
        if pos_size == 0:
            return False
        exit_side = 'sell' if number(pos.get('size', 0)) > 0 else 'buy'
        try:
            tools = self.client.available_tools()
            if 'close_all_positions' in tools:
                self.client.call('close_all_positions', {})
            elif 'place_order' in tools:
                self.client.call('place_order', {
                    'product_id': product_id,
                    'size': pos_size,
                    'side': exit_side,
                    'order_type': 'market_order',
                    'time_in_force': 'ioc',
                })
            ALERTS.dispatch('EMERGENCY_CLOSE_TRIGGERED', 'CRITICAL', f'Emergency closed unprotected position for product {product_id}')
            return True
        except Exception as exc:
            ALERTS.dispatch('EMERGENCY_CLOSE_FAILED', 'FATAL', f'Emergency close failed: {exc}')
            return False

    def _attempt_bracket_recovery(self, unprotected_positions, product):
        """Attempt to place bracket orders on unprotected positions.

        Uses Delta's place_bracket_order with nested stop_loss_order and
        take_profit_order objects as documented in the MCP interface.
        Position brackets cover the entire open position; size is not required.

        Returns True if recovery succeeds, False otherwise.
        """
        if self._bracket_recovery_attempts >= self._max_bracket_recovery_attempts:
            return False

        self._bracket_recovery_attempts += 1

        for pos in unprotected_positions:
            product_id = pos.get('product_id')
            pos_size = abs(number(pos.get('size', 0)))
            entry_price = number(pos.get('entry_price', 0))

            if pos_size == 0 or entry_price == 0:
                continue

            # Look up the last acknowledged intent for this product to get SL/TP.
            last_intent = self.db.execute(
                "SELECT payload FROM intents WHERE state='ACKNOWLEDGED' ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if not last_intent:
                continue

            intent_payload = json.loads(last_intent[0])
            if intent_payload.get('product_id') != product_id:
                continue

            sl_price = intent_payload.get('bracket_stop_loss_price')
            tp_price = intent_payload.get('bracket_take_profit_price')
            if not sl_price or not tp_price:
                continue

            # Check if place_bracket_order is available.
            tools = self.client.available_tools()
            if 'place_bracket_order' not in tools:
                return False

            try:
                result = self.client.call('place_bracket_order', {
                    'product_id': product_id,
                    'stop_loss_order': {
                        'order_type': 'market_order',
                        'stop_price': str(sl_price),
                    },
                    'take_profit_order': {
                        'order_type': 'market_order',
                        'stop_price': str(tp_price),
                    },
                    'bracket_stop_trigger_method': 'last_traded_price',
                })
                unwrap(result)
                return True
            except Exception:
                continue

        return False

    def _reconcile_fill_size(self, order_response, sent_payload):
        """Reconcile actual filled size vs requested size and compute real-fill risk.

        Returns dict with reconciliation details including actual_fill_price.
        """
        order = order_response if isinstance(order_response, dict) else {}
        result_order = order.get('result', order) if isinstance(order, dict) else order

        requested_size = number(sent_payload.get('size', 0))
        order_size = number(result_order.get('size', requested_size))
        unfilled = number(result_order.get('unfilled_size', 0))
        filled_size = order_size - unfilled

        # Use average fill price if available, otherwise fall back to limit price.
        avg_fill = result_order.get('average_fill_price')
        if avg_fill is not None:
            try:
                actual_entry = number(avg_fill)
            except (ValueError, TypeError):
                actual_entry = number(sent_payload.get('limit_price', 0))
        else:
            actual_entry = number(sent_payload.get('limit_price', 0))

        return {
            'requested_size': requested_size,
            'filled_size': filled_size,
            'unfilled_size': unfilled,
            'is_partial': 0 < filled_size < requested_size,
            'is_fully_filled': filled_size == requested_size,
            'is_unfilled': filled_size == 0,
            'actual_entry_price': actual_entry,
        }

    def _reconcile_closed_trades(self, positions, product, limits, now):
        """Record closed trades in the durable trades table for verified performance accounting."""
        active_ids = {p.get('product_id') for p in positions if number(p.get('size', 0)) != 0}
        unclosed = self.db.execute(
            "SELECT id, day, payload, response FROM intents WHERE state='ACKNOWLEDGED' AND id NOT IN (SELECT setup_id FROM trades) ORDER BY rowid ASC"
        ).fetchall()
        for ident, day, payload_json, resp_json in unclosed:
            payload = json.loads(payload_json) if payload_json else {}
            product_id = payload.get('product_id')
            if product_id in active_ids:
                continue  # Position is still active on the exchange

            # Position has closed!
            resp = json.loads(resp_json) if resp_json else {}
            fill_info = resp.get('_fill_reconciliation', {})
            direction = payload.get('side', 'buy')
            buy = direction == 'buy'
            actual_entry = fill_info.get('actual_entry_price') or number(payload.get('limit_price', 0))
            actual_fills = fill_info.get('filled_size', number(payload.get('size', 1)))
            stop = number(payload.get('bracket_stop_loss_price', 0))
            target = number(payload.get('bracket_take_profit_price', 0))
            unit = number(product.get('contract_value', 0.01))

            actual_exit = None
            try:
                fills = self.client.call('get_fills', {'product_id': product_id, 'page_size': 10})
                fill_rows = unwrap(fills)
                if isinstance(fill_rows, list):
                    exit_side = 'sell' if buy else 'buy'
                    exit_fills = [f for f in fill_rows if f.get('side') == exit_side]
                    if exit_fills:
                        actual_exit = number(exit_fills[0].get('price', target))
            except Exception:
                pass

            if actual_exit is None:
                actual_exit = target

            exit_reason = 'take_profit' if abs(actual_exit - target) <= abs(actual_exit - stop) else 'stop_loss'
            side_mult = 1 if buy else -1
            gross_pnl_usd = (actual_exit - actual_entry) * side_mult * unit * actual_fills
            entry_fee = actual_entry * unit * actual_fills * limits['fee_bps_per_side'] / 10000.0
            exit_fee = actual_exit * unit * actual_fills * limits['fee_bps_per_side'] / 10000.0
            actual_fees_usd = entry_fee + exit_fee
            net_pnl_usd = gross_pnl_usd - actual_fees_usd
            net_pnl_inr = net_pnl_usd * limits['quote_to_inr']

            risk_per_contract = abs(actual_entry - stop) * unit * limits['quote_to_inr'] + (actual_entry + stop) * limits['fee_bps_per_side'] / 10000.0 * unit * limits['quote_to_inr']
            initial_risk_inr = risk_per_contract * actual_fills
            r_multiple = net_pnl_inr / initial_risk_inr if initial_risk_inr > 0 else 0.0

            sig_time = 0
            if '-' in ident:
                try:
                    sig_time = int(ident.split('-')[1])
                except Exception:
                    sig_time = int(now)
            duration = max(0.0, now - sig_time) if sig_time > 0 else 0.0

            self.db.execute(
                '''INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (
                    ident, sig_time, direction, actual_entry, actual_fills, actual_entry,
                    actual_exit, actual_exit, stop, target, actual_fees_usd, 0.0, 0.0,
                    abs(actual_entry - number(payload.get('limit_price', actual_entry))),
                    gross_pnl_usd, net_pnl_inr, r_multiple, exit_reason, duration,
                    datetime.now(timezone.utc).isoformat()
                )
            )
            self.db.commit()
            ALERTS.dispatch('TRADE_CLOSED', 'INFO', f'Closed trade {ident}: exit={actual_exit}, net_inr={net_pnl_inr:.2f}, R={r_multiple:.2f}',
                            {'setup_id': ident, 'net_pnl_inr': net_pnl_inr, 'r_multiple': r_multiple, 'exit_reason': exit_reason})

    def step(self, allow_entry=True):
        with self.lock:
            self.initialize()
            now = time.time()
            try:
                product = self.market('get_product', {'symbol': 'ETHUSD'})
                if product.get('symbol') != 'ETHUSD' or product.get('state') != 'live':
                    raise ValueError('ETHUSD product unavailable')
                tick = number(product['tick_size'])
                rows = self.market('get_candles', {'symbol': 'ETHUSD', 'resolution': '15m',
                                                  'start': int(now)-900*100, 'end': int(now)})
                bars = closed_candles(rows, now)
                sig = signal(rows, now, tick)
                self.status.update(candles=bars[-96:], signal=sig, as_of=now, error=None)
                positions = unwrap(self.client.call('get_margined_positions'))
                if not isinstance(positions, list):
                    raise ValueError('Invalid positions response')
                orders = self.pages('get_open_orders', {})
                self.reconcile()
                active = [p for p in positions if number(p['size']) != 0]
                self.status['positions'] = positions
                self.status['open_orders'] = orders
                limits = self.config['risk_limits']
                missing = [key for key in LIMITS if limits.get(key) is None or limits.get(key) == '']
                self.status['missing_settings'] = missing
                if missing:
                    names = {
                        'max_trades_per_day': 'maximum daily entry attempts',
                        'max_contracts': 'maximum contracts',
                        'risk_per_trade_inr': 'risk per trade (INR)',
                        'daily_profit_inr': 'daily profit stop (INR)',
                        'daily_loss_inr': 'daily loss stop (INR)',
                        'quote_to_inr': 'USD-to-INR settlement conversion',
                        'fee_bps_per_side': 'fees and taxes per side',
                        'max_spread_bps': 'maximum spread',
                        'max_slippage_bps': 'entry slippage cap',
                    }
                    message = 'Complete and save Risk limits: ' + ', '.join(names[key] for key in missing) + '.'
                    self.status.update(state='SETUP_REQUIRED', reason=message, error=message, daily_net_inr=None)
                    return
                day_start = datetime.fromtimestamp(now, IST).replace(hour=0, minute=0, second=0, microsecond=0)
                day = day_start.date().isoformat()
                transactions = self.pages('get_wallet_transactions', {'start_time_us': int(day_start.timestamp()*1e6), 'end_time_us': int(now*1e6)})
                self.status['transactions'] = transactions
                # Net account trading cashflow: deposits/transfers are excluded; unknown types fail closed.
                pnl = 0.0
                ignored = {'deposit', 'external_deposit', 'withdrawal', 'sub_account_transfer',
                           'withdrawal_cancellation', 'referral_bonus', 'promo_credit', 'trading_credits'}
                trading = {'realized_pnl', 'settlement', 'commission', 'commission_rebate', 'funding',
                           'liquidation_fee', 'gst', 'tax'}
                for row in transactions:
                    kind = row.get('transaction_type')
                    if kind in ignored:
                        continue
                    if kind == 'cashflow' and row.get('product_id'):
                        kind = 'realized_pnl'
                    if kind not in trading:
                        raise ValueError(f'Unmapped wallet transaction type: {kind}; INR risk accounting blocked')
                    asset = row.get('asset_symbol')
                    if asset not in {'INR', 'USD'}:
                        raise ValueError('Unsupported settlement currency in daily accounting')
                    conversion = 1 if asset == 'INR' else number(limits['quote_to_inr'])
                    if conversion <= 0:
                        raise ValueError('Configure the Delta USD/INR settlement conversion')
                    pnl += number(row['amount']) * conversion
                self.status['daily_net_inr'] = pnl
                count = self.db.execute('SELECT COUNT(*) FROM intents WHERE day=?', (day,)).fetchone()[0]
                self.status['trades_today'] = count
                self.status['state'] = 'ARMED' if self.enabled else 'OFF'
                self.status['reason'] = sig['reason']

                # --- Continuous bracket/protection verification ---
                if active:
                    protected, unprotected, protection_detail = self._verify_position_protection(positions, orders)
                    self.status['protection_status'] = protection_detail
                    self.status['protection_verified'] = protected

                    if not protected:
                        # Attempt automatic bracket recovery.
                        recovered = self._attempt_bracket_recovery(unprotected, product)
                        if recovered:
                            # Re-verify after recovery.
                            orders_after = self.pages('get_open_orders', {})
                            protected, _, protection_detail = self._verify_position_protection(positions, orders_after)
                            self.status['open_orders'] = orders_after
                            self.status['protection_status'] = protection_detail + ' (recovered)'
                            self.status['protection_verified'] = protected
                            if protected:
                                self._bracket_recovery_attempts = 0  # Reset only on verified success.

                        if not protected:
                            # Optional emergency close
                            if self.emergency_close_unprotected:
                                for unp in unprotected:
                                    self._emergency_close_position(unp)
                            # Circuit breaker: UNPROTECTED_POSITION state.
                            self.enabled = False
                            ALERTS.dispatch('UNPROTECTED_POSITION', 'CRITICAL', f'{protection_detail}. Entries disabled.')
                            self.status.update(
                                state='UNPROTECTED_POSITION',
                                reason=f'CRITICAL: {protection_detail}. Entries disabled. '
                                       f'Recovery attempted {self._bracket_recovery_attempts}/{self._max_bracket_recovery_attempts} times. '
                                       f'Inspect Delta immediately and manually verify/place bracket orders.',
                                error=f'Position without SL/TP protection detected. Automatic recovery failed.'
                            )
                            return

                # Exchange brackets continue managing existing positions even when entries are OFF.
                if active or orders:
                    self.status['reason'] = 'Existing position/order: no additional entry; exchange exits remain active'
                    return
                if not self.enabled or not allow_entry:
                    return
                # Reset recovery counter and reconcile closed trades when no active positions.
                self._bracket_recovery_attempts = 0
                self._reconcile_closed_trades(positions, product, limits, now)
                if self.db.execute("SELECT 1 FROM intents WHERE state IN ('SUBMITTING','UNKNOWN')").fetchone():
                    raise ValueError('Uncertain previous submission; no new entries')
                if count >= limits['max_trades_per_day'] or pnl >= limits['daily_profit_inr'] or pnl <= -limits['daily_loss_inr']:
                    ALERTS.dispatch('DAILY_STOP', 'WARN', 'Daily trade count, profit or loss limit reached')
                    self.status.update(state='DAILY_STOP', reason='Daily trade count, profit or loss limit reached')
                    return
                if sig['side'] == 'hold':
                    return
                if self.db.execute('SELECT 1 FROM intents WHERE id=?', (sig['id'],)).fetchone():
                    self.status['reason'] = 'This CRT setup was already submitted'
                    return
                ticker = self.market('get_ticker', {'symbol': 'ETHUSD'})
                quote = ticker['quotes']
                bid, ask = number(quote['best_bid']), number(quote['best_ask'])
                if not 0 < bid <= ask or (ask-bid)/bid*10000 > limits['max_spread_bps']:
                    raise ValueError('Invalid or excessive bid/ask spread')
                # Refuse delayed requests and stale candle decisions immediately before submitting.
                if time.time() - now > 30 or time.time() - sig['signal_time'] > 90:
                    raise ValueError('Market snapshot too old to submit')
                buy = sig['side'] == 'buy'
                cap = sig['entry'] * (1 + (1 if buy else -1)*limits['max_slippage_bps']/10000)
                rounding = ROUND_FLOOR if buy else ROUND_CEILING
                cap = float((Decimal(str(cap))/Decimal(str(tick))).to_integral_value(rounding=rounding)*Decimal(str(tick)))
                if (buy and ask > cap) or (not buy and bid < cap):
                    self.status['reason'] = 'Price moved beyond the entry slippage cap'
                    return
                if not (sig['stop'] < cap < sig['target'] if buy else sig['target'] < cap < sig['stop']):
                    raise ValueError('Entry price outside CRT stop/target')
                if product.get('contract_type') != 'perpetual_futures' or product.get('contract_unit_currency') != 'ETH':
                    raise ValueError('Unsupported contract specification')
                unit = number(product['contract_value']) * limits['quote_to_inr']
                if unit <= 0:
                    raise ValueError('Invalid contract value')
                costs = (cap + sig['stop']) * limits['fee_bps_per_side']/10000 * unit
                risk = abs(cap-sig['stop'])*unit + costs
                reward = abs(sig['target']-cap)*unit - (cap+sig['target'])*limits['fee_bps_per_side']/10000*unit
                if reward <= 0:
                    self.status['reason'] = 'Target does not cover configured transaction costs'
                    return
                budget = min(limits['risk_per_trade_inr'], limits['daily_loss_inr']+pnl)
                size = min(limits['max_contracts'], int(budget/risk))
                if size < 1:
                    self.status['reason'] = 'One contract exceeds the remaining INR risk budget'
                    return
                wallets = unwrap(self.client.call('get_wallet_balances'))
                settlement = product['settling_asset']['symbol']
                if settlement not in {'INR', 'USD'}:
                    raise ValueError('Unsupported settlement collateral')
                available = sum(number(w['available_balance']) * (limits['quote_to_inr'] if settlement == 'USD' else 1)
                                for w in wallets if w.get('asset_symbol') == settlement)
                # Fully collateralized notional check avoids assuming the account's leverage.
                if available < size*(cap*unit+costs):
                    raise ValueError('Insufficient INR collateral for conservative notional check')
                if time.time() - now > 30 or time.time() - sig['signal_time'] > 90:
                    raise ValueError('Snapshot expired during preflight')
                payload = dict(product_id=int(product['id']), size=size, side=sig['side'], order_type='limit_order',
                               limit_price=str(cap), time_in_force='ioc', client_order_id=sig['id'],
                               bracket_stop_loss_price=str(sig['stop']), bracket_take_profit_price=str(sig['target']),
                               bracket_stop_trigger_method='last_traded_price', dry_run=False)
                self.db.execute('INSERT INTO intents VALUES (?,?,?,?,?)', (sig['id'], day, json.dumps(payload), None, 'SUBMITTING'))
                self.db.commit()  # Persist identity BEFORE any money-changing network call.
                ALERTS.dispatch('INTENT_CREATED', 'INFO', f'Order intent created: {sig["id"]} {sig["side"]} {size} contracts')
                try:
                    response = self.client.call('place_order', payload)
                    order = unwrap(response)
                    if not isinstance(order, dict) or not order.get('id') or response.get('dry_run'):
                        raise ValueError('Order acknowledgment could not be verified')
                    if (number(order.get('bracket_stop_loss_price')) != number(payload['bracket_stop_loss_price']) or
                            number(order.get('bracket_take_profit_price')) != number(payload['bracket_take_profit_price'])):
                        raise ValueError('Order protection not confirmed; inspect Delta immediately')

                    # --- Partial-fill reconciliation and real-fill risk ---
                    fill_info = self._reconcile_fill_size(response, payload)
                    response_record = response if isinstance(response, dict) else {}
                    response_record = dict(response_record,
                                           _fill_reconciliation=fill_info)

                    if fill_info['is_unfilled']:
                        # IOC was fully cancelled; no position or risk exposure.
                        self.db.execute('UPDATE intents SET response=?, state=? WHERE id=?',
                                        (json.dumps(response_record), 'CANCELLED_UNFILLED', sig['id']))
                        self.db.commit()
                        self.status['reason'] = 'IOC order fully cancelled (zero fill); no position opened'
                        ALERTS.dispatch('ORDER_CANCELLED', 'INFO', f'IOC order {sig["id"]} fully cancelled (zero fill)')
                        return

                    if fill_info['is_partial']:
                        # Partial fill: position exists but is smaller than requested.
                        # Log it and acknowledge — bracket protection covers the filled portion.
                        response_record['_partial_fill_warning'] = (
                            f"Requested {fill_info['requested_size']} contracts, "
                            f"filled {fill_info['filled_size']}, "
                            f"unfilled {fill_info['unfilled_size']}"
                        )
                        ALERTS.dispatch('PARTIAL_FILL', 'WARN', f'Order {sig["id"]} partially filled: {fill_info["filled_size"]}/{fill_info["requested_size"]}')

                    # Compute actual risk from real fill price, not the IOC limit cap.
                    actual_entry = fill_info['actual_entry_price']
                    if actual_entry > 0:
                        actual_risk_per_contract = abs(actual_entry - sig['stop']) * number(product['contract_value']) * limits['quote_to_inr']
                        actual_total_risk = actual_risk_per_contract * fill_info['filled_size']
                        response_record['_actual_risk_inr'] = actual_total_risk
                        response_record['_actual_entry_price'] = actual_entry

                    self.db.execute('UPDATE intents SET response=?, state=? WHERE id=?',
                                    (json.dumps(response_record), 'ACKNOWLEDGED', sig['id']))
                    self.db.commit()
                    self.status['reason'] = 'CRT bracket order acknowledged by Delta'
                    if fill_info['is_partial']:
                        self.status['reason'] += f" (partial fill: {fill_info['filled_size']}/{fill_info['requested_size']})"
                    ALERTS.dispatch('ORDER_ACKNOWLEDGED', 'INFO', f'Order {sig["id"]} acknowledged by Delta')
                except Exception:
                    self.db.execute("UPDATE intents SET state='UNKNOWN' WHERE id=?", (sig['id'],))
                    self.db.commit()
                    self.enabled = False
                    ALERTS.dispatch('ORDER_UNKNOWN', 'CRITICAL', f'Order outcome uncertain for {sig["id"]}')
                    raise ValueError('Order outcome/protection uncertain. Entries OFF; inspect Delta before reconciliation')
            except Exception as exc:
                self.status.update(state='BLOCKED', error=str(exc), reason=str(exc), daily_net_inr=None)

    def get_status(self):
        with self.lock:
            value = copy.deepcopy(self.status)
            value.update(enabled=self.enabled, risk_limits=self.config['risk_limits'], strategy='CRT · ETHUSD · 15m',
                         live_permitted=os.environ.get('CRT_LIVE_ENABLED') == '1', win_rate=None,
                         performance_note='No verified CRT closed-trade performance yet',
                         bracket_recovery_attempts=self._bracket_recovery_attempts,
                         max_bracket_recovery_attempts=self._max_bracket_recovery_attempts,
                         emergency_close_unprotected=self.emergency_close_unprotected)
            if self.db:
                value['intents'] = [dict(zip(('id','day','payload','response','state'), row)) for row in self.db.execute('SELECT * FROM intents ORDER BY rowid DESC LIMIT 30')]
                trades = self.db.execute('SELECT setup_id, net_pnl, r_multiple, duration, exit_reason FROM trades ORDER BY rowid DESC').fetchall()
                if trades:
                    total_trades = len(trades)
                    wins = sum(1 for t in trades if t[1] > 0)
                    win_rate = round(wins / total_trades, 4)
                    value['win_rate'] = win_rate
                    value['closed_trades_count'] = total_trades
                    value['performance_note'] = f"{total_trades} verified closed trade(s), win rate: {win_rate*100:.1f}%"
                    value['recent_trades'] = [dict(zip(('setup_id', 'net_pnl', 'r_multiple', 'duration', 'exit_reason'), t)) for t in trades[:10]]
                else:
                    value['closed_trades_count'] = 0
                    value['win_rate'] = None
                    value['performance_note'] = 'No verified CRT closed-trade performance yet'
                    value['recent_trades'] = []
            return value

    def start(self):
        self.initialize()
        def loop():
            while not self.quit.is_set():
                self.step()
                self.quit.wait(10)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.enabled = False
        self.quit.set()
        if self.thread:
            self.thread.join(timeout=35)
        if not self.thread or not self.thread.is_alive():
            if self.db:
                self.db.close()
                self.db = None
            if hasattr(self, 'lease'):
                self.lease.close()
