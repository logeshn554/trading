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
    def __init__(self, client, config, state_dir=None):
        self.client = client
        self.config = copy.deepcopy(config)
        self.root = Path(state_dir or os.environ.get('CRT_STATE_DIR', 'runtime/crt'))
        self.lock = threading.RLock()
        self.quit = threading.Event()
        self.thread = None
        self.enabled = False  # Every process restart requires explicit arming.
        self.status = {'state': 'OFF', 'reason': 'Live entries disabled', 'candles': [], 'signal': {}}
        self.db = None

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
            msvcrt.locking(self.lease.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.root / 'state.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS intents (id TEXT PRIMARY KEY, day TEXT, payload TEXT, response TEXT, state TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, payload TEXT)')
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
                if terminal_empty or protected:
                    self.db.execute('UPDATE intents SET state=?,response=? WHERE id=?',
                                    ('CANCELLED_UNFILLED' if terminal_empty else 'ACKNOWLEDGED', json.dumps(raw), identity))
                    self.db.commit()
            except Exception:
                continue  # A missing/failed lookup is not proof that no order was placed.

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
                # Exchange brackets continue managing existing positions even when entries are OFF.
                if active or orders:
                    self.status['reason'] = 'Existing position/order: no additional entry; exchange exits remain active'
                    return
                if not self.enabled or not allow_entry:
                    return
                if self.db.execute("SELECT 1 FROM intents WHERE state IN ('SUBMITTING','UNKNOWN')").fetchone():
                    raise ValueError('Uncertain previous submission; no new entries')
                if count >= limits['max_trades_per_day'] or pnl >= limits['daily_profit_inr'] or pnl <= -limits['daily_loss_inr']:
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
                try:
                    response = self.client.call('place_order', payload)
                    order = unwrap(response)
                    if not isinstance(order, dict) or not order.get('id') or response.get('dry_run'):
                        raise ValueError('Order acknowledgment could not be verified')
                    if (number(order.get('bracket_stop_loss_price')) != number(payload['bracket_stop_loss_price']) or
                            number(order.get('bracket_take_profit_price')) != number(payload['bracket_take_profit_price'])):
                        raise ValueError('Order protection not confirmed; inspect Delta immediately')
                    self.db.execute('UPDATE intents SET response=?, state=? WHERE id=?', (json.dumps(response), 'ACKNOWLEDGED', sig['id']))
                    self.db.commit()
                    self.status['reason'] = 'CRT bracket order acknowledged by Delta'
                except Exception:
                    self.db.execute("UPDATE intents SET state='UNKNOWN' WHERE id=?", (sig['id'],))
                    self.db.commit()
                    self.enabled = False
                    raise ValueError('Order outcome/protection uncertain. Entries OFF; inspect Delta before reconciliation')
            except Exception as exc:
                self.status.update(state='BLOCKED', error=str(exc), reason=str(exc), daily_net_inr=None)

    def get_status(self):
        with self.lock:
            value = copy.deepcopy(self.status)
            value.update(enabled=self.enabled, risk_limits=self.config['risk_limits'], strategy='CRT · ETHUSD · 15m',
                         live_permitted=os.environ.get('CRT_LIVE_ENABLED') == '1', win_rate=None,
                         performance_note='No verified CRT closed-trade performance yet')
            if self.db:
                value['intents'] = [dict(zip(('id','day','payload','response','state'), row)) for row in self.db.execute('SELECT * FROM intents ORDER BY rowid DESC LIMIT 30')]
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
