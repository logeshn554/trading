import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from ethresearch.crt import signal, closed_candles
from ethresearch.crt_live import CRTTrader, LIMITS

ROWS = [dict(time=0, open=110, high=120, low=100, close=112),
        dict(time=900, open=112, high=115, low=90, close=110)]
SETTINGS = dict(max_trades_per_day=5, max_contracts=1, risk_per_trade_inr=100,
                daily_profit_inr=500, daily_loss_inr=500, quote_to_inr=85,
                fee_bps_per_side=6, max_spread_bps=10, max_slippage_bps=10)


class FakeClient:
    allow_trading = True
    environment = 'india_testnet'
    def __init__(self):
        self.sent = []
        self.fail = False
        self.transactions = []
        self.positions = []
        self.rows = copy.deepcopy(ROWS)
    def available_tools(self):
        return {'place_order'}
    def tool_schema(self, name):
        return {'properties': dict.fromkeys(('bracket_stop_loss_price','bracket_take_profit_price','client_order_id','time_in_force'))}
    def call(self, name, args=None):
        if name == 'get_product':
            data = dict(id=3136,symbol='ETHUSD',state='live',tick_size='.05',contract_value='.01',
                        contract_type='perpetual_futures',contract_unit_currency='ETH',settling_asset={'symbol':'USD'})
        elif name == 'get_candles': data = self.rows
        elif name == 'get_margined_positions': data = self.positions
        elif name == 'get_open_orders': data = []
        elif name == 'get_wallet_transactions': data = self.transactions
        elif name == 'get_ticker': data = {'quotes': {'best_bid': '110', 'best_ask': '110.01'}}
        elif name == 'get_wallet_balances': data = [{'asset_symbol':'USD','available_balance':'1000'}]
        elif name == 'place_order':
            self.sent.append(args)
            if self.fail: raise TimeoutError('uncertain')
            data = dict(args, id=123)
        else: raise AssertionError(name)
        return {'success': True, 'result': data}


class SignalTests(unittest.TestCase):
    def test_network_timeout_is_actionable_and_does_not_submit(self):
        from urllib.error import URLError
        client = FakeClient()
        engine = CRTTrader(client, {'risk_limits': SETTINGS})
        with patch.object(client, 'call', side_effect=RuntimeError('MCP unavailable')):
            with patch('ethresearch.crt_live.urlopen', side_effect=URLError('handshake timed out')):
                with self.assertRaisesRegex(ConnectionError, 'Retrying automatically'):
                    engine.market('get_product', {'symbol': 'ETHUSD'})
        self.assertEqual(client.sent, [])

    def test_long(self):
        s=signal(ROWS,1801,.05)
        self.assertEqual(s['side'],'buy'); self.assertEqual(s['target'],120)
        self.assertAlmostEqual(s['stop'],89.95)
    def test_short(self):
        rows=copy.deepcopy(ROWS); rows[1].update(high=130,low=105)
        s=signal(rows,1801,.05)
        self.assertEqual(s['side'],'sell'); self.assertEqual(s['target'],100)
    def test_double_sweep_and_boundary_close_skip(self):
        rows=copy.deepcopy(ROWS); rows[1]['high']=130
        self.assertEqual(signal(rows,1801,.05)['side'],'hold')
        rows[1]['high']=115; rows[1]['close']=100
        self.assertEqual(signal(rows,1801,.05)['side'],'hold')
    def test_no_forming_or_late_entries(self):
        self.assertEqual(signal(ROWS,1799,.05)['side'],'hold')
        self.assertEqual(signal(ROWS,1891,.05)['side'],'hold')
        future=dict(time=1800,open=110,high=500,low=1,close=300)
        self.assertEqual(signal(ROWS+[future],1801,.05),signal(ROWS,1801,.05))
    def test_bad_data_rejected(self):
        with self.assertRaises(ValueError): closed_candles(ROWS+ROWS,1801)
        rows=copy.deepcopy(ROWS);rows[1]['close']=float('nan')
        with self.assertRaises(ValueError): closed_candles(rows,1801)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.client=FakeClient()
        self.engine=CRTTrader(self.client,{'risk_limits':SETTINGS},self.tmp.name)
        self.clock=patch('ethresearch.crt_live.time.time',return_value=1801);self.clock.start()
    def tearDown(self):
        self.engine.stop();self.clock.stop();self.tmp.cleanup()
    def test_off_never_submits(self):
        self.engine.step();self.assertEqual(self.client.sent,[])
    def test_missing_conversion_is_explained_without_float_error(self):
        self.engine.config['risk_limits']['quote_to_inr'] = None
        self.client.transactions = [dict(transaction_type='commission',asset_symbol='USD',amount='-.1')]
        self.engine.step()
        self.assertEqual(self.engine.status['state'], 'SETUP_REQUIRED')
        self.assertIn('USD-to-INR settlement conversion', self.engine.status['reason'])
        self.assertIsNone(self.engine.status['daily_net_inr'])
        self.assertEqual(len(self.engine.status['candles']), 2)
        self.assertEqual(self.client.sent, [])
        self.engine.configure(SETTINGS)
        self.engine.step()
        self.assertEqual(self.engine.status['state'], 'OFF')
        self.assertEqual(self.engine.status['missing_settings'], [])
    def test_null_numeric_input_is_validation_error(self):
        with self.assertRaisesRegex(ValueError, 'numeric value'):
            self.engine.configure(dict(SETTINGS,quote_to_inr=None))
    def test_bracket_once_and_durable_restart(self):
        self.engine.enabled=True;self.engine.step();self.engine.step()
        self.assertEqual(len(self.client.sent),1)
        order=self.client.sent[0]
        self.assertEqual(order['time_in_force'],'ioc');self.assertLess(float(order['bracket_stop_loss_price']),float(order['limit_price']))
        self.engine.stop();self.engine=CRTTrader(self.client,{'risk_limits':SETTINGS},self.tmp.name)
        self.assertFalse(self.engine.enabled)
        self.engine.enabled=True;self.engine.step();self.assertEqual(len(self.client.sent),1)
    def test_timeout_not_retried(self):
        self.client.fail=True;self.engine.enabled=True;self.engine.step()
        self.assertFalse(self.engine.enabled)
        self.engine.enabled=True;self.engine.step();self.assertEqual(len(self.client.sent),1)
        self.assertIn('Uncertain',self.engine.status['reason'])
    def test_daily_net_includes_fees_and_funding(self):
        self.client.transactions=[{'transaction_type':k,'asset_symbol':'USD','amount':a,'product_id':3136}
                                  for k,a in [('cashflow','2'),('commission','-.1'),('funding','-.2')]]
        self.engine.step();self.assertAlmostEqual(self.engine.status['daily_net_inr'],144.5)
    def test_daily_loss_stops_entries(self):
        self.client.transactions=[dict(transaction_type='settlement',asset_symbol='INR',amount=-600)]
        self.engine.enabled=True;self.engine.step()
        self.assertEqual(self.engine.status['state'],'DAILY_STOP');self.assertEqual(self.client.sent,[])
    def test_unknown_cashflow_and_currency_block(self):
        self.client.transactions=[dict(transaction_type='cashflow',asset_symbol='USD',amount=100)]
        self.engine.enabled=True;self.engine.step();self.assertEqual(self.client.sent,[])
        self.assertEqual(self.engine.status['state'],'BLOCKED')
    def test_existing_position_prevents_entry(self):
        self.client.positions=[dict(size=1)];self.engine.enabled=True;self.engine.step()
        self.assertEqual(self.client.sent,[])
    def test_limits_persist_and_bad_values_rejected(self):
        self.engine.configure(SETTINGS)
        with self.assertRaises(ValueError): self.engine.configure(dict(SETTINGS,max_contracts=1.5))
        with self.assertRaises(ValueError): self.engine.configure(dict(SETTINGS,risk_per_trade_inr=float('nan')))
    def test_live_permission_required(self):
        with patch.dict(os.environ,{'CRT_LIVE_ENABLED':'0'}):
            with self.assertRaises(ValueError): self.engine.arm(True)
        self.engine.arm(False)


class RouteTests(unittest.TestCase):
    def test_auth_and_legacy_routes(self):
        import serve_dashboard as app
        server=ThreadingHTTPServer(('127.0.0.1',0),app.DashboardHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with patch.object(app,'PUBLIC_MODE',True):
                with self.assertRaises(HTTPError) as err: urlopen(base+'/api/crt')
                self.assertEqual(err.exception.code,401)
            with patch.object(app,'PUBLIC_MODE',False):
                with self.assertRaises(HTTPError) as err:
                    urlopen(Request(base+'/api/crt/toggle',data=b'{"enabled":true}',headers={'Content-Type':'application/json'}))
                self.assertEqual(err.exception.code,403)
                with self.assertRaises(HTTPError) as err: urlopen(base+'/api/gtrxl/signal')
                self.assertEqual(err.exception.code,404)
                self.assertEqual(json.load(urlopen(base+'/health'))['strategy'],'CRT_15M')
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__': unittest.main()
