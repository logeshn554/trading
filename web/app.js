const byId = id => document.getElementById(id);
const numberFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 6});
const moneyFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});

function num(value, digits = 6) {
  if (value === null || value === undefined || value === '') return '—';
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return (digits === 2 ? moneyFmt : numberFmt).format(n);
}

function time(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

function field(row, ...keys) {
  for (const key of keys) {
    if (row && row[key] !== null && row[key] !== undefined && row[key] !== '') return row[key];
  }
  return null;
}

function renderRows(id, rows, columns, emptyText) {
  const body = byId(id);
  body.replaceChildren();
  if (!Array.isArray(rows) || rows.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = columns.length;
    td.className = 'empty';
    td.textContent = emptyText;
    tr.append(td);
    body.append(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const getValue of columns) {
      const td = document.createElement('td');
      const value = getValue(row);
      td.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
      if (/^[-+]?\d/.test(td.textContent) && td.textContent.startsWith('-')) td.classList.add('negative');
      tr.append(td);
    }
    body.append(tr);
  }
}

function pnlMap(map) {
  const entries = Object.entries(map || {});
  return entries.length ? entries.map(([asset, value]) => `${num(value, 2)} ${asset}`).join(' · ') : '—';
}

function product(row) {
  return field(row, 'product_symbol', 'symbol') || row.product?.symbol || (row.product_id ? `#${row.product_id}` : '—');
}

function settlementAsset(row) {
  return field(row, 'settling_asset_symbol') || row.product?.settling_asset?.symbol || '—';
}

function setNotice(message, kind = 'warn') {
  const el = byId('notice');
  el.textContent = message;
  el.className = `notice ${kind}`;
}

function updateIpDisplay(ip) {
  if (!ip || ip === 'Unavailable') return;
  const pill = byId('server-ip');
  if (pill) {
    pill.textContent = `IP: ${ip}`;
    pill.title = `Click to copy ${ip}`;
  }
  const display = byId('ip-address-display');
  if (display) {
    display.textContent = ip;
  }
}

function render(data) {
  const connected = data.connection === 'connected';
  byId('sign-out').hidden = !data.public_mode;
  byId('environment').textContent = data.environment === 'india_testnet' ? 'INDIA TESTNET' : 'INDIA PRODUCTION';
  byId('connection').textContent = connected ? 'ACCOUNT CONNECTED' : 'ACCOUNT NOT CONNECTED';
  byId('connection').className = `connection ${connected ? 'connected' : 'disconnected'}`;
  byId('as-of').textContent = data.as_of ? `Last checked ${time(data.as_of)}` : 'Account data unavailable';

  if (data.outbound_ip) {
    updateIpDisplay(data.outbound_ip);
  }

  if (data.error) {
    setNotice(data.error, 'error');
  } else if (data.connection === 'needs_read_data_key') {
    const ipMsg = data.outbound_ip && data.outbound_ip !== 'Unavailable' ? ` (Server IP: ${data.outbound_ip})` : '';
    setNotice(`Delta MCP is running. Connect a Delta API key with Read Data permission${ipMsg}.`, 'warn');
  } else if (!connected) {
    setNotice('Delta MCP account connection is unavailable.', 'error');
  } else if (Object.keys(data.errors || {}).length) {
    setNotice(`Connected with incomplete data: ${Object.entries(data.errors).map(([key, value]) => `${key}: ${value}`).join('; ')}`, 'warn');
  } else {
    setNotice('Live Delta account data received. No paper balances or simulated P&L are shown.', 'ok');
  }

  const ticker = data.ticker?.result || data.ticker || {};
  byId('eth-price').textContent = num(field(ticker, 'mark_price', 'close', 'last_price', 'price'), 2);
  byId('market-note').textContent = data.errors?.ticker || 'Delta ETHUSD market data';

  const wallets = Array.isArray(data.wallets) ? data.wallets : [];
  const preferred = wallets.find(w => ['INR', 'USDT', 'USD'].includes(w.asset_symbol) && Number(w.balance) !== 0) || wallets[0];
  byId('balance').textContent = preferred ? `${num(preferred.balance, 2)} ${preferred.asset_symbol || ''}` : '—';
  byId('available').textContent = preferred ? `${num(preferred.available_balance, 2)} ${preferred.asset_symbol || ''}` : '—';
  byId('balance-note').textContent = preferred ? 'Selected wallet · all assets below' : 'No wallet data';
  byId('available-note').textContent = preferred ? 'Selected wallet · all assets below' : 'No wallet data';
  byId('realized').textContent = pnlMap(data.realized_pnl_open_positions);
  byId('unrealized').textContent = pnlMap(data.unrealized_pnl_open_positions);

  byId('wallet-count').textContent = connected ? `${wallets.length} asset${wallets.length === 1 ? '' : 's'}` : '—';
  renderRows('wallet-rows', wallets, [
    r => field(r, 'asset_symbol'), r => num(r.balance), r => num(r.available_balance),
    r => num(r.position_margin), r => num(r.order_margin), r => num(r.strategy_blocked_amount),
  ], data.errors?.wallets || (connected ? 'No wallet assets returned' : 'Connect a Read Data key'));

  const positions = Array.isArray(data.positions) ? data.positions : [];
  byId('position-count').textContent = connected ? `${positions.length} open` : '—';
  renderRows('position-rows', positions, [
    product, r => num(r.size), r => num(r.entry_price, 2), r => num(r.mark_price, 2),
    r => num(r.liquidation_price, 2), r => `${num(r.margin)} ${settlementAsset(r)}`,
    r => `${num(r.realized_pnl, 2)} ${settlementAsset(r)}`, r => `${num(r.unrealized_pnl, 2)} ${settlementAsset(r)}`,
    r => num(r.realized_funding),
  ], data.errors?.positions || (connected ? 'No open positions' : 'Connect a Read Data key'));

  const fills = Array.isArray(data.fills) ? data.fills : [];
  byId('fill-count').textContent = connected ? `${fills.length} shown` : '—';
  renderRows('fill-rows', fills, [
    r => time(r.created_at), product, r => field(r, 'side'), r => num(r.size),
    r => num(r.price, 2), r => num(r.commission), r => field(r, 'settling_asset_symbol'),
  ], data.errors?.fills || (connected ? 'No fills in this window' : 'Connect a Read Data key'));
  byId('fills-note').textContent = !connected ? 'Connect a Read Data key to view fills.' : data.fills_after ? 'Showing the first 100 of the last 30 days. Additional pages exist on Delta.' : 'Last 30 days; no additional page reported by Delta.';

  const txs = Array.isArray(data.transactions) ? data.transactions : [];
  byId('transaction-count').textContent = connected ? `${txs.length} shown` : '—';
  renderRows('transaction-rows', txs, [
    r => time(r.created_at), r => field(r, 'asset_symbol'), r => field(r, 'transaction_type'),
    r => num(r.amount), r => num(r.balance),
  ], data.errors?.transactions || (connected ? 'No wallet transactions in this window' : 'Connect a Read Data key'));
  byId('transactions-note').textContent = !connected ? 'Connect a Read Data key to view wallet history.' : data.transactions_after ? 'Showing the first 100 of the last 30 days. Additional pages exist on Delta.' : 'Last 30 days; transaction types remain separate.';

  const orders = Array.isArray(data.open_orders) ? data.open_orders : [];
  byId('order-count').textContent = connected ? `${orders.length} open / pending` : '—';
  renderRows('order-rows', orders, [
    product, r => field(r, 'side'), r => field(r, 'order_type'), r => num(r.size),
    r => num(field(r, 'limit_price', 'stop_price'), 2), r => field(r, 'state'), r => field(r, 'id', 'client_order_id'),
  ], data.errors?.open_orders || (connected ? 'No open orders' : 'Connect a Read Data key'));

  const strategy = data.strategy || {};
  byId('strategy-name').textContent = strategy.id || 'Primary strategy';
  byId('strategy-reason').textContent = strategy.reason || 'Live Trading Configuration';
  const isLive = Boolean(strategy.live_orders_enabled);
  const stateEl = byId('strategy-state');
  stateEl.textContent = isLive ? 'VALIDATED · LIVE ACTIVE' : (strategy.validation || 'BLOCKED');
  stateEl.className = isLive ? 'ready-pill' : 'blocked-pill';

  const limits = strategy.risk_limits || {};
  const limitValue = value => value === null || value === undefined ? 'Not set' : `₹${num(value, 2)}`;
  byId('max-trades').value = limits.max_trades_per_day ? `${limits.max_trades_per_day} trades/day` : 'Not set';
  byId('profit-target').value = limitValue(limits.daily_net_profit_target);
  byId('daily-loss').value = limitValue(limits.daily_max_loss);
  byId('stop-loss').value = limitValue(limits.per_trade_stop_loss);
  byId('take-profit').value = limitValue(limits.per_trade_take_profit);

  const switchBtn = byId('live-switch');
  switchBtn.textContent = isLive ? 'LIVE ON · ACTIVE' : 'OFF · PAUSED';
  switchBtn.className = isLive ? 'live-active-btn' : 'live-off-btn';
  switchBtn.disabled = false;

  const controlNote = byId('control-note');
  if (controlNote) {
    controlNote.textContent = isLive
      ? 'Live order execution is ACTIVE. Delta Trading Key is enabled with conservative 1-contract sizing and strict INR risk limits.'
      : 'Live order execution is currently PAUSED. Click button above to resume.';
  }

  const blockers = strategy.trading_readiness?.blockers || [];
  const blockerList = byId('live-blockers');
  blockerList.replaceChildren();
  for (const reason of blockers) {
    const item = document.createElement('li');
    item.textContent = reason;
    blockerList.append(item);
  }
}

let busy = false;
async function refresh(fresh = false) {
  if (busy) return;
  busy = true;
  const button = byId('refresh');
  button.disabled = true;
  button.textContent = 'Refreshing…';
  try {
    const response = await fetch(`/api/snapshot${fresh ? '?fresh=1' : ''}`, {cache: 'no-store'});
    if (response.status === 401) {
      window.location.assign('/auth/login');
      return;
    }
    const data = await response.json();
    render(data);
  } catch (error) {
    setNotice(`Cannot reach the local dashboard: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Refresh';
    busy = false;
  }
}

byId('refresh').addEventListener('click', () => refresh(true));

function copyIp() {
  const display = byId('ip-address-display');
  const ip = display ? display.textContent.trim() : '';
  if (ip && ip !== 'Detecting IP…' && ip !== 'Unavailable') {
    navigator.clipboard.writeText(ip).then(() => {
      const btn = byId('copy-ip-btn');
      if (btn) {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy IP'; }, 2000);
      }
    });
  }
}

const copyBtn = byId('copy-ip-btn');
if (copyBtn) copyBtn.addEventListener('click', copyIp);
const ipPill = byId('server-ip');
if (ipPill) ipPill.addEventListener('click', copyIp);

const liveSwitch = byId('live-switch');
if (liveSwitch) {
  liveSwitch.addEventListener('click', async () => {
    liveSwitch.disabled = true;
    try {
      const res = await fetch('/api/trade/toggle', {method: 'POST'});
      await res.json();
      refresh(true);
    } catch (err) {
      alert('Failed to toggle live trading: ' + err.message);
    } finally {
      liveSwitch.disabled = false;
    }
  });
}

const testTradeBtn = byId('test-trade-btn');
if (testTradeBtn) {
  testTradeBtn.addEventListener('click', async () => {
    if (!confirm('Submit a 1-contract test market BUY order on Delta India ETHUSD?')) return;
    testTradeBtn.disabled = true;
    testTradeBtn.textContent = 'Submitting…';
    try {
      const res = await fetch('/api/trade/order', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({side: 'buy', size: 1})
      });
      const data = await res.json();
      if (data.status === 'success') {
        alert('Test trade submitted successfully to Delta Exchange India!');
        refresh(true);
      } else {
        alert('Delta order response: ' + (data.error || JSON.stringify(data)));
      }
    } catch (err) {
      alert('Order failed: ' + err.message);
    } finally {
      testTradeBtn.disabled = false;
      testTradeBtn.textContent = 'Test Trade (1 Contract)';
    }
  });
}

// Fetch outbound IP immediately on page load
fetch('/api/my-ip')
  .then(r => r.json())
  .then(d => { if (d.outbound_ip) updateIpDisplay(d.outbound_ip); })
  .catch(() => {});

refresh();
setInterval(() => refresh(), 30000);

