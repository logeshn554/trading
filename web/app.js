const byId = id => document.getElementById(id);
const numberFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 6});
const moneyFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});

let latestData = null;
let selectedLots = 1;
let lockoutInterval = null;
let busy = false;

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
  if (el) {
    el.textContent = message;
    el.className = `notice ${kind}`;
  }
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

/* ==================== OTP AUTHENTICATION & LOCKOUT ==================== */

function showOtpModal() {
  const modal = byId('otp-modal');
  if (modal) modal.style.display = 'flex';
  const pinInput = byId('otp-pin');
  if (pinInput && !pinInput.disabled) {
    pinInput.focus();
  }
}

function hideOtpModal() {
  const modal = byId('otp-modal');
  if (modal) modal.style.display = 'none';
  const alertEl = byId('otp-alert');
  if (alertEl) alertEl.style.display = 'none';
}

function startLockoutTimer(seconds) {
  if (lockoutInterval) clearInterval(lockoutInterval);
  let remaining = Math.max(1, Math.round(seconds));

  const pinInput = byId('otp-pin');
  const submitBtn = byId('otp-submit-btn');
  const timerBadge = byId('lockout-timer');
  const timerSecs = byId('lockout-seconds');
  const alertEl = byId('otp-alert');

  if (pinInput) pinInput.disabled = true;
  if (submitBtn) submitBtn.disabled = true;
  if (timerBadge) timerBadge.style.display = 'flex';
  if (timerSecs) timerSecs.textContent = remaining;
  if (alertEl) {
    alertEl.textContent = '❌ Incorrect PIN entered. Security cooldown active: Please wait 30 seconds.';
    alertEl.style.display = 'block';
  }

  lockoutInterval = setInterval(() => {
    remaining -= 1;
    if (timerSecs) timerSecs.textContent = remaining;

    if (remaining <= 0) {
      clearInterval(lockoutInterval);
      lockoutInterval = null;
      if (pinInput) {
        pinInput.disabled = false;
        pinInput.value = '';
        pinInput.focus();
      }
      if (submitBtn) submitBtn.disabled = false;
      if (timerBadge) timerBadge.style.display = 'none';
      if (alertEl) {
        alertEl.textContent = 'Cooldown finished. You may enter your 6-digit PIN now.';
        alertEl.style.display = 'block';
      }
    }
  }, 1000);
}

async function checkOtpStatus() {
  try {
    const res = await fetch('/api/auth/otp-status', {cache: 'no-store'});
    const data = await res.json();
    if (!data.authenticated) {
      showOtpModal();
      if (data.lockout_remaining > 0) {
        startLockoutTimer(data.lockout_remaining);
      }
      return false;
    } else {
      hideOtpModal();
      return true;
    }
  } catch (err) {
    showOtpModal();
    return false;
  }
}

const otpForm = byId('otp-form');
if (otpForm) {
  otpForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const pinInput = byId('otp-pin');
    const submitBtn = byId('otp-submit-btn');
    const alertEl = byId('otp-alert');
    const dialog = document.querySelector('.otp-dialog');
    const pin = pinInput ? pinInput.value.trim() : '';

    if (!pin || pin.length !== 6) {
      if (alertEl) {
        alertEl.textContent = 'Please enter all 6 digits of your PIN.';
        alertEl.style.display = 'block';
      }
      return;
    }

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Verifying…';
    }

    try {
      const res = await fetch('/api/auth/verify-otp', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({otp: pin})
      });
      const data = await res.json();

      if (res.ok && data.success) {
        hideOtpModal();
        if (pinInput) pinInput.value = '';
        refresh(true);
      } else {
        if (dialog) {
          dialog.classList.remove('shake');
          void dialog.offsetWidth; // re-flow
          dialog.classList.add('shake');
        }
        const lockout = data.lockout_remaining || 30;
        startLockoutTimer(lockout);
      }
    } catch (err) {
      if (alertEl) {
        alertEl.textContent = 'Network error while verifying OTP: ' + err.message;
        alertEl.style.display = 'block';
      }
    } finally {
      if (submitBtn && (!lockoutInterval)) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Unlock Terminal';
      }
    }
  });
}

const lockBtn = byId('lock-btn');
if (lockBtn) {
  lockBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/auth/logout', {method: 'POST'});
    } catch (e) {}
    showOtpModal();
    const pinInput = byId('otp-pin');
    if (pinInput && !pinInput.disabled) {
      pinInput.value = '';
      pinInput.focus();
    }
  });
}

/* ==================== LOT SIZING & BALANCE VALIDATION ==================== */

function updateLots(newLotCount) {
  const maxLots = (latestData?.strategy?.risk_limits?.contract_size) || 100;
  selectedLots = Math.max(1, Math.min(maxLots, parseInt(newLotCount, 10) || 1));

  const lotInput = byId('lot-size-input');
  if (lotInput && parseInt(lotInput.value, 10) !== selectedLots) {
    lotInput.value = selectedLots;
  }

  // Update preset chips
  document.querySelectorAll('.preset-chip').forEach(chip => {
    const chipLots = parseInt(chip.dataset.lots, 10);
    if (chipLots === selectedLots) {
      chip.classList.add('active');
    } else {
      chip.classList.remove('active');
    }
  });

  // Re-calculate margins and check balance
  recalculateSizingAndBalance();
}

function recalculateSizingAndBalance() {
  if (!latestData) return;

  const ticker = latestData.ticker?.result || latestData.ticker || {};
  const markPrice = Number(field(ticker, 'mark_price', 'close', 'last_price', 'price')) || 2600;

  // Delta India ETHUSD: 1 lot = 0.001 ETH
  const ethPerLot = 0.001;
  const totalEth = selectedLots * ethPerLot;
  const notionalUsd = totalEth * markPrice;
  const usdToInr = 87.0; // Current approximate USD/INR conversion rate
  const notionalInr = notionalUsd * usdToInr;

  // 10x leverage margin requirement (~10%)
  const leverage = 10;
  const reqMarginInr = notionalInr / leverage;
  const reqMarginUsd = notionalUsd / leverage;

  // Extract available balances
  const wallets = Array.isArray(latestData.wallets) ? latestData.wallets : [];
  let availInr = 0;
  let availUsd = 0;
  let preferredAsset = 'INR';

  wallets.forEach(w => {
    const symbol = String(w.asset_symbol || '').toUpperCase();
    const ab = Number(w.available_balance) || 0;
    if (symbol === 'INR') availInr += ab;
    if (symbol === 'USDT' || symbol === 'USD') availUsd += ab;
  });

  let availBal = 0;
  let reqMargin = 0;
  if (availInr > 0 || (availUsd === 0 && wallets.length > 0)) {
    availBal = availInr;
    reqMargin = reqMarginInr;
    preferredAsset = 'INR';
  } else {
    availBal = availUsd;
    reqMargin = reqMarginUsd;
    preferredAsset = 'USDT';
  }

  // Update DOM labels
  const lotsDisp = byId('disp-selected-lots');
  if (lotsDisp) lotsDisp.textContent = `${selectedLots} Lot${selectedLots > 1 ? 's' : ''} (${totalEth.toFixed(3)} ETH)`;

  const notionalDisp = byId('disp-notional-val');
  if (notionalDisp) notionalDisp.textContent = `Notional: ~₹${moneyFmt.format(notionalInr)}`;

  const reqMarginDisp = byId('disp-req-margin');
  if (reqMarginDisp) reqMarginDisp.textContent = `~₹${moneyFmt.format(reqMarginInr)} INR`;

  const availBalDisp = byId('disp-avail-balance');
  if (availBalDisp) availBalDisp.textContent = `${moneyFmt.format(availBal)} ${preferredAsset}`;

  const availNote = byId('disp-avail-note');
  if (availNote) availNote.textContent = wallets.length ? `${wallets.length} active wallet asset(s)` : 'No wallet assets found';

  document.querySelectorAll('.dyn-lot-label').forEach(el => {
    el.textContent = selectedLots;
  });

  // Balance Sufficiency Evaluation
  const isLive = Boolean(latestData.strategy?.live_orders_enabled);
  const isFunded = availBal > 0;
  const isSufficient = isFunded && (availBal >= reqMargin);

  const statusBadge = byId('balance-status-badge');
  const buyingStatus = byId('disp-buying-status');
  const buyingNote = byId('disp-buying-note');
  const warningBanner = byId('balance-warning-alert');
  const warningText = byId('balance-warning-text');
  const buyBtn = byId('buy-order-btn');
  const sellBtn = byId('sell-order-btn');

  if (isSufficient) {
    if (statusBadge) {
      statusBadge.className = 'badge-sufficient';
      statusBadge.textContent = '🟢 Sufficient Balance';
    }
    if (buyingStatus) {
      buyingStatus.className = 'positive';
      buyingStatus.textContent = 'Sufficient';
    }
    if (buyingNote) buyingNote.textContent = 'Ready to execute live';
    if (warningBanner) warningBanner.style.display = 'none';

    if (buyBtn) {
      buyBtn.disabled = !isLive;
      buyBtn.title = isLive ? `Buy / Long ${selectedLots} lot(s)` : 'Live trading is paused';
    }
    if (sellBtn) {
      sellBtn.disabled = !isLive;
      sellBtn.title = isLive ? `Sell / Short ${selectedLots} lot(s)` : 'Live trading is paused';
    }
  } else {
    // Insufficient Balance
    if (statusBadge) {
      statusBadge.className = 'badge-insufficient';
      statusBadge.textContent = '🔴 Insufficient Balance';
    }
    if (buyingStatus) {
      buyingStatus.className = 'negative';
      buyingStatus.textContent = 'Insufficient';
    }
    if (buyingNote) {
      buyingNote.textContent = `Need at least ~₹${moneyFmt.format(reqMarginInr)}`;
    }
    if (warningBanner) {
      warningBanner.style.display = 'block';
      if (warningText) {
        if (!isFunded) {
          warningText.textContent = `Your available balance is ₹0.00. Cannot place orders for ${selectedLots} lot(s). Please deposit funds to Delta India first.`;
        } else {
          warningText.textContent = `Insufficient balance: Placing ${selectedLots} lot(s) requires ~₹${moneyFmt.format(reqMarginInr)} margin, but your available balance is only ${moneyFmt.format(availBal)} ${preferredAsset}. Please deposit funds or reduce lot count.`;
        }
      }
    }

    if (buyBtn) {
      buyBtn.disabled = true;
      buyBtn.title = 'Cannot buy: Insufficient balance';
    }
    if (sellBtn) {
      sellBtn.disabled = true;
      sellBtn.title = 'Cannot sell: Insufficient balance';
    }
  }
}

// Stepper and Preset listeners
const decBtn = byId('lot-dec-btn');
if (decBtn) decBtn.addEventListener('click', () => updateLots(selectedLots - 1));

const incBtn = byId('lot-inc-btn');
if (incBtn) incBtn.addEventListener('click', () => updateLots(selectedLots + 1));

const lotInput = byId('lot-size-input');
if (lotInput) {
  lotInput.addEventListener('change', () => updateLots(lotInput.value));
  lotInput.addEventListener('input', () => updateLots(lotInput.value));
}

document.querySelectorAll('.preset-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    const lots = parseInt(btn.dataset.lots, 10);
    if (lots) updateLots(lots);
  });
});

/* ==================== DASHBOARD RENDERING ==================== */

function render(data) {
  latestData = data;
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
  switchBtn.textContent = isLive ? 'LIVE ON · ACTIVE' : 'LIVE OFF · PAUSED';
  switchBtn.className = isLive ? 'live-active-btn' : 'live-off-btn';
  switchBtn.disabled = false;

  const controlNote = byId('control-note');
  if (controlNote) {
    controlNote.textContent = isLive
      ? 'Live order execution is ACTIVE. Delta Trading Key is enabled with conservative sizing and strict INR risk limits.'
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

  // Update lot sizing and sufficient balance checks
  recalculateSizingAndBalance();
}

async function refresh(fresh = false) {
  if (busy) return;
  busy = true;
  const button = byId('refresh');
  button.disabled = true;
  button.textContent = 'Refreshing…';
  try {
    const response = await fetch(`/api/snapshot${fresh ? '?fresh=1' : ''}`, {cache: 'no-store'});
    if (response.status === 401) {
      const body = await response.json().catch(() => ({}));
      if (body.requires_otp) {
        showOtpModal();
      } else {
        window.location.assign('/auth/login');
      }
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

// Live Trading Toggle Switch
const liveSwitch = byId('live-switch');
if (liveSwitch) {
  liveSwitch.addEventListener('click', async () => {
    liveSwitch.disabled = true;
    try {
      const res = await fetch('/api/trade/toggle', {method: 'POST'});
      const data = await res.json();
      if (data.requires_otp) {
        showOtpModal();
        return;
      }
      refresh(true);
    } catch (err) {
      alert('Failed to toggle live trading: ' + err.message);
    } finally {
      liveSwitch.disabled = false;
    }
  });
}

// Order Submission Handler
async function submitTradeOrder(side) {
  const isLive = Boolean(latestData?.strategy?.live_orders_enabled);
  if (!isLive) {
    alert('Live Trading is currently OFF. Please turn ON Live Trading first.');
    return;
  }

  const promptMsg = `Confirm Market ${side.toUpperCase()} Order:\n\nInstrument: ETHUSD (Delta India)\nLots: ${selectedLots} Contract(s)\nSide: ${side.toUpperCase()}\n\nDo you want to submit this live order now?`;
  if (!confirm(promptMsg)) return;

  const buyBtn = byId('buy-order-btn');
  const sellBtn = byId('sell-order-btn');
  if (buyBtn) buyBtn.disabled = true;
  if (sellBtn) sellBtn.disabled = true;

  try {
    const res = await fetch('/api/trade/order', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({side, size: selectedLots})
    });
    const data = await res.json();

    if (data.requires_otp) {
      showOtpModal();
      return;
    }

    if (res.ok && data.status === 'success') {
      alert(`Success! Market ${side.toUpperCase()} order for ${selectedLots} lot(s) placed on Delta India.`);
      refresh(true);
    } else {
      alert(`Order Failed: ${data.error || JSON.stringify(data)}`);
      refresh(true);
    }
  } catch (err) {
    alert(`Order execution error: ${err.message}`);
  } finally {
    recalculateSizingAndBalance();
  }
}

const buyOrderBtn = byId('buy-order-btn');
if (buyOrderBtn) {
  buyOrderBtn.addEventListener('click', () => submitTradeOrder('buy'));
}

const sellOrderBtn = byId('sell-order-btn');
if (sellOrderBtn) {
  sellOrderBtn.addEventListener('click', () => submitTradeOrder('sell'));
}

// Initial checks & poll loops
checkOtpStatus().then(authed => {
  if (authed) {
    refresh();
  }
});

fetch('/api/my-ip')
  .then(r => r.json())
  .then(d => { if (d.outbound_ip) updateIpDisplay(d.outbound_ip); })
  .catch(() => {});

setInterval(() => {
  if (!byId('otp-modal') || byId('otp-modal').style.display === 'none') {
    refresh();
  }
}, 30000);
