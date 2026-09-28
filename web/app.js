const byId = id => document.getElementById(id);
const numberFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 6});
const moneyFmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});

let latestData = null;
let selectedLots = 1;
let lockoutInterval = null;
let busy = false;
let limitsUserDirty = false;

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
  if (!body) return;
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

/* ==================== OTP AUTHENTICATION & LOCKOUT (ZERO INLINE STYLES) ==================== */

function showOtpModal() {
  const modal = byId('otp-modal');
  if (modal) {
    modal.classList.remove('hidden');
  }
  const pinInput = byId('otp-pin');
  if (pinInput && !pinInput.disabled) {
    setTimeout(() => pinInput.focus(), 100);
  }
}

function hideOtpModal() {
  const modal = byId('otp-modal');
  if (modal) {
    modal.classList.add('hidden');
  }
  const alertEl = byId('otp-alert');
  if (alertEl) {
    alertEl.classList.add('hidden');
  }
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
  if (timerBadge) timerBadge.classList.remove('hidden');
  if (timerSecs) timerSecs.textContent = remaining;
  if (alertEl) {
    alertEl.textContent = '❌ Incorrect PIN entered. Security cooldown active: Please wait 30 seconds.';
    alertEl.classList.remove('hidden');
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
      if (timerBadge) timerBadge.classList.add('hidden');
      if (alertEl) {
        alertEl.textContent = 'Cooldown finished. You may enter your 6-digit PIN now.';
        alertEl.classList.remove('hidden');
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
        alertEl.classList.remove('hidden');
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
        alertEl.classList.remove('hidden');
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

/* ==================== ALGO LOT SIZING & BALANCE VALIDATION ==================== */

function updateLots(newLotCount, fromUser = false) {
  const parsed = parseInt(newLotCount, 10);
  selectedLots = Math.max(1, Math.min(100, isNaN(parsed) ? 1 : parsed));

  const lotInput = byId('lot-size-input');
  if (lotInput && parseInt(lotInput.value, 10) !== selectedLots) {
    lotInput.value = selectedLots;
  }

  const algoLotInput = byId('algo-lot-size');
  if (algoLotInput && parseInt(algoLotInput.value, 10) !== selectedLots) {
    algoLotInput.value = selectedLots;
  }

  if (fromUser) {
    limitsUserDirty = true;
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
  const usdToInr = 87.0; // Approximate USD/INR conversion rate
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

  // Balance Sufficiency Evaluation
  const isFunded = availBal > 0;
  const isSufficient = isFunded && (availBal >= reqMargin);

  const statusBadge = byId('balance-status-badge');
  const buyingStatus = byId('disp-buying-status');
  const buyingNote = byId('disp-buying-note');
  const warningBanner = byId('balance-warning-alert');
  const warningText = byId('balance-warning-text');

  if (isSufficient) {
    if (statusBadge) {
      statusBadge.className = 'badge-sufficient';
      statusBadge.textContent = '🟢 Sufficient Balance';
    }
    if (buyingStatus) {
      buyingStatus.className = 'positive';
      buyingStatus.textContent = 'Sufficient';
    }
    if (buyingNote) buyingNote.textContent = 'Ready for auto-orders';
    if (warningBanner) {
      warningBanner.classList.add('hidden');
    }
  } else {
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
      warningBanner.classList.remove('hidden');
      if (warningText) {
        if (!isFunded) {
          warningText.textContent = `Your available balance is ₹0.00. Cannot place automated orders for ${selectedLots} lot(s). Please deposit funds to Delta India first.`;
        } else {
          warningText.textContent = `Insufficient balance: ${selectedLots} lot(s) require ~₹${moneyFmt.format(reqMarginInr)} margin, but available balance is only ${moneyFmt.format(availBal)} ${preferredAsset}. Deposit funds or reduce lot size.`;
        }
      }
    }
  }
}

// Stepper and Preset listeners
const decBtn = byId('lot-dec-btn');
if (decBtn) decBtn.addEventListener('click', () => updateLots(selectedLots - 1, true));

const incBtn = byId('lot-inc-btn');
if (incBtn) incBtn.addEventListener('click', () => updateLots(selectedLots + 1, true));

const lotInput = byId('lot-size-input');
if (lotInput) {
  lotInput.addEventListener('change', () => updateLots(lotInput.value, true));
  lotInput.addEventListener('input', () => updateLots(lotInput.value, true));
}

const algoLotInput = byId('algo-lot-size');
if (algoLotInput) {
  algoLotInput.addEventListener('change', () => updateLots(algoLotInput.value, true));
  algoLotInput.addEventListener('input', () => updateLots(algoLotInput.value, true));
}

document.querySelectorAll('.preset-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    const lots = parseInt(btn.dataset.lots, 10);
    if (lots) updateLots(lots, true);
  });
});

// Mark dirty when user types into any limit input so background polling doesn't overwrite
['max-trades', 'algo-lot-size', 'profit-target', 'daily-loss', 'stop-loss', 'take-profit'].forEach(id => {
  const el = byId(id);
  if (el) {
    el.addEventListener('input', () => { limitsUserDirty = true; });
  }
});

/* ==================== DASHBOARD RENDERING ==================== */

function render(data) {
  latestData = data;
  const connected = data.connection === 'connected';
  const signOutEl = byId('sign-out');
  if (signOutEl) signOutEl.hidden = !data.public_mode;

  const envEl = byId('environment');
  if (envEl) envEl.textContent = data.environment === 'india_testnet' ? 'INDIA TESTNET' : 'INDIA PRODUCTION';

  const connEl = byId('connection');
  if (connEl) {
    connEl.textContent = connected ? 'ACCOUNT CONNECTED' : 'ACCOUNT NOT CONNECTED';
    connEl.className = `connection ${connected ? 'connected' : 'disconnected'}`;
  }

  const asOfEl = byId('as-of');
  if (asOfEl) asOfEl.textContent = data.as_of ? `Last checked ${time(data.as_of)}` : 'Account data unavailable';

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
    setNotice('Live Delta account data received. Algorithmic trade execution active.', 'ok');
  }

  const ticker = data.ticker?.result || data.ticker || {};
  const ethPrice = byId('eth-price');
  if (ethPrice) ethPrice.textContent = num(field(ticker, 'mark_price', 'close', 'last_price', 'price'), 2);

  const marketNote = byId('market-note');
  if (marketNote) marketNote.textContent = data.errors?.ticker || 'Delta ETHUSD market data';

  const wallets = Array.isArray(data.wallets) ? data.wallets : [];
  const preferred = wallets.find(w => ['INR', 'USDT', 'USD'].includes(w.asset_symbol) && Number(w.balance) !== 0) || wallets[0];
  const balanceEl = byId('balance');
  if (balanceEl) balanceEl.textContent = preferred ? `${num(preferred.balance, 2)} ${preferred.asset_symbol || ''}` : '—';

  const availEl = byId('available');
  if (availEl) availEl.textContent = preferred ? `${num(preferred.available_balance, 2)} ${preferred.asset_symbol || ''}` : '—';

  const balNote = byId('balance-note');
  if (balNote) balNote.textContent = preferred ? 'Selected wallet · all assets below' : 'No wallet data';

  const availNote = byId('available-note');
  if (availNote) availNote.textContent = preferred ? 'Selected wallet · all assets below' : 'No wallet data';

  const realizedEl = byId('realized');
  if (realizedEl) realizedEl.textContent = pnlMap(data.realized_pnl_open_positions);

  const unrealizedEl = byId('unrealized');
  if (unrealizedEl) unrealizedEl.textContent = pnlMap(data.unrealized_pnl_open_positions);

  const walletCount = byId('wallet-count');
  if (walletCount) walletCount.textContent = connected ? `${wallets.length} asset${wallets.length === 1 ? '' : 's'}` : '—';

  renderRows('wallet-rows', wallets, [
    r => field(r, 'asset_symbol'), r => num(r.balance), r => num(r.available_balance),
    r => num(r.position_margin), r => num(r.order_margin), r => num(r.strategy_blocked_amount),
  ], data.errors?.wallets || (connected ? 'No wallet assets returned' : 'Connect a Read Data key'));

  const positions = Array.isArray(data.positions) ? data.positions : [];
  const posCount = byId('position-count');
  if (posCount) posCount.textContent = connected ? `${positions.length} open` : '—';

  renderRows('position-rows', positions, [
    product, r => num(r.size), r => num(r.entry_price, 2), r => num(r.mark_price, 2),
    r => num(r.liquidation_price, 2), r => `${num(r.margin)} ${settlementAsset(r)}`,
    r => `${num(r.realized_pnl, 2)} ${settlementAsset(r)}`, r => `${num(r.unrealized_pnl, 2)} ${settlementAsset(r)}`,
    r => num(r.realized_funding),
  ], data.errors?.positions || (connected ? 'No open positions' : 'Connect a Read Data key'));

  const fills = Array.isArray(data.fills) ? data.fills : [];
  const fillCount = byId('fill-count');
  if (fillCount) fillCount.textContent = connected ? `${fills.length} shown` : '—';

  renderRows('fill-rows', fills, [
    r => time(r.created_at), product, r => field(r, 'side'), r => num(r.size),
    r => num(r.price, 2), r => num(r.commission), r => field(r, 'settling_asset_symbol'),
  ], data.errors?.fills || (connected ? 'No fills in this window' : 'Connect a Read Data key'));

  const fillsNote = byId('fills-note');
  if (fillsNote) fillsNote.textContent = !connected ? 'Connect a Read Data key to view fills.' : data.fills_after ? 'Showing the first 100 of the last 30 days. Additional pages exist on Delta.' : 'Last 30 days; no additional page reported by Delta.';

  const txs = Array.isArray(data.transactions) ? data.transactions : [];
  const txCount = byId('transaction-count');
  if (txCount) txCount.textContent = connected ? `${txs.length} shown` : '—';

  renderRows('transaction-rows', txs, [
    r => time(r.created_at), r => field(r, 'asset_symbol'), r => field(r, 'transaction_type'),
    r => num(r.amount), r => num(r.balance),
  ], data.errors?.transactions || (connected ? 'No wallet transactions in this window' : 'Connect a Read Data key'));

  const txNote = byId('transactions-note');
  if (txNote) txNote.textContent = !connected ? 'Connect a Read Data key to view wallet history.' : data.transactions_after ? 'Showing the first 100 of the last 30 days. Additional pages exist on Delta.' : 'Last 30 days; transaction types remain separate.';

  const orders = Array.isArray(data.open_orders) ? data.open_orders : [];
  const orderCount = byId('order-count');
  if (orderCount) orderCount.textContent = connected ? `${orders.length} open / pending` : '—';

  renderRows('order-rows', orders, [
    product, r => field(r, 'side'), r => field(r, 'order_type'), r => num(r.size),
    r => num(field(r, 'limit_price', 'stop_price'), 2), r => field(r, 'state'), r => field(r, 'id', 'client_order_id'),
  ], data.errors?.open_orders || (connected ? 'No open orders' : 'Connect a Read Data key'));

  const strategy = data.strategy || {};
  const stratName = byId('strategy-name');
  if (stratName) stratName.textContent = strategy.id || 'Primary strategy';

  const stratReason = byId('strategy-reason');
  if (stratReason) stratReason.textContent = strategy.reason || 'Algorithmic Execution Active';

  const isLive = Boolean(strategy.live_orders_enabled);
  const stateEl = byId('strategy-state');
  if (stateEl) {
    stateEl.textContent = isLive ? 'VALIDATED · LIVE ACTIVE' : (strategy.validation || 'BLOCKED');
    stateEl.className = isLive ? 'ready-pill' : 'blocked-pill';
  }

  // Populate risk limits form only if user has not modified inputs
  const limits = strategy.risk_limits || {};
  const serverLot = limits.contract_size || 1;

  if (!limitsUserDirty) {
    const setVal = (id, val) => {
      const el = byId(id);
      if (el && val !== undefined && val !== null) {
        el.value = val;
      }
    };
    setVal('max-trades', limits.max_trades_per_day ?? 5);
    setVal('algo-lot-size', serverLot);
    setVal('lot-size-input', serverLot);
    setVal('profit-target', limits.daily_net_profit_target ?? 1200);
    setVal('daily-loss', limits.daily_max_loss ?? 1000);
    setVal('stop-loss', limits.per_trade_stop_loss ?? 300);
    setVal('take-profit', limits.per_trade_take_profit ?? 600);
    updateLots(serverLot, false);
  }

  const switchBtn = byId('live-switch');
  if (switchBtn) {
    switchBtn.textContent = isLive ? 'ALGO LIVE · ACTIVE' : 'ALGO PAUSED · OFF';
    switchBtn.className = isLive ? 'live-active-btn' : 'live-off-btn';
    switchBtn.disabled = false;
  }

  const algoStatus = byId('algo-run-status');
  if (algoStatus) {
    algoStatus.textContent = isLive ? 'AUTO-TRADING ACTIVE' : 'AUTO-TRADING PAUSED';
    algoStatus.className = isLive ? 'ready-pill' : 'blocked-pill';
  }

  const controlNote = byId('control-note');
  if (controlNote) {
    controlNote.textContent = isLive
      ? 'Algorithmic execution is ACTIVE. Automated signals execute on Delta India within strict INR risk limits.'
      : 'Algorithmic execution is currently PAUSED. Click button above to resume automated trading.';
  }

  const blockers = strategy.trading_readiness?.blockers || [];
  const blockerList = byId('live-blockers');
  if (blockerList) {
    blockerList.replaceChildren();
    for (const reason of blockers) {
      const item = document.createElement('li');
      item.textContent = reason;
      blockerList.append(item);
    }
  }

  recalculateSizingAndBalance();
  if (data.gtrxl_trader) {
    renderGTrXL(data.gtrxl_trader);
  }
}

function renderGTrXL(trader) {
  if (!trader) return;
  const signalEl = byId('gtrxl-signal');
  const confEl = byId('gtrxl-signal-conf');
  const probsEl = byId('gtrxl-probs');
  const valEl = byId('gtrxl-val');
  const memEl = byId('gtrxl-mem');
  const agentStatusEl = byId('gtrxl-agent-status');
  const lastStepEl = byId('gtrxl-last-step');
  const tradesCountEl = byId('gtrxl-trades-count');
  const logsEl = byId('gtrxl-logs');

  const signal = trader.last_signal || 'HOLD';
  if (signalEl) {
    signalEl.textContent = signal;
    signalEl.className = signal === 'BUY' ? 'signal-buy' : (signal === 'SELL' ? 'signal-sell' : 'signal-hold');
  }

  if (confEl) {
    const conf = trader.confidence ? (Number(trader.confidence) * 100).toFixed(1) : '—';
    confEl.textContent = `Confidence: ${conf}%`;
  }

  if (probsEl && Array.isArray(trader.action_probs) && trader.action_probs.length === 3) {
    const [h, l, s] = trader.action_probs.map(p => Math.round(Number(p) * 100));
    probsEl.textContent = `H: ${h}% · L: ${l}% · S: ${s}%`;
  }

  if (valEl) {
    const v = Number(trader.value_estimate) || 0;
    valEl.textContent = `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
    valEl.className = v >= 0 ? 'positive' : 'negative';
  }

  if (memEl) {
    memEl.textContent = `${trader.memory_bars || 0} Bars`;
  }

  if (agentStatusEl) {
    const status = trader.status || 'ONLINE';
    agentStatusEl.textContent = `GTrXL ${status}`;
    agentStatusEl.className = (status === 'RUNNING' || status === 'WARMED_UP') ? 'ready-pill' : 'blocked-pill';
  }

  if (lastStepEl && trader.last_evaluation_time) {
    lastStepEl.innerHTML = `<strong>Last Step:</strong> ${time(trader.last_evaluation_time)}`;
  }

  if (tradesCountEl) {
    tradesCountEl.textContent = `Trades today: ${trader.trades_today || 0}`;
  }

  if (logsEl && Array.isArray(trader.recent_logs) && trader.recent_logs.length > 0) {
    logsEl.replaceChildren();
    for (const logLine of trader.recent_logs) {
      const lineDiv = document.createElement('div');
      lineDiv.className = 'gtrxl-log-line';
      lineDiv.textContent = logLine;
      logsEl.append(lineDiv);
    }
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

const evalBtn = byId('gtrxl-eval-btn');
if (evalBtn) {
  evalBtn.addEventListener('click', async () => {
    evalBtn.disabled = true;
    evalBtn.textContent = 'Evaluating…';
    try {
      const res = await fetch('/api/gtrxl/evaluate', { method: 'POST' });
      if (res.ok) {
        const trader = await res.json();
        renderGTrXL(trader);
      }
    } catch (e) {
      console.error('Manual evaluate failed', e);
    } finally {
      evalBtn.disabled = false;
      evalBtn.textContent = '⚡ Run AI Step';
    }
  });
}

async function refresh(fresh = false) {
  if (busy) return;
  busy = true;
  const button = byId('refresh');
  if (button) {
    button.disabled = true;
    button.textContent = 'Refreshing…';
  }
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
    if (button) {
      button.disabled = false;
      button.textContent = 'Refresh';
    }
    busy = false;
  }
}

const refreshBtn = byId('refresh');
if (refreshBtn) refreshBtn.addEventListener('click', () => refresh(true));

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

// Reset button listener
const resetLimitsBtn = byId('reset-limits-btn');
if (resetLimitsBtn) {
  resetLimitsBtn.addEventListener('click', () => {
    limitsUserDirty = false;
    if (latestData) {
      render(latestData);
    }
    const statusMsg = byId('save-limits-status');
    if (statusMsg) {
      statusMsg.textContent = '↺ Reset to current saved limits.';
      statusMsg.className = 'save-status-msg';
      setTimeout(() => { if (statusMsg) statusMsg.textContent = ''; }, 3000);
    }
  });
}

// Save Algorithmic Risk Limits
const saveLimitsBtn = byId('save-limits-btn');
if (saveLimitsBtn) {
  saveLimitsBtn.addEventListener('click', async () => {
    const statusMsg = byId('save-limits-status');
    const maxTrades = parseInt(byId('max-trades')?.value, 10);
    const lotSize = parseInt(byId('algo-lot-size')?.value || byId('lot-size-input')?.value, 10);
    const profitTarget = parseFloat(byId('profit-target')?.value);
    const dailyLoss = parseFloat(byId('daily-loss')?.value);
    const stopLoss = parseFloat(byId('stop-loss')?.value);
    const takeProfit = parseFloat(byId('take-profit')?.value);

    if (isNaN(maxTrades) || maxTrades < 1) {
      if (statusMsg) {
        statusMsg.textContent = '❌ Max trades per day must be at least 1.';
        statusMsg.className = 'save-status-msg err';
      }
      return;
    }

    if (isNaN(lotSize) || lotSize < 1 || lotSize > 100) {
      if (statusMsg) {
        statusMsg.textContent = '❌ Trade lot size must be between 1 and 100 contracts.';
        statusMsg.className = 'save-status-msg err';
      }
      return;
    }

    saveLimitsBtn.disabled = true;
    if (statusMsg) {
      statusMsg.textContent = 'Saving…';
      statusMsg.className = 'save-status-msg';
    }

    try {
      const res = await fetch('/api/strategy/risk_limits', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          max_trades_per_day: maxTrades,
          contract_size: lotSize,
          daily_net_profit_target: isNaN(profitTarget) ? 1200 : profitTarget,
          daily_max_loss: isNaN(dailyLoss) ? 1000 : dailyLoss,
          per_trade_stop_loss: isNaN(stopLoss) ? 300 : stopLoss,
          per_trade_take_profit: isNaN(takeProfit) ? 600 : takeProfit
        })
      });
      const data = await res.json();
      if (data.requires_otp) {
        showOtpModal();
        if (statusMsg) statusMsg.textContent = '';
        return;
      }
      if (res.ok && data.status === 'success') {
        limitsUserDirty = false;
        if (statusMsg) {
          statusMsg.textContent = '✅ Algo risk limits & lot size saved!';
          statusMsg.className = 'save-status-msg ok';
          setTimeout(() => { if (statusMsg) statusMsg.textContent = ''; }, 4000);
        }
        refresh(true);
      } else {
        if (statusMsg) {
          statusMsg.textContent = `❌ ${data.error || 'Failed to save risk limits.'}`;
          statusMsg.className = 'save-status-msg err';
        }
      }
    } catch (err) {
      if (statusMsg) {
        statusMsg.textContent = `❌ Network error: ${err.message}`;
        statusMsg.className = 'save-status-msg err';
      }
    } finally {
      saveLimitsBtn.disabled = false;
    }
  });
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
  const modal = byId('otp-modal');
  if (!modal || modal.classList.contains('hidden')) {
    refresh();
  }
}, 30000);
