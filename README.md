# Delta ETHUSD — Deterministic 15-Minute Candle Range Theory (CRT)

A deterministic, rule-based algorithmic trading engine and local web dashboard for Delta Exchange ETHUSD perpetual futures. Built for local execution without external dependencies or mandatory cloud logins.

> **CRITICAL DISCLAIMER**:
> This software is an engineering framework for evaluating Candle Range Theory (CRT). **NO claim of strategy profitability, positive expectancy, or verified win rate is made.** Historical backtesting on Binance ETHUSDT spot proxy data showed a win rate of ~1.1% (1 win / 89 diagnostic trades) under standard transaction costs. Stop-loss orders can experience slippage or gap risk; configured INR stops are estimates, NOT guaranteed loss limits. Live trading carries real financial risk.

---

## 1. Strategy Implementation (Core Rules)

The single source of truth for strategy signals is `ethresearch/crt.py::signal`. Live trading and backtesting share the exact same function.

- **Instrument**: ETHUSD perpetual futures (`contract_unit_currency: ETH`, settlement in `USD`).
- **Timeframe**: 15-minute completed candles (`INTERVAL = 900` seconds).
- **Two-Candle Pattern**:
  - **Candle N-1 (Reference Candle)**: Establishes reference range `[reference.low, reference.high]`.
  - **Candle N (Sweep Candle)**: Must sweep exactly one boundary and close strictly inside the reference range.
- **Entry Rules**:
  - **Low Sweep (BUY / LONG)**: `sweep.low < reference.low` AND `reference.low < sweep.close < reference.high`.
  - **High Sweep (SELL / SHORT)**: `sweep.high > reference.high` AND `reference.low < sweep.close < reference.high`.
- **Exclusion Filters (NO TRADE)**:
  - **Double Sweep**: Both high and low swept on the same candle.
  - **Boundary Close**: Sweep candle closes exactly at `reference.low` or `reference.high`.
  - **Close Outside Range**: Sweep candle closes outside reference range.
  - **Missing / Non-Consecutive**: Timestamp delta between reference and sweep != 900 seconds.
  - **Expired Window**: Signal evaluation occurs > 90 seconds after sweep candle close.
  - **Forming Candle**: Current unclosed candle is never used (no lookahead bias).
- **Exit Rules**:
  - **Stop Loss**: Exactly one exchange tick beyond the sweep extreme:
    - Long: `floor(sweep.low / tick - 1) * tick`
    - Short: `ceil(sweep.high / tick + 1) * tick`
  - **Take Profit Target**: The opposite edge of the reference candle:
    - Long: `reference.high`
    - Short: `reference.low`
- **Strict Strategy Constraints**:
  - No partial midpoint exits.
  - No martingale or position doubling.
  - No adaptive or machine learning overrides.
  - No trend filters.

---

## 2. Localhost-First Operation (No Google Login Required)

The application is designed to operate locally on loopback (`127.0.0.1`):
```bash
DASHBOARD_PUBLIC=0
HOST=127.0.0.1
PORT=8000
```
When `DASHBOARD_PUBLIC=0`, **Google OAuth is completely bypassed**. No Google Client ID, secret, or browser OAuth flow is required. If `HOST` is configured to anything other than `127.0.0.1` or `localhost` while `DASHBOARD_PUBLIC=0`, the server **fails closed** on startup.

### Quick Start (Localhost):

1. **Install Python 3.10+ and dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your local `.env`**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Delta testnet API credentials:
   ```ini
   DASHBOARD_PUBLIC=0
   HOST=127.0.0.1
   PORT=8000
   DELTA_MCP_ENV=india_testnet
   CRT_LIVE_ENABLED=0
   DELTA_API_KEY=your_testnet_api_key
   DELTA_API_SECRET=your_testnet_api_secret
   CRT_STATE_DIR=runtime/crt
   ```

3. **Start the local server**:
   ```bash
   python serve_dashboard.py
   ```
   Open `http://127.0.0.1:8000` in your browser.

---

## 3. Backtest Results & Verification

Run the deterministic backtest engine using the exact same signal function as live execution:
```bash
# Run with sample candles or specify your historical candle dataset:
python -m ethresearch.backtest --candles-file artifacts/crt/sample_candles.json

# Run with custom parameters and sensitivity analysis:
python -m ethresearch.backtest \
  --tick 0.05 \
  --fee-bps 6.0 \
  --slippage-bps 10.0 \
  --usd-inr 85.0 \
  --risk-inr 100 \
  --max-contracts 1 \
  --sensitivity
```

### Backtest Output Metrics:
- **Setups Detected & Executed Trades**
- **Win Rate & Wilson 95% Confidence Interval**
- **Gross Profit / Loss & Net Profit (USD and INR)**
- **Expectancy & Profit Factor**
- **Average R-Multiple**
- **Max Drawdown (USD & INR) & Max Consecutive Losses**
- **Average Holding Period (Minutes & Bars)**
- **Long vs Short Performance Breakdown**
- **Monthly, Day-of-Week, and Session (IST) Breakdown**
- **Fee and Slippage Sensitivity Grid**

---

## 4. Testnet Validation Status

The execution layer in `ethresearch/crt_live.py` and test suite `tests/test_delta_testnet.py` explicitly validate the following matrix:

| Scenario | Test Status | Exchange Behavior Description |
|---|---|---|
| **Normal Full Fill** | ✅ VALIDATED | IOC limit order fills 100%, intent marked `ACKNOWLEDGED`. |
| **IOC Unfilled (Zero Fill)** | ✅ VALIDATED | Zero fill marked `CANCELLED_UNFILLED`; no position opened. |
| **Partial Fill (25%, 50%, 99%)** | ✅ VALIDATED | Partial fill recognized, real fill price used for risk sizing, warning logged. |
| **Protective Watchdog (SL/TP)** | ✅ VALIDATED | Verifies presence and contract quantity of SL and TP orders. |
| **Missing Protection / Circuit Breaker** | ✅ VALIDATED | Missing SL or TP triggers `UNPROTECTED_POSITION` circuit breaker; disables new entries. |
| **Automatic Bracket Recovery** | ✅ VALIDATED | Attempts up to 3 recovery submissions using Delta nested bracket schema. |
| **Emergency Close (Optional)** | ✅ VALIDATED | If `emergency_close_unprotected=True`, safely exits unprotected position. |
| **Connection Timeout After Submit** | ✅ VALIDATED | Intent written to SQLite before network call; timeout leaves `UNKNOWN` state. |
| **Engine Restart with UNKNOWN Intent** | ✅ VALIDATED | Engine refuses to arm or submit orders until uncertain intent is reconciled. |
| **Duplicate Submission Gate** | ✅ VALIDATED | Skips repeated entry within the same signal/candle timestamp window. |
| **Insufficient Collateral Gate** | ✅ VALIDATED | Rejects order if available balance < notional + costs. |
| **Stale Market Data Gate** | ✅ VALIDATED | Rejects order if signal is > 90 seconds old or snapshot > 30s old. |
| **Live Delta Exchange Fills** | ⚠️ NOT VERIFIED | Requires manual verification with real testnet API keys. |

---

## 5. Production Readiness & Safety Procedure

**Live trading entries MUST REMAIN OFF (`CRT_LIVE_ENABLED=0`) until you complete controlled testnet validation.**

### Step-by-Step Procedure to Safely Transition to Live Trading:

1. **Step 1: Testnet Validation**
   Set:
   ```ini
   DELTA_MCP_ENV=india_testnet
   CRT_LIVE_ENABLED=0
   ```
   Start dashboard. Save risk limits. Arm entries temporarily on testnet to verify order acknowledgments, bracket placement, and reconciliation.

2. **Step 2: Inspect Exchange Behavior**
   Confirm on the Delta testnet UI that:
   - Orders are placed as IOC limit orders with matching brackets.
   - Stop-loss and take-profit orders appear in the Open Orders tab with correct trigger prices.
   - Closed positions correctly trigger take-profit or stop-loss executions.

3. **Step 3: Production Preparation**
   Rotate your credentials. Ensure production API keys have withdrawal permissions DISABLED.
   Set:
   ```ini
   DELTA_MCP_ENV=india_prod
   CRT_LIVE_ENABLED=0
   ```
   Start the server. Verify account balances, ETHUSD ticker, and risk limits in the dashboard.

4. **Step 4: Explicit Arming**
   Every application restart starts in state **`OFF`**. To enable live orders:
   - Ensure all 9 risk limits are saved and non-zero.
   - Set `CRT_LIVE_ENABLED=1` in your environment.
   - Click "Turn live entries ON" in the dashboard.
   - The engine conducts preflight safety verification before arming.

---

## 6. Known Limitations

1. **Single Bot Instance**: The application uses OS file locking (`engine.lock`). Never run two bot instances with the same API key or pointing to the same state directory.
2. **Slippage and Gap Risk**: Bracket orders are exchange stop-market or limit orders. Highly volatile markets or liquidity gaps can result in fills significantly worse than the modeled stop price.
3. **Collateral Requirement**: The risk engine enforces conservative full notional collateralization (`available >= size * (entry + costs)`). High leverage is never assumed.
4. **Historical Secret Disclosure**: An obsolete test OTP (`477554`) was present in early git history (commit `54bafab`). It has been removed from all active code. Automated secret scanning (`scripts/scan_secrets.py`) runs in CI.

---

## 7. Automated Testing & Verification Commands

```bash
# Run complete test suite (unit tests, testnet suite, property/fuzz tests):
pytest -v

# Run automated secret scanner:
python scripts/scan_secrets.py

# Run backtest engine:
python -m ethresearch.backtest
```
