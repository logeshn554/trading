# Delta ETHUSD — 15-minute Candle Range Theory

The active application is now deterministic CRT. The dashboard and Docker image do not load GTrXL, reinforcement learning, paper trading, or legacy manual-order routes. Obsolete GTrXL source, training scripts, tests and model artifacts have been removed from the current checkout; Git history retains them.

Rules: take two consecutive completed 15-minute Delta candles. The second must sweep exactly one extreme of the first and close strictly inside the first candle's range. Low sweep → long; high sweep → short. Skip double sweeps, boundary closes, missing candles, and decisions more than 90 seconds after close. Stop one exchange tick beyond the sweep extreme; full-position target at the opposite reference edge. There is no partial midpoint exit, trend filter, martingale, or adaptive learning.

Entries are IOC limit orders capped relative to the signal close. Each includes Delta bracket stop-loss and take-profit fields. Product ID, tick size and contract value come from Delta. One position/order at a time across the account; do not run another bot or manually trade this account while CRT is armed.

## Run locally

Install Python 3.12 and:
pip install delta-exchange-mcp==0.7.0 google-auth==2.40.3 requests==2.32.5

Run: python serve_dashboard.py

Open http://127.0.0.1:8000/. Public candle data can load without account credentials; balances and execution require a working Delta MCP session. Never expose the local unauthenticated mode to the network.

## Render setup

Use the included Dockerfile and render.yaml with one instance and the persistent /app/runtime disk. The Blueprint enables deployment on commits to the connected branch. Existing Render services may need Auto-Deploy enabled in their settings. Deployment does not arm trading: after each restart, save limits if needed and explicitly turn live entries ON.

Required environment variables:
- DASHBOARD_PUBLIC=1
- GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
- GOOGLE_ALLOWED_EMAIL=logeslogesh.n554@gmail.com
- DASHBOARD_SESSION_SECRET: a random secret at least 32 characters
- DELTA_API_KEY and DELTA_API_SECRET
- DELTA_MCP_ENV=india_prod (india_testnet for execution validation)
- CRT_STATE_DIR=/app/runtime/crt
- CRT_LIVE_ENABLED=0 initially

Create a Google OAuth web client. Add https://YOUR-RENDER-HOST/auth/callback as its authorized redirect URI. Enable the Google consent screen and add the allowed account as a test user if the consent screen remains in testing. Render supplies RENDER_EXTERNAL_HOSTNAME; set DASHBOARD_ALLOWED_HOSTS for any custom domain. Public startup fails if OAuth/secrets are missing. Public access uses Google only; OTP was removed.

For live capability, supply a Delta trading-enabled key in Render and set CRT_LIVE_ENABLED=1. Save every displayed limit and explicitly turn entries ON in the dashboard. Every process restart starts OFF. This software was not live-validated during implementation: validate exchange behavior in testnet before production activation.

## Limits and accounting

All risk values start empty; no old strategy risk defaults carry over. Configure maximum daily entry attempts, maximum contracts, estimated INR risk per trade, daily net profit/loss entry stops, the correct Delta USD-to-INR settlement conversion, fees including taxes per side in basis points, maximum spread and entry slippage.

The observed public ETHUSD contract on 2026-09-29 reported 0.01 ETH per contract, tick 0.05 and USD settlement. These are read dynamically, not hardcoded. INR displays for USD amounts use the operator's configured conversion; verify this against Delta's settlement policy rather than assuming a spot FX rate.

Daily net is account-wide trading wallet cashflow since midnight IST, including recognized commissions/funding. It is not a CRT-only closed-trade win rate. Deposits/withdrawals and recognized transfers are excluded. Unknown cashflow types/currencies or incomplete history block entries. Generic cashflow is counted as trading only when a product ID is present. This mapping still needs validation against your actual account statement. Daily stops prevent new entries; they do not force-close existing positions. Bracket exits stay active when the switch is OFF.

Risk sizing includes configured fee allowance. Full notional collateral is required; no assumed 10x margin. Stop fills and gaps can exceed estimated risk. No guarantee of an exact monetary stop or profit limit is made.

## Restart and order reconciliation

SQLite intent records are committed before sending. Each setup has one stable client order ID. A timeout never triggers an automatic resend. Unknown submissions halt entries and are looked up by client ID on later polls; only a protected acknowledgment or a confirmed unfilled cancellation clears uncertainty. Missing orders are never assumed safe to retry. Restart leaves entries OFF.

Keep the persistent disk and one service instance. The file lock protects processes sharing that directory, not independent deployments with different disks. Do not deploy another copy with the same trading key. Retain state during redeploys. OFF leaves existing exchange exits intact.

If protection cannot be confirmed, inspect the exchange position and protective orders immediately. The current implementation halts and reports uncertainty; it does not guarantee automated emergency liquidation or bracket repair. External cancellation of protective orders is not automatically repaired. These are material production validation gaps.

## Verification

Run: python -m unittest discover -s tests -p test_crt.py -v
Run: node --check web/app.js

Tests use a fake exchange and cover causal long/short signals, invalid candles, entry expiry, duplicate prevention, restart, uncertain submission, daily loss, transaction costs, and route authentication. They do not prove exchange fills, bracket/OCO behavior, partial-fill protection, slippage, Google login end-to-end, or profitability. No CRT backtest or live winning percentage is claimed.

Official API reference: https://docs.delta.exchange/ (order brackets, client-order lookup, contract specification and wallet transactions).

