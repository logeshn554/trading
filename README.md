# Trading

Private, read-only Delta Exchange account dashboard for ETH research. It shows live ETHUSD market data, wallet balances, available funds, open positions, current-position realized and unrealized P&L, recent fills, wallet transactions, and open orders. It refreshes every 30 seconds while open.

**Live order execution is blocked.** The main research candidate `EXP_021_023_WF_2025_flow10_strict_candle_h12` had 29 wins in 40 saved simulated trades (72.5%) and +$350.47 after modeled costs, but failed its selection gate. Its Delta-specific execution and forward validation remain incomplete. The dashboard's live switch is disabled; its INR trade and profit/loss limits are unset until supplied and implemented as enforced risk controls. The view must not be interpreted as an automatic trading bot.

Adding a Delta key with Trading permission to Render does **not** activate order execution. The MCP bridge has a read-only tool allowlist, and the dashboard shows the exact blockers that keep the switch OFF. Delta's own trading tools have no rehearsal step or size cap; a validated signal feed, exchange-specific contract sizing and stop orders, durable order reconciliation, risk-limit enforcement, and testnet/forward validation are required before a real ON state can be implemented.

## Deploy on Render

This repository contains a small Docker image and `render.yaml` Blueprint for an always-on paid Render web service. The image copies only the account dashboard, the Delta MCP bridge, and the strategy status JSON. It contains no datasets, research reports, local credentials, or order-submission code. Render's free web tier sleeps after inactivity, so the Blueprint uses its paid Starter service plan.

1. Keep this repository **private**. In Google Cloud, create a Google OAuth **Web application** client with authorized redirect URI `https://logeshn554-trading-dashboard.onrender.com/auth/callback`. Configure the OAuth consent screen so `logeslogesh.n554@gmail.com` may sign in; if the app is in Testing mode, add that address as a test user. The app accepts only a Google-verified identity with that exact email.
2. In Render, choose **New → Blueprint** and connect this repository. Review the paid service price before creating it. Render will use the service name `logeshn554-trading-dashboard` unless it must assign a different hostname. If it does, update the redirect URI in Google Cloud to the actual HTTPS hostname.
3. In Render's prompted secret environment fields, enter `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `DELTA_API_KEY`, and the matching `DELTA_API_SECRET`. Use a Delta key with **Read Data** permission only; rotate the previously shared key before cloud deployment. Render generates `DASHBOARD_SESSION_SECRET`. Never commit any of these values or paste them into chat.
4. Wait for `/health` to pass, open the Render HTTPS URL, and sign in with the allowed Google account. Confirm `ACCOUNT CONNECTED` and compare wallet values against Delta Exchange. If the actual Render hostname differs from the OAuth redirect URI, update the URI in Google Cloud and retry.

The service refuses to start without all required secrets. It checks the Render hostname, verifies Google's signed ID token and login nonce, and uses a secure, HTTP-only session cookie. If you use a custom domain, add it to `DASHBOARD_ALLOWED_HOSTS` and register its callback URI in Google Cloud. `/health` reveals only service availability. Account and page routes require sign-in. No API route can submit an order.

## Run locally

Python 3.12 or newer and `uvx` are required. Install the Delta MCP server with `uvx delta-exchange-mcp==0.7.0`, then set up a Read Data key using `uvx delta-exchange-mcp==0.7.0 login` in a secure terminal. Start `python serve_dashboard.py 8000` and open `http://127.0.0.1:8000`. Local mode binds to loopback and does not require Google sign-in. The public mode is activated only by `DASHBOARD_PUBLIC=1` and requires Google and Delta secrets.

Realized P&L in the overview is Delta's `realized_pnl` field for **currently open positions**. It is not lifetime closed-trade profit. Fills and wallet transactions are shown separately; their first 100 rows from the past 30 days may have further pages. Amounts retain their original asset units and are never summed across currencies.

Run `python -m unittest tests.test_delta_dashboard -v` for focused account/authentication checks. The research datasets and reports remain in the original local workspace and are intentionally excluded from this deployment repository.
