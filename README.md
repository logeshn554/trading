# Pure Delta Exchange GTrXL Automated Trading Platform

Autonomous, production-grade algorithmic trading system and live dashboard built exclusively for **Delta Exchange India (ETHUSD)**.

### Architecture
- **Engine**: GTrXL (Gated Transformer-XL) Reinforcement Learning network with 4 layers, 4 heads Rel-MHA, GRUGate, and Identity Map Reordering.
- **Venue**: 100% Pure Delta Exchange India (native `ETHUSD` contracts, 0.001 ETH notional, 10x leverage, INR settlement). Zero external venue dependencies.
- **Data Streaming**: Real-time 1m candle harvesting and orderbook flow directly via Delta India MCP tools (`get_candles`, `get_ticker`, `get_product`, `place_order`).
- **Autonomous Self-Healing**: Dynamic root-cause diagnostics and auto-sizing when margin/balance breaches or tool desync occurs.
- **Online Failure Adaptation**: Automatically triggers policy gradient adaptation whenever a stop-loss is hit or an adverse trade closes, updating weights online to prevent repeating the failure in similar market regimes.

## Deploy on Render
The application is containerized via `Dockerfile` and configured with `render.yaml` Blueprint for an always-on Render web service.
1. Connect this repository to Render Blueprint.
2. Provide your Delta India API credentials (`DELTA_API_KEY`, `DELTA_API_SECRET`) and Google OAuth secrets.
3. Access the live dashboard with real-time AI signal stream, risk controls, and automated trading controls.

