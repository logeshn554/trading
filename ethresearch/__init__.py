"""Numerical ETH research and algorithmic execution engine."""
from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError
from ethresearch.gtrxl import (
    GRUGate,
    GTrXLActorCritic,
    GTrXLBlock,
    GTrXLLoss,
    GTrXLStreamingInferenceEngine,
    RelMultiHeadAttention,
    rel_shift,
)
from ethresearch.trader import GTrXLAutomatedTrader, compute_bar_features

__all__ = [
    "DeltaMcpClient",
    "DeltaMcpError",
    "GRUGate",
    "GTrXLActorCritic",
    "GTrXLBlock",
    "GTrXLLoss",
    "GTrXLStreamingInferenceEngine",
    "RelMultiHeadAttention",
    "rel_shift",
    "GTrXLAutomatedTrader",
    "compute_bar_features",
]
