"""GTrXL (Gated Transformer-XL) Reinforcement Learning Architecture.

Production-grade implementation featuring:
- Identity Map Reordering (Pre-LN + Gating)
- Learnable GRUGate with gate bias initialization
- Dai et al. Relative Multi-Head Attention (Rel-MHA) with memory cache
- Segmented BPTT and continuous sliding KV-cache streaming for intermediate horizons (30m - 2h)
- Actor-Critic heads with auxiliary multi-horizon return prediction
- Full support for Training Mode (chunk length L) and Inference Mode (step-by-step L=1)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SECTION 1 & 5: GRUGate Module
# ============================================================================

class GRUGate(nn.Module):
    """GRU-style Gating Unit replacing additive residual skip connections.

    Implements Parisotto et al. (2020) 'Stabilizing Transformers in RL':
        r = sigmoid(W_r * x + U_r * y_hat + b_r)
        z = sigmoid(W_z * x + U_z * y_hat + b_z)
        h_tilde = tanh(W_g * x + U_g * (r * y_hat) + b_g)
        output = (1 - z) * x + z * h_tilde

    With initial_bias < 0 (e.g. -2.0), z is initialized close to 0 (sigmoid(-2) ~ 0.119),
    biasing the layer toward identity mapping (output ~ x) at initialization.
    This guarantees pristine gradient flow at t=0 and avoids premature policy collapse.
    """

    def __init__(self, dim: int, initial_bias: float = -2.0) -> None:
        super().__init__()
        self.dim = dim
        self.initial_bias = initial_bias

        # Linear projections for input x (residual stream)
        self.w_r = nn.Linear(dim, dim, bias=False)
        self.w_z = nn.Linear(dim, dim, bias=False)
        self.w_g = nn.Linear(dim, dim, bias=False)

        # Linear projections for candidate transformation y_hat (sublayer output)
        self.u_r = nn.Linear(dim, dim, bias=False)
        self.u_z = nn.Linear(dim, dim, bias=True)
        self.u_g = nn.Linear(dim, dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Standard Xavier uniform initialization
        for layer in (self.w_r, self.w_z, self.w_g, self.u_r, self.u_g):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.u_z.weight)
        # Crucial GTrXL property: negative gate bias to initialize update gate z ~ 0
        nn.init.constant_(self.u_z.bias, self.initial_bias)

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Residual input tensor of shape (Batch, SeqLen, Dim).
        y_hat: Sublayer transformation candidate of shape (Batch, SeqLen, Dim).

        Returns:
            Gated tensor y of shape (Batch, SeqLen, Dim).
        """
        r = torch.sigmoid(self.w_r(x) + self.u_r(y_hat))
        z = torch.sigmoid(self.w_z(x) + self.u_z(y_hat))
        h_tilde = torch.tanh(self.w_g(x) + self.u_g(r * y_hat))
        return (1.0 - z) * x + z * h_tilde


# ============================================================================
# SECTION 1 & 5: Relative Positional Encoding & Attention Shift
# ============================================================================

def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Performs the relative shift trick from Transformer-XL (Dai et al.).

    Given attention matrix `x` of shape (B, H, L, TotalLen) where TotalLen = L + M,
    pads along the relative column axis and reshapes so that each query row i
    aligns with relative offsets [M + i, ..., 0].

    Input:  (B, H, L, TotalLen)
    Output: (B, H, L, TotalLen) with shifted relative index alignment.
    """
    b, h, l, total_len = x.shape
    # Pad 1 column on the left of relative distance axis
    zero_pad = torch.zeros((b, h, l, 1), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)  # (b, h, l, total_len + 1)

    # Flatten the last two dimensions (l, total_len + 1)
    x_flat = x_padded.view(b, h, total_len + 1, l)

    # Slice off the first dummy row and reshape back to (b, h, l, total_len)
    x_shifted = x_flat[:, :, 1:].view(b, h, l, total_len)
    return x_shifted


class PositionalEmbedding(nn.Module):
    """Sinusoidal Relative Positional Embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        """pos_seq: 1D tensor of relative positions (TotalLen,).

        Returns: (TotalLen, Dim) relative positional embeddings.
        """
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return pos_emb


# ============================================================================
# SECTION 1 & 5: Relative Multi-Head Self-Attention with Memory Cache
# ============================================================================

class RelMultiHeadAttention(nn.Module):
    """Transformer-XL Relative Multi-Head Attention layer with recurrent memory.

    Decomposes the attention score matrix:
        A_{i,j}^{rel} = q_i k_j^T + q_i (W_{k,R} R_{i-j})^T + u k_j^T + v (W_{k,R} R_{i-j})^T
    where:
        - Term (a) q_i k_j^T is content-to-content attention
        - Term (b) q_i (W_{k,R} R_{i-j})^T is content-dependent positional bias
        - Term (c) u k_j^T is global content bias (learnable parameter u)
        - Term (d) v (W_{k,R} R_{i-j})^T is global positional bias (learnable parameter v)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else (d_model // n_heads)
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Projections
        self.q_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.r_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)

        self.out_proj = nn.Linear(self.n_heads * self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable global bias vectors u and v: (n_heads, d_head)
        self.u_bias = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        nn.init.normal_(self.u_bias, std=0.02)
        nn.init.normal_(self.v_bias, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        x: Current sequence chunk of shape (B, L, D_model).
        r_emb: Relative positional embeddings of shape (L + M, D_model).
        mem: Cached key/value memory from preceding chunk (B, M, D_model) or None.
        mask: Optional causal attention mask (L, L + M).

        Returns:
            Output tensor of shape (B, L, D_model).
        """
        b, l, _ = x.shape

        # Concatenate recurrent memory along sequence dimension
        if mem is not None and mem.size(1) > 0:
            full_x = torch.cat([mem, x], dim=1)  # (B, M + L, D_model)
        else:
            full_x = x

        total_len = full_x.size(1)  # M + L

        # 1. Project Q, K, V
        # Q is computed from current chunk x only: (B, L, H, D_head)
        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        # K and V are computed over the full concatenated context (M + L):
        k = self.k_proj(full_x).view(b, total_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(full_x).view(b, total_len, self.n_heads, self.d_head).transpose(1, 2)

        # 2. Project relative positional embeddings R: (TotalLen, H, D_head)
        r = self.r_proj(r_emb).view(total_len, self.n_heads, self.d_head).transpose(0, 1)

        # 3. Content attention: AC = (Q + u) * K^T
        # Shape: (B, H, L, TotalLen)
        q_u = q + self.u_bias.unsqueeze(0).unsqueeze(2)  # (B, H, L, D_head)
        ac_score = torch.matmul(q_u, k.transpose(-1, -2))

        # 4. Positional attention: BD = (Q + v) * R^T shifted
        q_v = q + self.v_bias.unsqueeze(0).unsqueeze(2)  # (B, H, L, D_head)
        bd_score = torch.matmul(q_v, r.transpose(-1, -2))  # (B, H, L, TotalLen)
        bd_score = rel_shift(bd_score)

        # 5. Combine and scale attention logits
        attn_logits = (ac_score + bd_score) * self.scale

        # 6. Apply causal mask (prevent attending to future timesteps)
        if mask is not None:
            # mask expected shape: (L, TotalLen) or broadcastable (1, 1, L, TotalLen)
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        # 7. Softmax and weighted sum
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attn_out: (B, H, L, D_head) -> transpose to (B, L, H, D_head) -> reshape (B, L, D_model)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head)
        return self.out_proj(attn_out)


# ============================================================================
# SECTION 1 & 5: GTrXL Block (Pre-LN + Rel-MHA + GRUGate + Gated FFN)
# ============================================================================

class GTrXLBlock(nn.Module):
    """Single GTrXL Layer implementing Identity Map Reordering:

        x -> LayerNorm(x) -> RelMHA(LN(x), mem) -> y_hat1 -> GRUGate1(x, y_hat1) -> x_mid
        x_mid -> LayerNorm(x_mid) -> FFN(LN(x_mid)) -> y_hat2 -> GRUGate2(x_mid, y_hat2) -> x_out
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: Optional[int] = None,
        d_inner: Optional[int] = None,
        dropout: float = 0.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        d_inner = d_inner if d_inner is not None else 4 * d_model

        # Sublayer 1: Relative Attention with Pre-LN & GRUGate
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RelMultiHeadAttention(d_model, n_heads, d_head, dropout=dropout)
        self.gate1 = GRUGate(d_model, initial_bias=gate_bias)

        # Sublayer 2: Feed-Forward Network with Pre-LN & GRUGate
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.gate2 = GRUGate(d_model, initial_bias=gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        x: (B, L, D_model)
        r_emb: (L + M, D_model)
        mem: (B, M, D_model)
        mask: (L, L + M)

        Returns:
            Layer output of shape (B, L, D_model).
        """
        # Identity Map Reordering 1: Pre-LN -> Rel-MHA -> GRUGate
        norm_x = self.ln1(x)
        norm_mem = self.ln1(mem) if (mem is not None and mem.size(1) > 0) else None
        attn_out = self.attn(norm_x, r_emb, mem=norm_mem, mask=mask)
        x_mid = self.gate1(x, attn_out)

        # Identity Map Reordering 2: Pre-LN -> FFN -> GRUGate
        ffn_out = self.ffn(self.ln2(x_mid))
        x_out = self.gate2(x_mid, ffn_out)
        return x_out


# ============================================================================
# SECTION 4 & 5: Complete GTrXL Actor-Critic Backbone
# ============================================================================

class GTrXLActorCritic(nn.Module):
    """Complete GTrXL Reinforcement Learning Network.

    Features:
    - Input feature projection + LayerNorm embedding
    - Recurrent KV-memory cache across N layers
    - Shared representation trunk with decoupled Actor (policy) and Critic (value) heads
    - Auxiliary multi-horizon return prediction head (stabilizes representations)
    - Full dual-mode operation:
        * Training Mode: processes chunks of length L with incoming memory M
        * Continuous Inference Mode: step-by-step (L=1) rolling KV-cache execution
    """

    def __init__(
        self,
        d_in: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mem_len: int = 128,
        d_inner: Optional[int] = None,
        dropout: float = 0.0,
        continuous_action: bool = False,
        aux_horizons: Tuple[int, ...] = (1, 4, 12),
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mem_len = mem_len
        self.continuous_action = continuous_action
        self.aux_horizons = aux_horizons

        # Input feature projection
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Sinusoidal relative positional embedding generator
        self.pos_emb = PositionalEmbedding(d_model)

        # Stack of GTrXL Blocks
        self.layers = nn.ModuleList([
            GTrXLBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_head=d_model // n_heads,
                d_inner=d_inner if d_inner is not None else 4 * d_model,
                dropout=dropout,
                gate_bias=-2.0,
            )
            for _ in range(n_layers)
        ])

        # Final trunk normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Policy (Actor) Head
        if continuous_action:
            self.actor_mean = nn.Linear(d_model, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_logits = nn.Linear(d_model, action_dim)

        # Value (Critic) Head
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Auxiliary multi-horizon return predictor head
        self.aux_head = nn.Linear(d_model, len(aux_horizons))

    def init_memory(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initializes empty memory cache for all layers.

        Each layer cache has shape (Batch, 0, D_model) initially.
        """
        return [
            torch.empty(batch_size, 0, self.d_model, device=device)
            for _ in range(self.n_layers)
        ]

    def _create_causal_mask(self, q_len: int, mem_len: int, device: torch.device) -> torch.Tensor:
        """Constructs causal attention mask of shape (1, 1, q_len, mem_len + q_len).

        Tokens at index i (0 <= i < q_len) can attend to all `mem_len` historical tokens
        and current-chunk tokens up to index i.
        """
        total_len = q_len + mem_len
        # Causal mask for chunk: lower triangular of shape (q_len, q_len)
        chunk_mask = torch.tril(torch.ones(q_len, q_len, device=device))
        # Memory mask: all 1s (all memory is in the past, hence valid to attend to)
        if mem_len > 0:
            mem_mask = torch.ones(q_len, mem_len, device=device)
            full_mask = torch.cat([mem_mask, chunk_mask], dim=1)  # (q_len, total_len)
        else:
            full_mask = chunk_mask
        return full_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, total_len)

    def _update_memory(
        self,
        prev_mems: List[torch.Tensor],
        layer_hiddens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Updates and truncates KV-cache memory per layer with stop_gradient (detach).

        Memory concatenation formula:
            mem_{t} = stop_gradient([mem_{t-1}, h_{t-1}][-mem_len:])
        """
        new_mems: List[torch.Tensor] = []
        for prev_m, h in zip(prev_mems, layer_hiddens):
            # Concatenate along time dimension (dim=1) and detach to prevent BPTT blowup
            if prev_m.size(1) > 0:
                cat_m = torch.cat([prev_m, h], dim=1)
            else:
                cat_m = h
            # Retain only the most recent mem_len tokens
            new_m = cat_m[:, -self.mem_len:].detach()
            new_mems.append(new_m)
        return new_mems

    def forward(
        self,
        x: torch.Tensor,
        memories: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Forward pass supporting both Training and Inference.

        Args:
            x: Input feature tensor of shape (B, L, D_in).
            memories: List of N layer memory tensors, each (B, M, D_model).
                      If None, initialized to empty caches.

        Returns:
            outputs: Dictionary containing:
                - "policy": Logits (B, L, ActionDim) or (Mean, LogStd) for continuous
                - "value": State value estimates V(s_t) of shape (B, L, 1)
                - "aux_returns": Auxiliary multi-horizon predictions of shape (B, L, K)
                - "features": Latent hidden features (B, L, D_model)
            new_memories: List of updated detached memory tensors for next chunk/step.
        """
        b, l, _ = x.shape
        device = x.device

        if memories is None:
            memories = self.init_memory(b, device)

        curr_mem_len = memories[0].size(1)
        total_len = l + curr_mem_len

        # Generate relative positional sequence: [total_len - 1, ..., 0]
        pos_seq = torch.arange(total_len - 1, -1, -1.0, device=device)
        r_emb = self.pos_emb(pos_seq)  # (total_len, D_model)

        # Causal mask for the current chunk
        mask = self._create_causal_mask(l, curr_mem_len, device)

        # Input feature projection
        h = self.in_proj(x)  # (B, L, D_model)

        layer_hiddens: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            layer_hiddens.append(h)
            h = layer(h, r_emb, mem=memories[idx], mask=mask)

        # Pass through final trunk norm
        latent = self.final_norm(h)  # (B, L, D_model)

        # Actor head
        if self.continuous_action:
            mean = self.actor_mean(latent)
            log_std = self.actor_log_std.expand_as(mean)
            policy_out = {"mean": mean, "log_std": log_std}
        else:
            policy_out = self.actor_logits(latent)

        # Critic head
        val_out = self.critic(latent)  # (B, L, 1)

        # Auxiliary return prediction head
        aux_out = self.aux_head(latent)  # (B, L, len(aux_horizons))

        # Update recurrent cache with detach() semantics
        new_memories = self._update_memory(memories, layer_hiddens)

        outputs = {
            "policy": policy_out,
            "value": val_out,
            "aux_returns": aux_out,
            "features": latent,
        }
        return outputs, new_memories


# ============================================================================
# SECTION 3: Exact RL Loss & Optimization Pipeline
# ============================================================================

class GTrXLLoss(nn.Module):
    """Exact Multi-Objective Loss Formulation for GTrXL RL.

    L_total = L_policy + c1 * L_value - c2 * Entropy(pi) + c3 * L_aux

    Features:
    - PPO Clipped Surrogate Objective with Generalized Advantage Estimation (GAE)
    - Clipped Value Function Loss (prevents value network divergence)
    - Policy Entropy Bonus
    - Auxiliary Self-Supervised Multi-Horizon Return Prediction (Huber Loss)
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        val_clip_eps: float = 0.2,
        c1_val: float = 0.5,
        c2_ent: float = 0.01,
        c3_aux: float = 0.1,
        burn_in_steps: int = 0,
    ) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.val_clip_eps = val_clip_eps
        self.c1_val = c1_val
        self.c2_ent = c2_ent
        self.c3_aux = c3_aux
        self.burn_in_steps = burn_in_steps

    def forward(
        self,
        policy_logits: torch.Tensor,
        old_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        aux_predictions: Optional[torch.Tensor] = None,
        target_aux_returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

        policy_logits: (B, L, ActionDim)
        old_policy_logits: (B, L, ActionDim)
        actions: (B, L)
        advantages: (B, L)
        values: (B, L, 1)
        old_values: (B, L, 1)
        returns: (B, L, 1)
        aux_predictions: (B, L, K)
        target_aux_returns: (B, L, K)
        """
        # Slice off burn-in prefix if burn-in is enabled
        s = self.burn_in_steps
        if s > 0:
            policy_logits = policy_logits[:, s:]
            old_policy_logits = old_policy_logits[:, s:]
            actions = actions[:, s:]
            advantages = advantages[:, s:]
            values = values[:, s:]
            old_values = old_values[:, s:]
            returns = returns[:, s:]
            if aux_predictions is not None and target_aux_returns is not None:
                aux_predictions = aux_predictions[:, s:]
                target_aux_returns = target_aux_returns[:, s:]

        # 1. PPO Policy Loss
        dist = torch.distributions.Categorical(logits=policy_logits)
        old_dist = torch.distributions.Categorical(logits=old_policy_logits)

        log_probs = dist.log_prob(actions)
        old_log_probs = old_dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 2. Value Function Loss (Clipped)
        v_clipped = old_values + torch.clamp(values - old_values, -self.val_clip_eps, self.val_clip_eps)
        vf_loss1 = (values - returns).pow(2)
        vf_loss2 = (v_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        # 3. Auxiliary Multi-Horizon Return Loss (Smooth L1 / Huber)
        if aux_predictions is not None and target_aux_returns is not None:
            aux_loss = F.smooth_l1_loss(aux_predictions, target_aux_returns)
        else:
            aux_loss = torch.tensor(0.0, device=values.device)

        # 4. Total Combined Loss
        total_loss = policy_loss + self.c1_val * value_loss - self.c2_ent * entropy + self.c3_aux * aux_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "aux_loss": aux_loss,
        }


# ============================================================================
# SECTION 6: Real-Time Continuous Streaming Inference Engine
# ============================================================================

class GTrXLStreamingInferenceEngine:
    """Production-grade step-by-step rolling inference engine for intermediate timeframes.

    Maintains the persistent KV-cache across live market bars (30m, 1h, 2h).
    Handles:
    - Step-by-step L=1 forward evaluation
    - Sliding window memory eviction (O(1) memory bound)
    - Regime continuity without retraining or cold-starts
    """

    def __init__(self, model: GTrXLActorCritic, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()
        self.memories: Optional[List[torch.Tensor]] = None
        self.step_count: int = 0

    def reset(self) -> None:
        """Clears the recurrent memory cache."""
        self.memories = None
        self.step_count = 0

    @torch.no_grad()
    def step(self, bar_features: torch.Tensor) -> Dict[str, Union[int, float, List[float]]]:
        """Executes a single live market step (L=1).

        Args:
            bar_features: 1D or 2D tensor of shape (D_in,) or (1, D_in).

        Returns:
            Dictionary with action, value, action_probs, and aux_horizon_returns.
        """
        if bar_features.dim() == 1:
            x = bar_features.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, D_in)
        elif bar_features.dim() == 2:
            x = bar_features.unsqueeze(1).to(self.device)  # (1, 1, D_in)
        else:
            x = bar_features.to(self.device)

        outputs, self.memories = self.model(x, self.memories)
        self.step_count += 1

        policy = outputs["policy"]
        if isinstance(policy, dict):
            # Continuous action
            mean = policy["mean"].squeeze().cpu().numpy().tolist()
            log_std = policy["log_std"].squeeze().cpu().numpy().tolist()
            action_data = {"mean": mean, "log_std": log_std}
        else:
            # Discrete action
            logits = policy.squeeze(0).squeeze(0)  # (ActionDim,)
            probs = F.softmax(logits, dim=-1).cpu().numpy().tolist()
            action = int(torch.argmax(logits).item())
            action_data = {"action": action, "action_probs": probs}

        value = float(outputs["value"].squeeze().item())
        aux = outputs["aux_returns"].squeeze().cpu().numpy().tolist()

        return {
            "step": self.step_count,
            "value": value,
            "aux_returns": aux,
            **action_data,
        }
