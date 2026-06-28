"""DeepSeekMoE — Step 5 deliverable (ROADMAP.md §5).

The MoE paradigm (fine-grained experts + shared expert, aux-loss-free balancing) is
INHERITED from V2/V3 (ROADMAP §0.2) — this ports the stable structure and implements
ONLY the verified V4 deltas, parity-checked against transformers deepseek_v4
(DeepseekV4SparseMoeBlock / TopKRouter / HashRouter / Experts, pinned 9ded3dbbfc):

  delta (a)  affinity scoring Sigmoid -> Sqrt(Softplus(.))      (config.scoring_func)
  delta (b)  first n_hash_layers MoE layers use FROZEN hash routing (tid2eid[input_ids])
  delta (c)  V3's n_group / topk_group constraint REMOVED (plain global top-k)
  delta (d)  aux-loss-free balancing (e_score_correction_bias, INHERITED) + a NEW
             sequence-wise balance loss

Aux-loss-free balancing (Wang 2024 / V3, NOT a V4 invention): the bias is added to the
affinity scores for SELECTION ONLY; the routing weights use the raw (unbiased) scores.
The bias is nudged by a load-based SIGN rule (no gradient — keep it out of the optimizer),
so balancing costs no loss-degrading aux term. The complementary sequence-wise balance
loss discourages within-sequence collapse.

Toy config (config.py): n_routed_experts 8, top-k 2, shared 1 — DO NOT use V4-Pro's 384 /
Flash's 256. Success is routing BEHAVIOUR (load CV, specialization, controller convergence),
NOT "beats dense" (structurally unreachable at this scale — ROADMAP §5).

torch.compile: top-k select + token scatter/gather graph-break — keep routing/dispatch
EAGER and compile only the per-expert FFN (ROADMAP §5). The dispatch/indexing is where the
bugs live; the parity case (parity.py) pins the forward to the reference at <1e-4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _score(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Affinity activation. V4 delta (a): sqrt(softplus(.)) (matches ACT2FN['sqrtsoftplus']
    = softplus(x).sqrt()); 'sigmoid' is the V3 form kept for the A/B routing study.

    The softplus is floored at 1e-12 before the sqrt: sqrt'(0)=inf, so a single very-negative
    logit (softplus underflowing to 0) would otherwise NaN the backward. The floor only moves
    values where softplus < 1e-12 (logit < ~-27), far below any realistic / parity input, so
    the forward — and the weight-copy parity vs deepseek_v4 — is unaffected (still 0.00e+00)."""
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).clamp_min(1e-12).sqrt()
    if scoring_func == "sigmoid":
        return torch.sigmoid(logits)
    raise ValueError(scoring_func)


class MoEExperts(nn.Module):
    """Routed experts as 3D weight tensors (mirrors DeepseekV4Experts). Clamped SwiGLU:
    gate clamped to <= limit, up clamped to [-limit, limit], then silu(gate) * up."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        H, I = config.n_embd, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, H, I))
        self.limit = config.swiglu_limit
        self.quant_mode = config.quant_mode          # Step 7 FP4 fake-quant (experts only)
        self.quant_per_channel = config.quant_per_channel
        self.reset_parameters()

    def reset_parameters(self):
        # raw nn.Parameters are NOT touched by GPT._init_weights (which only inits
        # Linear/Embedding) — init here, like the reference's _init_weights for Experts.
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def _w(self, weight: torch.Tensor) -> torch.Tensor:
        # Step 7: FP4 fake-quant the expert weights. qat -> always (STE so grads flow);
        # ptq -> eval only (train full precision, quantize the final weights at inference).
        if self.quant_mode == "qat" or (self.quant_mode == "ptq" and not self.training):
            from components.quant import fp4_fake_quant
            return fp4_fake_quant(weight, ste=self.quant_mode == "qat",
                                  per_channel=self.quant_per_channel)
        return weight

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # hidden_states [T, H]; top_k_index/top_k_weights [T, top_k]. Loop over experts that
        # actually received tokens, gather their tokens, run the expert, scatter-add back
        # (index_add_), weighting by the routing weight. Matches the reference op-for-op.
        gate_up_proj, down_proj = self._w(self.gate_up_proj), self._w(self.down_proj)
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], gate_up_proj[expert_idx]))
            current = F.linear(current, down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class _RouterBase(nn.Module):
    """Shared affinity head: logits = x @ weight.T, scores = score_fn(logits). Subclasses
    decide which experts are SELECTED (learned top-k vs frozen hash)."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.n_active_experts
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.n_embd
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)   # raw Parameter — init here (see Experts)

    def _weights(self, scores, indices):
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor


class TopKRouter(_RouterBase):
    """Learned top-k routing with aux-loss-free balancing. The bias is added to the scores
    for SELECTION ONLY (delta d / inherited); the returned weights use the raw scores."""

    def __init__(self, config):
        super().__init__(config)
        self.bias_update_rate = config.bias_update_rate
        # buffer, not a parameter: updated by a load-based sign rule, never by the optimizer.
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = _score(logits, self.scoring_func)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        if self.training and self.bias_update_rate > 0:
            self._update_bias(indices)
        return logits, self._weights(scores, indices), indices

    @torch.no_grad()
    def _update_bias(self, indices):
        # Aux-loss-free controller: nudge bias toward the mean load by a SIGN step
        # (Wang 2024). Overloaded experts get their bias decreased, starving ones raised.
        load = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
        err = load.mean() - load
        self.e_score_correction_bias += self.bias_update_rate * torch.sign(err)


class HashRouter(_RouterBase):
    """Frozen hash routing (delta b): expert selection is a fixed tid2eid[input_ids] lookup,
    not a learned argmax. The learned `weight` still produces the scores that weight the
    selected experts; only WHICH experts are static. tid2eid is a frozen buffer (no grad)."""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(self, hidden_states, input_ids):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = _score(logits, self.scoring_func)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        return logits, self._weights(scores, indices), indices


class DeepSeekMoE(nn.Module):
    """V4 MoE block: a (hash or top-k) router + fine-grained routed experts + one always-on
    shared expert (mirrors DeepseekV4SparseMoeBlock). Layers [0, n_hash_layers) hash-route.

    Stashes the last forward's router logits + selected indices on the module so the GPT can
    aggregate the sequence-wise balance loss; the aux-loss-free bias update happens inside
    the TopKRouter during training. Returns ONLY the tensor, so model.py's residual math is
    unchanged (ffn_type='moe' swaps exactly one variable)."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_hash = layer_idx < config.n_hash_layers
        self.n_experts = config.n_routed_experts
        self.top_k = config.n_active_experts
        self.balance_loss_coef = config.balance_loss_coef
        self.scoring_func = config.scoring_func
        self.gate = HashRouter(config) if self.is_hash else TopKRouter(config)
        self.experts = MoEExperts(config)
        # shared expert = plain (unclamped) SwiGLU, matching DeepseekV4MLP at moe_intermediate_size.
        from model import SwiGLU
        self.shared_experts = SwiGLU(config.n_embd, config.moe_intermediate_size)
        self.last_logits = None
        self.last_indices = None

    def forward(self, x, input_ids=None):
        B, S, H = x.shape
        flat = x.view(-1, H)
        if self.is_hash:
            if input_ids is None:
                raise ValueError("hash_moe layer needs input_ids (thread it through the block)")
            logits, weights, indices = self.gate(x, input_ids)
        else:
            logits, weights, indices = self.gate(x)
        routed = self.experts(flat, indices, weights).view(B, S, H)
        self.last_logits, self.last_indices = logits, indices       # [B*S, E], [B*S, top_k]
        return routed + self.shared_experts(x)

    def balance_loss(self, batch_seq: int) -> torch.Tensor:
        """Sequence-wise balance loss (delta d, V3-style complementary term). For each
        sequence: f_i = fraction of its top-k slots taken by expert i, P_i = mean routing
        prob mass on expert i; loss = E * mean_seq sum_i f_i * P_i. Minimized when load is
        spread evenly within each sequence (discourages within-sequence collapse)."""
        E, k = self.n_experts, self.top_k
        logits = self.last_logits.view(batch_seq, -1, E)            # [B, S, E]
        idx = self.last_indices.view(batch_seq, -1, k)              # [B, S, k]
        probs = _score(logits, self.scoring_func)
        probs = probs / (probs.sum(-1, keepdim=True) + 1e-20)
        P = probs.mean(dim=1)                                       # [B, E]
        f = F.one_hot(idx, E).sum(dim=(1, 2)).float() / (idx.shape[1] * k)   # [B, E]
        return E * (f * P).sum(-1).mean()
