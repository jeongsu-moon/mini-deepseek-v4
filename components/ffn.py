"""DeepSeekMoE — Step 5 deliverable (ROADMAP.md §5).

Study ONLY the verified V4 deltas (the MoE paradigm itself is inherited from V2/V3):
  - affinity activation Sigmoid -> Sqrt(Softplus(.))   (scoring_func='sqrtsoftplus')
  - first 3 MoE layers use frozen Hash routing (hash_moe), tid2eid[input_ids]
  - V3's n_group / topk_group constraint removed
  - aux-loss-free balancing (inherited: Wang 2024 / V3) + a new sequence-wise balance loss

Toy config (config.py): n_routed_experts 8-16, n_active 2, shared 1.
Do NOT instantiate V4-Pro's 384 / Flash's 256 experts here. Success is routing
behaviour (load CV, specialization, controller convergence), NOT "beats dense".

torch.compile note: top-k select + token scatter/gather graph-break — keep
routing/dispatch EAGER, compile only the per-expert FFN. See ROADMAP.md §5.
"""
import torch.nn as nn


class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "DeepSeekMoE (ffn_type='moe') is a Step 5 deliverable — not yet implemented.\n"
            "Build it in components/ffn.py per ROADMAP.md §5 (V4 deltas only), then\n"
            "parity-check vs transformers deepseek_v4 MoE submodule (pinned commit)."
        )
