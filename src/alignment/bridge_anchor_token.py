"""Token-level Bridge Anchor alignment layer with Cross-Attention Pooling (CAP).

Key differentiator from the CLS variant: instead of treating each sample as a
single (B, D) CLS vector, this layer consumes the full token sequence
(B, T, D) and pools token contributions per anchor with a softmax over tokens.

Forward contract:
    input:  z (B, T, D) token features or (B, D) CLS fallback
            mask (B, T) optional — 1 = valid token, 0 = padding
    output: (B, K) L2-normalized profile (same shape as CLS variant, so
            CLIPLoss / STRUCTURE reg are unchanged)

Reference: bridge-anchors/bridge_anchors.py::_compute_profile(cross_attn path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


class BottleneckProjector(nn.Module):
    """Lightweight residual bottleneck projector.

    Zero-initialised on the up projection so the layer starts as identity.
    """

    def __init__(self, d_in: int, d_mid: int):
        super().__init__()
        self.down = nn.Linear(d_in, d_mid)
        self.act = nn.GELU()
        self.up = nn.Linear(d_mid, d_in)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


@AlignmentFactory.register()
class BridgeAnchorTokenAlignmentLayer(BaseAlignmentLayer):
    """Token-level BA with cross-attention pooling over tokens."""

    def __init__(
        self,
        input_dim: int,
        num_anchors: int | None = None,
        pool_temperature: float = 0.05,
        projector_dim: int = 0,
        init_method: str = "random",
        dim_alignment: int | None = None,
        pool_method: str = "cap",
    ):
        super().__init__(input_dim=input_dim)

        # dim_alignment alias survives STRUCTURE's YAML deep-merge (default.yaml
        # injects dim_alignment=256 into every alignment_layer_kwargs).
        if num_anchors is None:
            num_anchors = dim_alignment if dim_alignment is not None else 128
        self.num_anchors = num_anchors
        self.pool_temperature = pool_temperature
        self.projector_dim = projector_dim
        self.init_method = init_method
        self.pool_method = pool_method

        self.anchors = nn.Parameter(torch.empty(num_anchors, input_dim))
        if init_method in ("random", "normal"):
            nn.init.normal_(self.anchors)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, dim=-1)

        self.projector = (
            BottleneckProjector(input_dim, projector_dim) if projector_dim > 0 else None
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # CLS-fallback path for 2D input (used at eval time when only CLS is
        # available, or for backwards compat with CLS-only configs).
        if z.dim() == 2:
            z_norm = F.normalize(z, dim=-1)
            a_norm = F.normalize(self.anchors, dim=-1)
            profile = z_norm @ a_norm.T  # (B, K)
            return F.normalize(profile, dim=-1)

        # Token-level path: z is (B, T, D)
        if self.projector is not None:
            z = self.projector(z)

        z_norm = F.normalize(z, dim=-1)              # (B, T, D)
        a_norm = F.normalize(self.anchors, dim=-1)   # (K, D)
        sim = z_norm @ a_norm.T                       # (B, T, K)

        if getattr(self, "pool_method", "cap") == "mean":
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()   # (B, T, 1)
                profile = (sim * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                profile = sim.mean(dim=1)             # (B, K)
            return F.normalize(profile, dim=-1)

        # CAP path (default)
        logits = sim / self.pool_temperature         # (B, T, K)

        if mask is not None:
            logits = logits.masked_fill(
                ~mask.bool().unsqueeze(-1), float("-inf")
            )

        attn = F.softmax(logits, dim=1)              # softmax over tokens
        attn = attn.nan_to_num(0.0)                   # all-masked safety

        profile = (attn * sim).sum(dim=1)            # (B, K)
        return F.normalize(profile, dim=-1)
