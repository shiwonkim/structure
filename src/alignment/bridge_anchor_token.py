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
    """Token-level BA with cross-attention pooling over tokens.

    When ``cls_attn_prior=True``, the per-anchor softmax logits are
    additively biased by ``beta_k * log(cls_attn + eps)`` where
    ``cls_attn`` is the DINOv2 CLS-token self-attention pattern
    (patches only, shape ``(B, num_patches)``), and ``beta_k`` is a
    learnable per-anchor scalar initialized to ``cls_attn_beta_init``.
    This softly steers each anchor toward patches that the backbone's
    own CLS token considers globally important, while letting each
    anchor learn how much weight to place on that prior (``beta`` can
    drive toward 0 to ignore it).

    The ``cls_attn`` tensor is shape ``(B, P)`` with one entry per
    *patch* (no CLS position). When the sequence being pooled has
    length ``T > P`` (e.g. ``T = 1 + P`` with the CLS token at position
    0), zero-padding is inserted at the front so the log-prior at the
    CLS position becomes ``log(eps)`` and effectively suppresses it
    from the attention pool. This matches the reference
    bridge-anchors implementation.

    When ``cls_attn`` is ``None`` (no extraction available), the layer
    silently reverts to standard CAP — no error, no branch surprise.
    """

    def __init__(
        self,
        input_dim: int,
        num_anchors: int | None = None,
        pool_temperature: float = 0.05,
        projector_dim: int = 0,
        init_method: str = "random",
        dim_alignment: int | None = None,
        cls_attn_prior: bool = False,
        cls_attn_beta_init: float = 1.0,
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

        # CLS-attention prior: one learnable scalar per anchor. Only
        # instantiated when enabled so checkpoints for existing runs
        # stay compatible.
        self.cls_attn_prior = bool(cls_attn_prior)
        self.cls_attn_beta_init = float(cls_attn_beta_init)
        if self.cls_attn_prior:
            self.beta = nn.Parameter(
                torch.full((num_anchors,), self.cls_attn_beta_init)
            )
        else:
            self.beta = None

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        cls_attn: torch.Tensor | None = None,
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

        logits = sim / self.pool_temperature         # (B, T, K)

        # CLS-attention prior. Silently skipped if cls_attn is None or
        # the feature is disabled, so all existing call sites keep
        # working without modification.
        if self.cls_attn_prior and cls_attn is not None:
            T = logits.shape[1]
            P = cls_attn.shape[1]
            if P < T:
                # cls_attn is over patches only; pad with zeros at the
                # front so the CLS position (and any other non-patch
                # prefix) has log(eps) prior and is suppressed.
                pad = torch.zeros(
                    cls_attn.shape[0],
                    T - P,
                    device=cls_attn.device,
                    dtype=cls_attn.dtype,
                )
                cls_attn_padded = torch.cat([pad, cls_attn], dim=1)
            else:
                cls_attn_padded = cls_attn[:, :T]
            log_prior = torch.log(
                cls_attn_padded.to(logits.dtype) + 1e-8
            ).unsqueeze(-1)                          # (B, T, 1)
            # self.beta has shape (K,); broadcasts over the K axis.
            betas = self.beta.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
            logits = logits + betas * log_prior       # (B, T, K)

        if mask is not None:
            logits = logits.masked_fill(
                ~mask.bool().unsqueeze(-1), float("-inf")
            )

        attn = F.softmax(logits, dim=1)              # softmax over tokens
        attn = attn.nan_to_num(0.0)                   # all-masked safety

        profile = (attn * sim).sum(dim=1)            # (B, K)
        return F.normalize(profile, dim=-1)
