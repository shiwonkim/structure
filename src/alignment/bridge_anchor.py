"""Bridge Anchor alignment layer for STRUCTURE framework.

Core idea: instead of projecting embeddings to a shared space (like Linear/MLP),
measure each embedding's similarity to K learnable anchor points, producing a
K-dimensional "distance profile" that serves as the aligned representation.

This is the vanilla version — no cross-attention pooling, no projectors,
no multi-expert, no auxiliary losses. Just anchors + cosine + L2 norm.

Each modality gets its own instance with its own anchors (STRUCTURE instantiates
one alignment layer per modality), matching the original BA design where
anchors_img and anchors_txt are separate parameter sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


@AlignmentFactory.register()
class BridgeAnchorAlignmentLayer(BaseAlignmentLayer):
    """Aligns embeddings via cosine similarity to K learnable anchor points.

    Input:  (B, D) — CLS/pooled embedding from one modality
    Output: (B, K) — L2-normalized similarity profile to the anchors
    """

    def __init__(
        self,
        input_dim: int,
        num_anchors: int | None = None,
        init_method: str = "random",
        dim_alignment: int | None = None,
    ):
        super().__init__(input_dim=input_dim)
        # For BA the output dim IS the anchor count, so dim_alignment aliases num_anchors.
        # num_anchors takes precedence if both are given (STRUCTURE's default.yaml sets dim_alignment).
        if num_anchors is None:
            if dim_alignment is None:
                num_anchors = 128
            else:
                num_anchors = dim_alignment
        self.num_anchors = num_anchors
        self.init_method = init_method

        self.anchors = nn.Parameter(torch.empty(num_anchors, input_dim))
        if init_method in ("random", "normal"):
            nn.init.normal_(self.anchors)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z, dim=-1)
        a_norm = F.normalize(self.anchors, dim=-1)
        profile = z_norm @ a_norm.T  # (B, K)
        return F.normalize(profile, dim=-1)
