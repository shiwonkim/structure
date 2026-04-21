"""SAIL StarMLP alignment layer (GLU variant).

Reproduces the StarMLP projector from SAIL (CVPR 2025: "Assessing and
Learning Alignment of Unimodal Vision and Language Models"). The forward
is a gated linear unit:

    out = g(ReLU6(f1(x)) * f2(x))

where f1, f2 expand to ``width_factor * input_dim`` and g projects down
to ``dim_alignment``. Both CLS (B, D) and token-level (B, T, D) inputs
are supported — the latter applies masked mean-pool after the GLU,
following the same convention as LinearAlignmentLayer and ResLowRankHead.

Learnable ``logit_scale`` and ``logit_bias`` for SigLipLoss are stored
here so they join the optimizer with the alignment parameters. When used
with CLIPLoss (fixed temperature), these are ignored.

Reference: /home/shiwon/SAIL/model/linear.py::StarMLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer
from src.alignment.linear_alignment_layer import _masked_mean_pool


@AlignmentFactory.register()
class SAILStarMLP(BaseAlignmentLayer):

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int = 512,
        width_factor: int = 8,
    ):
        super().__init__(input_dim=input_dim)
        self.dim_alignment = dim_alignment
        self.width_factor = width_factor
        hidden = width_factor * input_dim

        self.ln = nn.LayerNorm(input_dim)
        self.f1 = nn.Linear(input_dim, hidden)
        self.f2 = nn.Linear(input_dim, hidden)
        self.g = nn.Linear(hidden, dim_alignment)
        self.act = nn.ReLU6()

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self.ln(z)
        x1 = torch.clamp(self.f1(z), min=-1e3, max=1e3)
        x2 = torch.clamp(self.f2(z), min=-1e3, max=1e3)
        projected = self.g(self.act(x1) * x2)

        if projected.dim() == 3:
            projected = _masked_mean_pool(projected, mask)

        return F.normalize(projected, p=2, dim=-1)
