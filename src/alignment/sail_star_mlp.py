"""SAIL StarMLP alignment layer (GLU variant).

Reproduces the StarMLP projector from SAIL (CVPR 2025: "Assessing and
Learning Alignment of Unimodal Vision and Language Models"). The forward
is a gated linear unit:

    out = g(ReLU6(f1(x)) * f2(x))

where f1, f2 expand to ``width_factor * input_dim`` and g projects down
to ``dim_alignment``.

Supports SAIL's default ``concat`` aggregation mode for images:
    image: cat(CLS, mean(patches)) → (B, 2*D) → StarMLP(2*D → dim_alignment)
    text:  mean-pool tokens → (B, D) → StarMLP(D → dim_alignment)

Since STRUCTURE creates one alignment layer per modality with the same
class and kwargs, both modalities' components live on each instance.
``set_modality('image' | 'text')`` selects the active branch, similar
to FreezeAlignAlignmentLayer.

Reference: /home/shiwon/SAIL/model/linear.py::StarMLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer
from src.alignment.linear_alignment_layer import _masked_mean_pool


def _build_star_mlp(input_dim, dim_alignment, width_factor):
    hidden = width_factor * input_dim
    return nn.ModuleDict({
        "ln": nn.LayerNorm(input_dim),
        "f1": nn.Linear(input_dim, hidden),
        "f2": nn.Linear(input_dim, hidden),
        "g": nn.Linear(hidden, dim_alignment),
    })


@AlignmentFactory.register()
class SAILStarMLP(BaseAlignmentLayer):

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int = 512,
        width_factor: int = 8,
        concat_cls_patch: bool = True,
    ):
        super().__init__(input_dim=input_dim)
        self.dim_alignment = dim_alignment
        self.width_factor = width_factor
        self.concat_cls_patch = concat_cls_patch
        self._modality: str | None = None

        # Image projector: input_dim is 2*D if concat mode, else D
        img_input_dim = 2 * input_dim if concat_cls_patch else input_dim
        self.image_mlp = _build_star_mlp(img_input_dim, dim_alignment, width_factor)

        # Text projector: always input_dim (mean-pool, no concat)
        self.text_mlp = _build_star_mlp(input_dim, dim_alignment, width_factor)

        self.act = nn.ReLU6()

    def set_modality(self, modality: str) -> None:
        if modality not in ("image", "text"):
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
        self._modality = modality
        logger.debug(f"SAILStarMLP.set_modality({modality!r})")

    def _star_forward(self, z: torch.Tensor, mlp: nn.ModuleDict) -> torch.Tensor:
        z = mlp["ln"](z)
        x1 = torch.clamp(mlp["f1"](z), min=-1e3, max=1e3)
        x2 = torch.clamp(mlp["f2"](z), min=-1e3, max=1e3)
        return mlp["g"](self.act(x1) * x2)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        modality = self._modality
        if modality is None:
            modality = "text" if mask is not None else "image"

        if modality == "image":
            if z.dim() == 3 and self.concat_cls_patch:
                cls_token = z[:, 0, :]
                patch_mean = z[:, 1:, :].mean(dim=1)
                z = torch.cat([cls_token, patch_mean], dim=-1)  # (B, 2*D)
                projected = self._star_forward(z, self.image_mlp)
            elif z.dim() == 3:
                projected = self._star_forward(z, self.image_mlp)
                projected = _masked_mean_pool(projected, mask)
            else:
                # 2D CLS-only fallback
                if self.concat_cls_patch:
                    z = torch.cat([z, z], dim=-1)  # (B, 2*D) — duplicate CLS as proxy
                projected = self._star_forward(z, self.image_mlp)
        else:
            # Text: mean-pool then project
            if z.dim() == 3:
                z = _masked_mean_pool(z, mask)
            projected = self._star_forward(z, self.text_mlp)

        return F.normalize(projected, p=2, dim=-1)

    # Structure reg uses the base class default: mean-pool all tokens.
    # Matching the concat forward (cat(CLS, mean(patches))) was tested
    # on FreezeAlign (similar CLS+patches architecture) and performed
    # worse due to 2× magnitude causing structure loss to dominate.
