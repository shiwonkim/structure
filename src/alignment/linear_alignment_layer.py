import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


def _masked_mean_pool(
    x: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


@AlignmentFactory.register()
class LinearAlignmentLayer(BaseAlignmentLayer):

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int,
        normalize_to_hypersphere: bool = False,
    ):
        super().__init__(input_dim=input_dim)
        self.normalize_to_hypersphere = normalize_to_hypersphere
        self.linear_mapping = torch.nn.Linear(input_dim, dim_alignment)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        projected = self.linear_mapping(z)

        if projected.dim() == 3:
            projected = _masked_mean_pool(projected, mask)

        if self.normalize_to_hypersphere:
            return F.normalize(projected, p=2, dim=-1)
        return projected
