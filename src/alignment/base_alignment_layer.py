from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAlignmentLayer(ABC, nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reduce_for_structure_reg(self, z: torch.Tensor) -> torch.Tensor:
        """Reduce 3D token tensors to 2D for structure regularization.

        Subclasses override this to match their pooling architecture.
        Default: mean-pool all tokens (correct for Linear, MLP, BA).
        """
        if z.dim() == 3:
            return z.mean(dim=1)
        return z
