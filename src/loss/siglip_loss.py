"""SigLip sigmoid loss for SAIL integration.

Unlike CLIPLoss (softmax cross-entropy over rows/columns), SigLipLoss
treats each (image_i, text_j) pair independently via sigmoid binary
classification: positive pairs (i==j) get label +1, negatives get -1.

    logits = scale * norm(img) @ norm(txt).T + bias
    loss = -mean(logsigmoid(labels * logits))

The learnable ``logit_scale`` and ``logit_bias`` live on the alignment
layer (SAILStarMLP), not here — this loss reads them from the layer.

Reference: /home/shiwon/SAIL/model/loss.py::SigLipLoss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.clip_loss import structure_reg, DistanceFunction, Centering


class SigLipLoss(nn.Module):

    def __init__(
        self,
        structure_lambda: float = 0,
        structure_levels: int = 1,
        structure_margin: float = 0.0,
        structure_weighting: str = "none",
        logit_scale_init: float = 20.0,
        logit_bias_init: float = -10.0,
        **kwargs,
    ):
        super().__init__()
        self.structure_lambda = structure_lambda
        self.structure_use_only_unimodal = False
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.logit_bias = nn.Parameter(torch.tensor(logit_bias_init))

    def step(self):
        pass

    def forward(
        self,
        image_embeddings_aligned: torch.Tensor,
        text_embeddings_aligned: torch.Tensor,
        logit_scale: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
        image_embeddings_original: torch.Tensor | None = None,
        text_embeddings_original: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if logit_scale is None:
            logit_scale = self.logit_scale
        if logit_bias is None:
            logit_bias = self.logit_bias
        image_norm = F.normalize(image_embeddings_aligned, p=2, dim=-1)
        text_norm = F.normalize(text_embeddings_aligned, p=2, dim=-1)

        logits = logit_scale.exp() * image_norm @ text_norm.T + logit_bias

        N = logits.shape[0]
        labels = 2 * torch.eye(N, device=logits.device, dtype=logits.dtype) - 1

        loss_matrix = F.logsigmoid(labels * logits)
        siglip_loss = -loss_matrix.mean()

        with torch.no_grad():
            diag = torch.diagonal(loss_matrix).sum()
            pos_loss = -diag / (N * N)
            neg_loss = -(loss_matrix.sum() - diag) / (N * N)

        total_loss = siglip_loss
        output = {
            "siglip_loss": siglip_loss.item(),
            "positive_loss": pos_loss.item(),
            "negative_loss": neg_loss.item(),
        }

        if (
            self.structure_lambda > 0
            and image_embeddings_original is not None
            and text_embeddings_original is not None
        ):
            img_str = structure_reg(
                image_embeddings_original, image_embeddings_aligned,
                levels=1, temperature=0.1,
            )
            txt_str = structure_reg(
                text_embeddings_original, text_embeddings_aligned,
                levels=1, temperature=0.1,
            )
            str_loss = (img_str + txt_str) / 2
            total_loss = total_loss + self.structure_lambda * str_loss
            output["structure_loss"] = self.structure_lambda * str_loss.item()

        output["overall_loss"] = total_loss
        return output
