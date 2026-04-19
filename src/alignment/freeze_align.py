"""Freeze-Align projector baseline (Maniparambil et al., CVPR 2025).

Faithful reproduction of the official Freeze-Align ``combined_vis_cls`` model,
adapted to STRUCTURE's per-modality alignment-layer contract. The official
implementation runs both modalities through a single ``CLIP`` module; STRUCTURE
instantiates one alignment layer per modality with the SAME class and SAME
kwargs, so we put both modalities' components in this class and route forward
via ``set_modality(...)``.

References:
    - Original code:
        freeze-align/train/models/clip_adjustable_combined_vis_cls.py
    - Verified STRUCTURE-independent reproduction (audited side-by-side):
        bridge-anchors/src/models/freeze_align.py

Architecture (vis_cls combined config):
    Vision (token):
        local = local_vision_proj(z)                    # (B, T, embed)
        local = local[:, 1:, :].mean(dim=1)             # mean pool patches
        cls   = cls_vision_proj(z[:, 0, :])             # original CLS
        feat  = local + cls
        feat  = L2_normalize(feat)
    Vision (CLS fallback):
        feat  = L2_normalize(cls_vision_proj(z))
    Text (token):
        proj  = local_text_proj(z)                      # (B, S, embed)
        feat  = masked_mean(proj, mask)                 # (B, embed)
        feat  = text_proj(feat)
        feat  = L2_normalize(feat)
    Text (CLS fallback):
        feat  = L2_normalize(text_proj(z))

Components:
    PatchProjection: y = Linear(x) + [Linear -> GELU -> Linear](x)
        The linear branch IS the residual; no explicit x+ skip. GELU not ReLU.
    ProjectionHead (text MLP):
        proj = Linear(x); h = GELU(proj) -> Linear -> Dropout
        out  = LayerNorm(h + proj)

Notes vs. the original paper:
    - Temperature: the original uses a learnable nn.Parameter clamped to
      [0.001, 0.5]. STRUCTURE's CLIPLoss already owns the temperature
      (config-level, fixed at 0.05 by default), so we do NOT add a learnable
      temp here. Same loss across all alignment methods = fair comparison.
    - SPARC fine-grained loss: omitted. STRUCTURE's training loop only consumes
      the (B, K) aligned features, and the SPARC loss is a separate alignment
      objective.
    - Both modalities' components are instantiated on every layer instance
      (image + text). Only the active modality contributes to the loss, so the
      unused half receives no gradient and is effectively dead weight in
      memory. This trade-off is the cost of fitting STRUCTURE's "one class,
      one kwargs dict" alignment-factory contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


class PatchProjection(nn.Module):
    """Token projector: residual sum of linear and non-linear branches.

    output = Linear(x) + [Linear(x) -> GELU -> Linear(x)]
    The linear branch serves as the residual/skip path.
    """

    def __init__(self, embedding_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.linear_projection = nn.Linear(embedding_dim, projection_dim)
        self.non_linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_projection(x) + self.non_linear_projection(x)


class ProjectionHead(nn.Module):
    """MLP projection head with residual connection and LayerNorm.

    projected = Linear(x); h = GELU(projected) -> Linear -> Dropout
    output    = LayerNorm(h + projected)
    """

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


@AlignmentFactory.register()
class FreezeAlignAlignmentLayer(BaseAlignmentLayer):
    """Freeze-Align projector baseline for STRUCTURE.

    Both modalities' components are instantiated on every layer instance;
    ``set_modality('image' | 'text')`` selects the active branch for forward.
    If ``set_modality`` is never called, ``forward`` auto-detects: a non-None
    ``mask`` argument means text, otherwise image.

    Args:
        input_dim: Encoder output dimension. Required by STRUCTURE's contract.
            Both modalities use ``input_dim`` for their first-layer width;
            STRUCTURE happens to pair encoders with matching dims (1024/1024
            for the large-model defaults, 384/384 for ViT-S + MiniLM).
        embed_dim: Shared projection width. Falls back to ``dim_alignment``
            (alias) and finally to ``input_dim`` so the layer remains
            workable even when only the YAML-merged default is provided.
        dropout: Dropout rate inside the projectors and the text MLP head.
        dim_alignment: alias for ``embed_dim``; survives STRUCTURE's YAML
            deep-merge of ``alignment_layer_kwargs.dim_alignment`` from
            ``configs/default.yaml``.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int | None = None,
        dropout: float = 0.1,
        dim_alignment: int | None = None,
    ) -> None:
        super().__init__(input_dim=input_dim)

        if embed_dim is None:
            embed_dim = dim_alignment if dim_alignment is not None else input_dim
        self.embed_dim = int(embed_dim)
        self.dropout_p = float(dropout)
        # set_modality() flips this; until then forward() auto-detects via mask
        self._modality: str | None = None

        # --- Vision components ---
        # local_vision_proj: applied to ALL tokens, then patches mean-pooled
        self.local_vision_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(self.dropout_p),
            PatchProjection(input_dim, self.embed_dim),
        )
        # cls_vision_proj: applied to ORIGINAL CLS token (before local proj)
        self.cls_vision_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(self.dropout_p),
            PatchProjection(input_dim, self.embed_dim),
        )

        # --- Text components ---
        # local_text_proj: PatchProjection on ALL text tokens (CLS included)
        self.local_text_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(self.dropout_p),
            PatchProjection(input_dim, self.embed_dim),
        )
        # text_proj: ONE MLP head shared by both token and CLS-fallback paths.
        # The CLS fallback routes the single CLS token through local_text_proj
        # first (treated as a length-1 sequence), so text_proj always sees an
        # embed_dim input regardless of whether input_dim matches.
        self.text_proj = ProjectionHead(
            embedding_dim=self.embed_dim,
            projection_dim=self.embed_dim,
            dropout=self.dropout_p,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_modality(self, modality: str) -> None:
        if modality not in ("image", "text"):
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
        self._modality = modality
        logger.debug(f"FreezeAlignAlignmentLayer.set_modality({modality!r})")

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        modality = self._modality
        if modality is None:
            # Auto-detect when set_modality wasn't called: presence of a mask
            # is the universal signal for the text branch in STRUCTURE.
            modality = "text" if mask is not None else "image"

        if modality == "image":
            return self._forward_image(z)
        return self._forward_text(z, mask)

    # ------------------------------------------------------------------
    # Modality-specific forwards
    # ------------------------------------------------------------------
    def _forward_image(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            # CLS fallback: only the CLS projector is exercised.
            feat = self.cls_vision_proj(z)
            return F.normalize(feat, dim=-1)

        # Token-level path: (B, T, D)
        cls_token_orig = z[:, 0, :]                        # (B, D)
        all_projected = self.local_vision_proj(z)          # (B, T, embed)
        local_feat = all_projected[:, 1:, :].mean(dim=1)   # mean over patches
        cls_feat = self.cls_vision_proj(cls_token_orig)    # (B, embed)
        image_feat = local_feat + cls_feat
        return F.normalize(image_feat, dim=-1)

    def _forward_text(
        self, z: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if z.dim() == 2:
            # CLS fallback: route the single CLS token through the full token
            # pipeline as a length-1 sequence. Mean-pool over 1 token = squeeze.
            z_seq = z.unsqueeze(1)                          # (B, D) -> (B, 1, D)
            projected = self.local_text_proj(z_seq)         # (B, 1, embed)
            pooled = projected.squeeze(1)                   # (B, embed)
            feat = self.text_proj(pooled)                   # (B, embed)
            return F.normalize(feat, dim=-1)

        if mask is None:
            raise ValueError(
                "FreezeAlignAlignmentLayer text token mode requires an "
                "attention mask"
            )
        # Token-level path: (B, S, D)
        projected = self.local_text_proj(z)                # (B, S, embed)
        m = mask.to(dtype=projected.dtype).unsqueeze(-1)   # (B, S, 1)
        denom = m.sum(dim=1).clamp(min=1)
        pooled = (projected * m).sum(dim=1) / denom        # (B, embed)
        text_feat = self.text_proj(pooled)                 # (B, embed)
        return F.normalize(text_feat, dim=-1)

    # ------------------------------------------------------------------
    # Structure reg reduction
    # ------------------------------------------------------------------
    # Uses the base class default: mean-pool all tokens (v2).
    # v3 (patches_mean + CLS) was tested but performed worse — the
    # summed representation has ~2× magnitude which makes the structure
    # loss dominate and collapse retrieval.

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def active_param_count(self) -> int:
        """Parameters touched by the active modality's forward pass."""
        if self._modality == "image":
            mods = [self.local_vision_proj, self.cls_vision_proj]
        elif self._modality == "text":
            mods = [self.local_text_proj, self.text_proj]
        else:
            return sum(p.numel() for p in self.parameters())
        return sum(p.numel() for m in mods for p in m.parameters())
