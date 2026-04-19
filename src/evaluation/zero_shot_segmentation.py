"""Zero-shot segmentation evaluation for the STRUCTURE alignment suite.

Unified framework: four localization methods × two text-encoding strategies,
all consuming the same shared evaluation loop. Methods differ only in how
they produce per-patch image descriptors and per-class text descriptors; the
rest of the pipeline — cosine similarity, grid upscaling, confusion-matrix
accumulation, per-class IoU, and mean-IoU over foreground classes — is
shared.

Improvements over the freeze-align reference (``train/zero_shot_segmentation.py``):

- Standard mIoU computed from a single (C, C) confusion matrix accumulated
  across the val set, not per-image averaging.
- Patch grid size ``h = w = int(sqrt(P))`` derived from the token count, no
  hard-coded 18×18.
- ``F.interpolate(mode='bilinear', align_corners=False)`` for similarity
  maps; GT masks consumed at native resolution (no lossy resize on targets).
- ``argmax`` over classes, no threshold hack — standard semantic segmentation.
- ``ignore_index=255`` excluded from both prediction and target when updating
  the confusion matrix.
- Ported prompt-ensembling: 80 OpenAI ImageNet templates, averaged per class,
  reusing ``src/evaluation/consts.py::DATASETS_TO_TEMPLATES['imagenet']``.

Methods
-------
1. ``direct_cosine``   — raw encoder patches vs raw text embeddings
                         (no alignment layer; baseline)
2. ``freezealign``     — FreezeAlign ``local_vision_proj`` per-patch features
                         vs FreezeAlign full text forward
3. ``anchor_codebook`` — BA-token: per-patch K-dim anchor similarity vector
                         vs per-class CAP profile; both sides live in the
                         anchor codebook
4. ``attention_map``   — BA-token: per-anchor softmax attention over patches
                         vs per-class CAP profile; image side is "where each
                         anchor looks", text side is "which anchors each class
                         activates"

Text strategies
---------------
- ``raw``      — class name as-is (single-template format ``"{}"``)
- ``ensemble`` — 80 ImageNet OpenAI templates averaged per class

Usage
-----
    python src/evaluation/zero_shot_segmentation.py \\
        --config configs/ba/vits_minilm/token_k256.yaml \\
        --checkpoint results/alignment-.../checkpoint-epochN.pth \\
        --layer-img 11 --layer-txt 6 \\
        --dataset voc2012 \\
        --methods anchor_codebook,attention_map,direct_cosine \\
        --text-strategies raw,ensemble \\
        --gpu 0

Checkpoint-free mode (runs only ``direct_cosine``)::

    python src/evaluation/zero_shot_segmentation.py \\
        --config configs/default.yaml \\
        --layer-img 11 --layer-txt 6 \\
        --methods direct_cosine \\
        --text-strategies raw,ensemble

Output: console table + per-class IoU CSV at ``--output-csv``.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import _ensure_rgb_image
from src.evaluation.consts import DATASETS_TO_TEMPLATES
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.models.text.models import load_llm, load_tokenizer

# ------------------------------------------------------------------
# Dataset specs
# ------------------------------------------------------------------

# PASCAL VOC 2012 segmentation: 0 = background, 1..20 = foreground,
# 255 = ignore. Class names match torchvision's VOCSegmentation encoding.
VOC2012_CLASSES: List[str] = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Natural-language rewrites for the tokenizer. Helps both raw and ensemble
# strategies since the encoder's vocab rarely contains the joined forms.
VOC2012_CLASS_PROMPTS: Dict[str, str] = {
    "background": "background",
    "aeroplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorbike",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "sofa",
    "train": "train",
    "tvmonitor": "tv monitor",
}

IGNORE_INDEX = 255


# Pascal Context 59 foreground classes (from https://cs.stanford.edu/~roozbeh/
# pascal-context/59_labels.txt), index 0 reserved for background. Matches
# the PASCAL-Context-59 benchmark used by MaskCLIP / TCL / GroupViT.
PASCAL_CONTEXT_CLASSES: List[str] = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "table", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor", "bag", "bed",
    "bench", "book", "building", "cabinet", "ceiling", "cloth", "computer",
    "cup", "door", "fence", "floor", "flower", "food", "grass", "ground",
    "keyboard", "light", "mountain", "mouse", "curtain", "platform", "sign",
    "plate", "road", "rock", "shelves", "sidewalk", "sky", "snow",
    "bedclothes", "track", "tree", "truck", "wall", "water", "window", "wood",
]

PASCAL_CONTEXT_PROMPTS: Dict[str, str] = {
    c: c for c in PASCAL_CONTEXT_CLASSES
}
PASCAL_CONTEXT_PROMPTS.update({
    "pottedplant": "potted plant",
    "tvmonitor": "tv monitor",
    "bedclothes": "bedclothes",
})


# ADE20K 150 classes from the ADEChallenge2016 scene-parsing benchmark.
# Index 0 is ignore (annotation label 0 = "unknown") — treat as ignore, not
# background. Classes 1..150 are real objects. We add a dummy "background"
# at index 0 so the argmax dimension matches torchvision's label encoding.
ADE20K_CLASSES: List[str] = [
    "background",
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
    "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]
assert len(ADE20K_CLASSES) == 151, f"expected 151 (1 bg + 150 classes), got {len(ADE20K_CLASSES)}"
ADE20K_PROMPTS: Dict[str, str] = {c: c for c in ADE20K_CLASSES}
# ADE20K uses label 0 = unknown/ignore in the raw PNG; we remap to our own
# ignore index so the confusion matrix excludes those pixels.
ADE20K_IGNORE_INDEX = 0


# Cityscapes 19 evaluation classes (from the standard training/eval label set).
CITYSCAPES_CLASSES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
CITYSCAPES_PROMPTS: Dict[str, str] = {c: c for c in CITYSCAPES_CLASSES}
# Cityscapes 255 = ignore; we won't add a "background" pseudo-class, so
# foreground mIoU == all-class mIoU (exclude_background=False at call site).
CITYSCAPES_IGNORE_INDEX = 255


@dataclass
class DatasetSpec:
    name: str
    classes: Sequence[str]
    prompts: Dict[str, str]
    num_classes: int
    ignore_index: int
    #: Whether to exclude class 0 from foreground mIoU computation. True for
    #: Pascal VOC / Pascal Context where index 0 is "background"; False for
    #: ADE20K (class 0 is already an ignore label) and Cityscapes (no
    #: background class at all).
    has_background: bool = True

    @classmethod
    def for_voc2012(cls) -> "DatasetSpec":
        return cls(
            name="voc2012",
            classes=VOC2012_CLASSES,
            prompts=VOC2012_CLASS_PROMPTS,
            num_classes=len(VOC2012_CLASSES),
            ignore_index=IGNORE_INDEX,
            has_background=True,
        )

    @classmethod
    def for_pascal_context(cls) -> "DatasetSpec":
        return cls(
            name="pascal_context",
            classes=PASCAL_CONTEXT_CLASSES,
            prompts=PASCAL_CONTEXT_PROMPTS,
            num_classes=len(PASCAL_CONTEXT_CLASSES),
            # Pascal Context uses label 0 for background; unlabeled/void is
            # typically -1 or absent. The .mat labels from the Stanford
            # release use 0..59 directly so no per-dataset ignore is needed.
            # We still pass IGNORE_INDEX so any out-of-range pixels are skipped.
            ignore_index=IGNORE_INDEX,
            has_background=True,
        )

    @classmethod
    def for_ade20k(cls) -> "DatasetSpec":
        return cls(
            name="ade20k",
            classes=ADE20K_CLASSES,
            prompts=ADE20K_PROMPTS,
            num_classes=len(ADE20K_CLASSES),
            ignore_index=ADE20K_IGNORE_INDEX,
            has_background=False,   # the "background" at index 0 is unknown, not a real class
        )

    @classmethod
    def for_cityscapes(cls) -> "DatasetSpec":
        return cls(
            name="cityscapes",
            classes=CITYSCAPES_CLASSES,
            prompts=CITYSCAPES_PROMPTS,
            num_classes=len(CITYSCAPES_CLASSES),
            ignore_index=CITYSCAPES_IGNORE_INDEX,
            has_background=False,
        )


# ------------------------------------------------------------------
# Text strategies
# ------------------------------------------------------------------

def get_text_templates(strategy: str) -> Sequence[str]:
    """Return a list of format strings for the chosen text strategy."""
    if strategy == "raw":
        return ["{}"]
    if strategy == "ensemble":
        return list(DATASETS_TO_TEMPLATES["imagenet"])
    raise ValueError(
        f"unknown text strategy: {strategy!r} (expected 'raw' or 'ensemble')"
    )


# ------------------------------------------------------------------
# Method interface — each method knows how to produce (P, D_m) patch
# descriptors and (C, D_m) class descriptors in its own aligned space.
# ------------------------------------------------------------------

class SegmentationMethod:
    name: str = "<abstract>"
    #: ``"cls"`` if strict CLS is the natural text pooling; ``"avg"`` or
    #: ``"none"`` otherwise. Only consumed by the shared text path.
    pool_txt: str = "avg"

    def get_patch_features(
        self, layer_feats: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Return a (P, D_method) per-patch descriptor (CLS stripped)."""
        raise NotImplementedError

    def get_text_features(
        self,
        classnames: Sequence[str],
        templates: Sequence[str],
        tokenizer,
        language_model,
        layer_txt: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a (C, D_method) per-class descriptor in the same space."""
        raise NotImplementedError


class DirectCosineMethod(SegmentationMethod):
    """Raw DINOv2 patches vs raw text encoder embeddings.

    Baseline with no alignment head — gives a floor for how much value the
    trained alignment layer adds on dense prediction.
    """
    name = "direct_cosine"

    def __init__(self, pool_txt: str):
        self.pool_txt = pool_txt

    def get_patch_features(self, layer_feats, device):
        # layer_feats: (T, D) with CLS at index 0. Return P patches only.
        return layer_feats[1:, :].to(device)

    def get_text_features(
        self, classnames, templates, tokenizer, language_model,
        layer_txt, device,
    ):
        return build_zero_shot_classifier(
            language_model=language_model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            dataset=None,
            layer_index=layer_txt,
            alignment_layer=None,
            num_classes_per_batch=8,
            device=device,
            pool_txt=self.pool_txt,
            save_path=None,
            token_level=False,
        ).to(device)


class FreezeAlignMethod(SegmentationMethod):
    """FreezeAlign ``local_vision_proj`` per-patch features vs full text forward.

    Image side mirrors FreezeAlign's token-path local branch *without* the
    mean pool and CLS addition — projects every patch through
    ``local_vision_proj``, keeps all patches, strips the CLS position.

    Text side uses the full trained FA text pipeline (``local_text_proj`` +
    ``text_proj``), identical to training.
    """
    name = "freezealign"
    pool_txt = "none"

    def __init__(self, alignment_image, alignment_text):
        self.alignment_image = alignment_image
        self.alignment_text = alignment_text

    def get_patch_features(self, layer_feats, device):
        with torch.no_grad():
            z = layer_feats.unsqueeze(0).to(device)          # (1, T, D)
            projected = self.alignment_image.local_vision_proj(z)  # (1, T, E)
            return projected.squeeze(0)[1:, :]                # (P, E)

    def get_text_features(
        self, classnames, templates, tokenizer, language_model,
        layer_txt, device,
    ):
        # Token-level path: alignment_text is routed through set_modality('text'),
        # so build_zero_shot_classifier will call it as
        # alignment_text(tokens, mask=...) and get (B, E) per prompt.
        return build_zero_shot_classifier(
            language_model=language_model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            dataset=None,
            layer_index=layer_txt,
            alignment_layer=self.alignment_text,
            num_classes_per_batch=8,
            device=device,
            pool_txt="none",
            save_path=None,
            token_level=True,
        ).to(device)


class BAAnchorCodebookMethod(SegmentationMethod):
    """BridgeAnchors: each patch is encoded as a K-dim anchor-similarity
    vector; each class is encoded as a K-dim CAP profile. Both sides live
    in the same anchor codebook.
    """
    name = "anchor_codebook"
    pool_txt = "none"

    def __init__(self, alignment_image, alignment_text):
        self.alignment_image = alignment_image
        self.alignment_text = alignment_text

    def get_patch_features(self, layer_feats, device):
        with torch.no_grad():
            z = layer_feats[1:, :].to(device)  # (P, D)
            z_n = F.normalize(z, dim=-1)
            a_n = F.normalize(self.alignment_image.anchors, dim=-1)  # (K, D)
            return z_n @ a_n.T  # (P, K)

    def get_text_features(
        self, classnames, templates, tokenizer, language_model,
        layer_txt, device,
    ):
        return build_zero_shot_classifier(
            language_model=language_model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            dataset=None,
            layer_index=layer_txt,
            alignment_layer=self.alignment_text,
            num_classes_per_batch=8,
            device=device,
            pool_txt="none",
            save_path=None,
            token_level=True,
        ).to(device)


class BAAttentionMapMethod(SegmentationMethod):
    """BridgeAnchors attention-map method: per-anchor softmax attention over
    patches as the image descriptor. Interpretable as "anchor k attends to
    patch t"; comparing with the class-side K-dim profile answers "class c
    activates anchor k" and the product gives "patch t belongs to class c".

    Uses the same temperature ``pool_temperature`` as training.
    """
    name = "attention_map"
    pool_txt = "none"

    def __init__(self, alignment_image, alignment_text, pool_temperature: float):
        self.alignment_image = alignment_image
        self.alignment_text = alignment_text
        self.pool_temperature = pool_temperature

    def get_patch_features(self, layer_feats, device):
        with torch.no_grad():
            z = layer_feats[1:, :].to(device)  # (P, D)
            z_n = F.normalize(z, dim=-1)
            a_n = F.normalize(self.alignment_image.anchors, dim=-1)  # (K, D)
            sim = z_n @ a_n.T                                          # (P, K)
            attn = F.softmax(sim / self.pool_temperature, dim=0)       # softmax over P
            return attn  # (P, K)

    def get_text_features(
        self, classnames, templates, tokenizer, language_model,
        layer_txt, device,
    ):
        return build_zero_shot_classifier(
            language_model=language_model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            dataset=None,
            layer_index=layer_txt,
            alignment_layer=self.alignment_text,
            num_classes_per_batch=8,
            device=device,
            pool_txt="none",
            save_path=None,
            token_level=True,
        ).to(device)


class LinearPerPatchMethod(SegmentationMethod):
    """Per-patch projection through a trained Linear or MLP (ResLowRankHead)
    alignment layer, WITHOUT the mean-pool step. Produces (P, D_out) per-patch
    descriptors that can be compared against (C, D_out) text descriptors.

    For Linear: patches → layer.linear_mapping → (P, D_out)
    For ResLowRankHead: patches → P(z) + α·W₂(GELU(W₁(z))) → (P, D_out)

    Text side uses the same alignment_text layer with token_level=True
    so text features are in the same D_out space.
    """
    name = "linear_perpatch"
    pool_txt = "none"

    def __init__(self, alignment_image, alignment_text):
        self.alignment_image = alignment_image
        self.alignment_text = alignment_text

    def get_patch_features(self, layer_feats, device):
        with torch.no_grad():
            patches = layer_feats[1:, :].to(device)  # (P, D), strip CLS
            cls_name = type(self.alignment_image).__name__
            if cls_name == "LinearAlignmentLayer":
                return self.alignment_image.linear_mapping(patches)  # (P, D_out)
            elif cls_name == "ResLowRankHead":
                ai = self.alignment_image
                z0 = ai.P(patches)
                delta = ai.W2(ai.act(ai.W1(patches)))
                return z0 + ai.alpha * delta  # (P, D_out)
            elif cls_name == "MLPAlignmentLayer":
                return self.alignment_image.mlp(patches)  # (P, D_out)
            else:
                raise ValueError(
                    f"linear_perpatch not supported for {cls_name}"
                )

    def get_text_features(
        self, classnames, templates, tokenizer, language_model,
        layer_txt, device,
    ):
        return build_zero_shot_classifier(
            language_model=language_model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            dataset=None,
            layer_index=layer_txt,
            alignment_layer=self.alignment_text,
            num_classes_per_batch=8,
            device=device,
            pool_txt="none",
            save_path=None,
            token_level=True,
        ).to(device)


# ------------------------------------------------------------------
# Encoder + dataset + transform helpers
# ------------------------------------------------------------------

def build_vision_encoder(cfg: dict, device: torch.device):
    """Load the DINOv2 backbone and wrap it in a per-layer feature extractor.

    Returns the model plus a Resize→ToTensor→Normalize transform suited for
    segmentation (square resize, no center crop — we don't want to drop GT
    pixels implicitly at eval).
    """
    lvm_model_name = cfg["alignment"]["lvm_model_name"]
    img_size = int(cfg["features"]["img_size"])

    model_kwargs = {"img_size": img_size}
    vision_model = timm.create_model(lvm_model_name, pretrained=True, **model_kwargs)
    data_cfg = resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    if "vit" in lvm_model_name:
        return_nodes = [
            f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
        ]
    else:
        raise NotImplementedError(f"unknown vision model {lvm_model_name}")
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    vision_model = vision_model.to(device).eval()

    image_transform = transforms.Compose([
        transforms.Lambda(_ensure_rgb_image),
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return vision_model, image_transform, img_size


def build_language_encoder(cfg: dict, device: torch.device):
    llm_model_name = cfg["alignment"]["llm_model_name"]
    tokenizer = load_tokenizer(llm_model_name)
    language_model = load_llm(llm_model_name).to(device).eval()
    return language_model, tokenizer


class PascalContext59Dataset:
    """Pascal Context with the 59-class benchmark split.

    Expects::

        data_root/
            images/       ← symlink or copy of VOCdevkit/VOC2010/JPEGImages
            trainval/     ← .mat files from the Stanford Context release
            59_labels.txt
            val.txt       ← line-per-image id for the Context-59 val split

    The Context annotations are ``.mat`` files with variable-length integer
    label maps. We map the raw 400+ Context classes down to the 59-class
    benchmark via the ``context_labels`` mapping below (following MaskCLIP
    and TCL convention).
    """

    def __init__(self, data_root: str):
        import scipy.io  # deferred import — only needed for Pascal Context

        self.root = Path(data_root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "trainval"
        if not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Pascal Context annotations missing: {self.ann_dir}"
            )
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Pascal Context images missing: {self.images_dir}. "
                "Provide VOC 2010 JPEGImages as a symlink or copy."
            )

        # The 59-class benchmark split file is typically distributed
        # separately. If not present, fall back to enumerating every .mat
        # file — this matches the unofficial "all trainval" protocol.
        split_file = self.root / "val.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.ids = [line.strip() for line in f if line.strip()]
        else:
            logger.warning(
                f"No val.txt at {split_file}; enumerating all .mat files "
                "as the val split (not the official Context-59 protocol)."
            )
            self.ids = sorted(p.stem for p in self.ann_dir.glob("*.mat"))

        self._scipy_io = scipy.io
        # Build the 460 -> 59 label mapping lazily from 59_labels.txt if
        # available. Pascal Context ships an explicit file mapping the 59
        # foreground classes to their raw-label indices; anything else maps
        # to 0 (background).
        labels_file = self.root / "labels.txt"
        if labels_file.exists():
            raw_label_to_idx = {}
            with open(labels_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx, name = line.split(":", 1)
                    raw_label_to_idx[name.strip()] = int(idx)
            self._context_map = np.zeros(max(raw_label_to_idx.values()) + 2, dtype=np.int64)
            for new_idx, name in enumerate(PASCAL_CONTEXT_CLASSES):
                if name == "background":
                    continue
                # Some names differ slightly between our list and labels.txt;
                # try exact match then lower-case fallback.
                raw = raw_label_to_idx.get(name) or raw_label_to_idx.get(name.lower())
                if raw is not None and raw < self._context_map.shape[0]:
                    self._context_map[raw] = new_idx
        else:
            logger.warning(
                "labels.txt missing; Pascal Context label remapping disabled"
            )
            self._context_map = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = self.images_dir / f"{image_id}.jpg"
        mat_path = self.ann_dir / f"{image_id}.mat"

        img = Image.open(img_path).convert("RGB")
        mat = self._scipy_io.loadmat(str(mat_path))
        # The .mat files contain a 'LabelMap' key (HxW int array).
        raw = mat.get("LabelMap")
        if raw is None:
            raise KeyError(
                f".mat missing LabelMap: {mat_path} (keys: {list(mat.keys())})"
            )
        raw = raw.astype(np.int64)
        if self._context_map is not None:
            remapped = self._context_map[np.clip(raw, 0, self._context_map.shape[0] - 1)]
        else:
            remapped = raw
        # Return a PIL image for the mask so the shared eval loop can call
        # np.array() on it uniformly with torchvision's VOCSegmentation.
        return img, Image.fromarray(remapped.astype(np.uint8))


class ADE20KDataset:
    """ADE20K ADEChallenge2016 val split.

    Expects the layout produced by unzipping ``ADEChallengeData2016.zip``::

        data_root/
            ADEChallengeData2016/
                images/validation/     ← 2000 JPEGs
                annotations/validation/← 2000 PNGs (label 0 = unknown)

    Annotations are 8-bit PNGs where pixel value = class index in [0, 150].
    Pixel 0 is an "unknown" label we treat as ignore (see
    ``DatasetSpec.for_ade20k`` with ``ignore_index=0``).
    """

    def __init__(self, data_root: str):
        root = Path(data_root) / "ADEChallengeData2016"
        if not root.exists():
            raise FileNotFoundError(
                f"ADEChallengeData2016 not found at {root}; unzip the "
                "official release into data/ade20k/."
            )
        self.img_dir = root / "images" / "validation"
        self.ann_dir = root / "annotations" / "validation"
        self.ids = sorted(p.stem for p in self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = Image.open(self.img_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.ann_dir / f"{image_id}.png")
        return img, mask


def build_dataset(name: str, data_root: str, download: bool):
    if name == "voc2012":
        ds = VOCSegmentation(
            root=data_root, year="2012", image_set="val", download=download,
        )
        return ds, DatasetSpec.for_voc2012()
    if name == "pascal_context":
        ds = PascalContext59Dataset(data_root=data_root)
        return ds, DatasetSpec.for_pascal_context()
    if name == "ade20k":
        ds = ADE20KDataset(data_root=data_root)
        return ds, DatasetSpec.for_ade20k()
    if name == "cityscapes":
        from torchvision.datasets import Cityscapes
        ds = Cityscapes(
            root=data_root, split="val", mode="fine", target_type="semantic",
        )
        return ds, DatasetSpec.for_cityscapes()
    raise ValueError(f"unsupported segmentation dataset {name!r}")


# ------------------------------------------------------------------
# Confusion matrix + IoU
# ------------------------------------------------------------------

def update_confusion_matrix(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> None:
    """Accumulate a ``(C, C)`` confusion matrix in-place.

    Rows are GT classes, columns are predicted classes.
    """
    mask = gt != ignore_index
    gt_v = gt[mask].astype(np.int64)
    pr_v = pred[mask].astype(np.int64)
    # Clip predictions to [0, C-1] just in case (shouldn't happen with argmax)
    pr_v = np.clip(pr_v, 0, num_classes - 1)
    gt_v = np.clip(gt_v, 0, num_classes - 1)
    idx = num_classes * gt_v + pr_v
    bincount = np.bincount(idx, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes).astype(np.int64)


def compute_iou_from_confusion(
    confusion: np.ndarray, exclude_background: bool = True
) -> Dict[str, float]:
    """Return ``{'miou_all', 'miou_fg', 'per_class'}`` from a confusion matrix.

    ``per_class`` is a plain list of IoU values in class-index order.
    """
    cm = confusion.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    # Classes that never appear in GT and are never predicted have denom=0
    # and should be excluded from the mean rather than counted as 0 IoU.
    valid = denom > 0
    iou = np.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid]
    per_class = iou.tolist()
    miou_all = float(iou[valid].mean()) if valid.any() else float("nan")
    if exclude_background and len(iou) > 1:
        fg_valid = valid.copy()
        fg_valid[0] = False  # drop background row
        miou_fg = float(iou[fg_valid].mean()) if fg_valid.any() else float("nan")
    else:
        miou_fg = miou_all
    return {
        "miou_all": miou_all,
        "miou_fg": miou_fg,
        "per_class": per_class,
    }


# ------------------------------------------------------------------
# Shared evaluation loop
# ------------------------------------------------------------------

def run_eval(
    method: SegmentationMethod,
    strategy: str,
    dataset,
    dataset_spec: DatasetSpec,
    vision_model,
    image_transform,
    img_size: int,
    tokenizer,
    language_model,
    layer_img: int,
    layer_txt: int,
    device: torch.device,
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate one (method, strategy) pair and return IoU metrics."""
    templates = get_text_templates(strategy)
    prompt_names = [dataset_spec.prompts[c] for c in dataset_spec.classes]

    text_feats = method.get_text_features(
        classnames=prompt_names,
        templates=templates,
        tokenizer=tokenizer,
        language_model=language_model,
        layer_txt=layer_txt,
        device=device,
    )
    text_feats = F.normalize(text_feats.to(device), dim=-1)

    confusion = np.zeros(
        (dataset_spec.num_classes, dataset_spec.num_classes), dtype=np.int64,
    )

    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    for i in tqdm(range(n), desc=f"{method.name}/{strategy}", leave=False):
        pil_img, pil_mask = dataset[i]
        img_t = image_transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            lvm_out = vision_model(img_t)
            # lvm_out is an OrderedDict[blocks.i.add_1 -> (B, T, D)]
            layer_key = list(lvm_out.keys())[layer_img]
            feats = lvm_out[layer_key].squeeze(0)  # (T, D)

        patch_feats = method.get_patch_features(feats, device)  # (P, D_m)
        patch_feats = F.normalize(patch_feats, dim=-1)
        sim = patch_feats @ text_feats.T  # (P, C)

        P = sim.shape[0]
        h = int(round(math.sqrt(P)))
        if h * h != P:
            raise RuntimeError(
                f"non-square patch grid: P={P} tokens for img_size={img_size}"
            )
        sim_map = sim.view(h, h, dataset_spec.num_classes)
        sim_map = sim_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, h, h)

        gt = np.array(pil_mask)
        H, W = gt.shape
        sim_up = F.interpolate(
            sim_map.float(), size=(H, W), mode="bilinear", align_corners=False,
        )
        pred = sim_up.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)

        update_confusion_matrix(
            confusion, gt, pred,
            num_classes=dataset_spec.num_classes,
            ignore_index=dataset_spec.ignore_index,
        )

    return compute_iou_from_confusion(
        confusion, exclude_background=dataset_spec.has_background,
    )


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_results_table(results: List[dict], dataset_spec: DatasetSpec) -> None:
    header = f"{'Method':<20} {'Strategy':<10} {'mIoU-fg':>9} {'mIoU-all':>10}"
    sep = "-" * len(header)
    print(); print(sep); print(header); print(sep)
    for r in results:
        print(
            f"{r['method']:<20} {r['strategy']:<10} "
            f"{r['miou_fg']*100:>8.2f}% {r['miou_all']*100:>9.2f}%"
        )
    print(sep)


def save_per_class_csv(
    results: List[dict], dataset_spec: DatasetSpec, path: str
) -> None:
    import csv
    header = ["method", "strategy"] + list(dataset_spec.classes)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            row = [r["method"], r["strategy"]] + [
                f"{v:.4f}" for v in r["per_class"]
            ]
            w.writerow(row)
    logger.info(f"Per-class IoU CSV written to {path}")


# ------------------------------------------------------------------
# Method dispatch
# ------------------------------------------------------------------

def build_method(
    name: str, alignment_image, alignment_text, cfg: dict
) -> SegmentationMethod:
    pool_txt = cfg["features"].get("pool_txt", "avg")
    if name == "direct_cosine":
        return DirectCosineMethod(pool_txt=pool_txt)
    if name == "freezealign":
        if alignment_image is None:
            raise ValueError("freezealign method requires --checkpoint")
        return FreezeAlignMethod(
            alignment_image=alignment_image, alignment_text=alignment_text,
        )
    if name == "anchor_codebook":
        if alignment_image is None:
            raise ValueError("anchor_codebook requires --checkpoint")
        return BAAnchorCodebookMethod(
            alignment_image=alignment_image, alignment_text=alignment_text,
        )
    if name == "attention_map":
        if alignment_image is None:
            raise ValueError("attention_map requires --checkpoint")
        pool_temperature = (
            cfg["training"]
            .get("alignment_layer_kwargs", {})
            .get("pool_temperature", 0.05)
        )
        return BAAttentionMapMethod(
            alignment_image=alignment_image,
            alignment_text=alignment_text,
            pool_temperature=pool_temperature,
        )
    if name == "linear_perpatch":
        if alignment_image is None:
            raise ValueError("linear_perpatch requires --checkpoint")
        return LinearPerPatchMethod(
            alignment_image=alignment_image, alignment_text=alignment_text,
        )
    raise ValueError(f"unknown method {name!r}")


def auto_filter_methods(
    requested: List[str], alignment_image, alignment_text, cfg: dict,
) -> List[str]:
    """Drop methods that can't run with the given checkpoint/config.

    - No checkpoint     → only ``direct_cosine`` is allowed.
    - BA checkpoint     → BA methods + direct_cosine OK, FreezeAlign dropped.
    - FA checkpoint     → FreezeAlign + direct_cosine OK, BA methods dropped.
    """
    available = set(requested)
    if alignment_image is None:
        return [m for m in requested if m == "direct_cosine"]

    layer_cls = type(alignment_image).__name__
    is_ba = "BridgeAnchor" in layer_cls
    is_fa = "FreezeAlign" in layer_cls
    is_linear_mlp = layer_cls in ("LinearAlignmentLayer", "ResLowRankHead", "MLPAlignmentLayer")
    keep = []
    for m in requested:
        if m == "direct_cosine":
            keep.append(m)
        elif m == "freezealign" and is_fa:
            keep.append(m)
        elif m in ("anchor_codebook", "attention_map") and is_ba:
            keep.append(m)
        elif m == "linear_perpatch" and is_linear_mlp:
            keep.append(m)
        else:
            logger.warning(
                f"Dropping method {m!r}: incompatible with checkpoint class {layer_cls}"
            )
    return keep


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.load(f, Loader=Loader)
    return merge_dicts(raw.get("defaults", {}), raw.get("overrides", {}))


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot segmentation eval (STRUCTURE × BA × FreezeAlign)",
    )
    parser.add_argument("--config", required=True, help="Training yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint-epochN.pth (omit for direct_cosine only)",
    )
    parser.add_argument("--layer-img", type=int, required=True)
    parser.add_argument("--layer-txt", type=int, required=True)
    parser.add_argument(
        "--dataset", default="voc2012",
        choices=["voc2012", "pascal_context", "ade20k", "cityscapes"],
    )
    parser.add_argument("--data-root", default="data/pascal_voc")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--methods",
        default="direct_cosine,freezealign,anchor_codebook,attention_map,linear_perpatch",
        help="Comma-separated method names",
    )
    parser.add_argument(
        "--text-strategies", default="raw,ensemble",
        help="Comma-separated: raw,ensemble",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Optional cap on val set size for quick smoke tests",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Per-class IoU CSV output path (default: <dataset>_seg_iou.csv next to ckpt)",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)

    # Encoders
    vision_model, image_transform, img_size = build_vision_encoder(cfg, device)
    language_model, tokenizer = build_language_encoder(cfg, device)

    # Alignment layers from checkpoint (optional)
    alignment_image = alignment_text = None
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        alignment_image = ckpt["alignment_image"].to(device).eval()
        alignment_text = ckpt["alignment_text"].to(device).eval()
        for m in (alignment_image, alignment_text):
            if hasattr(m, "set_modality"):
                pass  # modality set below
        if hasattr(alignment_image, "set_modality"):
            alignment_image.set_modality("image")
        if hasattr(alignment_text, "set_modality"):
            alignment_text.set_modality("text")
        logger.info(
            f"Alignment class: {type(alignment_image).__name__}"
        )

    # Methods + filtering
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = auto_filter_methods(requested, alignment_image, alignment_text, cfg)
    if not methods:
        raise RuntimeError(
            "No valid methods to run; check --methods and --checkpoint"
        )

    strategies = [s.strip() for s in args.text_strategies.split(",") if s.strip()]

    # Dataset
    dataset, dataset_spec = build_dataset(
        args.dataset, data_root=args.data_root, download=args.download,
    )
    logger.info(
        f"Dataset {dataset_spec.name}: {len(dataset)} val images, "
        f"{dataset_spec.num_classes} classes"
    )

    # Run each method × strategy combination
    results = []
    for method_name in methods:
        method = build_method(method_name, alignment_image, alignment_text, cfg)
        for strategy in strategies:
            logger.info(f"==> {method_name} / {strategy}")
            res = run_eval(
                method=method,
                strategy=strategy,
                dataset=dataset,
                dataset_spec=dataset_spec,
                vision_model=vision_model,
                image_transform=image_transform,
                img_size=img_size,
                tokenizer=tokenizer,
                language_model=language_model,
                layer_img=args.layer_img,
                layer_txt=args.layer_txt,
                device=device,
                max_images=args.max_images,
            )
            results.append({"method": method_name, "strategy": strategy, **res})

    print_results_table(results, dataset_spec)

    csv_path = args.output_csv
    if csv_path is None:
        ckpt_dir = (
            Path(args.checkpoint).parent if args.checkpoint else Path("results")
        )
        csv_path = str(ckpt_dir / f"{dataset_spec.name}_seg_iou.csv")
    save_per_class_csv(results, dataset_spec, csv_path)


if __name__ == "__main__":
    main()
