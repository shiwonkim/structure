"""Token feature extraction for a single selected layer pair.

Usage:
    python src/extract_token_features.py \\
        --config_path configs/dryrun_ba_token.yaml \\
        --img_layer 11 --txt_layer 6 \\
        --split train

The output is a set of .pt files in results/features/ following STRUCTURE's
naming convention — the features use ``pool_img="none"`` / ``pool_txt="none"``
which STRUCTURE already supports for single-layer token extraction. We reuse
AlignmentTrainer.get_image_features/get_text_features to do the heavy lifting,
only overriding the config to the none-pool single-layer mode.

The text attention mask is saved alongside the text features because the
alignment layer needs it for CAP.

Notes
-----
* Layer selection (phase 1) must be done separately via the standard CLS
  pipeline. This script assumes the layer indices are already known.
* Image mask is not needed (vision transformers use all 1+num_patches tokens).
* File naming: ``{model}-{dataset}-{split}-none_layer-{L}.pt`` for features,
  ``{model}-{dataset}-{split}-none_layer-{L}_mask.pt`` for the text mask.
"""

import argparse
import copy
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as transforms
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets
from src.trainers.alignment_trainer import AlignmentTrainer


def _load_dataset(config, data_path: Path) -> Tuple[DataLoader, DataLoader]:
    """Reproduce train_alignment.load_dataset but without the extras."""
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset, val_dataset = get_datasets(
        dataset=config["features"]["dataset"],
        transform=transform,
        root_dir=data_path,
    )
    if config["features"]["dataset"] not in ("coco", "flickr30"):
        train_dataset = ImageTextDataset(
            dataset=train_dataset,
            label_templates=config["features"]["label_templates"],
            template_key=config["features"]["template_key"],
            precompute_captions=config["features"]["precompute_captions"],
        )
        val_dataset = ImageTextDataset(
            dataset=val_dataset,
            label_templates=config["features"]["label_templates"],
            template_key=config["features"]["template_key"],
            precompute_captions=config["features"]["precompute_captions"],
        )
        train_dataset.name = config["features"]["dataset"]
        val_dataset.name = config["features"]["dataset"]

    loader_kwargs = dict(
        batch_size=config["features"]["batch_size"],
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=config["features"]["num_workers"],
        collate_fn=ImageTextDataset.collate_fn,
    )
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs),
    )


def extract_text_mask(trainer: AlignmentTrainer, loader, llm_model_name: str, suffix: str):
    """Extract and save the text attention mask for the given loader.

    This complements ``get_text_features(pool_txt='none')`` which saves the
    features themselves but not the mask. We re-run the tokenizer over the
    dataframe (no model forward needed) to get masks in a separate file.
    """
    dataset_name = (
        loader.dataset.name
        if hasattr(loader.dataset, "name")
        else type(loader.dataset).__name__
    )
    save_path = AlignmentTrainer.get_feature_save_path(
        m_name=llm_model_name,
        d_name=dataset_name,
        save_path=trainer.save_path,
        suffix=suffix,
    )
    mask_path = save_path.with_name(save_path.stem + "_mask" + save_path.suffix)
    if mask_path.exists():
        logger.debug(f"Text mask already cached: {mask_path}")
        return

    # Ensure tokenizer is attached and df is tokenised (the features extraction
    # path already does this as a side effect; calling it again is cheap and
    # idempotent because apply_tokenizer mutates the df).
    _, tokenizer = trainer.get_llm(llm_model_name=llm_model_name)
    loader.dataset.tokenizer = tokenizer
    loader.dataset.apply_tokenizer()

    masks = []
    from tqdm import tqdm
    import sys

    for batch in tqdm(loader, total=len(loader), file=sys.stdout, desc="text-mask"):
        _, token_inputs = batch
        masks.append(token_inputs["attention_mask"].cpu())
    mask = torch.cat(masks, dim=0)  # (N, T)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mask": mask}, mask_path)
    logger.info(f"Saved text mask to: {mask_path} shape={tuple(mask.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--img_layer", type=int, required=True)
    parser.add_argument("--txt_layer", type=int, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to extract tokens for.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Override input image resolution (controls number of tokens).",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    config = merge_dicts(config.get("defaults", {}), config.get("overrides", {}))

    # Override pooling to single-layer token mode. STRUCTURE's existing
    # extraction code already handles this path.
    token_config = copy.deepcopy(config)
    token_config["features"]["pool_img"] = "none"
    token_config["features"]["pool_txt"] = "none"
    token_config["features"]["layer_img"] = args.img_layer
    token_config["features"]["layer_txt"] = args.txt_layer
    if args.img_size is not None:
        token_config["features"]["img_size"] = int(args.img_size)
    if args.batch_size is not None:
        token_config["features"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        token_config["features"]["num_workers"] = int(args.num_workers)
    if args.dataset is not None:
        token_config["features"]["dataset"] = args.dataset

    data_path = Path(token_config["paths"]["data_path"])
    train_loader, val_loader = _load_dataset(token_config, data_path)

    # Build a minimal trainer just to borrow the extraction methods.
    trainer_kwargs = {
        "config": token_config,
        "train_dataset": train_loader,
        "val_dataset": val_loader,
        "eval_zero_shot_datasets": [],
        "eval_retrieval_datasets": [],
    } | token_config["alignment"]
    trainer = AlignmentTrainer(**trainer_kwargs)

    img_size = token_config["features"].get("img_size")
    res_tag = f"-r{int(img_size)}" if img_size is not None else ""
    img_suffix_template = (
        "{split}-none_layer-" + str(args.img_layer) + res_tag
    )
    txt_suffix_template = "{split}-none_layer-" + str(args.txt_layer)

    for split in args.splits:
        loader = train_loader if split == "train" else val_loader
        img_suffix = img_suffix_template.format(split=split)
        txt_suffix = txt_suffix_template.format(split=split)

        logger.info(f"[{split}] Extracting image tokens (layer {args.img_layer})")
        img_feats = trainer.get_image_features(
            loader=loader,
            lvm_model_name=trainer.lvm_model_name,
            suffix=img_suffix,
        )
        logger.info(f"[{split}] image tokens shape: {tuple(img_feats.shape)}")

        logger.info(f"[{split}] Extracting text tokens (layer {args.txt_layer})")
        txt_feats = trainer.get_text_features(
            loader=loader,
            llm_model_name=trainer.llm_model_name,
            suffix=txt_suffix,
        )
        logger.info(f"[{split}] text tokens shape: {tuple(txt_feats.shape)}")

        logger.info(f"[{split}] Saving text attention mask")
        extract_text_mask(
            trainer=trainer,
            loader=loader,
            llm_model_name=trainer.llm_model_name,
            suffix=txt_suffix,
        )

    logger.info("Token feature extraction complete.")


if __name__ == "__main__":
    main()
