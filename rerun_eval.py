"""Standalone re-eval: load a trained checkpoint and run zero-shot
classification + retrieval on a different dataset list. Used to extend
the comparison table with DTD, Flowers102, GTSRB, Flickr30k after
training has finished.

Usage:
    python rerun_eval.py --config_path configs/... \
        --ckpt path/to/checkpoint-epochN.pth \
        --label run_label_for_log \
        [--zs dtd,flowers,gtsrb] \
        [--rt flickr30]
"""
import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as _F
import yaml
from loguru import logger
from torch.utils.data import DataLoader

# keep ToTensor robust (legacy fix, now handled in source but harmless here)
_orig_to_tensor = _F.to_tensor


def _safe_to_tensor(pic):
    if isinstance(pic, torch.Tensor):
        return pic
    return _orig_to_tensor(pic)


_F.to_tensor = _safe_to_tensor

from src.core.src.datasets.image_text_dataset import ImageTextDataset  # noqa: E402
from src.core.src.utils.loader import Loader, merge_dicts  # noqa: E402
from src.dataset_preparation.data_utils import (  # noqa: E402
    get_datasets,
    get_default_transforms,
)
from src.trainers.alignment_trainer import AlignmentTrainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True, help="Run label for log output")
    p.add_argument("--zs", default="", help="Comma-separated zero-shot datasets")
    p.add_argument("--rt", default="", help="Comma-separated retrieval datasets")
    p.add_argument("--img_layer", type=int, default=11)
    p.add_argument("--txt_layer", type=int, default=6)
    args = p.parse_args()

    cfg_path = Path(args.config_path)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=Loader)
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    zs_list = [x.strip() for x in args.zs.split(",") if x.strip()]
    rt_list = [x.strip() for x in args.rt.split(",") if x.strip()]
    cfg["evaluation"]["zero_shot_datasets"] = zs_list
    cfg["evaluation"]["retrieval_datasets"] = rt_list

    # Disable wandb via env (we still need a real offline run because
    # base_trainer reads wandb.run.dir at __init__ time)
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    wandb.init(
        project="rerun_eval",
        name=f"rerun_eval_{args.label}",
        mode="offline",
        reinit=True,
    )

    data_path = Path(cfg["paths"]["data_path"])
    pre_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Minimal train/val loaders needed by the AlignmentTrainer constructor —
    # we never actually train, but the base class reads these. Use the same
    # COCO paths the trained run used (keeps features cache path consistent).
    train_ds, val_ds = get_datasets(
        dataset=cfg["features"]["dataset"],
        transform=pre_transform,
        root_dir=data_path,
    )
    if cfg["features"]["dataset"] not in ("coco", "flickr30"):
        train_ds = ImageTextDataset(
            dataset=train_ds,
            label_templates=cfg["features"]["label_templates"],
            template_key=cfg["features"]["template_key"],
            precompute_captions=cfg["features"]["precompute_captions"],
        )
        val_ds = ImageTextDataset(
            dataset=val_ds,
            label_templates=cfg["features"]["label_templates"],
            template_key=cfg["features"]["template_key"],
            precompute_captions=cfg["features"]["precompute_captions"],
        )
        train_ds.name = cfg["features"]["dataset"]
        val_ds.name = cfg["features"]["dataset"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["features"]["batch_size"],
        num_workers=cfg["features"]["num_workers"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=ImageTextDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["features"]["batch_size"],
        num_workers=cfg["features"]["num_workers"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=ImageTextDataset.collate_fn,
    )

    # Build eval dataset list
    eval_zs = []
    for name in zs_list:
        try:
            _, ds = get_datasets(
                dataset=name, transform=get_default_transforms(), root_dir=data_path
            )
            eval_zs.append((name, ds))
            logger.info(f"  ZS loaded '{name}' size={len(ds)}")
        except Exception as e:
            logger.error(f"  ZS failed '{name}': {type(e).__name__}: {e}")

    eval_rt = []
    for name in rt_list:
        try:
            _, ds = get_datasets(
                dataset=name, transform=get_default_transforms(), root_dir=data_path
            )
            eval_rt.append((name, ds))
            logger.info(f"  RT loaded '{name}' size={len(ds)}")
        except Exception as e:
            logger.error(f"  RT failed '{name}': {type(e).__name__}: {e}")

    trainer = AlignmentTrainer(
        config=cfg,
        train_dataset=train_loader,
        val_dataset=val_loader,
        eval_zero_shot_datasets=eval_zs,
        eval_retrieval_datasets=eval_rt,
        **cfg["alignment"],
    )
    trainer.wandb_logging = False

    # Load checkpoint
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    alignment_image = ckpt["alignment_image"].to(trainer.device)
    alignment_text = ckpt["alignment_text"].to(trainer.device)
    alignment_image.eval()
    alignment_text.eval()
    if hasattr(alignment_image, "set_modality"):
        alignment_image.set_modality("image")
    if hasattr(alignment_text, "set_modality"):
        alignment_text.set_modality("text")

    layer_img = args.img_layer
    layer_txt = args.txt_layer
    layer_comb_str = f"img_{layer_img}_txt_{layer_txt}"

    logger.info(f"=== RERUN EVAL  label={args.label}  ckpt={args.ckpt} ===")
    logger.info(
        f"  ckpt epoch={ckpt.get('epoch')} best_epoch={ckpt.get('best_epoch')} "
        f"alignment_layer={cfg['training']['alignment_layer_name']}"
    )

    with torch.no_grad():
        if eval_zs:
            trainer.evaluate_zero_shot_classification(
                epoch=0,
                train_step=0,
                alignment_image=alignment_image,
                alignment_text=alignment_text,
                alignment_layer_combination=(layer_img, layer_txt),
                alignment_layer_combination_str=layer_comb_str,
                additional_result_dict={},
            )
        if eval_rt:
            trainer.evaluate_retrieval(
                epoch=0,
                train_step=0,
                alignment_image=alignment_image,
                alignment_text=alignment_text,
                alignment_layer_combination=(layer_img, layer_txt),
                alignment_layer_combination_str=layer_comb_str,
                additional_result_dict={},
            )


if __name__ == "__main__":
    main()
