"""Training data scaling experiment: COCO + LAION.

Sweep: COCO-only (83K), +80K, +160K, +320K, +560K, +917K (total ~1M).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/laion/05_train_data_scaling.py \
        --config configs/ba/vitl_roberta/token_k512_laion.yaml
"""
import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.train_alignment import load_dataset
from src.trainers.alignment_trainer import AlignmentTrainer

LAION_STEPS = [30_000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--wandb_notes", type=str, default=None)
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.load(f, Loader=Loader)
    config = merge_dicts(config.get("defaults", {}), config.get("overrides", {}))

    data_path = Path(config["paths"]["data_path"])
    train_dataset, val_dataset = load_dataset(
        dataset_name=config["features"]["dataset"],
        data_path=data_path,
        batch_size=config["features"]["batch_size"],
        num_workers=config["features"]["num_workers"],
        label_templates=config["features"]["label_templates"],
        template_key=config["features"]["template_key"],
    )

    eval_zero_shot_datasets = []
    eval_retrieval_datasets = []
    for d_name, l_data in [
        ("zero_shot_datasets", eval_zero_shot_datasets),
        ("retrieval_datasets", eval_retrieval_datasets),
    ]:
        for dataset_name in config["evaluation"][d_name]:
            try:
                _, ds_val = get_datasets(
                    dataset=dataset_name,
                    transform=get_default_transforms(),
                    root_dir=data_path,
                )
                l_data.append((dataset_name, ds_val))
                logger.info(f"Loaded '{dataset_name}', test size: {len(ds_val)}")
            except Exception as e:
                logger.error(f"Error on {dataset_name}: {e}")

    trainer_kwargs = {
        "config": config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "eval_zero_shot_datasets": eval_zero_shot_datasets,
        "eval_retrieval_datasets": eval_retrieval_datasets,
        "wandb_notes": args.wandb_notes,
    }
    trainer_kwargs = trainer_kwargs | config["alignment"]

    for n_laion in LAION_STEPS:
        config["random_state"] = 42
        logger.info(f"\n{'='*60}")
        logger.info(f"LAION addition: {n_laion:,} (total: {83000 + n_laion:,})")
        logger.info(f"{'='*60}")
        trainer = AlignmentTrainer(**trainer_kwargs)
        trainer.fit(n_random_additional_feats=n_laion)
        del trainer


if __name__ == "__main__":
    main()
