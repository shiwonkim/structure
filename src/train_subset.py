"""Data-size sweep for representation alignment.

Trains an alignment layer on random subsamples of the training set (seed 42)
to trace the data-scaling curve. Subsample sizes are given via ``--samples``
(default ``1000,5000,10000,50000``); pass a subset or a single value to run
only those points — e.g. to resume an interrupted sweep with ``--samples
10000,50000``. Full-data training is handled by the main training script, not
here.
"""
import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.train_alignment import load_dataset
from src.trainers.alignment_trainer import AlignmentTrainer

parser = argparse.ArgumentParser(
    description="Data-size sweep for the subsampled Representation Alignment.",
)
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
parser.add_argument(
    "--samples",
    type=str,
    default="1000,5000,10000,50000",
    help="Comma-separated subsample sizes to sweep (e.g. '10000,50000').",
)
parser.add_argument(
    "--wandb_notes",
    type=str,
    help="Notes for the wandb run.",
)
args = parser.parse_args()

if __name__ == "__main__":
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    # merge defaults with overrides (overrides take precedence)
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

    # our evaluation datasets
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
                logger.info(
                    f"Successfully loaded '{dataset_name}', test size: {len(ds_val)}"
                )
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

    sample_sizes = [int(s.strip()) for s in args.samples.split(",")]
    for n_samples in sample_sizes:
        config["random_state"] = 42
        trainer = AlignmentTrainer(**trainer_kwargs)
        trainer.fit(n_random_subsample_train=n_samples)
        del trainer
