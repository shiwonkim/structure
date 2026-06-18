"""Resume data sweep from 10K/50K (1K and 5K already done)."""
import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.train_alignment import load_dataset
from src.trainers.alignment_trainer import AlignmentTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--samples", type=str, default="1000,5000,10000,50000",
                    help="Comma-separated subsample sizes")
args = parser.parse_args()

if __name__ == "__main__":
    args.config_path = Path(args.config_path)
    with open(args.config_path, "r") as f:
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
                logger.info(f"Successfully loaded '{dataset_name}', test size: {len(ds_val)}")
            except Exception as e:
                logger.error(f"Error on {dataset_name}: {e}")

    trainer_kwargs = {
        "config": config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "eval_zero_shot_datasets": eval_zero_shot_datasets,
        "eval_retrieval_datasets": eval_retrieval_datasets,
    } | config["alignment"]

    sample_sizes = [int(s.strip()) for s in args.samples.split(",")]
    for n_samples in sample_sizes:
        config["random_state"] = 42
        trainer = AlignmentTrainer(**trainer_kwargs)
        trainer.fit(n_random_subsample_train=n_samples)
        del trainer
