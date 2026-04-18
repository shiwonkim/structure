import ast
import copy
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from deepspeed.runtime.lr_schedules import WarmupDecayLR, WarmupLR
from loguru import logger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm, trange

from src.alignment.alignment_factory import AlignmentFactory
from src.core.src.datasets.downstream_tasks.coco_dataset import LoadingType
from src.core.src.optimizers.utils import get_optimizer_type
from src.core.src.utils.plotting import embedding_plot, embedding_plot_w_markers
from src.core.src.utils.utils import EarlyStopping, clip_gradients, save_checkpoint
from src.dataset_preparation.data_utils import (
    FeatureDataset,
    _ensure_rgb_image,
    get_meta_dict,
)
from src.evaluation.consts import (
    DATASETS_TO_CLASSES,
    DATASETS_TO_TEMPLATES,
    SIMPLE_PROMPT_TEMPLATE,
)
from src.evaluation.retrieval import retrieval_metrics_df
from src.evaluation.zero_shot_classifier import (
    build_zero_shot_classifier,
    chunked_logits,
)
from src.loss.clip_loss import CLIPLoss
from src.measure_alignment import compute_score
from src.models.text.models import load_llm, load_tokenizer
from src.trainers.base_trainer import Trainer
from src.utils.utils import (
    continuity,
    safe_normalize,
    set_transform_dataset,
    trustworthiness,
)


class AlignmentTrainer(Trainer):
    def __init__(
        self,
        config: dict,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        llm_model_name: str,
        lvm_model_name: str,
        eval_zero_shot_datasets: Optional[List[DataLoader]] = None,
        eval_retrieval_datasets: Optional[List[DataLoader]] = None,
        cache_features: bool = False,
        print_model_summary: bool = True,
        wandb_logging: bool = True,
        wandb_project_name: str = "representation-alignment",
        wandb_notes: Optional[str] = None,
    ):
        self.exp_name = f"alignment-{AlignmentTrainer.get_model_name(llm_model_name)}-{AlignmentTrainer.get_model_name(lvm_model_name)}"
        config["llm_model_name"] = llm_model_name
        config["lvm_model_name"] = lvm_model_name
        config["experiment_name"] = self.exp_name
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            experiment_name=self.exp_name,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
            wandb_notes=wandb_notes,
        )
        self.cache_features = cache_features
        self.print_model_summary = print_model_summary
        self.save_path = Path(config["paths"]["save_path"])
        self.llm_model_name = llm_model_name
        self.lvm_model_name = lvm_model_name
        self.eval_zero_shot_datasets = eval_zero_shot_datasets
        self.eval_retrieval_datasets = eval_retrieval_datasets

        # cache dummies
        self.image_features_val = None
        self.image_features_train = None
        self.text_features_val = None
        self.text_features_train = None

        # the dataframe we use to store the scores
        self.df_scores_zero_shot = None
        self.df_scores_retrieval = None

        # make sure that our experiment folder is there
        (self.save_path / self.exp_name).mkdir(parents=True, exist_ok=True)
        (self.save_path / wandb.run.name).mkdir(parents=True, exist_ok=True)

    def __del__(self):
        # do garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_model_name(m_name: str):
        return m_name.replace("/", "_").replace("-", "_")

    @staticmethod
    def get_feature_save_path(
        m_name: str, d_name: str, save_path: Path, suffix: str = ""
    ):
        m_name = AlignmentTrainer.get_model_name(m_name=m_name)
        return save_path / "features" / f"{m_name}-{d_name}-{suffix}.npy"

    def add_exp_suffix_to_name(self, base_name: str):
        save_name = f"{base_name}"
        save_name += f"_{self.config['layer_selection']['n_samples']}"
        save_name += f"_{self.config['layer_selection']['metric']}"
        if self.config["layer_selection"].get("metric_kwargs", None) is not None:
            save_name += "_".join(
                map(str, self.config["layer_selection"]["metric_kwargs"].values())
            )
        return save_name

    def get_llm(self, llm_model_name: str):
        language_model = load_llm(llm_model_name)
        # since we're using huggingface's automapping
        # we don't need to move it to the device
        language_model = language_model.eval()
        tokenizer = load_tokenizer(llm_model_name)
        return language_model, tokenizer

    def get_lvm(self, lvm_model_name: str):
        img_size = self.config["features"].get("img_size")
        model_kwargs = {}
        if img_size is not None:
            model_kwargs["img_size"] = int(img_size)
        vision_model = timm.create_model(
            lvm_model_name, pretrained=True, **model_kwargs
        )
        data_config = resolve_data_config(
            vision_model.pretrained_cfg, model=vision_model
        )
        if img_size is not None:
            data_config["input_size"] = (3, int(img_size), int(img_size))
            data_config["crop_pct"] = 1.0
        transform = create_transform(**data_config)
        transform.transforms = [_ensure_rgb_image] + transform.transforms

        if "vit" in lvm_model_name:
            return_nodes = [
                f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
            ]
        else:
            raise NotImplementedError(f"unknown model {lvm_model_name}")
        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        vision_model = vision_model.to(self.device)
        vision_model = vision_model.eval()
        return vision_model, transform

    def get_text_features(
        self,
        loader,
        llm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
    ):
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
        save_path = AlignmentTrainer.get_feature_save_path(
            m_name=llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )

        if save_path.exists():
            llm_feats = torch.load(save_path, weights_only=False)["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return llm_feats

        language_model, tokenizer = self.get_llm(llm_model_name=llm_model_name)
        loader.dataset.tokenizer = tokenizer
        if hasattr(loader.dataset, "loading_type"):
            # for optimizing the loading and looping
            loader.dataset.loading_type = LoadingType.TXT_ONLY
        _df = loader.dataset.df
        loader.dataset.apply_tokenizer()
        # ensure this is still the same ordering
        assert loader.dataset.df.equals(_df)
        del _df

        pool_txt_mode = self.config["features"]["pool_txt"]
        llm_feats = None
        offset = 0
        total_n = len(loader.dataset)
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            _, token_inputs = batch
            token_inputs = {
                k: v.to(self.device).long() for (k, v) in token_inputs.items()
            }
            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )
                if pool_txt_mode == "avg":
                    # swap the backsize to the first dimension
                    # (BS, Layers, Tokens, Dim)
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    # make the mask compatible with the dimension
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    # average along the token dimension
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif pool_txt_mode == "last":
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif pool_txt_mode == "none":
                    assert self.config["features"].get("layer_txt") is not None
                    feats = torch.stack(list(llm_output["hidden_states"]))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, self.config["features"]["layer_txt"], :, :]
                else:
                    raise NotImplementedError(f"unknown pooling {pool_txt_mode}")

                if pool_txt_mode == "none":
                    feats_cpu = feats.to(dtype=torch.float16).cpu()
                    if llm_feats is None:
                        _, T, D = feats_cpu.shape
                        llm_feats = torch.empty(
                            (total_n, T, D), dtype=torch.float16
                        )
                    bs = feats_cpu.shape[0]
                    llm_feats[offset : offset + bs] = feats_cpu
                    offset += bs
                else:
                    if llm_feats is None:
                        llm_feats = []
                    llm_feats.append(feats.cpu())
        if pool_txt_mode == "none":
            if offset < total_n:
                llm_feats = llm_feats[:offset]
        else:
            llm_feats = torch.cat(llm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": llm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del language_model
        return llm_feats

    def get_image_features(
        self,
        loader,
        lvm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
        allow_image_dedup: bool = True,
    ):
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
        save_path = AlignmentTrainer.get_feature_save_path(
            m_name=lvm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )

        if save_path.exists():
            lvm_feats = torch.load(save_path, weights_only=False)["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return lvm_feats

        vision_model, image_transform = self.get_lvm(lvm_model_name=lvm_model_name)
        set_transform_dataset(
            dataset=loader.dataset,
            image_transform=image_transform,
        )

        pool_mode = self.config["features"]["pool_img"]
        # Image-side dedup at extraction: pool=none caches duplicate the same
        # image 5x in COCO (one row per caption). Detect that and iterate
        # only the first-occurrence rows of each unique image_path. Saves
        # ~5x extraction time and ~5x disk on the train cache. fit() detects
        # the deduped layout via shape and skips the redundant per-row
        # dedup it would otherwise apply.
        is_deduped_extraction = False
        unique_to_full_idx = None
        if (
            pool_mode == "none"
            and allow_image_dedup
            and self._should_dedup_image_extraction(loader)
        ):
            df = loader.dataset.df
            first_idx_mask = (df.groupby("image_path").cumcount() == 0).values
            keep_indices = np.where(first_idx_mask)[0].tolist()
            n_unique = len(keep_indices)
            n_full = len(df)
            logger.info(
                f"Image dedup at extraction: {n_full:,} caption-image pairs -> "
                f"{n_unique:,} unique images "
                f"(saves {(1 - n_unique / n_full) * 100:.1f}% of vision forwards)"
            )
            unique_view = self._indexed_dataset_view(loader.dataset, keep_indices)
            from torch.utils.data import DataLoader as _DataLoader

            iter_loader = _DataLoader(
                unique_view,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=getattr(loader, "num_workers", 0),
                pin_memory=False,
                drop_last=False,
                collate_fn=getattr(loader, "collate_fn", None),
            )
            iter_total_n = n_unique
            # Build the caption -> image-row mapping for the sidecar
            unique_paths = df.loc[first_idx_mask, "image_path"].reset_index(drop=True)
            path_to_pos = {p: i for i, p in enumerate(unique_paths)}
            unique_to_full_idx = torch.tensor(
                df["image_path"].map(path_to_pos).values, dtype=torch.long
            )
            is_deduped_extraction = True
        else:
            iter_loader = loader
            iter_total_n = len(loader.dataset)

        # Streaming allocation path for pool=none: token features are huge
        # (591K * 257 * 384 * 4 = 234 GB for full COCO float32), so we
        # pre-allocate a single float16 tensor and fill it by offset to
        # avoid torch.cat's 2x peak memory.
        lvm_feats = None
        offset = 0
        total_n = iter_total_n
        for batch in tqdm(iter_loader, total=len(iter_loader), file=sys.stdout):
            images, _ = batch
            with torch.no_grad():
                images = images.to(self.device, non_blocking=True)
                lvm_output = vision_model(images)
                if pool_mode == "cls":
                    # extract the class token for all layers
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif pool_mode == "none":
                    assert self.config["features"].get("layer_img") is not None
                    feats = torch.stack(list(lvm_output.values()))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, self.config["features"]["layer_img"], :, :]
                else:
                    raise NotImplementedError(f"unknown pooling {pool_mode}")

                if pool_mode == "none":
                    feats_cpu = feats.to(dtype=torch.float16).cpu()
                    if lvm_feats is None:
                        _, T, D = feats_cpu.shape
                        lvm_feats = torch.empty(
                            (total_n, T, D), dtype=torch.float16
                        )
                    bs = feats_cpu.shape[0]
                    lvm_feats[offset : offset + bs] = feats_cpu
                    offset += bs
                else:
                    if lvm_feats is None:
                        lvm_feats = []
                    lvm_feats.append(feats.cpu())
        if pool_mode == "none":
            if offset < total_n:
                lvm_feats = lvm_feats[:offset]
        else:
            lvm_feats = torch.cat(lvm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": lvm_feats}
        if is_deduped_extraction:
            save_dict["is_image_deduped"] = True
            save_dict["unique_to_full_idx"] = unique_to_full_idx
            # Persist the unique-row df so future loaders can rebuild
            # mappings without re-reading the original annotations.
            save_dict["dataframe"] = (
                loader.dataset.df.loc[
                    loader.dataset.df.groupby("image_path").cumcount() == 0
                ].reset_index(drop=True)
            )
        elif hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del vision_model
        return lvm_feats

    def compute_layer_alignment(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        # compute similarity between the models on training set
        logger.debug("Computing alignment between modalities")
        alignment_csv = (
            self.save_path
            / self.exp_name
            / (self.add_exp_suffix_to_name("df_alignment") + ".csv")
        )
        # only read from memory when we're not subsampling!
        if (
            alignment_csv.exists()
            and self.n_random_subsample_train is None
            and self.n_random_subsample_val is None
        ):
            df_alignment = pd.read_csv(alignment_csv)
            df_alignment["indices"] = df_alignment["indices"].apply(ast.literal_eval)
            logger.debug('Loaded "df_alignment" from disk.')
        else:
            if self.config["layer_selection"]["type"] == "random":
                sel_samples = np.random.choice(
                    image_features.shape[0],
                    min(
                        self.config["layer_selection"]["n_samples"],
                        image_features.shape[0],
                    ),
                    replace=False,
                )
            else:
                raise ValueError(
                    f"Unknown layer selection type: {self.config['layer_selection']['type']}"
                )
            _, _, alignment_list = compute_score(
                x_feats=image_features[sel_samples].float().to(self.device),
                y_feats=text_features[sel_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            df_alignment = pd.DataFrame(alignment_list)
            df_alignment["indices_x"] = df_alignment["indices"].apply(lambda x: x[0])
            df_alignment["indices_y"] = df_alignment["indices"].apply(lambda x: x[1])
            # remove all scores from the concatenated layers, since we're not interested in them!
            df_alignment = df_alignment[
                (df_alignment["indices_x"] != -1) & (df_alignment["indices_y"] != -1)
            ]
            df_alignment.to_csv(alignment_csv, index=False)

        n_score_bins = self.config["layer_selection"]["n_score_bins"]
        sampled_csv = (
            self.save_path
            / self.exp_name
            / (
                self.add_exp_suffix_to_name("sampled_df_alignment")
                + f"_bins{n_score_bins}"
                + ".csv"
            )
        )
        if (
            sampled_csv.exists()
            and self.n_random_subsample_train is None
            and self.n_random_subsample_val is None
        ):
            sampled_df_alignment = pd.read_csv(sampled_csv)
            sampled_df_alignment["indices"] = sampled_df_alignment["indices"].apply(
                ast.literal_eval
            )
            logger.debug('Loaded "sampled_df_alignment" from disk.')
        else:
            df_alignment["quantile_bin"] = pd.qcut(
                df_alignment["alignment_score"],
                q=self.config["layer_selection"]["n_score_bins"],
                labels=False,
                duplicates="drop",
            )
            sampled_df_alignment = (
                df_alignment.groupby("quantile_bin", group_keys=False)
                .apply(
                    lambda x: x.sample(n=1, random_state=self.config["random_state"])
                )
                .reset_index(drop=True)
            )
            sampled_df_alignment = sampled_df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            # make sure that the min and max is sampled
            df_alignment = df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            series_highest = df_alignment.iloc[0]
            series_lowest = df_alignment.iloc[-1]
            sampled_df_alignment.iloc[0] = series_highest
            sampled_df_alignment.iloc[-1] = series_lowest
            # make sure that we are always as well including the last layers as well
            df_alignment = df_alignment.sort_values(
                by=["indices_x", "indices_y"],
                ascending=False,
            )
            last_layer = df_alignment.iloc[0].copy()
            last_layer["indices"] = (-1, -1)
            last_layer["indices_x"] = -1
            last_layer["indices_y"] = -1
            sampled_df_alignment.loc[len(sampled_df_alignment)] = last_layer
            # make sure we drop the duplicates, if any
            sampled_df_alignment.drop_duplicates(subset="indices", inplace=True)
            sampled_df_alignment.to_csv(sampled_csv, index=False)

        if self.config["layer_selection"]["best_only"]:
            logger.debug("Selecting only best layer to align")
            sampled_df_alignment = sampled_df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            sampled_df_alignment = sampled_df_alignment.iloc[:1]
        elif self.config["layer_selection"]["last_only"]:
            logger.debug("Selecting only last layer to align")
            sampled_df_alignment = sampled_df_alignment[
                sampled_df_alignment["indices"] == (-1, -1)
            ]
        elif self.config["layer_selection"]["n_last_layers"] is not None:
            n_last_layers = self.config["layer_selection"]["n_last_layers"]
            df_alignment = df_alignment.sort_values(
                by=["indices_x", "indices_y"],
                ascending=False,
            )
            last_layer = df_alignment.iloc[0].copy()
            last_layer_index_x = last_layer["indices_x"]
            last_layer_index_y = last_layer["indices_y"]
            sel_layers_x = list(
                range(last_layer_index_x - n_last_layers + 1, last_layer_index_x + 1)
            )
            sel_layers_y = list(
                range(last_layer_index_y - n_last_layers + 1, last_layer_index_y + 1)
            )
            sampled_df_alignment = df_alignment[
                (df_alignment["indices_x"].isin(sel_layers_x))
                & (df_alignment["indices_y"].isin(sel_layers_y))
            ].copy()

        return sampled_df_alignment

    def _subsampled_loader(self, loader, max_samples: Optional[int]):
        """Return a DataLoader restricted to the first ``max_samples`` items.

        The returned loader wraps the same underlying dataset as ``loader``
        but delegates length to ``max_samples``. All attribute access falls
        through to the wrapped dataset so downstream code that reads
        ``dataset.df`` / ``dataset.name`` / ``dataset.tokenizer`` /
        ``dataset.apply_tokenizer`` still works. If ``max_samples`` is None
        or ``>=`` the dataset size, the original loader is returned unchanged.
        """
        base_dataset = loader.dataset
        n = len(base_dataset)
        if max_samples is None or max_samples >= n:
            return loader

        class _SubsetView:
            """Dataset proxy that truncates to the first ``k`` samples.

            The view forwards all attribute reads AND writes to the wrapped
            dataset so that ``loader.dataset.tokenizer = ...`` and
            ``loader.dataset.apply_tokenizer()`` mutate the real object's
            state. It only overrides ``__len__``, ``__getitem__`` and the
            ``df`` property (returning a sliced view).
            """

            _PROXY_ATTRS = {"_dataset", "_k"}

            def __init__(self, dataset, k):
                object.__setattr__(self, "_dataset", dataset)
                object.__setattr__(self, "_k", k)

            def __len__(self):
                return self._k

            def __getitem__(self, index):
                return self._dataset[index]

            def __getattr__(self, name):
                # only called when attribute is NOT found on self
                return getattr(self._dataset, name)

            def __setattr__(self, name, value):
                if name in type(self)._PROXY_ATTRS:
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._dataset, name, value)

            @property
            def df(self):
                inner_df = getattr(self._dataset, "df", None)
                if inner_df is None:
                    return None
                return inner_df.iloc[: self._k].reset_index(drop=True)

        view = _SubsetView(base_dataset, max_samples)

        from torch.utils.data import DataLoader

        collate_fn = getattr(loader, "collate_fn", None)
        return DataLoader(
            view,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _indexed_dataset_view(self, base_dataset, keep_indices):
        """Dataset proxy that yields only the rows in ``keep_indices``.

        Mirrors ``_SubsetView`` (read+write attribute delegation, df
        property override) but supports an arbitrary index list instead
        of a first-N truncation. Used by image-side dedup at extraction.
        """

        class _IndexedView:
            _PROXY_ATTRS = {"_dataset", "_indices"}

            def __init__(self, dataset, indices):
                object.__setattr__(self, "_dataset", dataset)
                object.__setattr__(self, "_indices", list(indices))

            def __len__(self):
                return len(self._indices)

            def __getitem__(self, i):
                return self._dataset[self._indices[i]]

            def __getattr__(self, name):
                return getattr(self._dataset, name)

            def __setattr__(self, name, value):
                if name in type(self)._PROXY_ATTRS:
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._dataset, name, value)

            @property
            def df(self):
                inner_df = getattr(self._dataset, "df", None)
                if inner_df is None:
                    return None
                return inner_df.iloc[self._indices].reset_index(drop=True)

        return _IndexedView(base_dataset, keep_indices)

    def _should_dedup_image_extraction(self, loader) -> bool:
        """Decide whether to apply image-dedup-at-extraction for this loader.

        Returns True only when:
          - the config flag ``features.image_dedup_extraction`` is on,
          - the dataset has a ``df`` with an ``image_path`` column,
          - that df has duplicate image_paths,
          - training.drop_duplicates is true and n_dup_samples == 1
            (so the trainer would otherwise dedup to first-occurrence anyway),
          - no n_random_subsample_train cap is set (the dryrun first-N path
            doesn't benefit and would interact awkwardly with dedup).
        """
        if not self.config["features"].get("image_dedup_extraction", False):
            return False
        ds = loader.dataset
        if not hasattr(ds, "df"):
            return False
        df = ds.df
        if df is None or "image_path" not in df.columns:
            return False
        if not df["image_path"].duplicated().any():
            return False
        if not self.config["training"].get("drop_duplicates", False):
            return False
        if int(self.config["training"].get("n_dup_samples", 1)) != 1:
            return False
        if self.config["training"].get("n_random_subsample_train") is not None:
            return False
        return True

    def _load_token_features_for_layer(
        self,
        img_layer_idx: int,
        txt_layer_idx: int,
    ):
        """Load / extract token features at the given layer pair.

        Returns a dict with keys:
            img_train, txt_train, txt_mask_train,
            img_val,   txt_val,   txt_mask_val

        Image features are (N, T_img, D_img) float tensors.
        Text features are (N, T_txt, D_txt) float tensors.
        Text masks are (N, T_txt) int64 tensors (1=valid, 0=pad).

        Image masks are not needed (ViT uses all 1+num_patches tokens).

        This method temporarily overrides ``features.pool_img`` /
        ``features.pool_txt`` / ``features.layer_{img,txt}`` to reuse
        STRUCTURE's existing ``pool=none`` single-layer extraction path,
        then restores the original config. It also respects the
        ``training.n_random_subsample_{train,val}`` limits by wrapping the
        loaders in a subset view — token features are heavy (ViT-S/14
        produces 1370 tokens per image at DINOv2's default 518×518 input),
        and the dry-run path would otherwise balloon disk + memory usage.
        """
        orig_cfg = self.config["features"]
        save_pool_img = orig_cfg.get("pool_img")
        save_pool_txt = orig_cfg.get("pool_txt")
        save_layer_img = orig_cfg.get("layer_img")
        save_layer_txt = orig_cfg.get("layer_txt")

        orig_cfg["pool_img"] = "none"
        orig_cfg["pool_txt"] = "none"
        orig_cfg["layer_img"] = int(img_layer_idx)
        orig_cfg["layer_txt"] = int(txt_layer_idx)

        # cap extraction to the effective training subset size
        max_train = self.config["training"].get("n_random_subsample_train")
        max_val = self.config["training"].get("n_random_subsample_val")
        train_loader = self._subsampled_loader(self.train_dataset, max_train)
        val_loader = self._subsampled_loader(self.val_dataset, max_val)
        # suffix gets a -n{N} tag when using a subset so the cache is
        # distinct from the "full" extraction
        train_tag = f"-n{max_train}" if max_train is not None else ""
        val_tag = f"-n{max_val}" if max_val is not None else ""

        img_size = orig_cfg.get("img_size")
        res_tag = f"-r{int(img_size)}" if img_size is not None else ""
        try:
            img_suffix = f"none_layer-{int(img_layer_idx)}{res_tag}"
            txt_suffix = f"none_layer-{int(txt_layer_idx)}"

            img_train = self.get_image_features(
                loader=train_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=f"train-{img_suffix}{train_tag}",
            )
            img_val = self.get_image_features(
                loader=val_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=f"val-{img_suffix}{val_tag}",
            )
            txt_train = self.get_text_features(
                loader=train_loader,
                llm_model_name=self.llm_model_name,
                suffix=f"train-{txt_suffix}{train_tag}",
            )
            txt_val = self.get_text_features(
                loader=val_loader,
                llm_model_name=self.llm_model_name,
                suffix=f"val-{txt_suffix}{val_tag}",
            )
            txt_mask_train = self._load_or_build_text_mask(
                loader=train_loader,
                llm_model_name=self.llm_model_name,
                suffix=f"train-{txt_suffix}{train_tag}",
            )
            txt_mask_val = self._load_or_build_text_mask(
                loader=val_loader,
                llm_model_name=self.llm_model_name,
                suffix=f"val-{txt_suffix}{val_tag}",
            )
        finally:
            # restore original pooling config so the rest of the pipeline
            # (eval, layer-selection metadata, etc.) behaves normally
            orig_cfg["pool_img"] = save_pool_img
            orig_cfg["pool_txt"] = save_pool_txt
            if save_layer_img is None:
                orig_cfg.pop("layer_img", None)
            else:
                orig_cfg["layer_img"] = save_layer_img
            if save_layer_txt is None:
                orig_cfg.pop("layer_txt", None)
            else:
                orig_cfg["layer_txt"] = save_layer_txt

        return {
            "img_train": img_train,
            "txt_train": txt_train,
            "txt_mask_train": txt_mask_train,
            "img_val": img_val,
            "txt_val": txt_val,
            "txt_mask_val": txt_mask_val,
        }

    def get_image_cls_attention(
        self,
        loader,
        lvm_model_name: str,
        suffix: str,
        layer_idx: int,
        allow_image_dedup: bool = True,
    ) -> torch.Tensor:
        """Extract per-image CLS→patch attention from the last block's attn.

        Returns a ``(N, P)`` float16 tensor where ``N`` is the number of
        unique images (with dedup) and ``P`` is the patch count. The
        attention is averaged over heads and renormalized so each row sums
        to 1.0. Cached to disk at
        ``results/features/<model>-<dataset>-<suffix>_cls_attn_layer-<L>-r<img_size>.npy``
        so subsequent runs skip the forward pass.

        Mirrors ``get_image_features`` conventions: same dedup gate, same
        save-path format, same sidecar shape expectations. The cache
        file is compatible with the ``pool=none`` image token cache,
        i.e. the N-axes line up with the dedup'd (118K for COCO) layout.
        """
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        else:
            dataset_name = type(loader.dataset).__name__

        img_size = self.config["features"].get("img_size")
        res_tag = f"-r{int(img_size)}" if img_size is not None else ""
        full_suffix = f"{suffix}_cls_attn_layer-{int(layer_idx)}{res_tag}"
        save_path = AlignmentTrainer.get_feature_save_path(
            m_name=lvm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=full_suffix,
        )
        if save_path.exists():
            cached = torch.load(save_path, weights_only=False)["features"]
            logger.debug(f"Loaded CLS attention from: {save_path}")
            return cached

        # Build a fresh unwrapped timm model — the feature-extractor wrapper
        # obscures the attention module's input, so a separate forward is
        # easier than patching the GraphModule. This is a single one-off
        # pass, amortised by the on-disk cache.
        model_kwargs = {"img_size": int(img_size)} if img_size is not None else {}
        timm_model = timm.create_model(
            lvm_model_name, pretrained=True, **model_kwargs
        )
        data_config = resolve_data_config(
            timm_model.pretrained_cfg, model=timm_model
        )
        if img_size is not None:
            data_config["input_size"] = (3, int(img_size), int(img_size))
            data_config["crop_pct"] = 1.0
        timm_transform = create_transform(**data_config)
        timm_transform.transforms = [_ensure_rgb_image] + timm_transform.transforms
        set_transform_dataset(
            dataset=loader.dataset,
            image_transform=timm_transform,
        )
        timm_model = timm_model.to(self.device).eval()

        # Apply dedup at extraction, same gate as the image token path,
        # so the N-axes line up between the token cache and the cls-attn
        # cache. One row per unique image_path.
        is_deduped = False
        if (
            allow_image_dedup
            and self._should_dedup_image_extraction(loader)
        ):
            df = loader.dataset.df
            first_idx_mask = (df.groupby("image_path").cumcount() == 0).values
            keep_indices = np.where(first_idx_mask)[0].tolist()
            n_unique = len(keep_indices)
            n_full = len(df)
            logger.info(
                f"CLS attn dedup: {n_full:,} rows -> {n_unique:,} unique "
                f"images (same dedup as token cache)"
            )
            unique_view = self._indexed_dataset_view(loader.dataset, keep_indices)
            from torch.utils.data import DataLoader as _DataLoader
            iter_loader = _DataLoader(
                unique_view,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=getattr(loader, "num_workers", 0),
                pin_memory=False,
                drop_last=False,
                collate_fn=getattr(loader, "collate_fn", None),
            )
            iter_total_n = n_unique
            is_deduped = True
            unique_paths = df.loc[first_idx_mask, "image_path"].reset_index(drop=True)
            path_to_pos = {p: i for i, p in enumerate(unique_paths)}
            unique_to_full_idx = torch.tensor(
                df["image_path"].map(path_to_pos).values, dtype=torch.long
            )
        else:
            iter_loader = loader
            iter_total_n = len(loader.dataset)
            unique_to_full_idx = None

        # Pre-allocate (N, P) float16 storage, filled by offset. Patch count
        # P is derived from the first batch's hooked input size.
        cls_attn_buf: Optional[torch.Tensor] = None
        offset = 0

        # Forward pre-hook on the last block's attention module captures
        # its (already-LN'd) input. After the main forward we recompute q/k
        # manually from that input and derive the CLS→patches attention.
        captured_input: Dict[str, torch.Tensor] = {}

        def _pre_hook(module, inputs):
            captured_input["x"] = inputs[0]

        last_attn = timm_model.blocks[int(layer_idx)].attn
        handle = last_attn.register_forward_pre_hook(_pre_hook)

        try:
            for batch in tqdm(
                iter_loader, total=len(iter_loader), file=sys.stdout,
                desc="CLS attention",
            ):
                images, _ = batch
                with torch.no_grad():
                    images = images.to(self.device, non_blocking=True)
                    _ = timm_model(images)

                    x = captured_input["x"]        # (B, N, C) LN'd input
                    B, N, C = x.shape
                    num_heads = last_attn.num_heads
                    head_dim = C // num_heads
                    qkv = last_attn.qkv(x).reshape(B, N, 3, num_heads, head_dim)
                    q, k, _v = torch.unbind(qkv, dim=2)
                    q = q.transpose(1, 2)          # (B, H, N, D_h)
                    k = k.transpose(1, 2)
                    scale = head_dim ** -0.5
                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    # CLS (index 0) → patches (index 1:)
                    cls_to_patches = attn_weights[:, :, 0, 1:]  # (B, H, P)
                    cls_attn = cls_to_patches.mean(dim=1)        # (B, P)
                    cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                    cls_attn_cpu = cls_attn.to(dtype=torch.float16).cpu()
                    if cls_attn_buf is None:
                        P = cls_attn_cpu.shape[1]
                        cls_attn_buf = torch.empty(
                            (iter_total_n, P), dtype=torch.float16
                        )
                    bs = cls_attn_cpu.shape[0]
                    cls_attn_buf[offset : offset + bs] = cls_attn_cpu
                    offset += bs
        finally:
            handle.remove()
            del timm_model
            torch.cuda.empty_cache()

        if offset < iter_total_n:
            cls_attn_buf = cls_attn_buf[:offset]

        save_dict = {"features": cls_attn_buf}
        if is_deduped and unique_to_full_idx is not None:
            save_dict["unique_to_full_idx"] = unique_to_full_idx
            save_dict["is_deduped_extraction"] = True
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)
        logger.info(
            f"Saved CLS attention cache: {save_path} "
            f"shape={tuple(cls_attn_buf.shape)} dtype={cls_attn_buf.dtype}"
        )
        return cls_attn_buf

    def _load_or_build_text_mask(
        self, loader, llm_model_name: str, suffix: str
    ) -> torch.Tensor:
        """Load cached text attention mask, or tokenise the loader to build one.

        The main ``get_text_features`` call doesn't persist masks — we write a
        companion file next to the features cache with the same base name
        plus ``_mask`` suffix.
        """
        dataset_name = (
            loader.dataset.name
            if hasattr(loader.dataset, "name")
            else type(loader.dataset).__name__
        )
        features_path = AlignmentTrainer.get_feature_save_path(
            m_name=llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )
        mask_path = features_path.with_name(
            features_path.stem + "_mask" + features_path.suffix
        )
        if mask_path.exists():
            payload = torch.load(mask_path, weights_only=False)
            logger.debug(f"Loaded text mask from: {mask_path}")
            return payload["mask"]

        # Build masks by re-running the tokenizer over the dataloader.
        _, tokenizer = self.get_llm(llm_model_name=llm_model_name)
        loader.dataset.tokenizer = tokenizer
        if hasattr(loader.dataset, "loading_type"):
            loader.dataset.loading_type = LoadingType.TXT_ONLY
        loader.dataset.apply_tokenizer()

        masks = []
        for batch in tqdm(
            loader, total=len(loader), file=sys.stdout, desc=f"text-mask[{suffix}]"
        ):
            _, token_inputs = batch
            masks.append(token_inputs["attention_mask"].cpu())
        mask = torch.cat(masks, dim=0)

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask}, mask_path)
        logger.debug(f"Saved text mask to: {mask_path}")
        return mask

    def _load_eval_token_features(
        self,
        eval_loader,
        img_layer_idx: int,
        txt_layer_idx: int,
    ):
        """Load (or extract) token features + text mask for a retrieval eval set.

        Temporarily flips ``pool_img``/``pool_txt`` to ``none`` and pins the
        layer indices so that ``get_image_features``/``get_text_features``
        write a single-layer token cache. Uses a distinct ``eval-*`` suffix
        so it does not collide with training caches.
        """
        orig_cfg = self.config["features"]
        save_pool_img = orig_cfg.get("pool_img")
        save_pool_txt = orig_cfg.get("pool_txt")
        save_layer_img = orig_cfg.get("layer_img")
        save_layer_txt = orig_cfg.get("layer_txt")

        orig_cfg["pool_img"] = "none"
        orig_cfg["pool_txt"] = "none"
        orig_cfg["layer_img"] = int(img_layer_idx)
        orig_cfg["layer_txt"] = int(txt_layer_idx)

        img_size = orig_cfg.get("img_size")
        res_tag = f"-r{int(img_size)}" if img_size is not None else ""
        img_suffix = f"eval-none_layer-{int(img_layer_idx)}{res_tag}"
        txt_suffix = f"eval-none_layer-{int(txt_layer_idx)}"
        try:
            # Eval val cache MUST stay aligned 1:1 with the eval text cache
            # (same row count) because the retrieval per-batch loop slices
            # both with a shared `i` index. Disable image dedup here.
            img_feats = self.get_image_features(
                loader=eval_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=img_suffix,
                allow_image_dedup=False,
            )
            txt_feats = self.get_text_features(
                loader=eval_loader,
                llm_model_name=self.llm_model_name,
                suffix=txt_suffix,
            )
            txt_mask = self._load_or_build_text_mask(
                loader=eval_loader,
                llm_model_name=self.llm_model_name,
                suffix=txt_suffix,
            )
        finally:
            orig_cfg["pool_img"] = save_pool_img
            orig_cfg["pool_txt"] = save_pool_txt
            if save_layer_img is None:
                orig_cfg.pop("layer_img", None)
            else:
                orig_cfg["layer_img"] = save_layer_img
            if save_layer_txt is None:
                orig_cfg.pop("layer_txt", None)
            else:
                orig_cfg["layer_txt"] = save_layer_txt
        return img_feats, txt_feats, txt_mask

    # ---------------------------------------------------------------------
    # Unified token-cache loaders
    # ---------------------------------------------------------------------
    # The unified pipeline extracts (N, T, D) image / (N, S, D) text token
    # tensors once per (model, dataset, split, layer, img_size). Both the
    # CLS-style (pool=cls/avg) and the token-level training paths can then
    # be served from the same files:
    #   image CLS  -> tokens[:, 0, :]
    #   image full -> tokens
    #   text  avg  -> masked mean over the sequence dim
    #   text  full -> tokens (+ mask)
    # The helpers below check for a token cache before falling back to the
    # legacy multi-layer extraction.

    def _unified_image_token_path(
        self, dataset_name: str, split_tag: str, layer_idx: int
    ) -> Path:
        img_size = self.config["features"].get("img_size")
        res_tag = f"-r{int(img_size)}" if img_size is not None else ""
        suffix = f"{split_tag}-none_layer-{int(layer_idx)}{res_tag}"
        return AlignmentTrainer.get_feature_save_path(
            m_name=self.lvm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )

    def _unified_text_token_path(
        self, dataset_name: str, split_tag: str, layer_idx: int
    ) -> Tuple[Path, Path]:
        suffix = f"{split_tag}-none_layer-{int(layer_idx)}"
        feats_path = AlignmentTrainer.get_feature_save_path(
            m_name=self.llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )
        mask_path = feats_path.with_name(
            feats_path.stem + "_mask" + feats_path.suffix
        )
        return feats_path, mask_path

    def _try_load_image_cls_from_tokens(
        self, dataset_name: str, split_tag: str, layer_idx: int
    ) -> Optional[torch.Tensor]:
        """If a unified image token cache exists, return CLS slice (N, D)."""
        path = self._unified_image_token_path(
            dataset_name=dataset_name, split_tag=split_tag, layer_idx=layer_idx
        )
        if not path.exists():
            return None
        payload = torch.load(path, weights_only=False)
        feats = payload["features"]
        # tokens[:, 0, :] is the CLS token under DINOv2's standard layout.
        cls = feats[:, 0, :].contiguous()
        logger.debug(
            f"Derived CLS from unified token cache: {path} "
            f"shape={tuple(cls.shape)} dtype={cls.dtype}"
        )
        return cls

    def _try_load_text_avg_from_tokens(
        self, dataset_name: str, split_tag: str, layer_idx: int
    ) -> Optional[torch.Tensor]:
        """If a unified text token + mask cache exists, return masked mean."""
        feats_path, mask_path = self._unified_text_token_path(
            dataset_name=dataset_name, split_tag=split_tag, layer_idx=layer_idx
        )
        if not (feats_path.exists() and mask_path.exists()):
            return None
        feats = torch.load(feats_path, weights_only=False)["features"]
        mask = torch.load(mask_path, weights_only=False)["mask"]
        # masked mean over the sequence axis: (feats * mask).sum(1) / mask.sum(1)
        m = mask.to(dtype=feats.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1)
        avg = (feats * m).sum(dim=1) / denom
        logger.debug(
            f"Derived masked-mean text features from unified token cache: "
            f"{feats_path} shape={tuple(avg.shape)} dtype={avg.dtype}"
        )
        return avg

    def fit(
        self,
        n_random_subsample_train: Optional[int] = None,
        n_random_subsample_val: Optional[int] = None,
        additional_unimodal_data: Optional[Dict[str, list]] = None,
        n_random_additional_feats: Optional[int] = None,
    ):
        # pre-compute the embeddings from both modalities
        # first embed the validation set since we're returning
        # the models for the training set
        # Only tag the cache with layer index when using single-layer
        # (pool=none) extraction. pool=cls/avg features include all layers,
        # so the suffix must stay layer-agnostic to hit the shared cache.
        # img_size in the suffix guards against mixing 224 and 518 caches.
        pool_img = self.config["features"]["pool_img"]
        pool_txt = self.config["features"]["pool_txt"]
        img_size_cfg = self.config["features"].get("img_size")
        res_tag = f"-r{int(img_size_cfg)}" if img_size_cfg is not None else ""
        cfg_layer_img = self.config["features"].get("layer_img")
        cfg_layer_txt = self.config["features"].get("layer_txt")

        # Unified-cache fast path: when the layer is pinned and a
        # (N, T, D) token cache already exists, derive the (N, D)
        # CLS / masked-mean view from it and skip the multi-layer
        # extraction. Also when token_level=true, skip the CLS pre-load
        # entirely — the token override later in fit() supplies the
        # real tensors and the CLS pre-load is wasted compute.
        token_level_cfg = bool(self.config["training"].get("token_level", False))

        def _train_dataset_name(loader):
            if hasattr(loader.dataset, "name"):
                return loader.dataset.name
            return type(loader.dataset).__name__

        train_ds_name = _train_dataset_name(self.train_dataset)
        val_ds_name = _train_dataset_name(self.val_dataset)

        image_val_suffix = f"val-{pool_img}"
        if self.image_features_val is None:
            if (
                pool_img == "none"
                and cfg_layer_img is not None
            ):
                image_val_suffix += f"_layer-{cfg_layer_img}"
            image_val_suffix += res_tag

            image_features_val = None
            if cfg_layer_img is not None and pool_img != "none":
                # Try unified-cache-derived CLS first (avoids multi-layer extraction).
                derived = self._try_load_image_cls_from_tokens(
                    dataset_name=val_ds_name,
                    split_tag="val",
                    layer_idx=cfg_layer_img,
                )
                if derived is not None:
                    image_features_val = derived
            if image_features_val is None:
                if token_level_cfg and cfg_layer_img is not None:
                    # Skip: tokens will be loaded later. Use a tiny stub
                    # that satisfies shape/dim checks without extraction.
                    image_features_val = torch.zeros(
                        (len(self.val_dataset.dataset), 1),
                        dtype=torch.float32,
                    )
                else:
                    image_features_val = self.get_image_features(
                        loader=self.val_dataset,
                        lvm_model_name=self.lvm_model_name,
                        suffix=image_val_suffix,
                    )
        else:
            image_features_val = self.image_features_val

        text_val_suffix = f"val-{pool_txt}"
        if self.text_features_val is None:
            if (
                pool_txt == "none"
                and cfg_layer_txt is not None
            ):
                text_val_suffix += f"_layer-{cfg_layer_txt}"

            text_features_val = None
            if cfg_layer_txt is not None and pool_txt != "none":
                derived = self._try_load_text_avg_from_tokens(
                    dataset_name=val_ds_name,
                    split_tag="val",
                    layer_idx=cfg_layer_txt,
                )
                if derived is not None:
                    text_features_val = derived
            if text_features_val is None:
                if token_level_cfg and cfg_layer_txt is not None:
                    text_features_val = torch.zeros(
                        (len(self.val_dataset.dataset), 1),
                        dtype=torch.float32,
                    )
                else:
                    text_features_val = self.get_text_features(
                        loader=self.val_dataset,
                        llm_model_name=self.llm_model_name,
                        suffix=text_val_suffix,
                    )
        else:
            text_features_val = self.text_features_val

        if self.image_features_train is None:
            image_features_train = None
            if cfg_layer_img is not None and pool_img != "none":
                derived = self._try_load_image_cls_from_tokens(
                    dataset_name=train_ds_name,
                    split_tag="train",
                    layer_idx=cfg_layer_img,
                )
                if derived is not None:
                    image_features_train = derived
            if image_features_train is None:
                if token_level_cfg and cfg_layer_img is not None:
                    image_features_train = torch.zeros(
                        (len(self.train_dataset.dataset), 1),
                        dtype=torch.float32,
                    )
                else:
                    image_features_train = self.get_image_features(
                        loader=self.train_dataset,
                        lvm_model_name=self.lvm_model_name,
                        suffix=image_val_suffix.replace("val-", "train-"),
                    )
        else:
            image_features_train = self.image_features_train

        if self.text_features_train is None:
            text_features_train = None
            if cfg_layer_txt is not None and pool_txt != "none":
                derived = self._try_load_text_avg_from_tokens(
                    dataset_name=train_ds_name,
                    split_tag="train",
                    layer_idx=cfg_layer_txt,
                )
                if derived is not None:
                    text_features_train = derived
            if text_features_train is None:
                if token_level_cfg and cfg_layer_txt is not None:
                    text_features_train = torch.zeros(
                        (len(self.train_dataset.dataset), 1),
                        dtype=torch.float32,
                    )
                else:
                    text_features_train = self.get_text_features(
                        loader=self.train_dataset,
                        llm_model_name=self.llm_model_name,
                        suffix=text_val_suffix.replace("val-", "train-"),
                    )
        else:
            text_features_train = self.text_features_train
        image_dim = image_features_train.shape[-1]
        text_dim = text_features_train.shape[-1]

        # cache features if wanted
        self.image_features_val = image_features_val
        self.image_features_train = image_features_train
        self.text_features_val = text_features_val
        self.text_features_train = text_features_train

        # Compute drop_duplicates index masks ONCE so both the CLS path
        # (mutates here) and the token-level override (later in the
        # combo loop) can apply the same selection. Without this, the
        # token override would silently bypass dedup and train on the
        # full caption-image cross product (591K) instead of the
        # deduped 118K.
        sel_train_indices = None
        sel_val_indices = None
        if (
            self.config["training"]["drop_duplicates"]
            and hasattr(self.train_dataset.dataset, "df")
            and "image_path" in self.train_dataset.dataset.df.columns
        ):
            sel_train_indices = (
                self.train_dataset.dataset.df.groupby("image_path").cumcount()
                < self.config["training"]["n_dup_samples"]
            )
            sel_val_indices = (
                self.val_dataset.dataset.df.groupby("image_path").cumcount()
                < self.config["training"]["n_dup_samples"]
            )

        # Apply the dedup mask per-tensor: when image was deduped at
        # extraction time (118K rows) but text is still 591K, we apply
        # the mask only to the tensor whose row count matches the mask
        # length. After this block all four tensors are at the same
        # row count.
        def _apply_if_full(tensor, mask_full, mask_bool):
            if (
                mask_bool is not None
                and tensor is not None
                and tensor.shape[0] == mask_full
            ):
                return tensor[mask_bool]
            return tensor

        if sel_train_indices is not None:
            full_train_n = len(sel_train_indices)
            image_features_train = _apply_if_full(
                image_features_train, full_train_n, sel_train_indices
            )
            text_features_train = _apply_if_full(
                text_features_train, full_train_n, sel_train_indices
            )
        if sel_val_indices is not None:
            full_val_n = len(sel_val_indices)
            image_features_val = _apply_if_full(
                image_features_val, full_val_n, sel_val_indices
            )
            text_features_val = _apply_if_full(
                text_features_val, full_val_n, sel_val_indices
            )

        # check that we have the same samples — assertion AFTER dedup so
        # the deduped-image-extraction case (118K vs 591K pre-dedup) is
        # handled correctly.
        assert image_features_train.shape[0] == text_features_train.shape[0]
        assert image_features_val.shape[0] == text_features_val.shape[0]

        # remember the subsample permutations so token-level loading can
        # replay them on the fresh token tensors
        token_subsample_train_idx = None
        token_subsample_val_idx = None
        if (
            n_random_subsample_train is not None
            and n_random_subsample_train < image_features_train.shape[0]
        ):
            logger.debug(f"Subsampling train set to {n_random_subsample_train}")
            self.n_random_subsample_train = n_random_subsample_train
            wandb.run.tags = wandb.run.tags + (
                f"TRAIN subsample {n_random_subsample_train}",
            )

            random_sequence = torch.randperm(image_features_train.shape[0])[
                :n_random_subsample_train
            ]
            image_features_train = image_features_train[random_sequence]
            text_features_train = text_features_train[random_sequence]
            token_subsample_train_idx = random_sequence
        if (
            n_random_subsample_val is not None
            and n_random_subsample_val < image_features_val.shape[0]
        ):
            logger.debug(f"Subsampling validation set to {n_random_subsample_val}")
            self.n_random_subsample_val = n_random_subsample_val
            wandb.run.tags = wandb.run.tags + (
                f"VAL subsample {n_random_subsample_val}",
            )

            random_sequence = torch.randperm(image_features_val.shape[0])[
                :n_random_subsample_val
            ]
            image_features_val = image_features_val[random_sequence]
            text_features_val = text_features_val[random_sequence]
            token_subsample_val_idx = random_sequence

        logger.debug(
            f"TRAIN - img: {image_features_train.shape}, txt: {text_features_train.shape}"
        )
        logger.debug(
            f"VAL - img: {image_features_val.shape}, txt: {text_features_val.shape}"
        )

        # additional unimodal data
        additional_image_features = None
        additional_text_features = None
        if additional_unimodal_data is not None:
            additional_image_features = []
            additional_text_features = []
            for modality, modality_datasets in additional_unimodal_data.items():
                for m_dataset_name, m_dataset in modality_datasets:
                    if modality == "text":
                        add_text_features = self.get_text_features(
                            loader=m_dataset,
                            llm_model_name=self.llm_model_name,
                            suffix=text_val_suffix.replace("val-", "train-"),
                            dataset_name=m_dataset_name,
                        )
                        additional_text_features.append(add_text_features)
                    else:
                        add_image_features = self.get_image_features(
                            loader=m_dataset,
                            lvm_model_name=self.lvm_model_name,
                            suffix=image_val_suffix.replace("val-", "train-"),
                            dataset_name=m_dataset_name,
                        )
                        additional_image_features.append(add_image_features)
            if len(additional_image_features) > 0:
                additional_image_features = torch.cat(additional_image_features).cpu()
            else:
                additional_image_features = None
            if len(additional_text_features) > 0:
                additional_text_features = torch.cat(additional_text_features).cpu()
            else:
                additional_text_features = None

        # only compute the best alignment if not specified
        if (
            self.config["features"].get("layer_img") is None
            and self.config["features"].get("layer_txt") is None
        ):
            sampled_df_alignment = self.compute_layer_alignment(
                image_features=image_features_train,
                text_features=text_features_train,
            )
        else:
            sampled_df_alignment = pd.DataFrame(columns=["indices", "alignment_score"])
            sampled_df_alignment.loc[len(sampled_df_alignment)] = [
                (
                    self.config["features"]["layer_img"],
                    self.config["features"]["layer_txt"],
                ),
                np.nan,
            ]

        # for each sampled combination
        # train the alignment between the representations
        print(sampled_df_alignment)
        comb_iter = sampled_df_alignment.iterrows()
        for i_comb, (_, layer_series) in enumerate(comb_iter):
            layer_comb = layer_series["indices"]
            image_layer_idx, text_layer_idx = layer_comb
            layer_comb_score = layer_series["alignment_score"]
            layer_comb_str = f"img_{image_layer_idx}_txt_{text_layer_idx}"

            # Slice features at the chosen layer. Inputs may be:
            #  - (N, L, D) legacy multi-layer CLS extraction
            #  - (N, D)    derived from unified token cache (single-layer CLS)
            #  - (N, 1)    placeholder when token_level skips the CLS pre-load
            def _layer_slice(feats, idx):
                if feats.ndim == 2:
                    return feats
                return feats[:, idx, :]

            layer_image_features_train = _layer_slice(
                image_features_train, image_layer_idx
            )
            layer_text_features_train = _layer_slice(
                text_features_train, text_layer_idx
            )
            layer_image_features_val = _layer_slice(
                image_features_val, image_layer_idx
            )
            layer_text_features_val = _layer_slice(
                text_features_val, text_layer_idx
            )

            # Token-level override: replace the (N, D) CLS slices with
            # (N, T, D) token features for the selected layer, plus masks.
            token_level = bool(
                self.config["training"].get("token_level", False)
            )
            layer_text_mask_train = None
            layer_text_mask_val = None
            if token_level:
                token_bundle = self._load_token_features_for_layer(
                    img_layer_idx=image_layer_idx,
                    txt_layer_idx=text_layer_idx,
                )
                layer_image_features_train = token_bundle["img_train"]
                layer_text_features_train = token_bundle["txt_train"]
                layer_image_features_val = token_bundle["img_val"]
                layer_text_features_val = token_bundle["txt_val"]
                layer_text_mask_train = token_bundle["txt_mask_train"]
                layer_text_mask_val = token_bundle["txt_mask_val"]

                # Apply drop_duplicates to token features so the token
                # path matches the CLS path's caption-image dedup
                # (591K -> 118K for COCO). Each tensor's shape is checked
                # independently against len(sel_*_indices) — that gates
                # both the legacy "everything full" case and the new
                # "image was deduped at extraction" case where the image
                # tensor is already 118K but text / mask are still 591K.
                def _apply_if_full(tensor, mask_full, mask_bool):
                    if (
                        mask_bool is not None
                        and tensor is not None
                        and tensor.shape[0] == mask_full
                    ):
                        return tensor[mask_bool]
                    return tensor

                if sel_train_indices is not None:
                    full_train_n = len(sel_train_indices)
                    layer_image_features_train = _apply_if_full(
                        layer_image_features_train, full_train_n, sel_train_indices
                    )
                    layer_text_features_train = _apply_if_full(
                        layer_text_features_train, full_train_n, sel_train_indices
                    )
                    layer_text_mask_train = _apply_if_full(
                        layer_text_mask_train, full_train_n, sel_train_indices
                    )
                if sel_val_indices is not None:
                    full_val_n = len(sel_val_indices)
                    layer_image_features_val = _apply_if_full(
                        layer_image_features_val, full_val_n, sel_val_indices
                    )
                    layer_text_features_val = _apply_if_full(
                        layer_text_features_val, full_val_n, sel_val_indices
                    )
                    layer_text_mask_val = _apply_if_full(
                        layer_text_mask_val, full_val_n, sel_val_indices
                    )

                # NOTE: in token_level mode the `_load_token_features_for_layer`
                # helper already applies the ``n_random_subsample_*`` cap at
                # extraction time (first-N view on the dataloader) to avoid
                # materialising 100+ GB of full-dataset tokens. The per-tensor
                # shape check above guards against double-application:
                #   - subset extraction (`-n{N}` cache) shape != full df ->
                #     dedup silently skipped (the subset is already sized)
                #   - dedup-at-extraction image cache is already 118K and
                #     != full 591K -> dedup silently skipped on image only,
                #     still applied to the still-full text + mask tensors

                logger.debug(
                    f"TOKEN TRAIN - img: {tuple(layer_image_features_train.shape)}, "
                    f"txt: {tuple(layer_text_features_train.shape)}, "
                    f"txt_mask: {tuple(layer_text_mask_train.shape)}"
                )
                logger.debug(
                    f"TOKEN VAL - img: {tuple(layer_image_features_val.shape)}, "
                    f"txt: {tuple(layer_text_features_val.shape)}, "
                    f"txt_mask: {tuple(layer_text_mask_val.shape)}"
                )

            # CLS-attention prior: load per-image CLS→patch attention from
            # the last DINOv2 block if the configured alignment layer has
            # cls_attn_prior=True. Cache is extracted on demand and shares
            # the N-axis with the image token cache (same dedup).
            image_cls_attn_train = None
            image_cls_attn_val = None
            if token_level:
                akw = self.config["training"].get("alignment_layer_kwargs", {})
                if akw.get("cls_attn_prior", False):
                    # Extraction layer for the CLS attention — use the
                    # same layer as the token features (selected layer).
                    # Reuse the subsampled loaders from the token loader
                    # helper to keep the N-axis aligned with the img cache.
                    max_train_cap = self.config["training"].get(
                        "n_random_subsample_train"
                    )
                    max_val_cap = self.config["training"].get(
                        "n_random_subsample_val"
                    )
                    train_loader_cls = self._subsampled_loader(
                        self.train_dataset, max_train_cap
                    )
                    val_loader_cls = self._subsampled_loader(
                        self.val_dataset, max_val_cap
                    )
                    train_tag = (
                        f"-n{max_train_cap}" if max_train_cap is not None else ""
                    )
                    val_tag = (
                        f"-n{max_val_cap}" if max_val_cap is not None else ""
                    )
                    try:
                        image_cls_attn_train = self.get_image_cls_attention(
                            loader=train_loader_cls,
                            lvm_model_name=self.lvm_model_name,
                            suffix=f"train{train_tag}",
                            layer_idx=int(image_layer_idx),
                        )
                        image_cls_attn_val = self.get_image_cls_attention(
                            loader=val_loader_cls,
                            lvm_model_name=self.lvm_model_name,
                            suffix=f"val{val_tag}",
                            layer_idx=int(image_layer_idx),
                        )
                        logger.info(
                            f"CLS attention loaded: "
                            f"train {tuple(image_cls_attn_train.shape)}, "
                            f"val {tuple(image_cls_attn_val.shape)}"
                        )

                        # Dedup application — same _apply_if_full logic as
                        # the image token tensor. image_cls_attn is already
                        # dedup'd if the image cache is; if sel_*_indices
                        # is full-length (i.e. image tensor is still full),
                        # apply the boolean mask to image_cls_attn too.
                        if sel_train_indices is not None:
                            image_cls_attn_train = _apply_if_full(
                                image_cls_attn_train,
                                len(sel_train_indices),
                                sel_train_indices,
                            )
                        if sel_val_indices is not None:
                            image_cls_attn_val = _apply_if_full(
                                image_cls_attn_val,
                                len(sel_val_indices),
                                sel_val_indices,
                            )
                    except Exception as e:
                        logger.warning(
                            f"CLS attention extraction failed: {type(e).__name__}: "
                            f"{e}. Falling back to standard CAP (cls_attn=None)."
                        )
                        image_cls_attn_train = None
                        image_cls_attn_val = None

            layer_additional_image_features = None
            if additional_image_features is not None:
                layer_additional_image_features = additional_image_features[
                    :, image_layer_idx, :
                ]

            layer_additional_text_features = None
            if additional_text_features is not None:
                layer_additional_text_features = additional_text_features[
                    :, text_layer_idx, :
                ]

            if (
                len(sampled_df_alignment) == 1
                or i_comb == len(sampled_df_alignment) - 1
            ):
                # clean up the memory if we're only doing one comb or its the last
                del image_features_train, text_features_train
                del image_features_val, text_features_val
                if additional_image_features is not None:
                    del additional_image_features
                if additional_text_features is not None:
                    del additional_text_features

            l_add_img_feats = []
            for add_img_feat_paths in self.config["features"].get(
                "add_img_feat_paths", []
            ):
                if Path(add_img_feat_paths).exists():
                    add_img_feats = torch.load(add_img_feat_paths, weights_only=False)[
                        "image_feats"
                    ]
                    l_add_img_feats.append(add_img_feats)
                    logger.debug(f"Loaded features from: {add_img_feat_paths}")
            if len(l_add_img_feats) > 1:
                l_add_img_feats = torch.cat(l_add_img_feats, dim=0)

            l_add_txt_feats = []
            for add_txt_feat_paths in self.config["features"].get(
                "add_txt_feat_paths", []
            ):
                if Path(add_txt_feat_paths).exists():
                    add_txt_feats = torch.load(add_txt_feat_paths, weights_only=False)[
                        "text_feats"
                    ]
                    l_add_txt_feats.append(add_txt_feats)
                    logger.debug(f"Loaded features from: {add_txt_feat_paths}")
            if len(l_add_txt_feats) > 1:
                l_add_txt_feats = torch.cat(l_add_txt_feats, dim=0)

            if n_random_additional_feats == 0:
                del l_add_img_feats, l_add_txt_feats
            else:
                if (
                    n_random_additional_feats is not None
                    and n_random_additional_feats < l_add_img_feats.shape[0]
                ):
                    logger.debug(f"Subsampling LAION to {n_random_additional_feats}")
                    wandb.run.tags = wandb.run.tags + (
                        f"LAION subsample {n_random_additional_feats}",
                    )
                    random_sequence = torch.randperm(l_add_img_feats.shape[0])[
                        :n_random_additional_feats
                    ]
                    l_add_img_feats = l_add_img_feats[random_sequence]
                    l_add_txt_feats = l_add_txt_feats[random_sequence]
                if len(l_add_img_feats) > 1:
                    layer_image_features_train = torch.cat(
                        (layer_image_features_train, l_add_img_feats), dim=0
                    )
                    logger.debug(
                        f"New train dim image: {layer_image_features_train.shape}"
                    )
                if len(l_add_txt_feats) > 1:
                    layer_text_features_train = torch.cat(
                        (layer_text_features_train, l_add_txt_feats), dim=0
                    )
                    logger.debug(
                        f"New train dim text: {layer_text_features_train.shape}"
                    )

            log_dict = {
                f"{layer_comb_str}/meta/layer_comb": layer_comb,
                f"{layer_comb_str}/meta/layer_comb_score": layer_comb_score,
            }
            if self.n_random_subsample_train is not None:
                log_dict["meta/n_random_subsample_train"] = (
                    self.n_random_subsample_train
                )
            if self.n_random_subsample_val is not None:
                log_dict["meta/n_random_subsample_val"] = self.n_random_subsample_val

            if (
                self.config["training"]["visualize_original_embeddings"]
                and not token_level
            ):
                # visualize the original embedding space (2D only)
                fig_emb_image = embedding_plot(
                    X=layer_image_features_val.float().numpy(),
                    return_figure=True,
                )
                fig_emb_text = embedding_plot(
                    X=layer_text_features_val.float().numpy(),
                    return_figure=True,
                )
                log_dict[f"{layer_comb_str}/embedding_plot/val_original_emb_image"] = (
                    wandb.Image(fig_emb_image)
                )
                log_dict[f"{layer_comb_str}/embedding_plot/val_original_emb_text"] = (
                    wandb.Image(fig_emb_text)
                )
                plt.close(fig_emb_image)
                plt.close(fig_emb_text)
            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

            logger.info(
                f"Training alignment for layers {layer_comb} (score: {layer_comb_score:.4f})"
            )

            # define the loss function
            self.loss = CLIPLoss(
                **self.config["training"]["clip_loss"],
            )
            self.loss = self.loss.to(self.device)

            alignment_image = AlignmentFactory.create(
                self.config["training"]["alignment_layer_name"],
                input_dim=image_dim,
                **self.config["training"]["alignment_layer_kwargs"],
            )
            alignment_text = AlignmentFactory.create(
                self.config["training"]["alignment_layer_name"],
                input_dim=text_dim,
                **self.config["training"]["alignment_layer_kwargs"],
            )
            # Some alignment layers (e.g. FreezeAlignAlignmentLayer) need to
            # know which modality they serve because they hold both modalities'
            # components in a single class. Backwards compatible: layers that
            # don't define set_modality see no change.
            if hasattr(alignment_image, "set_modality"):
                alignment_image.set_modality("image")
            if hasattr(alignment_text, "set_modality"):
                alignment_text.set_modality("text")
            if self.config["training"]["wandb_watch"]:
                wandb.watch(models=[alignment_image, alignment_text], log="all")

            if self.print_model_summary and not token_level:
                print("*" * 20 + " Alignment Image " + "*" * 20)
                input_size = (self.train_batch_size,)
                summary(
                    alignment_image,
                    input_size=input_size + (image_dim,),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                )
                print("*" * 20 + " Alignment Text " + "*" * 20)
                summary(
                    alignment_text,
                    input_size=input_size + (text_dim,),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                )
            elif self.print_model_summary and token_level:
                # torchinfo.summary with mask kwarg is awkward; just print parameter counts
                n_img = sum(p.numel() for p in alignment_image.parameters())
                n_txt = sum(p.numel() for p in alignment_text.parameters())
                print(
                    f"[token_level] alignment_image params={n_img}, "
                    f"alignment_text params={n_txt}, "
                    f"img_tokens={tuple(layer_image_features_train.shape)}, "
                    f"txt_tokens={tuple(layer_text_features_train.shape)}"
                )

            optimizer_cls = get_optimizer_type(
                optimizer_name=self.config["training"]["optimizer_name"].lower(),
            )
            if self.config["training"].get("use_lr_finder", False):
                logger.debug("Running learning rate finder...")
                optimal_lr = self.find_optimal_learning_rate(
                    image_features_train=layer_image_features_train,
                    text_features_train=layer_text_features_train,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    optimizer_cls=optimizer_cls,
                    wandb_prefix=f"{layer_comb_str}/",
                    text_mask_train=layer_text_mask_train,
                    **self.config["training"]["lr_finder"],
                )
                logger.debug(f"LR finder complete. Using learning rate: {optimal_lr}")
                self.config["training"]["learning_rate"] = optimal_lr

            params = list(alignment_image.parameters()) + list(
                alignment_text.parameters()
            )

            optimizer = optimizer_cls(
                params=params,
                lr=self.config["training"]["learning_rate"],
                **self.config["training"]["optimizer_kwargs"],
            )
            if self.config["training"]["scheduler_name"] is None:
                scheduler = None
            if self.config["training"]["scheduler_name"] == "WarmupDecayLR":
                scheduler = WarmupDecayLR(
                    optimizer,
                    total_num_steps=self.config["training"]["n_epochs"]
                    * max(
                        (len(layer_image_features_train) // self.train_batch_size), 1
                    ),
                    **self.config["training"]["scheduler_kwargs"],
                )
            elif self.config["training"]["scheduler_name"] == "WarmupLR":
                scheduler = WarmupLR(
                    optimizer,
                    **self.config["training"]["scheduler_kwargs"],
                )
            elif self.config["training"]["scheduler_name"] == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.config["training"]["scheduler_epoch_cycles"]
                    * max(
                        (len(layer_image_features_train) // self.train_batch_size), 1
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown learning rate scheduler: {self.config['training']['scheduler_name']}"
                )

            if self.config["training"]["early_stopping"]:
                early_stopping = EarlyStopping(
                    patience=self.config["training"]["early_stopping_patience"],
                    log_messages=True,
                )
            best_epoch = 0
            best_val_loss = float("inf")
            best_weights_alignment_image = copy.deepcopy(alignment_image.state_dict())
            best_weights_alignment_text = copy.deepcopy(alignment_text.state_dict())

            train_step = 0
            for epoch in (pbar := trange(self.config["training"]["n_epochs"])):
                alignment_image = alignment_image.to(self.device)
                alignment_text = alignment_text.to(self.device)

                steps, train_loss = self.train(
                    epoch=epoch,
                    train_step=train_step,
                    image_features=layer_image_features_train,
                    text_features=layer_text_features_train,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    additional_image_features=layer_additional_image_features,
                    additional_text_features=layer_additional_text_features,
                    wandb_prefix=f"{layer_comb_str}/",
                    text_mask=layer_text_mask_train,
                    image_cls_attn=image_cls_attn_train,
                )
                train_step += steps

                with torch.no_grad():
                    val_loss = self.validate(
                        epoch=epoch,
                        train_step=train_step,
                        image_features=layer_image_features_val,
                        text_features=layer_text_features_val,
                        alignment_image=alignment_image,
                        alignment_text=alignment_text,
                        wandb_prefix=f"{layer_comb_str}/",
                        text_mask=layer_text_mask_val,
                        image_cls_attn=image_cls_attn_val,
                    )
                pbar.set_description(
                    f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_weights_alignment_image = copy.deepcopy(
                        alignment_image.cpu().state_dict()
                    )
                    best_weights_alignment_text = copy.deepcopy(
                        alignment_text.cpu().state_dict()
                    )

                if self.config["training"]["early_stopping"]:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

            if self.config["training"]["early_stopping"]:
                # load the best model (if using early stopping)
                alignment_image.load_state_dict(best_weights_alignment_image)
                alignment_text.load_state_dict(best_weights_alignment_text)

            # save the alignment
            if self.config["training"]["wandb_watch"]:
                wandb.unwatch(models=[alignment_image, alignment_text])
            save_dict = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "train_step": train_step,
                "alignment_text": alignment_text,
                "alignment_image": alignment_image,
                "optimizer": optimizer.state_dict(),
                "config": self.config,
                "loss": self.loss.state_dict(),
            }
            save_checkpoint(
                run_dir=self.save_path
                / wandb.run.name
                / f"{layer_comb}_{layer_comb_score:.4f}",
                save_dict=save_dict,
                epoch=epoch,
            )

            # evaluate
            res_dict = {
                "layer_comb": layer_comb,
                "layer_comb_alignment": layer_comb_score,
                "epoch": epoch,
                "train_step": train_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            with torch.no_grad():
                self.evaluate_retrieval(
                    epoch=epoch,
                    train_step=train_step,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )
                gc.collect()
                self.evaluate_zero_shot_classification(
                    epoch=epoch,
                    train_step=train_step,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )
        # first stop the wandb run
        wandb.run.finish()
        wandb.finish()

    def train(
        self,
        epoch: int,
        train_step: int,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        optimizer,
        scheduler=None,
        additional_image_features: Optional[torch.Tensor] = None,
        additional_text_features: Optional[torch.Tensor] = None,
        wandb_prefix: str = "",
        text_mask: Optional[torch.Tensor] = None,
        image_cls_attn: Optional[torch.Tensor] = None,
    ):
        num_samples = image_features.shape[0]

        # randomly shuffle the embeddings since we didn't shuffle before
        random_sequence = torch.randperm(num_samples)
        image_features = image_features[random_sequence]
        text_features = text_features[random_sequence]
        if text_mask is not None:
            text_mask = text_mask[random_sequence]
        if image_cls_attn is not None:
            image_cls_attn = image_cls_attn[random_sequence]

        # NOTE: ablation from reviewers (fixed set for R_S)
        random_sequence_fixed = torch.randperm(num_samples)[: self.train_batch_size]
        image_features_fixed = image_features[random_sequence_fixed]
        text_features_fixed = text_features[random_sequence_fixed]

        # in order to efficiently loop over the splits we use splits and modulo
        if additional_image_features is not None:
            random_sequence = torch.randperm(additional_image_features.shape[0])
            additional_image_features = additional_image_features[random_sequence]
            additional_image_features = torch.split(
                additional_image_features, self.train_batch_size, dim=0
            )
        if additional_text_features is not None:
            random_sequence = torch.randperm(additional_text_features.shape[0])
            additional_text_features = additional_text_features[random_sequence]
            additional_text_features = torch.split(
                additional_text_features, self.train_batch_size, dim=0
            )

        alignment_image.train()
        alignment_text.train()

        l_aligned_image_feats = []
        l_aligned_text_feats = []

        # FuseMix setup
        mixup_alpha = self.config["training"].get("mixup_alpha", 0.0)

        loss_metric = torchmetrics.MeanMetric().to(self.device)
        for i in range(0, num_samples, self.train_batch_size):
            end_i = i + self.train_batch_size
            if end_i > num_samples and mixup_alpha > 0.0:
                continue  # Skip last batch if it's not full, to simplify mixup

            image_feats = image_features[i:end_i]
            text_feats = text_features[i:end_i]
            image_feats = image_feats.float().to(self.device)
            text_feats = text_feats.float().to(self.device)

            if self.config["training"].get("fixed_structure", False):
                image_features_fixed = image_features_fixed.float().to(self.device)
                text_features_fixed = text_features_fixed.float().to(self.device)

            if mixup_alpha > 0.0:
                # To get a second batch, we can simply roll the original tensor
                # This is an efficient way to pair each sample with a different one
                roll_amount = self.train_batch_size // 2
                image_feats2 = torch.roll(image_features, shifts=roll_amount, dims=0)[
                    i:end_i
                ]
                text_feats2 = torch.roll(text_features, shifts=roll_amount, dims=0)[
                    i:end_i
                ]
                image_feats2 = image_feats2.float().to(self.device)
                text_feats2 = text_feats2.float().to(self.device)
                # Apply Mixup
                # Sample a single interpolation coefficient
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Create the augmented features by interpolating between the two batches
                image_feats = lam * image_feats + (1 - lam) * image_feats2
                text_feats = lam * text_feats + (1 - lam) * text_feats2

            # step scheduler of the loss function
            self.loss.step()

            # zero out the gradients
            optimizer.zero_grad()

            # forward pass through alignment layers
            # Conditionally pass cls_attn prior: only when the layer opts
            # in via `cls_attn_prior=True` AND we actually have a tensor
            # for this batch. Keeps non-token layers (Linear, MLP, etc.)
            # unaffected — they never see the kwarg.
            if (
                image_cls_attn is not None
                and getattr(alignment_image, "cls_attn_prior", False)
            ):
                img_cls_attn_batch = image_cls_attn[i:end_i].to(self.device)
                aligned_image_feats = alignment_image(
                    image_feats, cls_attn=img_cls_attn_batch
                )
            else:
                aligned_image_feats = alignment_image(image_feats)
            if text_mask is not None:
                text_mask_batch = text_mask[i:end_i].to(self.device)
                aligned_text_feats = alignment_text(text_feats, mask=text_mask_batch)
            else:
                aligned_text_feats = alignment_text(text_feats)

            # additional unimodal data
            loss_kwargs = {}
            if additional_image_features is not None:
                N_splits_img = len(additional_image_features)
                add_image_feats = additional_image_features[i % N_splits_img]
                add_image_feats = add_image_feats.float().to(self.device)
                add_aligned_image_feats = alignment_image(add_image_feats)
                loss_kwargs["add_image_features"] = (
                    add_image_feats,
                    add_aligned_image_feats,
                )

            if additional_text_features is not None:
                N_splits_txt = len(additional_text_features)
                add_text_feats = additional_text_features[i % N_splits_txt]
                add_text_feats = add_text_feats.float().to(self.device)
                add_aligned_text_feats = alignment_text(add_text_feats)
                loss_kwargs["add_text_features"] = (
                    add_text_feats,
                    add_aligned_text_feats,
                )

            if self.config["training"].get("fixed_structure", False):
                # hack to do what they want
                self.loss.structure_use_only_unimodal = True
                aligned_image_features_fixed = alignment_image(image_features_fixed)
                aligned_text_features_fixed = alignment_text(text_features_fixed)
                loss_kwargs["add_image_features"] = (
                    image_features_fixed,
                    aligned_image_features_fixed,
                )
                loss_kwargs["add_text_features"] = (
                    text_features_fixed,
                    aligned_text_features_fixed,
                )

            # backward pass with clip loss
            loss_dict = self.loss(
                image_embeddings_aligned=aligned_image_feats,
                text_embeddings_aligned=aligned_text_feats,
                image_embeddings_original=image_feats,
                text_embeddings_original=text_feats,
                **loss_kwargs,
            )
            loss = loss_dict["overall_loss"]
            loss_metric.update(loss, weight=image_feats.size(0))
            loss.backward()
            clip_grad = self.config["training"]["clip_grad"]
            if clip_grad:
                _ = clip_gradients(alignment_image, clip_grad)
                _ = clip_gradients(alignment_text, clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # speeds up the training by only adding if we have yet to fill up the buffer
            if len(l_aligned_image_feats) * self.train_batch_size < 10_000:
                l_aligned_image_feats.append(aligned_image_feats.detach().cpu())
                l_aligned_text_feats.append(aligned_text_feats.detach().cpu())

            loss_dict = {f"{wandb_prefix}{k}": v for k, v in loss_dict.items()}
            log_dict = loss_dict | {
                f"{wandb_prefix}learning_rate": optimizer.param_groups[0]["lr"],
                f"{wandb_prefix}weight_decay": optimizer.param_groups[0][
                    "weight_decay"
                ],
                f"{wandb_prefix}structure_lambda": self.loss.structure_lambda,
                f"{wandb_prefix}train_loss": loss,
                "counters/epoch": epoch,
                "counters/train_step": train_step + i,
            }
            if self.wandb_logging:
                wandb.log(log_dict)

        log_dict = {
            f"{wandb_prefix}train_loss_avg": loss_metric.compute().item(),
            "counters/epoch": epoch,
            "counters/train_step": train_step + i,
        }
        if (
            self.config["training"].get("log_structural_preservation", False)
            or self.config["training"].get("log_repr_similarity", False)
            or epoch % self.config["training"]["embedding_visualization"] == 0
        ):
            l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
            l_aligned_text_feats = torch.cat(l_aligned_text_feats).cpu()
        if self.config["training"].get("log_structural_preservation", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            for mod, original, aligned in [
                ("image", image_features, l_aligned_image_feats),
                ("text", text_features, l_aligned_text_feats),
            ]:
                for k in self.config["training"].get(
                    "structural_preservation_k", [100]
                ):
                    tw = trustworthiness(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    cont = continuity(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    log_dict[f"{wandb_prefix}trustworthiness@{k}_{mod}_train"] = tw
                    log_dict[f"{wandb_prefix}continuity@{k}_{mod}_train"] = cont
        if self.config["training"].get("log_repr_similarity", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            alignment_score_img, _, _ = compute_score(
                x_feats=image_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_image_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            alignment_score_txt, _, _ = compute_score(
                x_feats=text_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_text_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_image_train"
            ] = alignment_score_img
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_text_train"
            ] = alignment_score_txt
        if epoch % self.config["training"]["embedding_visualization"] == 0:
            l_aligned_feats = torch.cat([l_aligned_image_feats, l_aligned_text_feats])
            l_aligned_targets = np.ones((len(l_aligned_feats),))
            l_aligned_targets[: len(l_aligned_image_feats)] = 0
            label_dict = {0: "images", 1: "texts"}

            fig_emb = embedding_plot(
                X=l_aligned_feats.numpy(),
                y=l_aligned_targets,
                label_dict=label_dict,
                return_figure=True,
            )
            log_dict[f"{wandb_prefix}train_aligned_emb"] = wandb.Image(fig_emb)
            log_dict[f"{wandb_prefix}train_modality_gap"] = (
                l_aligned_image_feats.mean(dim=0) - l_aligned_text_feats.mean(dim=0)
            ).norm(p=2)
            plt.close(fig_emb)
            plt.close("all")

        if self.wandb_logging:
            wandb.log(log_dict)
        del log_dict

        return i, loss_metric.compute().item()

    def validate(
        self,
        epoch: int,
        train_step: int,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        wandb_prefix: str = "",
        text_mask: Optional[torch.Tensor] = None,
        image_cls_attn: Optional[torch.Tensor] = None,
    ):
        num_samples = image_features.shape[0]

        alignment_image.eval()
        alignment_text.eval()

        l_aligned_image_feats = []
        l_aligned_text_feats = []

        loss_metric_val = torchmetrics.MeanMetric().to(self.device)
        for i in range(0, num_samples, self.train_batch_size):
            image_feats = image_features[i : i + self.train_batch_size]
            text_feats = text_features[i : i + self.train_batch_size]
            image_feats = image_feats.float().cuda()
            text_feats = text_feats.float().cuda()

            # Conditionally pass cls_attn prior — see train() for why.
            if (
                image_cls_attn is not None
                and getattr(alignment_image, "cls_attn_prior", False)
            ):
                img_cls_attn_batch = image_cls_attn[
                    i : i + self.train_batch_size
                ].to(self.device)
                aligned_image_feats = alignment_image(
                    image_feats, cls_attn=img_cls_attn_batch
                )
            else:
                aligned_image_feats = alignment_image(image_feats)
            if text_mask is not None:
                text_mask_batch = text_mask[i : i + self.train_batch_size].to(
                    self.device
                )
                aligned_text_feats = alignment_text(
                    text_feats, mask=text_mask_batch
                )
            else:
                aligned_text_feats = alignment_text(text_feats)

            # backward pass with clip loss
            loss_dict = self.loss(
                image_embeddings_aligned=aligned_image_feats,
                text_embeddings_aligned=aligned_text_feats,
                image_embeddings_original=image_feats,
                text_embeddings_original=text_feats,
            )
            # compute the median cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                aligned_image_feats, aligned_text_feats, dim=-1
            )
            median_cos = cos.median().item()
            loss = loss_dict["overall_loss"]
            loss_metric_val.update(loss, weight=image_feats.size(0))
            loss_dict = {f"{wandb_prefix}val_{k}": v for k, v in loss_dict.items()}
            loss_dict[f"{wandb_prefix}val_median_cos"] = median_cos
            log_dict = loss_dict | {
                "counters/epoch": epoch,
                "counters/train_step": train_step + i,
            }
            if self.wandb_logging:
                wandb.log(log_dict)

            # speeds up the training by only adding if we have yet to fill up the buffer
            if len(l_aligned_image_feats) * self.train_batch_size < 10_000:
                l_aligned_image_feats.append(aligned_image_feats.cpu())
                l_aligned_text_feats.append(aligned_text_feats.cpu())

        log_dict = {
            f"{wandb_prefix}val_loss_avg": loss_metric_val.compute().item(),
            "counters/epoch": epoch,
            "counters/train_step": train_step,
        }

        if (
            self.config["training"].get("log_repr_similarity", False)
            or self.config["training"].get("log_structural_preservation", False)
            or epoch % self.config["training"]["embedding_visualization"] == 0
        ):
            l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
            l_aligned_text_feats = torch.cat(l_aligned_text_feats).cpu()
        if self.config["training"].get("log_structural_preservation", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            for mod, original, aligned in [
                ("image", image_features, l_aligned_image_feats),
                ("text", text_features, l_aligned_text_feats),
            ]:
                for k in self.config["training"].get(
                    "structural_preservation_k", [100]
                ):
                    tw = trustworthiness(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    cont = continuity(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    log_dict[f"{wandb_prefix}trustworthiness@{k}_{mod}_val"] = tw
                    log_dict[f"{wandb_prefix}continuity@{k}_{mod}_val"] = cont
        if self.config["training"].get("log_repr_similarity", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            alignment_score_img, _, _ = compute_score(
                x_feats=image_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_image_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            alignment_score_txt, _, _ = compute_score(
                x_feats=text_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_text_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_image_val"
            ] = alignment_score_img
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_text_val"
            ] = alignment_score_txt
        if epoch % self.config["training"]["embedding_visualization"] == 0:
            l_aligned_feats = torch.cat([l_aligned_image_feats, l_aligned_text_feats])
            l_aligned_targets = np.ones((len(l_aligned_feats),))
            l_aligned_targets[: len(l_aligned_image_feats)] = 0
            label_dict = {0: "images", 1: "texts"}

            fig_emb = embedding_plot(
                X=l_aligned_feats.numpy(),
                y=l_aligned_targets,
                label_dict=label_dict,
                return_figure=True,
            )
            log_dict[f"{wandb_prefix}val_aligned_emb"] = wandb.Image(fig_emb)
            log_dict[f"{wandb_prefix}val_modality_gap"] = (
                l_aligned_image_feats.mean(dim=0) - l_aligned_text_feats.mean(dim=0)
            ).norm(p=2)
            plt.close(fig_emb)
            plt.close("all")

        if self.wandb_logging:
            wandb.log(log_dict)
        del log_dict

        return loss_metric_val.compute().item()

    def evaluate_zero_shot_classification(
        self,
        epoch: int,
        train_step: int,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
        if self.eval_zero_shot_datasets is None:
            return

        # move the layers and set evaluation mode
        alignment_image.eval()
        alignment_text.eval()

        alignment_image = alignment_image.to(self.device)
        alignment_text = alignment_text.to(self.device)

        vision_model, image_transform = self.get_lvm(lvm_model_name=self.lvm_model_name)
        language_model, tokenizer = self.get_llm(llm_model_name=self.llm_model_name)

        # Token-level zero-shot is opt-in and only meaningful when the
        # alignment layers are token-aware (i.e. training.token_level=true).
        token_level_zero_shot = bool(
            self.config["evaluation"].get("token_level_zero_shot", False)
        ) and bool(self.config["training"].get("token_level", False))

        for eval_dataset_name, e_dataset in self.eval_zero_shot_datasets:
            set_transform_dataset(
                dataset=e_dataset,
                image_transform=image_transform,
            )

            _img_size_cfg = self.config["features"].get("img_size")
            _res_tag = (
                f"-r{int(_img_size_cfg)}" if _img_size_cfg is not None else ""
            )
            if token_level_zero_shot:
                # Token-mode image cache: (N, T, D) at the selected layer.
                save_path_vision = AlignmentTrainer.get_feature_save_path(
                    m_name=self.lvm_model_name,
                    d_name=eval_dataset_name,
                    save_path=self.save_path,
                    suffix=(
                        f"eval-none_layer-{int(image_layer_idx)}{_res_tag}-zs"
                    ),
                )
            else:
                save_path_vision = AlignmentTrainer.get_feature_save_path(
                    m_name=self.lvm_model_name,
                    d_name=eval_dataset_name,
                    save_path=self.save_path,
                    suffix=f"eval-{self.config['features']['pool_img']}{_res_tag}",
                )
            save_path_language = AlignmentTrainer.get_feature_save_path(
                m_name=self.llm_model_name,
                d_name=eval_dataset_name,
                save_path=self.save_path,
                suffix=f"eval-{self.config['features']['pool_txt']}",
            )

            dataset_classes = DATASETS_TO_CLASSES[eval_dataset_name.lower()]
            zero_shot_classifier = build_zero_shot_classifier(
                language_model=language_model,
                alignment_layer=alignment_text,
                tokenizer=tokenizer,
                dataset=e_dataset,
                layer_index=text_layer_idx,
                classnames=dataset_classes,
                templates=(
                    DATASETS_TO_TEMPLATES[eval_dataset_name.lower()]
                    if self.config["evaluation"]["use_extended_prompts"]
                    else SIMPLE_PROMPT_TEMPLATE
                ),
                num_classes_per_batch=self.config["evaluation"][
                    "num_classes_per_batch"
                ],
                device=self.device,
                pool_txt=self.config["features"]["pool_txt"],
                save_path=save_path_language,
                sample_by_sample_embedding=self.config["evaluation"][
                    "sample_by_sample_embedding"
                ],
                token_level=token_level_zero_shot,
            )
            # we move it to the cpu since in the loop we move chunks back
            # (used to optimize memory for big models)
            zero_shot_classifier = zero_shot_classifier.cpu()

            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=False,
            )

            if save_path_vision is not None and save_path_vision.exists():
                cached = True
                feature_dataset = FeatureDataset(
                    feature_file=save_path_vision,
                    feature_name="features",
                    target_name="targets",
                )
                feature_loader = DataLoader(
                    feature_dataset,
                    batch_size=self.eval_batch_size,
                    num_workers=self.config["evaluation"]["num_workers"],
                    drop_last=False,
                    shuffle=False,
                    pin_memory=False,
                )
            else:
                cached = False
                lvm_feats = []

            i = 0
            all_targets = []

            metrics_kwargs = {"task": "multiclass", "num_classes": len(dataset_classes)}
            metrics_dict = {
                "top1_acc_micro": torchmetrics.classification.Accuracy(
                    top_k=1,
                    average="micro",
                    **metrics_kwargs,
                ),
                "top1_acc_macro": torchmetrics.classification.Accuracy(
                    top_k=1,
                    average="macro",
                    **metrics_kwargs,
                ),
                "top1_f1_micro": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="micro",
                    **metrics_kwargs,
                ),
                "top1_f1_macro": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="macro",
                    **metrics_kwargs,
                ),
                "top1_f1_weighted": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="weighted",
                    **metrics_kwargs,
                ),
                "top1_f1_per_class": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="none",
                    **metrics_kwargs,
                ),
                "confusion_matrix": torchmetrics.ConfusionMatrix(
                    **metrics_kwargs,
                ),
            }
            if len(dataset_classes) >= 5:
                metrics_dict = metrics_dict | {
                    "top5_acc_micro": torchmetrics.classification.Accuracy(
                        top_k=5,
                        average="micro",
                        **metrics_kwargs,
                    ),
                    "top5_acc_macro": torchmetrics.classification.Accuracy(
                        top_k=5,
                        average="macro",
                        **metrics_kwargs,
                    ),
                }

            l_original_image_feats = []
            l_aligned_image_feats = []
            for batch in tqdm(
                feature_loader if cached else eval_loader,
                total=len(eval_loader),
                desc=eval_dataset_name,
                file=sys.stdout,
            ):
                if cached:
                    lvm_output, target = batch
                    lvm_output = lvm_output.to(self.device)
                else:
                    if len(batch) == 2:
                        images, target = batch
                    elif len(batch) == 3:
                        images, _, target = batch
                    else:
                        raise ValueError(f"Unknown length of batch: {len(batch)}")

                    images = images.to(self.device, non_blocking=True)
                    lvm_output = vision_model(images)
                    if token_level_zero_shot:
                        # Token-mode: keep the full token sequence for the
                        # selected layer instead of CLS-slicing all layers.
                        # vision_model is a feature_extractor whose dict
                        # values are layer outputs in block order.
                        layer_outputs = list(lvm_output.values())
                        lvm_output = layer_outputs[image_layer_idx]  # (B, T, D)
                    elif self.config["features"]["pool_img"] == "cls":
                        # extract the class token for all layers
                        lvm_output = [v[:, 0, :] for v in lvm_output.values()]
                        lvm_output = torch.stack(lvm_output).permute(1, 0, 2)
                    else:
                        raise NotImplementedError(
                            f"unknown pooling {self.config['features']['pool_img']}"
                        )
                    lvm_feats.append(lvm_output.cpu())

                if token_level_zero_shot:
                    # cached path returns the layer-pinned (B, T, D) tensor
                    # already; on-the-fly path produced the same shape above.
                    image_feats = lvm_output.float()
                else:
                    # legacy CLS path: cached returns (B, L, D), on-the-fly
                    # returns (B, L, D); slice the layer.
                    image_feats = lvm_output[:, image_layer_idx, :].float()
                l_original_image_feats.append(image_feats.cpu())

                aligned = alignment_image(image_feats)
                aligned = safe_normalize(aligned, p=2, dim=-1)
                l_aligned_image_feats.append(aligned.cpu())

                # compute the logits by measuring the similarity
                logits = 100.0 * chunked_logits(
                    aligned,
                    zero_shot_classifier,
                    device=self.device,
                )
                all_targets.append(target.detach().cpu().numpy())
                for m in metrics_dict.values():
                    m.update(logits.cpu(), target.cpu())
                i += self.eval_batch_size

            if (
                not cached
                and save_path_vision is not None
                and not save_path_vision.exists()
            ):
                lvm_feats = torch.cat(lvm_feats).cpu()
                save_path_vision.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"features": lvm_feats, "targets": np.concatenate(all_targets)}
                    | get_meta_dict(e_dataset),
                    save_path_vision,
                )
                logger.debug(f"Saved eval features to: {save_path_vision}")

            if self.config["evaluation"].get("log_structural_preservation", False):
                n_samples = self.config["layer_selection"]["n_samples"]
                l_original_image_feats = torch.cat(l_original_image_feats).cpu()
                l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
                for mod, original, aligned in [
                    ("image", l_original_image_feats, l_aligned_image_feats),
                ]:
                    for k in self.config["evaluation"].get(
                        "structural_preservation_k", [100]
                    ):
                        tw = trustworthiness(
                            X=original[:n_samples].float().to(self.device),
                            Z=aligned[:n_samples].float().to(self.device),
                            k=k,
                            use_approx=True,
                        )
                        cont = continuity(
                            X=original[:n_samples].float().to(self.device),
                            Z=aligned[:n_samples].float().to(self.device),
                            k=k,
                            use_approx=True,
                        )
                        result_dict[
                            f"{eval_dataset_name}/trustworthiness@{k}_{mod}_train"
                        ] = tw
                        result_dict[
                            f"{eval_dataset_name}continuity@{k}_{mod}_train"
                        ] = cont

            log_str = f"{eval_dataset_name.capitalize()} -"
            for m_name, m in metrics_dict.items():
                if "per_class" in m_name:
                    score = m.compute().detach().numpy().tolist()
                    logger.info(m_name)
                    for i, s in enumerate(score):
                        logger.info(f"  - Class {dataset_classes[i]}: {s:.4f}")
                    result_dict[f"{eval_dataset_name}/{m_name}/std"] = (
                        torch.std(m.compute()).detach().item()
                    )
                elif "confusion_matrix" in m_name:
                    if len(dataset_classes) < 30:
                        fig_, ax_ = m.plot(labels=dataset_classes)
                        result_dict[f"{eval_dataset_name}/{m_name}"] = wandb.Image(fig_)
                        plt.close(fig_)
                        plt.close("all")
                else:
                    score = m.compute().item()
                    log_str += f" {m_name}: {score:.3f},"
                    result_dict[f"{eval_dataset_name}/{m_name}"] = score
            logger.info(log_str[:-1])
            log_dict = {
                f"{alignment_layer_combination_str}/{k}": v
                for k, v in result_dict.items()
            } | {
                "counters/epoch": epoch,
                "counters/train_step": train_step,
            }
            if self.config["evaluation"]["plot_embedding_space"]:
                if type(l_aligned_image_feats) is not torch.Tensor:
                    l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
                fig_emb = embedding_plot_w_markers(
                    X=l_aligned_image_feats.numpy(),
                    y=np.concatenate(all_targets),
                    text_X=zero_shot_classifier.cpu().numpy(),
                    text_y=np.arange(len(dataset_classes)),
                    label_dict={i: x for i, x in enumerate(dataset_classes)},
                )
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/val_aligned_emb"
                ] = wandb.Image(fig_emb)
                plt.close(fig_emb)
                plt.close("all")

            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

        if self.df_scores_zero_shot is None:
            self.df_scores_zero_shot = pd.DataFrame(columns=list(result_dict.keys()))
        self.df_scores_zero_shot.loc[len(self.df_scores_zero_shot)] = pd.Series(
            result_dict
        )
        self.df_scores_zero_shot.to_csv(
            f"{self.save_path / wandb.run.name / self.add_exp_suffix_to_name('zero_shot_results')}.csv",
            index=False,
        )

    def evaluate_retrieval(
        self,
        epoch: int,
        train_step: int,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
        if self.eval_retrieval_datasets is None:
            return

        # move the layers and set evaluation mode
        alignment_image.eval()
        alignment_text.eval()

        alignment_image = alignment_image.to(self.device)
        alignment_text = alignment_text.to(self.device)

        token_level_retrieval = bool(
            self.config["training"].get("token_level", False)
        )

        for eval_dataset_name, e_dataset in self.eval_retrieval_datasets:
            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=False,
            )
            eval_text_mask = None
            if token_level_retrieval:
                (
                    image_features_val,
                    text_features_val,
                    eval_text_mask,
                ) = self._load_eval_token_features(
                    eval_loader=eval_loader,
                    img_layer_idx=image_layer_idx,
                    txt_layer_idx=text_layer_idx,
                )
            else:
                _img_size_cfg = self.config["features"].get("img_size")
                _res_tag = (
                    f"-r{int(_img_size_cfg)}" if _img_size_cfg is not None else ""
                )
                image_features_val = self.get_image_features(
                    loader=eval_loader,
                    lvm_model_name=self.lvm_model_name,
                    suffix=f"eval-{self.config['features']['pool_img']}{_res_tag}",
                )
                text_features_val = self.get_text_features(
                    loader=eval_loader,
                    llm_model_name=self.llm_model_name,
                    suffix=f"eval-{self.config['features']['pool_txt']}",
                )
            num_samples = image_features_val.shape[0]

            # drop duplicates for fair comparison
            if (
                self.config["evaluation"]["drop_duplicates"]
                and hasattr(eval_loader.dataset, "df")
                and "image_path" in eval_loader.dataset.df.columns
            ):
                unique_val_indices = eval_loader.dataset.df.drop_duplicates(
                    subset="image_path"
                ).index
                image_features_val = image_features_val[unique_val_indices]
                text_features_val = text_features_val[unique_val_indices]

            aligned_image_feats = []
            aligned_text_feats = []
            for i in tqdm(
                range(0, num_samples, self.eval_batch_size),
                total=num_samples,
                desc=eval_dataset_name,
                file=sys.stdout,
            ):
                if token_level_retrieval:
                    # pool=none features are single-layer (N, T, D)
                    image_feats = image_features_val[i : i + self.eval_batch_size]
                    text_feats = text_features_val[i : i + self.eval_batch_size]
                else:
                    image_feats = image_features_val[
                        i : i + self.eval_batch_size, image_layer_idx
                    ]
                    text_feats = text_features_val[
                        i : i + self.eval_batch_size, text_layer_idx
                    ]
                image_feats = image_feats.float().to(self.device)
                text_feats = text_feats.float().to(self.device)

                if token_level_retrieval:
                    image_feats = alignment_image(image_feats)
                    if eval_text_mask is not None:
                        batch_mask = eval_text_mask[
                            i : i + self.eval_batch_size
                        ].to(self.device)
                        text_feats = alignment_text(text_feats, mask=batch_mask)
                    else:
                        text_feats = alignment_text(text_feats)
                else:
                    image_feats = alignment_image(image_feats)
                    text_feats = alignment_text(text_feats)

                aligned_image_feats.append(image_feats)
                aligned_text_feats.append(text_feats)

            aligned_image_feats = torch.cat(aligned_image_feats).cpu()
            aligned_text_feats = torch.cat(aligned_text_feats).cpu()

            df = e_dataset.df if hasattr(e_dataset, "df") else None
            recalls_i2t = retrieval_metrics_df(
                image_embeds=aligned_image_feats,
                text_embeds=aligned_text_feats,
                df=df,
                image_column="image_path",
                k_values=[1, 5, 10],
                batch_size=self.eval_batch_size,
            )
            recalls_t2i = retrieval_metrics_df(
                image_embeds=aligned_text_feats,
                text_embeds=aligned_image_feats,
                df=df,
                image_column="image_path",
                k_values=[1, 5, 10],
                batch_size=self.eval_batch_size,
            )
            recalls_i2t = {f"I2T-{k}": v for k, v in recalls_i2t.items()}
            recalls_t2i = {f"T2I-{k}": v for k, v in recalls_t2i.items()}
            recalls = recalls_i2t | recalls_t2i

            log_str = f"{eval_dataset_name.capitalize()} -"
            for m_name, score in recalls.items():
                log_str += f" {m_name}: {score:.3f},"
                result_dict[f"{eval_dataset_name}/{m_name}"] = score
            logger.info(log_str[:-1])
            log_dict = {
                f"{alignment_layer_combination_str}/{k}": v
                for k, v in result_dict.items()
            } | {
                "counters/epoch": epoch,
                "counters/train_step": train_step,
            }

            if self.config["evaluation"]["plot_embedding_space"]:
                l_aligned_feats = torch.cat(
                    [aligned_image_feats, aligned_text_feats]
                ).cpu()
                l_aligned_targets = np.ones((len(l_aligned_feats),))
                l_aligned_targets[: len(aligned_image_feats)] = 0
                label_dict = {0: "images", 1: "texts"}

                fig_emb = embedding_plot(
                    X=l_aligned_feats.numpy(),
                    y=l_aligned_targets,
                    label_dict=label_dict,
                    return_figure=True,
                )
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/val_aligned_emb"
                ] = wandb.Image(fig_emb)
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/modality_gap"
                ] = (
                    aligned_image_feats.mean(dim=0) - aligned_text_feats.mean(dim=0)
                ).norm(
                    p=2
                )
                plt.close(fig_emb)
                plt.close("all")

            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

        if self.df_scores_retrieval is None:
            self.df_scores_retrieval = pd.DataFrame(columns=list(result_dict.keys()))
        self.df_scores_retrieval.loc[len(self.df_scores_retrieval)] = pd.Series(
            result_dict
        )
        self.df_scores_retrieval.to_csv(
            f"{self.save_path / wandb.run.name / self.add_exp_suffix_to_name('retrieval_results')}.csv",
            index=False,
        )
