import random
import string
from typing import List, Tuple, Union

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .base_dataset import BaseDataset


class ImageTextDataset(Dataset):

    def __init__(
        self,
        dataset: BaseDataset,
        label_templates: List[str],
        template_key: str = "label",
        tokenizer=None,
        precompute_captions: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.label_templates = label_templates
        self.template_key = template_key
        self.tokenizer = tokenizer
        self.name = type(self.dataset).__name__

        check_templates = all(
            [
                self.template_key in ImageTextDataset.check_string_format_arguments(x)
                for x in self.label_templates
            ]
        )
        if not check_templates:
            raise ValueError(
                f"Label templates do not all have the template key: {template_key}"
            )

        # create a data structure for the labels
        if precompute_captions:
            l_captions = [
                self[i][-1]
                for i in tqdm(range(len(self)), desc="Precomputing captions")
            ]
        else:
            l_captions = []
            logger.warning(
                "Not precomputing captions, only use when you KNOW you have the embeddings."
            )
        self.df = pd.DataFrame(l_captions, columns=["captions"])
        self.apply_tokenizer()

    def apply_tokenizer(self) -> None:
        if self.tokenizer:
            self.tokens = self.tokenizer(
                list(self.df["captions"].values),
                padding="longest",
                return_tensors="pt",
            )

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, transform):
        self.dataset.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Union[str, dict]]:
        image, label = self.dataset.__getitem__(index=index)
        if self.tokenizer:
            caption = {k: v[index] for (k, v) in self.tokens.items()}
            return image, caption
        else:
            label = self.dataset.classes[label]
            if "," in label:
                label = random.choice(label.split(",")).strip()
            template = random.choice(self.label_templates)
            label_text = template.format(**{self.template_key: label}).lower().strip()
            return image, label_text

    @staticmethod
    def check_string_format_arguments(string_to_check):
        return [
            tup[1]
            for tup in string.Formatter().parse(string_to_check)
            if tup[1] is not None
        ]

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        if isinstance(images[0], Image.Image):
            images = list(images)
        else:
            images = torch.stack(images, dim=0)
        if isinstance(captions[0], dict):
            # Batch tokenized captions: each caption is a dict (e.g., input_ids, attention_mask).
            # We assume that all captions have the same keys.
            batch_captions = {
                key: torch.stack([caption[key] for caption in captions], dim=0)
                for key in captions[0]
            }
        elif isinstance(captions[0], str):
            batch_captions = list(captions)
        else:
            batch_captions = torch.stack(captions, dim=0)
        return images, batch_captions
