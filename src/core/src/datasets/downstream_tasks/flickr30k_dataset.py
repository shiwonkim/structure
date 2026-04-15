import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from turbojpeg import TurboJPEG

    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False

from .coco_dataset import LoadingType


class Flickr30kDataset(Dataset):

    def __init__(
        self,
        root_dir,
        meta_path,
        split: Optional[Union[str, List[str]]] = "train",
        transform=None,
        tokenizer=None,
        loading_type: LoadingType = LoadingType.STANDARD,
    ):
        self.root_dir = root_dir
        self.meta_path = meta_path
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.loading_type = loading_type

        self.df = pd.read_csv(self.meta_path, delimiter="|")
        self.df.columns = [x.strip() for x in self.df.columns]

        split_annotations = []
        for split in ["train", "val", "test"]:
            with open(os.path.join(self.root_dir, f"{split}.txt")) as file:
                split_indices = [line.rstrip() for line in file]
            split_indices = [(x, split) for x in split_indices]
            split_annotations += split_indices
        df_split = pd.DataFrame(split_annotations, columns=["image_name", "split"])
        df_split["image_name"] = df_split["image_name"] + ".jpg"
        df_split["image_path"] = df_split["image_name"].apply(
            lambda x: os.path.join(self.root_dir, "images", x)
        )
        self.df = self.df.merge(df_split, on="image_name", how="left")
        self.df.dropna(subset="comment", inplace=True)
        # select the correct dataset
        if self.split is not None:
            if type(self.split) is str:
                self.df = self.df[self.df["split"] == self.split]
            else:
                self.df = self.df[self.df["split"].isin(self.split)]
            self.df.reset_index(drop=True, inplace=True)
        self.apply_tokenizer()

        # create TurboJPEG object for image reading (if available)
        self.jpeg_reader = TurboJPEG() if TURBOJPEG_AVAILABLE else None

    def apply_tokenizer(self):
        if self.tokenizer:
            self.tokens = self.tokenizer(
                list(self.df["comment"].values),
                padding="longest",
                return_tensors="pt",
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        caption = self.df.iloc[idx].comment.strip()
        image_path = self.df.iloc[idx].image_path

        if (
            self.loading_type == LoadingType.STANDARD
            or self.loading_type == LoadingType.IMG_ONLY
        ):
            image = self.load_image(image_path)
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.Tensor(0)

        if (
            self.loading_type == LoadingType.STANDARD
            or self.loading_type == LoadingType.TXT_ONLY
        ):
            if self.tokenizer:
                caption = {k: v[idx] for (k, v) in self.tokens.items()}
        else:
            caption = torch.Tensor(0)

        return image, caption

    def load_image(self, f):
        # Returns a PIL.Image so the downstream transform pipeline's
        # ToTensor() runs on a PIL image (not a tensor). Mirrors the
        # CocoCaptionDataset fix.
        if self.jpeg_reader is not None:
            with open(f, "rb") as file:
                try:
                    image = self.jpeg_reader.decode(file.read())
                    if len(image.shape) == 2:
                        image = image[..., np.newaxis].repeat(3, axis=-1)
                    return Image.fromarray(image)
                except OSError:
                    print(
                        f"Failed to read file with TurboJPEG falling back on PIL: {f}"
                    )
        return Image.open(f).convert("RGB")
