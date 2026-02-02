import os
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from loguru import logger
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for datasets."""

    IMG_COL = "image"
    LBL_COL = "label"

    def __init__(self, transform=None, val_transform=None, **kwargs):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        """
        super().__init__()
        self.training = True
        self.transform = transform
        self.val_transform = val_transform
        self.meta_data = pd.DataFrame()
        self.labelencoder = sklearn.preprocessing.LabelEncoder()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def save_label_encoder(self, path: str):
        if self.labelencoder is not None:
            le_file_name = os.path.join(path, "label_encoder.pickle")
            le_file = open(le_file_name, "wb")
            pickle.dump(self.labelencoder, le_file)
            le_file.close()

    def get_class_weights(self):
        class_weight = {}
        total = len(self.meta_data)
        ser_vc = self.meta_data[self.LBL_COL].value_counts()
        for _, row in ser_vc.reset_index().iterrows():
            weight = 1 - (row[self.LBL_COL] / total)
            class_weight[row.name] = weight
        class_weight = dict(sorted(class_weight.items()))
        class_weight = list(class_weight.values())
        return torch.Tensor(class_weight)

    def check_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path needs to exist: {path}")
        return path

    def remove_data_quality_issues(
        self, data_quality_issues_list, drop_on_col: Optional[str] = None
    ):
        if data_quality_issues_list is not None:
            data_quality_issues_list = self.check_path(data_quality_issues_list)
            with open(data_quality_issues_list, "rb") as f:
                data_quality_issues = pickle.load(f)
                ind_irrelevants = []
                ind_near_dups = []
                ind_lbl_errors = []
                if "IrrelevantSamples" in data_quality_issues.keys():
                    ind_irrelevants = data_quality_issues["IrrelevantSamples"]
                if "NearDuplicates" in data_quality_issues.keys():
                    ind_near_dups = data_quality_issues["NearDuplicates"]
                if "LabelErrors" in data_quality_issues.keys():
                    ind_lbl_errors = data_quality_issues["LabelErrors"]
                # get only the indices from the near duplicates
                if len(ind_near_dups) > 0:
                    ind_near_dups = [x[0] for x in ind_near_dups]
                drop_indices = np.unique(
                    np.concatenate([ind_irrelevants, ind_near_dups])
                )
                if drop_on_col is None:
                    self.meta_data.drop(
                        list(drop_indices),
                        axis=0,
                        inplace=True,
                    )
                else:
                    self.meta_data = self.meta_data[
                        ~self.meta_data[drop_on_col].isin(drop_indices)
                    ]
                self.meta_data.reset_index(drop=True, inplace=True)
                logger.info(
                    f"Data Quality Issues ({type(self).__name__}):: "
                    f"Irrelevant Samples: {len(ind_irrelevants)}, "
                    f"Near Duplicates: {len(ind_near_dups)}, "
                    f"Label Errors: {len(ind_lbl_errors)}"
                )

    @staticmethod
    def find_files_with_extension(directory_path: Union[str, Path], extension: str):
        extension = extension.replace("*", "")
        l_files = []
        for entry in os.scandir(directory_path):
            if entry.is_file() and entry.name.endswith(extension):
                l_files.append(entry.path)
            elif entry.is_dir():
                l_files.extend(
                    BaseDataset.find_files_with_extension(
                        directory_path=entry.path,
                        extension=extension,
                    )
                )
        return l_files

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)
