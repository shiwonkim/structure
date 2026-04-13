"""Wrapper to fix CCA numerical stability + COCO double-ToTensor.
Monkey-patches cca_zoo's PCA to drop zero-variance components,
and fixes ToTensor double-application for COCO.
"""
import sys
import torch
import torchvision.transforms.functional as F

# Fix 1: ToTensor double-application
_original_to_tensor = F.to_tensor
def _patched_to_tensor(pic):
    if isinstance(pic, torch.Tensor):
        return pic
    return _original_to_tensor(pic)
F.to_tensor = _patched_to_tensor

# Fix 2: CCA PCA stability — truncate zero-variance components
import cca_zoo.linear._mcca as _mcca
from sklearn.decomposition import PCA

def _patched_apply_pca(self, views):
    self.pca_models = [PCA(n_components=0.999) for _ in views]
    return [self.pca_models[i].fit_transform(view) for i, view in enumerate(views)]

_mcca.MCCA._apply_pca = _patched_apply_pca

# Run the actual script
import runpy
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
