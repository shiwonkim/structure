"""Wrapper to fix CocoCaptionDataset double-ToTensor issue.
Monkey-patches torchvision.transforms.functional.to_tensor to pass through tensors,
then runs the actual training script.
"""
import sys
import torch
import torchvision.transforms.functional as F

# Save original
_original_to_tensor = F.to_tensor

def _patched_to_tensor(pic):
    if isinstance(pic, torch.Tensor):
        return pic
    return _original_to_tensor(pic)

# Monkey-patch
F.to_tensor = _patched_to_tensor

# Now run the actual training script
import runpy
sys.argv = sys.argv[1:]  # strip this script's name
runpy.run_path(sys.argv[0], run_name="__main__")
