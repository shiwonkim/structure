import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


def fix_random_seeds(seed=42):
    """Fix random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def flatten(t):
    return t.reshape(t.shape[0], -1)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def has_batchnorms(model: torch.nn.Module):
    bn_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def compare_models(model_1, model_2, log=False):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if log and (key_item_1[0] == key_item_2[0]):
                logger.error("Mismatch found at", key_item_1[0])
    return models_differ


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def init_distributed_mode():
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = rank % torch.cuda.device_count()
    # launched naively with `python main_XXX.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        logger.debug("Will run the code on one GPU.")
        rank, gpu, world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["LOCAL_RANK"] = "0"
    else:
        # if there is no GPU available we don't do anything
        return

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    logger.debug(f"STARTUP: Distributed init (rank {rank}): env://", flush=True)
    dist.barrier()
    setup_for_distributed(rank == 0)


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        logger.info("Pre-trained weights not found. Training from scratch.")
        return
    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                if len(msg.missing_keys) > 0:
                    k = next(iter(checkpoint[key]))
                    if "module." in k:
                        logger.debug(
                            f"=> Found `module` in {key}, trying to transform."
                        )
                        transf_state_dict = OrderedDict()
                        for k, v in checkpoint[key].items():
                            # remove the module from the key
                            # this is caused by the distributed training
                            k = k.replace("module.", "")
                            transf_state_dict[k] = v
                        msg = value.load_state_dict(transf_state_dict, strict=False)
                logger.debug(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logger.debug(
                        "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                    )
                except ValueError:
                    logger.error(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            logger.error(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    # reload variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def save_checkpoint(run_dir, save_dict, epoch, save_best=False):
    if is_main_process():
        run_dir = Path(run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        filename = str(run_dir / "checkpoints" / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(save_dict, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(run_dir / "checkpoints" / "model_best.pth")
            torch.save(save_dict, best_path)
            logger.info("Saving current best: model_best.pth ...")


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, log_messages: bool = True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.log_messages = log_messages
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.log_messages:
                logger.debug(
                    f"Early stopping counter {self.counter} of {self.patience}"
                )
            if self.counter >= self.patience:
                if self.log_messages:
                    logger.info("EarlyStopping, evaluation did not decrease.")
                self.early_stop = True


def p_value_stars(p: float, latex: bool = True) -> str:
    stars = "{}"
    if latex:
        stars = "{{{0}}}"
    if p < 0.001:
        return stars.format("***")
    elif p < 0.01:
        return stars.format("**")
    elif p < 0.05:
        return stars.format("*")
    else:
        return ""


def latex_median_quantile(arr: np.ndarray) -> str:
    median = np.median(arr)
    q_05 = np.quantile(arr, q=0.05)
    q_95 = np.quantile(arr, q=0.95)

    diff_05 = "{" + "{0:+.1f}".format(q_05 - median) + "}"
    diff_95 = "{" + "{0:+.1f}".format(q_95 - median) + "}"

    return f"{median:.1f}^{diff_95}_{diff_05}"
