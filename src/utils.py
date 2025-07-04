import os
import random
import torch
import numpy as np
from typing import Optional
from loguru import logger

__all__ = ["set_random_seed"]

def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(
            f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}"
        )
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(
            "The CUDNN is set to deterministic. This will increase reproducibility, "
            "but may slow down your training considerably."
        )
