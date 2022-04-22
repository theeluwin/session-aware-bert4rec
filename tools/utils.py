import random

import torch as t
import torch.backends.cudnn as cudnn
import numpy as np


__all__ = (
    'fix_random_seed',
)


# source: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch
def fix_random_seed(seed: int):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
