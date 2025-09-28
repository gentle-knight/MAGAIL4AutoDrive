import numpy as np
import torch
import random

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print('Random seed: {}'.format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)