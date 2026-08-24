import numpy as np
import torch


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, expr, mask_frac=0.15, seed=0):
        self.x = torch.tensor(expr, dtype=torch.float32)
        self.mask_frac = mask_frac
        self.g = expr.shape[1]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        m = torch.zeros(self.g, dtype=torch.bool)
        k = max(1, int(self.mask_frac * self.g))
        m[torch.from_numpy(self.rng.choice(self.g, k, replace=False))] = True
        return self.x[i], m


def standardize(expr):
    mu, sd = expr.mean(0), expr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (expr - mu) / sd, mu, sd


def split(expr, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(expr.shape[0])
    n = int(frac * len(idx))
    return expr[idx[:n]], expr[idx[n:]]
