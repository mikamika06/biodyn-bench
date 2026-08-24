import numpy as np


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    n = 0.0
    for p in pos:
        n += (p > neg).sum() + 0.5 * (p == neg).sum()
    return float(n / (len(pos) * len(neg)))
