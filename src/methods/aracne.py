import numpy as np
from methods.marginal import mi_matrix


def aracne(expr, bins=10, eps=0.0, mi=None):
    mi = mi_matrix(expr, bins) if mi is None else mi
    w = mi.copy()
    for k in range(w.shape[0]):
        mk = mi[k]
        weakest = np.minimum(mk[:, None], mk[None, :])
        w[mi < weakest - eps] = 0.0
    np.fill_diagonal(w, 0.0)
    return w
