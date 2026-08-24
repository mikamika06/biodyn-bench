import numpy as np
from methods.marginal import _standardize


def partial_corr_matrix(expr, ridge=1e-2):
    x = _standardize(expr)
    cov = np.cov(x, rowvar=False)
    prec = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
    d = np.sqrt(np.diag(prec))
    return -prec / np.outer(d, d)
