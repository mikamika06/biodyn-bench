import numpy as np


def _standardize(expr):
    x = expr - expr.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return x / sd


def corr_matrix(expr):
    x = _standardize(expr)
    return (x.T @ x) / x.shape[0]


def _bin_column(v, bins):
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, v, side="right")


def _bins(expr, bins):
    return np.column_stack([_bin_column(expr[:, j], bins) for j in range(expr.shape[1])])


def mi_matrix(expr, bins=10):
    n, p = expr.shape
    b = _bins(expr, bins)
    counts = np.zeros((p, bins))
    for k in range(bins):
        counts[:, k] = (b == k).sum(axis=0)
    px = counts / n
    hx = -np.sum(np.where(px > 0, px * np.log(px + 1e-300), 0.0), axis=1)
    onehot = np.zeros((p, bins, n))
    for k in range(bins):
        onehot[:, k, :] = (b == k).T
    joint = (onehot.reshape(p * bins, n) @ onehot.reshape(p * bins, n).T) / n
    mi = np.zeros((p, p))
    for i in range(p):
        ji = joint[i * bins:(i + 1) * bins].reshape(bins, p, bins)
        for j in range(i + 1, p):
            pj = ji[:, j, :]
            nz = pj > 0
            hxy = -np.sum(pj[nz] * np.log(pj[nz]))
            mi[i, j] = mi[j, i] = max(hx[i] + hx[j] - hxy, 0.0)
    return mi


def pair_mi(expr, pairs, bins=10):
    b = _bins(expr, bins)
    n = expr.shape[0]
    out = []
    for i, j in pairs:
        bi, bj = b[:, i], b[:, j]
        ki, kj = bi.max() + 1, bj.max() + 1
        joint = np.zeros((ki, kj))
        np.add.at(joint, (bi, bj), 1.0)
        joint /= n
        px, py = joint.sum(1), joint.sum(0)
        nz = joint > 0
        hxy = -np.sum(joint[nz] * np.log(joint[nz]))
        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        out.append(max(hx + hy - hxy, 0.0))
    return np.array(out)
