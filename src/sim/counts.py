import numpy as np


def to_counts(z, rng, mu_log=np.log(30.0), sigma=0.6):
    zs = (z - z.mean(axis=0)) / z.std(axis=0)
    lam = np.exp(mu_log + sigma * zs)
    return rng.poisson(lam).astype(np.float64)


def normalize(counts, target=1e4):
    depth = counts.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    return np.log1p(counts / depth * target)


def pair_corr(mat, i, j):
    a, b = mat[:, i], mat[:, j]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def make_counts(z, rng, counts_kw=None):
    if counts_kw is None:
        return to_counts(z, rng)
    from sim.realistic import to_counts_realistic
    return to_counts_realistic(z, rng, **counts_kw)


def ckey(counts_kw):
    return None if counts_kw is None else tuple(sorted(counts_kw.items()))
