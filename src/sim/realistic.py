import numpy as np

PARAMS = {
    "mean_shape": 0.6, "mean_rate": 0.3,
    "out_prob": 0.05, "out_loc": 4.0, "out_scale": 0.5,
    "lib_loc": 6.14, "lib_scale": 0.2,
    "bcv_common": 0.1, "bcv_df": 60,
    "sigma_z": 0.6,
    "dropout_mid": None, "dropout_shape": -1.0,
    "closure": True,
}


def to_counts_realistic(z, rng, **kw):
    p = {**PARAMS, **kw}
    n_cells, n_genes = z.shape

    base = rng.gamma(p["mean_shape"], 1.0 / p["mean_rate"], size=n_genes)
    is_out = rng.random(n_genes) < p["out_prob"]
    fac = rng.lognormal(p["out_loc"], p["out_scale"], size=n_genes)
    base = np.where(is_out, np.median(base) * fac, base)

    zs = (z - z.mean(axis=0)) / z.std(axis=0)
    mu = base[None, :] * np.exp(p["sigma_z"] * zs)

    lib = rng.lognormal(p["lib_loc"], p["lib_scale"], size=n_cells)
    if p["closure"]:
        # композиційне замикання: сума лічильників клітини фіксована глибиною
        # секвенування. Робить матрицю коваріацій майже виродженою.
        mu = mu / mu.sum(axis=1, keepdims=True) * lib[:, None]
    else:
        mu = mu * (lib[:, None] / np.exp(p["lib_loc"]))

    bcv = (p["bcv_common"] + 1.0 / np.sqrt(np.maximum(base, 1e-8))) * \
          np.sqrt(p["bcv_df"] / rng.chisquare(p["bcv_df"], size=n_genes))
    shape = 1.0 / bcv ** 2
    lam = rng.gamma(shape[None, :], mu / shape[None, :])
    counts = rng.poisson(lam).astype(np.float64)

    if p["dropout_mid"] is not None:
        logit = p["dropout_shape"] * (np.log(np.maximum(mu, 1e-8)) - p["dropout_mid"])
        keep = rng.random(counts.shape) >= 1.0 / (1.0 + np.exp(-logit))
        counts = counts * keep

    return counts


def sparsity(counts):
    return float((counts == 0).mean())
