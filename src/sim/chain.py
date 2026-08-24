import numpy as np
from sim.counts import make_counts, normalize, pair_corr, ckey

LINKS = {
    "linear": lambda a: a,
    "tanh": lambda a: np.tanh(a),
    "relu": lambda a: np.maximum(a, 0.0),
}

CFG = {"n_cells": 5000, "n_chain": 25, "n_direct": 25, "n_null": 30}
E1, E2, NOISE = 2.0, 2.0, 1.0
_CACHE = {}


def latent(cfg, k_direct, rng, link="linear"):
    f = LINKS[link]
    n = cfg["n_cells"]
    cols, chain, direct, null, mediators = [], [], [], [], []

    def add(v):
        cols.append(v)
        return len(cols) - 1

    for _ in range(cfg["n_chain"]):
        a = rng.standard_normal(n)
        b = E1 * f(a) + NOISE * rng.standard_normal(n)
        c = E2 * f(b) + NOISE * rng.standard_normal(n)
        ia, ib, ic = add(a), add(b), add(c)
        chain.append((ia, ic))
        mediators.append(ib)

    for _ in range(cfg["n_direct"]):
        x = rng.standard_normal(n)
        y = k_direct * f(x) + NOISE * rng.standard_normal(n)
        direct.append((add(x), add(y)))

    for _ in range(cfg["n_null"]):
        null.append((add(rng.standard_normal(n)), add(rng.standard_normal(n))))

    return np.column_stack(cols), {"D": direct, "T": chain, "N": null}, mediators


def build(seed, cfg, k_direct, hide_mediator=False, counts_kw=None, link="linear"):
    rng = np.random.default_rng(seed)
    z, idx, mediators = latent(cfg, k_direct, rng, link)
    counts = make_counts(z, rng, counts_kw)
    if hide_mediator:
        drop = set(mediators)
        keep = [i for i in range(counts.shape[1]) if i not in drop]
        remap = {o: n for n, o in enumerate(keep)}
        idx = {t: [(remap[i], remap[j]) for i, j in ps] for t, ps in idx.items()}
        counts = counts[:, keep]
    return normalize(counts), idx


def mean_stat(seed, cfg, k, typ, hide, counts_kw=None, link="linear",
              match="corr"):
    expr, idx = build(seed, cfg, k, hide, counts_kw, link)
    if match == "mi":
        from methods.marginal import pair_mi
        return float(np.mean(pair_mi(expr, idx[typ])))
    return float(np.mean([abs(pair_corr(expr, i, j)) for i, j in idx[typ]]))


def mean_corr(seed, cfg, k, typ, hide, counts_kw=None, link="linear"):
    return mean_stat(seed, cfg, k, typ, hide, counts_kw, link, "corr")


def tune(cfg, hide, seeds=(101, 102, 103), lo=0.3, hi=8.0, iters=30, counts_kw=None, link="linear", match="corr"):
    key = (tuple(sorted(cfg.items())), hide, ckey(counts_kw), link, match)
    if key in _CACHE:
        return _CACHE[key]
    target = float(np.mean([mean_stat(s, cfg, 2.0, "T", hide, counts_kw, link, match) for s in seeds]))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = np.mean([mean_stat(s, cfg, mid, "D", hide, counts_kw, link, match) for s in seeds])
        lo, hi = (mid, hi) if got < target else (lo, mid)
    res = (0.5 * (lo + hi), target)
    _CACHE[key] = res
    return res


def matched(seed, hide, cfg=None, counts_kw=None, link="linear", match="corr"):
    cfg = cfg or CFG
    k, target = tune(cfg, hide, counts_kw=counts_kw, link=link, match=match)
    expr, idx = build(seed, cfg, k, hide, counts_kw, link)
    return expr, idx, k, target
