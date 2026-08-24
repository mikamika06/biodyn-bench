import numpy as np
from sim.counts import make_counts, normalize, pair_corr, ckey

LINKS = {
    "linear": lambda a: a,
    "tanh": lambda a: np.tanh(a),
    "relu": lambda a: np.maximum(a, 0.0),
}

CFG = {"n_cells": 5000, "n_coll": 25, "n_direct": 25, "n_null": 30}
K_DIRECT, NOISE = 1.5896, 1.0
_CACHE = {}


def build(seed, cfg=None, k_direct=K_DIRECT, noise=NOISE, counts_kw=None, link="linear"):
    f = LINKS[link]
    cfg = cfg or CFG
    n = cfg["n_cells"]
    rng = np.random.default_rng(seed)
    cols, coll_pairs, direct_pairs, null_ids = [], [], [], []

    for _ in range(cfg["n_coll"]):
        a = rng.standard_normal(n)
        c = rng.standard_normal(n)
        b = f(a) + f(c) + noise * rng.standard_normal(n)
        ia, ic = len(cols), len(cols) + 2
        cols += [a, b, c]
        coll_pairs.append((ia, ic))

    for _ in range(cfg["n_direct"]):
        x = rng.standard_normal(n)
        y = k_direct * f(x) + noise * rng.standard_normal(n)
        direct_pairs.append((len(cols), len(cols) + 1))
        cols += [x, y]

    for _ in range(cfg["n_null"]):
        null_ids.append(len(cols))
        cols.append(rng.standard_normal(n))

    null_pairs = [(null_ids[i], null_ids[i + 1]) for i in range(0, len(null_ids) - 1, 2)]
    expr = normalize(make_counts(np.column_stack(cols), rng, counts_kw))
    return expr, {"D": direct_pairs, "K": coll_pairs, "N": null_pairs}


def mean_stat(seed, cfg, k, typ, counts_kw=None, link="linear", match="corr"):
    expr, idx = build(seed, cfg, k, NOISE, counts_kw, link)
    if match == "mi":
        from methods.marginal import pair_mi
        return float(np.mean(pair_mi(expr, idx[typ])))
    return float(np.mean([abs(pair_corr(expr, i, j)) for i, j in idx[typ]]))


def tune(cfg=None, seeds=(101, 102, 103), lo=0.3, hi=8.0, iters=30,
         counts_kw=None, link="linear", match="corr"):
    """Прямі пари зрівнюються з КОЛАЙДЕРНИМИ, а не з чужої панелі.
    Це робить позитивний контроль D-проти-N осмисленим при будь-якому звʼязку."""
    cfg = cfg or CFG
    key = (tuple(sorted(cfg.items())), ckey(counts_kw), link, match)
    if key in _CACHE:
        return _CACHE[key]
    target = float(np.mean([mean_stat(s, cfg, 1.0, "K", counts_kw, link, match)
                            for s in seeds]))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = np.mean([mean_stat(s, cfg, mid, "D", counts_kw, link, match) for s in seeds])
        lo, hi = (mid, hi) if got < target else (lo, mid)
    res = (0.5 * (lo + hi), target)
    _CACHE[key] = res
    return res


def matched(seed, cfg=None, counts_kw=None, link="linear", match="corr"):
    cfg = cfg or CFG
    k, target = tune(cfg, counts_kw=counts_kw, link=link, match=match)
    expr, idx = build(seed, cfg, k, NOISE, counts_kw, link)
    return expr, idx, k, target
