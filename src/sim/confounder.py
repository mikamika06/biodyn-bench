import numpy as np
from sim.counts import make_counts, normalize, pair_corr, ckey

LINKS = {
    "linear": lambda a: a,
    "tanh": lambda a: np.tanh(a),
    "relu": lambda a: np.maximum(a, 0.0),
}

DEFAULT = {"n_cells": 5000, "n_conf": 25, "n_direct": 25, "n_null": 30}
B_COEF, C_COEF, NOISE = 2.0, 3.0, 1.0
_CACHE = {}


def latent_panel(n_cells, n_conf, n_direct, n_null, k_direct, rng,
                 b_coef=B_COEF, c_coef=C_COEF, noise_sd=NOISE, link="linear"):
    f = LINKS[link]
    cols, meta = [], []

    def add(vec, role, group, gene):
        cols.append(vec)
        meta.append({"idx": len(cols) - 1, "role": role, "group": group, "gene": gene})

    for i in range(n_conf):
        a = rng.standard_normal(n_cells)
        b = b_coef * f(a) + noise_sd * rng.standard_normal(n_cells)
        c = c_coef * f(a) + noise_sd * rng.standard_normal(n_cells)
        add(a, "conf_root", f"C{i}", f"A{i}")
        add(b, "conf_leaf", f"C{i}", f"B{i}")
        add(c, "conf_leaf", f"C{i}", f"C{i}")

    for j in range(n_direct):
        x = rng.standard_normal(n_cells)
        y = k_direct * f(x) + noise_sd * rng.standard_normal(n_cells)
        add(x, "direct_src", f"D{j}", f"X{j}")
        add(y, "direct_tgt", f"D{j}", f"Y{j}")

    for m in range(n_null):
        add(rng.standard_normal(n_cells), "null", f"N{m}", f"N{m}")

    return np.column_stack(cols), meta


def pair_index(meta):
    conf, direct, null = [], [], []
    by_group = {}
    for m in meta:
        by_group.setdefault(m["group"], []).append(m)
    for g, members in by_group.items():
        if g.startswith("C"):
            leaves = [m["idx"] for m in members if m["role"] == "conf_leaf"]
            conf.append((leaves[0], leaves[1]))
        elif g.startswith("D"):
            s = [m["idx"] for m in members if m["role"] == "direct_src"][0]
            t = [m["idx"] for m in members if m["role"] == "direct_tgt"][0]
            direct.append((s, t))
    nulls = [m["idx"] for m in meta if m["role"] == "null"]
    for a in range(0, len(nulls) - 1, 2):
        null.append((nulls[a], nulls[a + 1]))
    return {"D": direct, "C": conf, "N": null}


def build(seed, cfg, k, hide_root=False, counts_kw=None, link="linear"):
    rng = np.random.default_rng(seed)
    z, meta = latent_panel(cfg["n_cells"], cfg["n_conf"], cfg["n_direct"],
                           cfg["n_null"], k, rng, link=link)
    counts = make_counts(z, rng, counts_kw)
    idx = pair_index(meta)
    if hide_root:
        roots = {m["idx"] for m in meta if m["role"] == "conf_root"}
        keep = [i for i in range(counts.shape[1]) if i not in roots]
        remap = {o: n for n, o in enumerate(keep)}
        idx = {t: [(remap[i], remap[j]) for i, j in ps] for t, ps in idx.items()}
        counts = counts[:, keep]
    return normalize(counts), idx


def mean_stat(seed, cfg, k, typ, hide_root, counts_kw=None, link="linear",
              match="corr"):
    expr, idx = build(seed, cfg, k, hide_root, counts_kw, link)
    if match == "mi":
        from methods.marginal import pair_mi
        return float(np.mean(pair_mi(expr, idx[typ])))
    return float(np.mean([abs(pair_corr(expr, i, j)) for i, j in idx[typ]]))


def mean_corr(seed, cfg, k, typ, hide_root, counts_kw=None, link="linear"):
    return mean_stat(seed, cfg, k, typ, hide_root, counts_kw, link, "corr")


def tune(cfg, hide_root=False, seeds=(101, 102, 103), lo=0.3, hi=8.0, iters=30, counts_kw=None, link="linear", match="corr"):
    key = (tuple(sorted(cfg.items())), hide_root, ckey(counts_kw), link, match)
    if key in _CACHE:
        return _CACHE[key]
    target = np.mean([mean_stat(s, cfg, 2.0, "C", hide_root, counts_kw, link, match) for s in seeds])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = np.mean([mean_stat(s, cfg, mid, "D", hide_root, counts_kw, link, match) for s in seeds])
        if got < target:
            lo = mid
        else:
            hi = mid
    res = (0.5 * (lo + hi), float(target))
    _CACHE[key] = res
    return res


def matched(seed, cfg=None, hide_root=False, counts_kw=None, link="linear",
            match="corr"):
    cfg = cfg or DEFAULT
    k, target = tune(cfg, hide_root, counts_kw=counts_kw, link=link, match=match)
    expr, idx = build(seed, cfg, k, hide_root, counts_kw, link)
    return expr, idx, k, target
