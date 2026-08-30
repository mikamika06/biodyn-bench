"""Зрівнювання пар за силою у ЗВʼЯЗНОМУ графі.

У сітці ізольованих модулів силу можна підігнати коефіцієнтом. У звʼязному
графі так не вийде — сила пари визначається всім графом. Тому зрівнюємо
пост-фактум: беремо підвибірки двох типів пар з ОДНАКОВИМ розподілом сили.
"""
import numpy as np
from sim.counts import pair_corr


def strengths(expr, pairs):
    return np.array([abs(pair_corr(expr, i, j)) for i, j in pairs])


def match_subsets(sa, sb, rng, bins=20, cap=None):
    """Індекси підвибірок A і B з однаковим розподілом сили.

    Кошики — квантилі обʼєднаного розподілу. З кожного кошика беремо
    min(|A|, |B|) пар кожного типу.
    """
    pool = np.concatenate([sa, sb])
    edges = np.unique(np.quantile(pool, np.linspace(0, 1, bins + 1)[1:-1]))
    ba, bb = np.searchsorted(edges, sa, "right"), np.searchsorted(edges, sb, "right")
    ia, ib = [], []
    for k in range(len(edges) + 1):
        ka, kb = np.where(ba == k)[0], np.where(bb == k)[0]
        n = min(len(ka), len(kb))
        if cap:
            n = min(n, cap)
        if n == 0:
            continue
        ia.append(rng.choice(ka, n, replace=False))
        ib.append(rng.choice(kb, n, replace=False))
    if not ia:
        return np.array([], int), np.array([], int)
    return np.concatenate(ia), np.concatenate(ib)


def matched_pairs(expr, pa, pb, seed=0, bins=20, cap=None):
    rng = np.random.default_rng(seed)
    sa, sb = strengths(expr, pa), strengths(expr, pb)
    ia, ib = match_subsets(sa, sb, rng, bins, cap)
    return ([pa[i] for i in ia], [pb[i] for i in ib],
            float(sa[ia].mean() - sb[ib].mean()) if len(ia) else 0.0,
            float(0.5 * (sa[ia].std() + sb[ib].std())) if len(ia) else 0.0,
            len(ia))


def r2_nodes(expr):
    X = np.log1p(np.asarray(expr, dtype=float))
    n, g = X.shape
    out = np.zeros(g)
    for j in range(g):
        o = [c for c in range(g) if c != j]
        A = np.c_[X[:, o], np.ones(n)]
        w = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        res = X[:, j] - A @ w
        out[j] = 1.0 - res.var() / max(1e-12, X[:, j].var())
    return out


def r2_central(expr, pairs, r2=None):
    r2 = r2_nodes(expr) if r2 is None else r2
    return np.array([max(r2[i], r2[j]) for i, j in pairs])


def matched_pairs_r2(expr, pa, pb, seed=0, bins=None, r2_bins=None, cap=None):
    rng = np.random.default_rng(seed)
    r2 = r2_nodes(expr)
    sa, sb = strengths(expr, pa), strengths(expr, pb)
    ra, rb = r2_central(expr, pa, r2), r2_central(expr, pb, r2)
    key_a = np.c_[sa, ra]
    key_b = np.c_[sb, rb]
    pool = np.r_[key_a, key_b]
    n_min = min(len(pa), len(pb))
    bins = bins or max(2, min(20, n_min // 5))
    r2_bins = r2_bins or max(2, min(5, n_min // 8))
    e1 = np.unique(np.quantile(pool[:, 0], np.linspace(0, 1, bins + 1)[1:-1]))
    e2 = np.unique(np.quantile(pool[:, 1], np.linspace(0, 1, r2_bins + 1)[1:-1]))
    cell = lambda k: np.searchsorted(e1, k[:, 0], "right") * (len(e2) + 1) + np.searchsorted(e2, k[:, 1], "right")
    ca, cb = cell(key_a), cell(key_b)
    ia, ib = [], []
    for c in np.unique(np.r_[ca, cb]):
        ka, kb = np.where(ca == c)[0], np.where(cb == c)[0]
        m = min(len(ka), len(kb))
        if cap:
            m = min(m, cap)
        if m == 0:
            continue
        ia.append(rng.choice(ka, m, replace=False))
        ib.append(rng.choice(kb, m, replace=False))
    if not ia:
        return [], [], {}
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    stats = {"n": int(len(ia)),
             "corr_gap": float(sa[ia].mean() - sb[ib].mean()),
             "r2_gap": float(ra[ia].mean() - rb[ib].mean()),
             "r2_gap_corr_only": None}
    return [pa[i] for i in ia], [pb[i] for i in ib], stats
