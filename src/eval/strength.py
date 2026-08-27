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
