"""Звʼязний scale-free орієнтований граф із похідними типами пар.

Мотиви тут не вмонтовуються — вони ВИНИКАЮТЬ у випадковому графі, і типи пар
виводяться з самого графа. Це наближає стенд до справжньої задачі, де
обумовлюватись треба на сотнях корельованих генів, а не на трьох чистих.
"""
import numpy as np
from sim.counts import make_counts, normalize

LINKS = {"linear": lambda a: a, "tanh": np.tanh,
         "relu": lambda a: np.maximum(a, 0.0)}

NOISES = {
    "gauss": lambda rng, n: rng.standard_normal(n),
    "uniform": lambda rng, n: (rng.random(n) - 0.5) * np.sqrt(12.0),
    "laplace": lambda rng, n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), n),
    "exp": lambda rng, n: rng.exponential(1.0, n) - 1.0,
}


def scale_free_dag(n, rng, m=2, alpha=1.0):
    """Орієнтований ациклічний граф із перевагою приєднання.

    Вузол j бере m батьків серед 0..j-1 з імовірністю, пропорційною
    (вихідний степінь + 1)^alpha. Дає важкий хвіст: кілька хабів, більшість
    вузлів із малим степенем — як у справжніх регуляторних мережах.
    """
    parents = [[] for _ in range(n)]
    outdeg = np.zeros(n)
    for j in range(1, n):
        k = min(m, j)
        w = (outdeg[:j] + 1.0) ** alpha
        idx = rng.choice(j, size=k, replace=False, p=w / w.sum())
        parents[j] = sorted(int(i) for i in idx)
        outdeg[idx] += 1
    return parents


def simulate(parents, n_cells, rng, link="linear", rho=0.0, w_lo=0.8, w_hi=2.0,
             noise=1.0, noise_dist="gauss"):
    n = len(parents)
    f = LINKS[link]  # вузли нормуються до одиничної дисперсії, інакше сигнал
                     # росте вниз по графу експоненційно і вбиває кореляції
    x = np.zeros((n_cells, n))
    W = {}
    for j in range(n):
        acc = noise * NOISES[noise_dist](rng, n_cells)
        for i in parents[j]:
            w = rng.uniform(w_lo, w_hi) * (-1.0 if rng.random() < rho else 1.0)
            W[(i, j)] = w
            acc = acc + w * f(x[:, i])
        sd = acc.std()
        x[:, j] = (acc - acc.mean()) / (sd if sd > 0 else 1.0)
    return x, W


def _reach(parents, n):
    """Матриця досяжності: anc[i, j] = чи є орієнтований шлях i -> j."""
    anc = np.zeros((n, n), dtype=bool)
    for j in range(n):
        for i in parents[j]:
            anc[i, j] = True
            anc[:, j] |= anc[:, i]
    return anc


def classify(parents, n, anc=None):
    """Типи пар, виведені з графа.

    D     пряме ребро i -> j
    CONF  спільний батько, шляху між ними немає
    CHAIN шлях довжиною >= 2, прямого ребра немає
    COLL  спільна дитина, ані шляху, ані спільного батька
    N     жодного звʼязку
    """
    anc = _reach(parents, n) if anc is None else anc
    kids = [[] for _ in range(n)]
    for j in range(n):
        for i in parents[j]:
            kids[i].append(j)
    par = [set(p) for p in parents]
    kid = [set(k) for k in kids]
    out = {t: [] for t in ("D", "CONF", "CHAIN", "COLL", "N")}
    for i in range(n):
        for j in range(i + 1, n):
            direct = anc[i, j] and j in kid[i] or anc[j, i] and i in kid[j]
            path = anc[i, j] or anc[j, i]
            if direct:
                out["D"].append((i, j)); continue
            share_par = bool(par[i] & par[j])
            share_kid = bool(kid[i] & kid[j])
            if path:
                out["CHAIN"].append((i, j))
            elif share_par:
                out["CONF"].append((i, j))
            elif share_kid:
                out["COLL"].append((i, j))
            else:
                out["N"].append((i, j))
    return out


def build(n_genes, n_cells, seed, m=2, link="linear", rho=0.0, alpha=1.0,
          counts_kw=None, noise_dist="gauss"):
    rng = np.random.default_rng(seed)
    parents = scale_free_dag(n_genes, rng, m=m, alpha=alpha)
    z, W = simulate(parents, n_cells, rng, link=link, rho=rho,
                    noise_dist=noise_dist)
    counts = make_counts(z, rng, counts_kw)
    return normalize(counts), classify(parents, n_genes), parents, W


def build_mixture(n_genes, n_cells, seed, k_types=3, share=0.6, m=2,
                  link="linear", rho=0.0, alpha=0.6, counts_kw=None,
                  shift=0.8, shift_frac=1.0):
    """Суміш популяцій: один скелет графа, у кожному типі клітин активна
    своя підмножина ребер плюс власний зсув базових рівнів.

    Тип клітини НЕ спостерігається. Він діє як прихований конфаундер, що
    корелює всі гени одразу — найчастіший реальний конфаундер, якого немає
    ні в ізольованій сітці, ні в однорідному графі.

    core  ребра, активні в УСІХ типах        справжня спільна регуляція
    var   ребра, активні лише в частині      специфічні для типу
    """
    rng = np.random.default_rng(seed)
    parents = scale_free_dag(n_genes, rng, m=m, alpha=alpha)
    edges = [(i, j) for j in range(n_genes) for i in parents[j]]
    active = {e: rng.random(k_types) < share for e in edges}
    core = [e for e in edges if active[e].all()]
    var = [e for e in edges if active[e].any() and not active[e].all()]

    per = n_cells // k_types
    blocks, labels = [], []
    for t in range(k_types):
        par_t = [[] for _ in range(n_genes)]
        for (i, j) in edges:
            if active[(i, j)][t]:
                par_t[j].append(i)
        z, _ = simulate(par_t, per, rng, link=link, rho=rho)
        sh = shift * rng.standard_normal(n_genes)
        if shift_frac < 1.0:
            # зсув торкається лише частини генів: решта НЕ може слугувати
            # проксі типу клітини, і обумовлення втрачає опору
            off = rng.random(n_genes) >= shift_frac
            sh[off] = 0.0
        z = z + sh[None, :]
        blocks.append(z)
        labels += [t] * per
    z = np.vstack(blocks)
    counts = make_counts(z, rng, counts_kw)
    anc = _reach(parents, n_genes)
    types = classify(parents, n_genes, anc)
    types["CORE"] = [(min(i, j), max(i, j)) for i, j in core]
    types["VAR"] = [(min(i, j), max(i, j)) for i, j in var]
    return normalize(counts), types, np.array(labels), parents
