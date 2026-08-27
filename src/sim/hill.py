"""Кінетика Хілла в стаціонарному стані — незалежний генератор.

Наш основний генератор адитивний: вплив батьків складається. SERGIO та схожі
симулятори використовують функції Хілла — насичувальні пороги з окремими
формами для активації та пригнічення. Якщо висновки стенду тримаються під
обома кінетиками, вони не є властивістю нашого симулятора.

Стаціонарний стан:  x_j = P_j(батьки) / lambda_j
де P_j = b_j + sum_i K_ij * h(x_i),  h — функція Хілла.
"""
import numpy as np
from sim.counts import make_counts, normalize
from sim.network import scale_free_dag, classify, _reach


def hill_activate(x, k, n):
    xp = np.maximum(x, 0.0) ** n
    return xp / (xp + k ** n)


def hill_repress(x, k, n):
    xp = np.maximum(x, 0.0) ** n
    return k ** n / (xp + k ** n)


def simulate_hill(parents, n_cells, rng, rho=0.0, n_coef=2.0, decay=0.8,
                  k_lo=0.5, k_hi=2.0, master_shape=2.0, master_scale=1.5,
                  noise=0.15):
    """rho — частка пригнічувальних взаємодій."""
    n = len(parents)
    x = np.zeros((n_cells, n))
    W = {}
    for j in range(n):
        if not parents[j]:
            # головний регулятор: розкид між клітинами задає варіацію системи
            x[:, j] = rng.gamma(master_shape, master_scale, size=n_cells)
            continue
        prod = np.full(n_cells, 0.05)
        for i in parents[j]:
            strength = rng.uniform(1.0, 4.0)
            k = rng.uniform(k_lo, k_hi) * max(np.median(x[:, i]), 1e-3)
            repress = rng.random() < rho
            W[(i, j)] = -strength if repress else strength
            h = hill_repress if repress else hill_activate
            prod = prod + strength * h(x[:, i], k, n_coef)
        val = prod / decay
        val = val * np.exp(noise * rng.standard_normal(n_cells))
        x[:, j] = val
    return x, W


def build_hill(n_genes, n_cells, seed, m=2, rho=0.0, alpha=0.6, counts_kw=None,
               **kw):
    rng = np.random.default_rng(seed)
    parents = scale_free_dag(n_genes, rng, m=m, alpha=alpha)
    z, W = simulate_hill(parents, n_cells, rng, rho=rho, **kw)
    z = np.log1p(z)
    z = (z - z.mean(0)) / np.where(z.std(0) == 0, 1.0, z.std(0))
    counts = make_counts(z, rng, counts_kw)
    return normalize(counts), classify(parents, n_genes), parents, W
