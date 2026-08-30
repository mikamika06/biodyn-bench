import numpy as np
from sim import grid
from sim.grid import Panel, SPEC, STRUCTURES, REFS, DEFAULT, tune, tune_ref, _signs
from sim.counts import make_counts, normalize


def s_feedback_latent(p, k, hide, iters=60, w=0.6, c=1.0):
    lat = p.new()
    sa, sb = p.sign(), p.sign()
    sl1, sl2 = p.sign(), p.sign()
    a, b = p.new(), p.new()
    ea, eb = p.eps(), p.eps()
    for _ in range(iters):
        a_new = w * sa * p.f(b) + c * sl1 * p.f(lat) + ea
        b = w * sb * p.f(a) + c * sl2 * p.f(lat) + eb
        a = a_new
    p.add(lat, hidden=True)
    p.pairs["S"].append((p.add(a), p.add(b)))


STRUCTURES2 = dict(STRUCTURES)
STRUCTURES2["feedback_latent"] = s_feedback_latent
SPEC2 = {k: dict(v) for k, v in SPEC.items()}
SPEC2["collider"]["neg"] = "M"
SPEC2["and"]["neg"] = "M"
SPEC2["and"]["asks"] = "чи знайде синергічне ребро проти пар вхід-вихід РІЗНИХ модулів (та сама роль)"
SPEC2["ffl"]["neg"] = "M"
SPEC2["ffl"]["asks"] = "чи відрізнить пряме ребро A→C від пари корінь-онук ланцюга (лише непрямий шлях), зрівняної за силою"
SPEC2["collider"]["asks"] = "чи ВИГАДАЄ ребро між батьками спільної дитини, проти батьків РІЗНИХ дітей (та сама роль, той самий R²)"
SPEC2["feedback_latent"] = {"pos": "S", "neg": "R", "correct": "high", "match": False, "ref": True,
                            "ref_fn": "confounder",
                            "asks": "чи відрізнить взаємну регуляцію з прихованим спільним драйвером від спільної причини"}

N_NUISANCE = 24
NUIS_K = 3


def r2_visible(z, visible):
    X = z[:, visible]
    n, g = X.shape
    out = np.zeros(g)
    for j in range(g):
        o = [c for c in range(g) if c != j]
        A = np.c_[X[:, o], np.ones(n)]
        w = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        res = X[:, j] - A @ w
        out[j] = 1.0 - res.var() / max(1e-12, X[:, j].var())
    return out


def balance_r2(z, hidden, rng, target=None, n_nuis=N_NUISANCE, k=NUIS_K):
    n, g = z.shape
    visible = [c for c in range(g) if c not in set(hidden)]
    nuis = np.column_stack([rng.standard_normal(n) for _ in range(n_nuis)])
    nuis = (nuis - nuis.mean(0)) / nuis.std(0)
    z = z.copy()
    z = (z - z.mean(0)) / np.where(z.std(0) == 0, 1.0, z.std(0))
    r2 = r2_visible(z, visible)
    tgt = float(np.median(r2)) if target is None else target
    for _ in range(2):
        for idx, j in enumerate(visible):
            cur = r2[idx]
            if abs(cur - tgt) < 0.02:
                continue
            if cur < tgt:
                w2 = (tgt - cur) / max(1e-6, 1.0 - cur)
                w2 = min(max(w2, 0.0), 0.95)
                pick = rng.choice(n_nuis, k, replace=False)
                u = nuis[:, pick].sum(1) / np.sqrt(k)
                z[:, j] = np.sqrt(1.0 - w2) * z[:, j] + np.sqrt(w2) * u
            else:
                w2 = 1.0 - tgt / max(1e-6, cur)
                w2 = min(max(w2, 0.0), 0.95)
                fresh = rng.standard_normal(n)
                fresh = (fresh - fresh.mean()) / fresh.std()
                z[:, j] = np.sqrt(1.0 - w2) * z[:, j] + np.sqrt(w2) * fresh
        z = (z - z.mean(0)) / np.where(z.std(0) == 0, 1.0, z.std(0))
        r2 = r2_visible(z, visible)
    return np.column_stack([z, nuis]), tgt


def role_matched(pairs, structure):
    S = pairs.get("S") or []
    if structure in ("collider", "and") and len(S) >= 2:
        return [(S[i][0], S[(i + 1) % len(S)][1]) for i in range(len(S))]
    return []


def s_chain_m(p, km):
    a = p.new()
    b = km * p.sign() * p.f(a) + p.eps()
    c = km * p.sign() * p.f(b) + p.eps()
    ia = p.add(a)
    p.add(b)
    p.pairs.setdefault("M", []).append((ia, p.add(c)))


_KM_CACHE = {}


def tune_km(cfg, link, rho, counts_kw, noise_dist, seeds=(101, 102, 103), lo=0.01, hi=40.0, iters=30):
    from sim.grid import _bisect
    from sim.counts import ckey
    key = (tuple(sorted(cfg.items())), link, rho, ckey(counts_kw), noise_dist)
    if key in _KM_CACHE:
        return _KM_CACHE[key]
    from eval.strength import strengths
    import numpy as _np

    def s_strength(km, typ):
        vals = []
        for sd in seeds:
            expr, pairs = build2("ffl", sd, cfg, 1.6, False, link, rho, counts_kw, 1.0, noise_dist, km=km)
            vals.append(float(_np.mean(_np.abs(strengths(expr, pairs[typ])))))
        return float(_np.mean(vals))

    target = s_strength(1.6, "S")
    km = _bisect(lambda mid: s_strength(mid, "M"), target, lo, hi, iters)
    _KM_CACHE[key] = (km, target)
    return km, target


def build2(structure, seed, cfg=None, k=1.6, hide=False, link="linear",
           rho=0.0, counts_kw=None, kr=1.0, noise_dist="gauss", balance=False, km=None):
    cfg = cfg or DEFAULT
    rng = np.random.default_rng(seed)
    p = Panel(cfg["n_cells"], rng, link, rho, noise_dist=noise_dist)
    fn = STRUCTURES2[structure]
    for _ in range(cfg["n_struct"]):
        fn(p, k, hide)
    for _ in range(cfg["n_direct"]):
        x = p.new()
        y = k * p.sign() * p.f(x) + p.eps()
        p.pairs["D"].append((p.add(x), p.add(y)))
    if SPEC2[structure]["ref"]:
        ref_fn = REFS[SPEC2[structure]["ref_fn"]]
        for _ in range(cfg.get("n_ref", cfg["n_direct"])):
            ref_fn(p, kr)
    nulls = [p.add(p.new()) for _ in range(cfg["n_null"])]
    p.pairs["N"] = [(nulls[i], nulls[i + 1]) for i in range(0, len(nulls) - 1, 2)]
    if structure == "ffl":
        for _ in range(cfg["n_struct"]):
            s_chain_m(p, km if km is not None else k)
    z, pairs, hidden = p.finish()
    if structure != "ffl":
        pairs["M"] = role_matched(pairs, structure)
    pairs.setdefault("M", [])
    if balance:
        z, _ = balance_r2(z, hidden, rng)
    counts = make_counts(z, rng, counts_kw)
    if hidden:
        drop = set(hidden)
        keep = [i for i in range(counts.shape[1]) if i not in drop]
        remap = {o: n for n, o in enumerate(keep)}
        pairs = {t: [(remap[i], remap[j]) for i, j in ps] for t, ps in pairs.items()}
        counts = counts[:, keep]
    return normalize(counts), pairs


class _use_v2:
    def __enter__(self):
        self.prev = (grid.BUILDER, grid.GEN_TAG, dict(grid.STRUCTURES), dict(grid.SPEC))
        grid.BUILDER = build2
        grid.GEN_TAG = "v2"
        grid.STRUCTURES.update(STRUCTURES2)
        grid.SPEC.update(SPEC2)
        return self

    def __exit__(self, *a):
        grid.BUILDER, grid.GEN_TAG = self.prev[0], self.prev[1]
        grid.STRUCTURES.clear(); grid.STRUCTURES.update(self.prev[2])
        grid.SPEC.clear(); grid.SPEC.update(self.prev[3])


def matched2(structure, seed, cfg=None, hide=False, link="linear", rho=0.0,
             counts_kw=None, stat="corr", noise_dist="gauss"):
    with _use_v2():
        k, target = tune(structure, cfg, hide, link, rho, counts_kw, stat=stat, noise_dist=noise_dist)
        kr, rtarget = tune_ref(structure, cfg, hide, link, rho, counts_kw, stat=stat, k=k, noise_dist=noise_dist)
        km = None
        if structure == "ffl":
            km, _ = tune_km(cfg or DEFAULT, link, rho, counts_kw, noise_dist)
        expr, pairs = build2(structure, seed, cfg, k, hide, link, rho, counts_kw, kr, noise_dist, km=km)
    return expr, pairs, k, (target if target is not None else rtarget)
