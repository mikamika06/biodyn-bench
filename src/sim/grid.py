"""Сітка причинних структур із закладеною істиною.

Кожна структура дає панель: N модулів структури + N прямих пар + N незалежних генів.
Пари зрівнюються за силою звʼязку, тому метод не може виграти на величині.

Ключі пар:
  D  пряме ребро X -> Y            (справжнє ребро, еталон)
  S  пара під випробуванням        (залежить від структури)
  R  еталон-НЕребро зрівняної сили (спільна причина з видимим коренем)
  N  незалежні гени                (негативний контроль)
"""
import numpy as np
from sim.counts import make_counts, normalize, pair_corr, ckey
from methods.marginal import pair_mi

LINKS = {
    "linear": lambda a: a,
    "tanh": np.tanh,
    "relu": lambda a: np.maximum(a, 0.0),
}
NOISE = 1.0
_CACHE = {}

# Форма шуму. Під ГАУСОВИМ шумом напрямок ребра не ідентифіковний у принципі
# (клас еквівалентності Маркова). Під негаусовим — ідентифіковний (LiNGAM).
NOISES = {
    "gauss": lambda rng, n: rng.standard_normal(n),
    "uniform": lambda rng, n: (rng.random(n) - 0.5) * np.sqrt(12.0),
    "laplace": lambda rng, n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), n),
    "exp": lambda rng, n: rng.exponential(1.0, n) - 1.0,
}


def _signs(n, rho, rng):
    """rho — частка пригнічувальних (відʼємних) ребер."""
    s = np.ones(n)
    if rho > 0:
        s[rng.random(n) < rho] = -1.0
    return s


class Panel:
    def __init__(self, n_cells, rng, link, rho, noise=NOISE, noise_dist="gauss"):
        self.n, self.rng, self.f = n_cells, rng, LINKS[link]
        self.rho, self.noise = rho, noise
        self.nd = NOISES[noise_dist]
        self.cols, self.pairs, self.hidden = [], {"D": [], "S": [], "R": [], "N": []}, []

    def new(self):
        return self.nd(self.rng, self.n)

    def eps(self):
        return self.noise * self.nd(self.rng, self.n)

    def sign(self):
        return -1.0 if self.rng.random() < self.rho else 1.0

    def add(self, vec, hidden=False):
        self.cols.append(vec)
        i = len(self.cols) - 1
        if hidden:
            self.hidden.append(i)
        return i

    def finish(self):
        return np.column_stack(self.cols), self.pairs, self.hidden


# ── структури ────────────────────────────────────────────────────────────────
# Кожна отримує Panel і коефіцієнт k, додає ОДИН модуль і реєструє пару S.

def s_confounder(p, k, hide):
    """A -> B, A -> C.  S = (B,C): ребра НЕМА, спільна причина."""
    a = p.new()
    b = 2.0 * p.sign() * p.f(a) + p.eps()
    c = 3.0 * p.sign() * p.f(a) + p.eps()
    ia = p.add(a, hidden=hide)
    p.pairs["S"].append((p.add(b), p.add(c)))


def s_chain(p, k, hide):
    """A -> B -> C.  S = (A,C): прямого ребра нема, вплив є через B."""
    a = p.new()
    b = 2.0 * p.sign() * p.f(a) + p.eps()
    c = 2.0 * p.sign() * p.f(b) + p.eps()
    ia = p.add(a)
    ib = p.add(b, hidden=hide)
    p.pairs["S"].append((ia, p.add(c)))


def s_collider(p, k, hide):
    """A -> B <- C, A і C НЕЗАЛЕЖНІ.  S = (A,C): ребра нема й кореляції нема."""
    a, c = p.new(), p.new()
    b = p.sign() * p.f(a) + p.sign() * p.f(c) + p.eps()
    ia = p.add(a)
    p.add(b, hidden=hide)
    p.pairs["S"].append((ia, p.add(c)))


def s_ffl(p, k, hide):
    """Feed-forward loop: A -> B, A -> C, B -> C.
    S = (A,C): ребро Є, але поруч іде й непрямий шлях через B."""
    a = p.new()
    b = 2.0 * p.sign() * p.f(a) + p.eps()
    c = 1.2 * p.sign() * p.f(a) + 1.5 * p.sign() * p.f(b) + p.eps()
    ia = p.add(a)
    p.add(b, hidden=hide)
    p.pairs["S"].append((ia, p.add(c)))


def s_feedback(p, k, hide, iters=60, w=0.6):
    """A <-> B, взаємна регуляція, стаціонарний стан.
    S = (A,B): ребро Є в обидва боки, напрямок не визначений."""
    sa, sb = p.sign(), p.sign()
    a, b = p.new(), p.new()
    ea, eb = p.eps(), p.eps()
    for _ in range(iters):
        a_new = w * sa * p.f(b) + ea
        b = w * sb * p.f(a) + eb
        a = a_new
    p.pairs["S"].append((p.add(a), p.add(b)))


def s_two_hidden(p, k, hide):
    """H1 -> B,C і H2 -> B,C — ДВА приховані батьки.
    S = (B,C): ребра нема, конфаундерів два."""
    h1, h2 = p.new(), p.new()
    b = 1.5 * p.sign() * p.f(h1) + 1.5 * p.sign() * p.f(h2) + p.eps()
    c = 1.5 * p.sign() * p.f(h1) + 1.5 * p.sign() * p.f(h2) + p.eps()
    p.add(h1, hidden=hide)
    p.add(h2, hidden=hide)
    p.pairs["S"].append((p.add(b), p.add(c)))


def s_and(p, k, hide):
    """C = A AND B — синергія. Поодинці кожен батько майже не впливає.
    S = (A,C): ребро Є, але маргінальний ефект слабкий."""
    a, b = p.new(), p.new()
    c = 2.5 * p.sign() * p.f(a) * p.f(b) + p.eps()
    ia = p.add(a)
    p.add(b)
    p.pairs["S"].append((ia, p.add(c)))


def s_or(p, k, hide):
    """C = A OR B (soft-OR) — надлишковість. Кожен батько окремо достатній.
    S = (A,C): ребро Є, але вимкнення одного батька мало що змінює."""
    a, b = p.new(), p.new()
    soft = np.maximum(p.f(a), p.f(b))
    c = 2.0 * p.sign() * soft + p.eps()
    ia = p.add(a)
    p.add(b)
    p.pairs["S"].append((ia, p.add(c)))


def s_ref(p, kr):
    """Еталон-НЕребро: спільна причина з ВИДИМИМ коренем, сила задається kr.
    Потрібен там, де пара S має ребро — щоб порівнювати не з шумом,
    а з не-ребром такої самої маргінальної сили."""
    a = p.new()
    b = kr * p.sign() * p.f(a) + p.eps()
    c = kr * p.sign() * p.f(a) + p.eps()
    p.add(a)
    p.pairs["R"].append((p.add(b), p.add(c)))


def s_oneway(p, kr, iters=60):
    """Еталон для стовпця напрямку: ОДНОСТОРОННЄ ребро, отримане ТИМ САМИМ
    ітеративним механізмом, що й двоцикл, але зі зворотною вагою нуль.

    Раніше еталон будувався одним множенням. Тоді двоцикл і еталон проходили
    різні процеси, і зрівнювання однієї статистики лишало другу розбалансованою
    на пʼять сігм — стовпець ставав тестом величини, а не структури.
    """
    sb = p.sign()
    a, b = p.new(), p.new()
    ea, eb = p.eps(), p.eps()
    for _ in range(iters):
        a_new = ea
        b = kr * sb * p.f(a) + eb
        a = a_new
    p.pairs["R"].append((p.add(a), p.add(b)))


def s_feedback_dir(p, k, hide, iters=60, w=0.6):
    """Той самий двоцикл, але порівнюється з ОДНОСТОРОННІМ ребром рівної сили.
    Питання стовпця: чи бачить метод НАПРЯМОК, а не наявність звʼязку."""
    s_feedback(p, k, hide, iters, w)


REFS = {"confounder": s_ref, "oneway": s_oneway}

STRUCTURES = {
    "confounder": s_confounder,
    "feedback_dir": s_feedback_dir,
    "chain": s_chain,
    "collider": s_collider,
    "ffl": s_ffl,
    "feedback": s_feedback,
    "two_hidden": s_two_hidden,
    "and": s_and,
    "or": s_or,
}

# Специфікація метрики для кожної структури.
#   pos/neg  які типи пар порівнюються
#   correct  "high" — метод має розрізнити; "0.5" — метод НЕ має вигадати ребро
#   match    чи зрівнювати силу звʼязку (можливо лише коли обидва типи ненульові)
#   asks     що саме питає стовпець
SPEC = {
    "confounder": {"pos": "D", "neg": "S", "correct": "high", "match": True, "ref": False,
                   "asks": "чи відрізнить пряме ребро від спільної причини"},
    "chain":      {"pos": "D", "neg": "S", "correct": "high", "match": True, "ref": False,
                   "asks": "чи відрізнить прямий вплив від транзитивного"},
    "two_hidden": {"pos": "D", "neg": "S", "correct": "high", "match": True, "ref": False,
                   "asks": "те саме, але конфаундерів ДВА і обидва приховані"},
    "collider":   {"pos": "S", "neg": "N", "correct": "0.5", "match": False, "ref": False,
                   "asks": "чи ВИГАДАЄ ребро між незалежними батьками"},
    "ffl":        {"pos": "S", "neg": "R", "correct": "high", "match": False, "ref": True,
                   "asks": "чи знайде пряме ребро, коли поруч іде обхідний шлях"},
    "feedback":   {"pos": "S", "neg": "R", "correct": "high", "match": False, "ref": True,
                   "asks": "чи відрізнить взаємну регуляцію від спільної причини"},
    "and":        {"pos": "S", "neg": "N", "correct": "high", "match": False, "ref": False,
                   "asks": "чи знайде ребро, невидиме маргінально (синергія)"},
    "or":         {"pos": "S", "neg": "R", "correct": "high", "match": False, "ref": True,
                   "asks": "чи знайде ребро при надлишковості батьків"},
    "feedback_dir": {"pos": "S", "neg": "R", "correct": "high", "match": False,
                     "ref": True, "ref_fn": "oneway",
                     "asks": "чи відрізнить взаємну регуляцію від ОДНОСТОРОННЬОЇ"},
}
for _v in SPEC.values():
    _v.setdefault("ref_fn", "confounder")
HAS_EDGE = {k: v["correct"] == "high" and v["pos"] == "S" or k in ("ffl", "feedback", "and", "or")
            for k, v in SPEC.items()}

DEFAULT = {"n_cells": 5000, "n_struct": 25, "n_direct": 25, "n_ref": 25, "n_null": 30}


def mean_abs_corr(expr, pairs):
    return float(np.mean([abs(pair_corr(expr, i, j)) for i, j in pairs]))


def mean_pair_mi(expr, pairs, bins=10):
    return float(np.mean(pair_mi(expr, pairs, bins)))


def mean_sq_corr(expr, pairs):
    """Кореляція КВАДРАТІВ стандартизованих значень.

    Ловить розбіжність у четвертому моменті (гостроверхість), до якої
    взаємна інформація на квантильних кошиках майже сліпа — зокрема для
    рівномірного шуму, у якого гостроверхість МЕНША за гаусову.
    """
    x = expr - expr.mean(axis=0)
    sd = x.std(axis=0)
    x = x / np.where(sd == 0, 1.0, sd)
    q = x ** 2
    q = q - q.mean(axis=0)
    qsd = q.std(axis=0)
    q = q / np.where(qsd == 0, 1.0, qsd)
    return float(np.mean([abs((q[:, i] * q[:, j]).mean()) for i, j in pairs]))


def mean_partial(expr, pairs):
    """Середня |часткова кореляція| пар — при фіксованих усіх інших генах.

    Три попередні статистики ПАРНІ: вони бачать лише двовимірний розподіл.
    Ця бачить те, що доступне обумовленню. Без неї міра ідентифіковності
    сліпа до конфаундера й ланцюга, які парно нерозрізненні за побудовою.
    """
    from methods.conditional import partial_corr_matrix
    w = np.abs(partial_corr_matrix(expr))
    return float(np.mean([w[i, j] for i, j in pairs]))


STATS = {"corr": mean_abs_corr, "mi": mean_pair_mi, "sq": mean_sq_corr,
         "partial": mean_partial}


def _strength(structure, seed, cfg, k, hide, link, rho, counts_kw, typ,
              stat="corr", kr=1.0, noise_dist="gauss"):
    expr, pairs = build(structure, seed, cfg, k, hide, link, rho, counts_kw, kr,
                        noise_dist)
    if not pairs[typ]:
        return 0.0
    return STATS[stat](expr, pairs[typ])


def _bisect(fn, target, lo, hi, iters):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def tune(structure, cfg=None, hide=False, link="linear", rho=0.0, counts_kw=None,
         seeds=(101, 102, 103), lo=0.01, hi=40.0, iters=40, stat="corr",
         noise_dist="gauss"):
    """Підбирає k прямих пар так, щоб їхня сила звʼязку зрівнялась зі структурною.

    stat  за якою статистикою рівняти: "corr" або "mi". Вибір НЕ нейтральний —
          він перевертає результат (див. docs/explain/08, частина 2), тому
          метод має проходити ОБИДВА зрівнювання.

    Має сенс лише коли SPEC[structure]["match"] — тобто коли обидва порівнювані
    типи мають ненульову силу. Для колайдера й синергії зрівнювання неможливе
    за побудовою: структурна пара має кореляцію біля нуля.
    """
    cfg = cfg or DEFAULT
    if not SPEC[structure]["match"]:
        return 1.6, None
    key = ("k", structure, tuple(sorted(cfg.items())), hide, link, rho,
           ckey(counts_kw), stat, noise_dist)
    if key in _CACHE:
        return _CACHE[key]
    args = (cfg, 1.0, hide, link, rho, counts_kw)
    target = float(np.mean([_strength(structure, s, *args, "S", stat,
                                      noise_dist=noise_dist) for s in seeds]))
    fn = lambda mid: np.mean([_strength(structure, s, cfg, mid, hide, link, rho,
                                        counts_kw, "D", stat,
                                        noise_dist=noise_dist) for s in seeds])
    k_hat = _bisect(fn, target, lo, hi, iters)
    got = float(fn(k_hat))
    if abs(got - target) > 0.15 * max(target, 1e-6):
        print(f"  УВАГА: зрівнювання {structure} не збіглось "
              f"(ціль {target:.4f}, вийшло {got:.4f}, k={k_hat:.3f})", flush=True)
    res = (k_hat, target)
    _CACHE[key] = res
    return res


def tune_ref(structure, cfg=None, hide=False, link="linear", rho=0.0, counts_kw=None,
             seeds=(101, 102, 103), lo=0.005, hi=40.0, iters=40, stat="corr", k=1.6,
             noise_dist="gauss"):
    """Підбирає силу еталонних НЕребер R так, щоб вона зрівнялась із парою S.

    Потрібно там, де пара S МАЄ ребро (ffl, feedback, or). Без цього порівняння
    йде проти незалежних генів, підлога стовпця дорівнює 1.000, і стовпець
    нічого не міряє.
    """
    cfg = cfg or DEFAULT
    if not SPEC[structure]["ref"]:
        return 1.0, None
    key = ("kr", structure, tuple(sorted(cfg.items())), hide, link, rho,
           ckey(counts_kw), stat, noise_dist)
    if key in _CACHE:
        return _CACHE[key]
    target = float(np.mean([_strength(structure, s, cfg, k, hide, link, rho,
                                      counts_kw, "S", stat,
                                      noise_dist=noise_dist) for s in seeds]))
    fn = lambda mid: np.mean([_strength(structure, s, cfg, k, hide, link, rho,
                                        counts_kw, "R", stat, mid,
                                        noise_dist) for s in seeds])
    res = (_bisect(fn, target, lo, hi, iters), target)
    _CACHE[key] = res
    return res


def matched(structure, seed, cfg=None, hide=False, link="linear", rho=0.0,
            counts_kw=None, stat="corr", noise_dist="gauss"):
    """Панель зі зрівняними парами. Повертає (expr, pairs, k, цільова сила)."""
    k, target = tune(structure, cfg, hide, link, rho, counts_kw, stat=stat,
                     noise_dist=noise_dist)
    kr, rtarget = tune_ref(structure, cfg, hide, link, rho, counts_kw, stat=stat,
                           k=k, noise_dist=noise_dist)
    expr, pairs = build(structure, seed, cfg, k, hide, link, rho, counts_kw, kr,
                        noise_dist)
    return expr, pairs, k, (target if target is not None else rtarget)


def build(structure, seed, cfg=None, k=1.6, hide=False, link="linear",
          rho=0.0, counts_kw=None, kr=1.0, noise_dist="gauss"):
    cfg = cfg or DEFAULT
    rng = np.random.default_rng(seed)
    p = Panel(cfg["n_cells"], rng, link, rho, noise_dist=noise_dist)
    fn = STRUCTURES[structure]
    for _ in range(cfg["n_struct"]):
        fn(p, k, hide)
    for _ in range(cfg["n_direct"]):
        x = p.new()
        y = k * p.sign() * p.f(x) + p.eps()
        p.pairs["D"].append((p.add(x), p.add(y)))
    if SPEC[structure]["ref"]:
        ref_fn = REFS[SPEC[structure]["ref_fn"]]
        for _ in range(cfg.get("n_ref", cfg["n_direct"])):
            ref_fn(p, kr)
    nulls = [p.add(p.new()) for _ in range(cfg["n_null"])]
    p.pairs["N"] = [(nulls[i], nulls[i + 1]) for i in range(0, len(nulls) - 1, 2)]

    z, pairs, hidden = p.finish()
    counts = make_counts(z, rng, counts_kw)
    if hide and hidden:
        drop = set(hidden)
        keep = [i for i in range(counts.shape[1]) if i not in drop]
        remap = {o: n for n, o in enumerate(keep)}
        pairs = {t: [(remap[i], remap[j]) for i, j in ps] for t, ps in pairs.items()}
        counts = counts[:, keep]
    return normalize(counts), pairs
