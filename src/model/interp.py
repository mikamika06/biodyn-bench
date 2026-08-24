import numpy as np
import torch


def _batches(x, bs):
    for i in range(0, x.shape[0], bs):
        yield x[i:i + bs]


@torch.no_grad()
def attention_network(model, x, dev, mask_frac=0.15, n_cells=512, seed=0, bs=64,
                      per_layer=False):
    model.eval()
    g = x.shape[1]
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    acc, n = None, 0
    for b in _batches(xs, bs):
        m = torch.zeros(b.shape[0], g, dtype=torch.bool)
        k = max(1, int(mask_frac * g))
        for r in range(b.shape[0]):
            m[r, torch.from_numpy(rng.choice(g, k, replace=False))] = True
        a = model.attention_maps(b.to(dev), m.to(dev)).mean(dim=0).cpu()
        acc = a if acc is None else acc + a
        n += 1
    per = (acc / n).numpy()
    per = np.maximum(per, np.transpose(per, (0, 2, 1)))
    if per_layer:
        return per
    w = per.mean(axis=0)
    return np.maximum(w, w.T)


def gradient_network(model, x, dev, mask_frac=0.15, n_cells=512, seed=0, bs=64):
    model.eval()
    g = x.shape[1]
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    w = np.zeros((g, g))
    cnt = np.zeros((g, g)) + 1e-9
    for b in _batches(xs, bs):
        m = torch.zeros(b.shape[0], g, dtype=torch.bool)
        k = max(1, int(mask_frac * g))
        for r in range(b.shape[0]):
            m[r, torch.from_numpy(rng.choice(g, k, replace=False))] = True
        bx = b.to(dev).clone().requires_grad_(True)
        md = m.to(dev)
        p = model(bx, md)
        for j in range(g):
            sel = md[:, j]
            if not sel.any():
                continue
            gr = torch.autograd.grad(p[sel, j].sum(), bx, retain_graph=True)[0]
            w[:, j] += gr[sel].abs().sum(0).detach().cpu().numpy()
            cnt[:, j] += int(sel.sum())
    w = w / cnt
    np.fill_diagonal(w, 0.0)
    return np.maximum(w, w.T)


@torch.no_grad()
def intervention_effect(model, x, dev, pairs, n_cells=512, seed=0, bs=128,
                        mask_frac=0.15):
    """Причинний вплив УСЕРЕДИНІ моделі: підмінюємо вхід гена i, міряємо зсув
    передбачення замаскованого гена j. Це «істина моделі», не істина даних.

    Маска тримає ту саму частку, що й на навчанні, інакше модель працює
    поза своїм розподілом. Ген i з маски виключено — інакше втручатись нікуди."""
    model.eval()
    g = x.shape[1]
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    perm = torch.from_numpy(rng.permutation(n_cells))
    k_extra = max(0, int(mask_frac * g) - 1)
    out = {}
    for (i, j) in pairs:
        pool = np.array([c for c in range(g) if c not in (i, j)])
        eff = []
        for s in range(0, n_cells, bs):
            b = xs[s:s + bs].to(dev)
            m = torch.zeros(b.shape[0], g, dtype=torch.bool)
            m[:, j] = True
            if k_extra:
                for r in range(b.shape[0]):
                    m[r, torch.from_numpy(rng.choice(pool, k_extra, replace=False))] = True
            m = m.to(dev)
            p0 = model(b, m)[:, j]
            b2 = b.clone()
            b2[:, i] = xs[perm[s:s + bs]][:, i].to(dev)
            p1 = model(b2, m)[:, j]
            eff.append((p1 - p0).abs().cpu().numpy())
        out[(i, j)] = float(np.concatenate(eff).mean())
    return out


@torch.no_grad()
def hidden_states(model, x, dev, mask_frac=0.15, n_cells=1024, seed=0, bs=64):
    model.eval()
    g = x.shape[1]
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    hs, ms = [], []
    for b in _batches(xs, bs):
        m = torch.zeros(b.shape[0], g, dtype=torch.bool)
        k = max(1, int(mask_frac * g))
        for r in range(b.shape[0]):
            m[r, torch.from_numpy(rng.choice(g, k, replace=False))] = True
        _, h = model(b.to(dev), m.to(dev), return_hidden=True)
        hs.append(h.cpu().numpy()); ms.append(m.numpy())
    return np.concatenate(hs), np.concatenate(ms), xs.numpy()


@torch.no_grad()
def _states_for(model, xs, dev, targets, g, rng, mask_frac, exclude, bs=128):
    """Приховані стани на позиціях targets, коли ці позиції замасковані,
    а гени зі списку exclude лишаються видимими."""
    k_extra = max(0, int(mask_frac * g) - 1)
    out = {}
    for j in targets:
        pool = np.array([c for c in range(g) if c != j and c not in exclude])
        hs = []
        for s in range(0, xs.shape[0], bs):
            b = xs[s:s + bs].to(dev)
            m = torch.zeros(b.shape[0], g, dtype=torch.bool)
            m[:, j] = True
            if k_extra and len(pool):
                for r in range(b.shape[0]):
                    m[r, torch.from_numpy(rng.choice(pool, min(k_extra, len(pool)),
                                                     replace=False))] = True
            _, h = model(b, m.to(dev), return_hidden=True)
            hs.append(h[:, j, :].cpu().numpy())
        out[j] = np.concatenate(hs)
    return out


def _ridge_r2(h, y, ridge=1.0, frac=0.7):
    n = len(y)
    cut = int(frac * n)
    htr, hte = h[:cut], h[cut:]
    ytr, yte = y[:cut], y[cut:]
    htr = np.c_[htr, np.ones(len(htr))]
    hte = np.c_[hte, np.ones(len(hte))]
    a = htr.T @ htr + ridge * np.eye(htr.shape[1])
    w = np.linalg.solve(a, htr.T @ ytr)
    pred = hte @ w
    ss = ((yte - yte.mean()) ** 2).sum()
    return float(1.0 - ((yte - pred) ** 2).sum() / max(ss, 1e-12))


def probe_effect(model, x, dev, pairs, n_cells=1024, seed=0, mask_frac=0.15):
    """Чи можна ЛІНІЙНО дістати значення гена i з прихованого стану позиції j."""
    model.eval()
    g = x.shape[1]
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    by_target = {}
    for i, j in pairs:
        by_target.setdefault(j, []).append(i)
    out = {}
    for j, srcs in by_target.items():
        st = _states_for(model, xs, dev, [j], g, rng, mask_frac, exclude=set(srcs))[j]
        for i in srcs:
            out[(i, j)] = max(_ridge_r2(st, x[:n_cells, i]), 0.0)
    return out


@torch.no_grad()
def patching_effect(model, x, dev, pairs, n_cells=512, seed=0, bs=128,
                    mask_frac=0.15, layers=None):
    """Patching активацій. Для пари (i, j):

      чистий прогін      -> p_clean
      зіпсований прогін  -> p_corr   (вхід гена i взято з іншої клітини)
      підмінений прогін  -> p_patch  (на шарі L позиції i повернуто ЧИСТУ активацію)

    Частка відновлення = |p_patch − p_corr| / |p_clean − p_corr|.
    Одиниця означає, що весь вплив гена i на ген j тече через цю активацію.
    """
    model.eval()
    g = x.shape[1]
    n_layers = len(model.encoder.layers)
    layers = layers if layers is not None else list(range(n_layers + 1))
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[:n_cells], dtype=torch.float32)
    perm = torch.from_numpy(rng.permutation(n_cells))
    k_extra = max(0, int(mask_frac * g) - 1)
    out = {}
    for (i, j) in pairs:
        pool = np.array([c for c in range(g) if c not in (i, j)])
        num = {L: [] for L in layers}
        den = []
        for s in range(0, n_cells, bs):
            b = xs[s:s + bs].to(dev)
            m = torch.zeros(b.shape[0], g, dtype=torch.bool)
            m[:, j] = True
            if k_extra:
                for r in range(b.shape[0]):
                    m[r, torch.from_numpy(rng.choice(pool, k_extra, replace=False))] = True
            m = m.to(dev)
            p_clean = model(b, m)[:, j]
            bc = b.clone()
            bc[:, i] = xs[perm[s:s + bs]][:, i].to(dev)
            p_corr = model(bc, m)[:, j]
            den.append((p_clean - p_corr).abs().cpu().numpy())
            states = model.layer_states(b, m)
            for L in layers:
                p_patch = model.run_patched(bc, m, [(L, i, states[L][:, i, :])])[:, j]
                num[L].append((p_patch - p_corr).abs().cpu().numpy())
        d = np.concatenate(den).mean()
        out[(i, j)] = {L: float(np.concatenate(v).mean() / max(d, 1e-9))
                       for L, v in num.items()}
        out[(i, j)]["_total"] = float(d)
    return out
