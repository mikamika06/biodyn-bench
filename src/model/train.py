import numpy as np
import torch
from model.data import CellDataset, split, standardize
from model.transformer import GeneTransformer


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def validate(m, dlv, dev):
    m.eval()
    v = []
    for x, msk in dlv:
        x, msk = x.to(dev), msk.to(dev)
        v.append((((m(x, msk) - x)[msk]) ** 2).mean().item())
    return float(np.mean(v))


def train(expr, steps=20000, d=192, n_layers=4, n_heads=4, batch=64, lr=1e-3,
          mask_frac=0.15, seed=0, verbose=True, dev=None, log_every=500,
          warmup=0.05, model=None):
    torch.manual_seed(seed)
    dev = dev or device()
    tr, te = split(expr, seed=seed)
    zs, mu, sd = standardize(expr)
    tr, te = (tr - mu) / sd, (te - mu) / sd

    dl = torch.utils.data.DataLoader(CellDataset(tr, mask_frac, seed),
                                     batch_size=batch, shuffle=True, drop_last=True)
    dlv = torch.utils.data.DataLoader(CellDataset(te, mask_frac, seed + 1), batch_size=128)
    m = (model or GeneTransformer(expr.shape[1], d, n_layers, n_heads)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, steps, pct_start=warmup)

    hist, step, run = [], 0, []
    while step < steps:
        m.train()
        for x, msk in dl:
            if step >= steps:
                break
            x, msk = x.to(dev), msk.to(dev)
            loss = ((m(x, msk) - x)[msk] ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step()
            run.append(loss.item()); step += 1
            if step % log_every == 0:
                va = validate(m, dlv, dev)
                hist.append((step, float(np.mean(run[-log_every:])), va))
                if verbose:
                    print(f"  крок {step:6d}  навч {hist[-1][1]:.4f}  валід {va:.4f}",
                          flush=True)
                m.train()
    return m, hist, (mu, sd)


def linear_ceiling(expr, n_probe=20, seed=0):
    zs, _, _ = standardize(expr)
    tr, te = split(zs, seed=seed)
    g = zs.shape[1]
    rng = np.random.default_rng(seed)
    out = []
    for j in rng.choice(g, n_probe, replace=False):
        o = [c for c in range(g) if c != j]
        Xb = np.c_[tr[:, o], np.ones(len(tr))]
        w = np.linalg.lstsq(Xb, tr[:, j], rcond=None)[0]
        out.append(float(((np.c_[te[:, o], np.ones(len(te))] @ w - te[:, j]) ** 2).mean()))
    return float(np.mean(out))
