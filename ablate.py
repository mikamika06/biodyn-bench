"""Задача 106: ВИЧЕРПНА абляція. Кожна голова, кожен нейрон, по черзі.

На великих моделях це неможливо за обсягом обчислень, тому поле бере відбір
кандидатів — тим самим методом, вірність якого й треба перевірити. Тут модель
мала, перебір повний, і можна виміряти, наскільки відбір спотворює висновок.
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np, torch
from contextlib import contextmanager
from sim.grid import SPEC, matched
from eval.metrics import auroc
from model.data import standardize, CellDataset, split
from model.interp import (attention_network, gradient_network,
                          head_attention, intervention_effect)
from model.train import train, device, linear_ceiling


@contextmanager
def ablated_many(model, units):
    """Вимикає кілька одиниць одночасно."""
    saved = []
    for kind, L, i in units:
        lay = model.encoder.layers[L]
        if kind == "head":
            w = lay.self_attn.out_proj.weight
            dh = model.d // lay.self_attn.num_heads
            sl = slice(i * dh, (i + 1) * dh)
            saved.append((w, sl, w.data[:, sl].clone()))
            w.data[:, sl] = 0.0
        else:
            w = lay.linear2.weight
            saved.append((w, i, w.data[:, i].clone()))
            w.data[:, i] = 0.0
    try:
        yield
    finally:
        for w, sl, old in saved:
            w.data[:, sl] = old


@contextmanager
def ablated(model, kind, layer, idx):
    """Тимчасово вимикає одну одиницю обчислення.

    head    зануляє стовпці out_proj, що належать цій голові — внесок голови
            у вихід блока стає нулем, решта голів не зачеплена
    neuron  зануляє стовпець linear2, тобто один нейрон мережі після уваги
    """
    L = model.encoder.layers[layer]
    if kind == "head":
        w = L.self_attn.out_proj.weight
        dh = model.d // L.self_attn.num_heads
        sl = slice(idx * dh, (idx + 1) * dh)
        old = w.data[:, sl].clone()
        w.data[:, sl] = 0.0
        try:
            yield
        finally:
            w.data[:, sl] = old
    else:
        w = L.linear2.weight
        old = w.data[:, idx].clone()
        w.data[:, idx] = 0.0
        try:
            yield
        finally:
            w.data[:, idx] = old


@torch.no_grad()
def val_mse(model, dlv, dev):
    model.eval()
    v = [(((model(x.to(dev), m.to(dev)) - x.to(dev))[m.to(dev)]) ** 2).mean().item()
         for x, m in dlv]
    return float(np.mean(v))


@torch.no_grad()
def pair_auroc(model, zs, dev, pairs, sp, n_cells, seed):
    eff = intervention_effect(model, zs, dev, [p for t in (sp["pos"], sp["neg"])
                                               for p in pairs[t]],
                              n_cells=n_cells, seed=seed)
    g = lambda t: [eff[(i, j)] for i, j in pairs[t]]
    return auroc(g(sp["pos"]), g(sp["neg"]))


ap = argparse.ArgumentParser()
ap.add_argument("--structure", default="confounder")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--cells", type=int, default=256)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--neurons", type=int, default=0, help="0 = всі")
ap.add_argument("--link", default="linear")
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--out", default="ablation.json")
a = ap.parse_args()

SMALL = {"n_cells": 5000, "n_struct": 10, "n_direct": 10, "n_ref": 10, "n_null": 10}
dev = device()
sp = SPEC[a.structure]
key = f"{a.structure}|{a.seed}|{a.link}|{a.rho}|{a.d}|{a.layers}"
dst = pathlib.Path("out") / a.out
report = json.loads(dst.read_text()) if dst.exists() else {}

expr, pairs, k, tgt = matched(a.structure, a.seed, SMALL, link=a.link, rho=a.rho)
zs, _, _ = standardize(expr)
te = split(expr, seed=a.seed)[1]
mu, sd = expr.mean(0), expr.std(0); sd = np.where(sd == 0, 1.0, sd)
dlv = torch.utils.data.DataLoader(CellDataset((te - mu) / sd, 0.15, a.seed + 1),
                                  batch_size=128)
print(f"панель {expr.shape[1]} генів, тренування…", flush=True)
model, hist, _ = train(expr, steps=a.steps, d=a.d, n_layers=a.layers,
                       n_heads=a.heads, seed=a.seed, verbose=False, dev=dev)

base_mse = val_mse(model, dlv, dev)
base_auc = pair_auroc(model, zs, dev, pairs, sp, a.cells, a.seed)
print(f"база: MSE {base_mse:.4f}  AUROC {base_auc:.3f}  стеля {linear_ceiling(expr):.4f}",
      flush=True)

d_ff = model.encoder.layers[0].linear1.out_features
n_neu = a.neurons or d_ff
units = ([("head", L, h) for L in range(a.layers) for h in range(a.heads)]
         + [("neuron", L, i) for L in range(a.layers) for i in range(n_neu)])

t0, rows = time.time(), []
for n, (kind, L, i) in enumerate(units):
    with ablated(model, kind, L, i):
        m = val_mse(model, dlv, dev)
        auc = pair_auroc(model, zs, dev, pairs, sp, a.cells, a.seed)
    rows.append({"kind": kind, "layer": L, "idx": i,
                 "d_mse": m - base_mse, "auroc": auc})
    if n % 200 == 0:
        print(f"  {n}/{len(units)}  {time.time()-t0:.0f} с", flush=True)

# ПАРНА АБЛЯЦІЯ ГОЛІВ: «ефекту нема» не означає «компонент неважливий».
# Якщо дві голови надлишкові, вимкнення кожної окремо не дає деградації,
# а вимкнення обох — дає. Саме цей крок пропускають, коли роблять висновок
# з одиночної абляції (пор. p=0.10 на абляції голів у 2602.17532).
head_units = [("head", L, h) for L in range(a.layers) for h in range(a.heads)]
single = {u: r["d_mse"] for u, r in zip(head_units, rows[:len(head_units)])}
pairs_res = []
for x in range(len(head_units)):
    for y in range(x + 1, len(head_units)):
        with ablated_many(model, [head_units[x], head_units[y]]):
            m2 = val_mse(model, dlv, dev)
            a2 = pair_auroc(model, zs, dev, pairs, sp, a.cells, a.seed)
        dx, dy = single[head_units[x]], single[head_units[y]]
        dxy = m2 - base_mse
        pairs_res.append({"a": list(head_units[x]), "b": list(head_units[y]),
                          "d_a": dx, "d_b": dy, "d_ab": dxy, "auroc": a2,
                          "redundancy": float(max(dx, dy) / dxy) if dxy > 1e-9 else None,
                          "interaction": dxy - dx - dy})
red = [r["redundancy"] for r in pairs_res if r["redundancy"] is not None]
inter = [r["interaction"] for r in pairs_res]
pair_stats = {"n_pairs": len(pairs_res),
              "max_single": float(max(single.values())),
              "max_pair": float(max(r["d_ab"] for r in pairs_res)),
              "median_redundancy": float(np.median(red)) if red else None,
              "frac_super_additive": float(np.mean(np.array(inter) > 0)),
              "median_interaction_share": float(np.median(
                  [abs(r["interaction"]) / max(abs(r["d_ab"]), 1e-9) for r in pairs_res]))}

# ВІДБІР-ЯК-У-ПОЛІ: голови ранжують за тим, наскільки їхня увага збігається
# з відомим списком ребер. Саме так робить Кендюхов (TRRUST-ranked heads).
ha = head_attention(model, zs, dev, n_cells=a.cells, seed=a.seed)
sel = []
for L in range(a.layers):
    for h in range(a.heads):
        w = ha[L, h]
        sel.append(auroc([w[i, j] for i, j in pairs["D"]],
                         [w[i, j] for i, j in pairs["N"]]))
sel = np.array(sel)

heads = [r for r in rows if r["kind"] == "head"]
exh_mse = np.array([r["d_mse"] for r in heads])
exh_auc = np.array([base_auc - r["auroc"] for r in heads])
rank = lambda v: np.argsort(np.argsort(-v))
spear = lambda a_, b_: float(np.corrcoef(rank(a_), rank(b_))[0, 1])

top = lambda v, k: set(np.argsort(-v)[:k])
recall = {k: len(top(sel, k) & top(exh_auc, k)) / k for k in (1, 2, 3)}
sel_cmp = {"selection_auroc": sel.tolist(),
           "spearman_sel_vs_mse": spear(sel, exh_mse),
           "spearman_sel_vs_causal": spear(sel, exh_auc),
           "spearman_mse_vs_causal": spear(exh_mse, exh_auc),
           "recall_at_k": recall}

report[key] = {"structure": a.structure, "base_mse": base_mse, "base_auroc": base_auc,
               "n_units": len(units), "d_ff": d_ff, "sec": round(time.time() - t0, 1),
               "selection": sel_cmp, "units": rows}
dst.parent.mkdir(exist_ok=True)
dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

neus = sorted([r for r in rows if r["kind"] == "neuron"], key=lambda r: -r["d_mse"])
print(f"\nголови ({len(heads)}):   відбір = AUROC уваги проти списку ребер")
for n_, r in enumerate(heads):
    print(f"  шар {r['layer']} голова {r['idx']}  ΔMSE {r['d_mse']:+.4f}  "
          f"причинна AUROC {r['auroc']:.3f}  відбір {sel[n_]:.3f}")
print(f"\nранговий збіг:")
print(f"  відбір проти ΔMSE       {sel_cmp['spearman_sel_vs_mse']:+.3f}")
print(f"  відбір проти причинної  {sel_cmp['spearman_sel_vs_causal']:+.3f}")
print(f"  ΔMSE проти причинної    {sel_cmp['spearman_mse_vs_causal']:+.3f}")
print(f"  recall@k відбору        {sel_cmp['recall_at_k']}")
print(f"\nнейрони: топ-10 із {len(neus)}")
for r in neus[:10]:
    print(f"  шар {r['layer']} нейрон {r['idx']:4d}  ΔMSE {r['d_mse']:+.4f}")
dm = np.array([r["d_mse"] for r in neus])
print(f"\nΔMSE нейронів: max {dm.max():+.4f}  медіана {np.median(dm):+.5f}  "
      f"частка >1% бази {(dm > 0.01*base_mse).mean():.3f}")
print(f"\n{time.time()-t0:.0f} с   out/{a.out}")
