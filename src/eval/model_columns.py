import json
import pathlib
import numpy as np
import torch
from eval.metrics import auroc
from model.interp import (attention_network, gradient_network,
                          intervention_effect, probe_effect)
from model.train import device, train
from model.data import standardize

OUT = pathlib.Path(__file__).resolve().parent.parent.parent / "out"


def pairs_of(idx, keys):
    out = []
    for t in keys:
        out += [(t, i, j) for i, j in idx[t]]
    return out


def evaluate(expr, idx, pos, neg, steps=12000, seed=0, n_cells=512, model=None):
    dev = device()
    if model is None:
        model, hist, _ = train(expr, steps=steps, seed=seed, verbose=False, dev=dev)
    zs, mu, sd = standardize(expr)

    per = attention_network(model, zs, dev, n_cells=n_cells, seed=seed, per_layer=True)
    nets = {"увага": np.maximum(per.mean(axis=0), per.mean(axis=0).T)}
    for li in range(per.shape[0]):
        nets[f"увага-шар{li + 1}"] = per[li]
    nets["градієнт"] = gradient_network(model, zs, dev, n_cells=n_cells, seed=seed)

    keys = [k for k in ("D", "C", "T", "K", "N") if k in idx]
    plist = pairs_of(idx, keys)
    pl = [(i, j) for _, i, j in plist]
    eff = intervention_effect(model, zs, dev, pl, n_cells=n_cells, seed=seed)
    pr = probe_effect(model, zs, dev, pl, n_cells=min(1024, 2 * n_cells), seed=seed)
    eff_by_type = {}
    for t, i, j in plist:
        eff_by_type.setdefault(t, []).append(eff[(i, j)])

    res = {"methods": {}, "model_truth": {}}
    for nm, w in nets.items():
        g = lambda t: np.array([w[i, j] for i, j in idx[t]])
        res["methods"][nm] = {
            "vs_planted": auroc(g(pos), g(neg)),
            "control": auroc(g("D"), g("N")) if "D" in idx else None,
            "neg_vs_null": auroc(g(neg), g("N")) if "N" in idx else None,
        }
    res["model_truth"] = {
        "vs_planted": auroc(np.array(eff_by_type[pos]), np.array(eff_by_type[neg])),
        "neg_vs_null": (auroc(np.array(eff_by_type[neg]), np.array(eff_by_type["N"]))
                        if "N" in eff_by_type else None),
        "control": (auroc(np.array(eff_by_type["D"]), np.array(eff_by_type["N"]))
                    if "D" in eff_by_type and "N" in eff_by_type else None),
        "means": {t: float(np.mean(v)) for t, v in eff_by_type.items()},
    }
    pr_by_type = {}
    for t, i, j in plist:
        pr_by_type.setdefault(t, []).append(pr[(i, j)])
    res["methods"]["проба"] = {
        "vs_planted": auroc(np.array(pr_by_type[pos]), np.array(pr_by_type[neg])),
        "control": (auroc(np.array(pr_by_type["D"]), np.array(pr_by_type["N"]))
                    if "D" in pr_by_type and "N" in pr_by_type else None),
        "neg_vs_null": (auroc(np.array(pr_by_type[neg]), np.array(pr_by_type["N"]))
                        if "N" in pr_by_type else None),
        "spearman_vs_model_truth": float(np.corrcoef(
            np.argsort(np.argsort(np.array([pr[(i, j)] for _, i, j in plist]))),
            np.argsort(np.argsort(np.array([eff[(i, j)] for _, i, j in plist]))))[0, 1]),
        "means": {t: float(np.mean(v)) for t, v in pr_by_type.items()},
    }
    for nm, w in nets.items():
        wv = np.array([w[i, j] for _, i, j in plist])
        ev = np.array([eff[(i, j)] for _, i, j in plist])
        res["methods"][nm]["spearman_vs_model_truth"] = float(
            np.corrcoef(np.argsort(np.argsort(wv)), np.argsort(np.argsort(ev)))[0, 1])
    return res, model


def save(report, name="model_columns.json"):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=1,
                                       default=float), encoding="utf-8")
    return OUT / name
