import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np, torch
from eval.panels import get as get_panel
from eval.metrics import auroc
from model.data import standardize
from model.interp import patching_effect
from model.train import device
from model.transformer import GeneTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=int, default=384)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--out", default="patching.json")
a = ap.parse_args()

PANELS = ["confounder", "confounder_hidden", "collider", "chain", "chain_hidden"]
SEEDS = [int(x) for x in a.seeds.split(",")]
dev = device()
report = {}


def one(tag, name):
    ck = torch.load(f"out/model_{tag}.pt", map_location="cpu", weights_only=False)
    m = GeneTransformer(ck["n_genes"], ck["d"], ck["layers"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    expr, idx, pos, neg, _ = get_panel(name, ck["seed"], ck["size"], link=ck["link"])
    zs, _, _ = standardize(expr)
    types = [t for t in ("D", "C", "T", "K", "N") if t in idx]
    plist = [(t, i, j) for t in types for i, j in idx[t]]
    res = patching_effect(m, zs, dev, [(i, j) for _, i, j in plist],
                          n_cells=a.cells, seed=ck["seed"])
    nl = ck["layers"]
    prof = {t: {L: [] for L in range(nl + 1)} for t in types}
    tot = {t: [] for t in types}
    for t, i, j in plist:
        for L in range(nl + 1):
            prof[t][L].append(res[(i, j)][L])
        tot[t].append(res[(i, j)]["_total"])
    score = {(i, j): max(res[(i, j)][L] for L in range(1, nl)) for _, i, j in plist}
    g = lambda t: np.array([score[(i, j)] for i, j in idx[t]])
    return {
        "layers": nl,
        "profile": {t: {str(L): float(np.mean(v)) for L, v in d.items()}
                    for t, d in prof.items()},
        "total": {t: float(np.mean(v)) for t, v in tot.items()},
        "auroc_vs_planted": auroc(g(pos), g(neg)),
        "control": auroc(g("D"), g("N")) if "D" in idx and "N" in idx else None,
    }


for name in PANELS:
    for sd in SEEDS:
        tag = name if sd == 0 else f"{name}_s{sd}"
        if not pathlib.Path(f"out/model_{tag}.pt").exists():
            continue
        report[tag] = one(tag, name)
        r = report[tag]
        print(f"{tag:24} AUROC {r['auroc_vs_planted']:.3f}  "
              f"контроль {r['control']}  "
              + "  ".join(f"{t} L1 {r['profile'][t]['1']:.3f}" for t in r["profile"]),
              flush=True)

pathlib.Path("out").mkdir(exist_ok=True)
json.dump(report, open(f"out/{a.out}", "w"), ensure_ascii=False, indent=1)

print()
print(f"{'панель':24}{'AUROC':>16}{'контроль':>16}{'n':>4}")
print("-" * 60)
for name in PANELS:
    rs = [v for k, v in report.items() if k == name or k.startswith(name + "_s")]
    if not rs:
        continue
    a1 = np.array([r["auroc_vs_planted"] for r in rs])
    c1 = np.array([r["control"] for r in rs if r["control"] is not None])
    print(f"{name:24}{a1.mean():>10.3f}±{a1.std():.3f}"
          f"{c1.mean():>10.3f}±{c1.std():.3f}{len(rs):>4}")
print(f"\nout/{a.out}")
