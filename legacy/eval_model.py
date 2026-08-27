import argparse, json, sys, pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np, torch
from eval.panels import get as get_panel
from model.transformer import GeneTransformer
from model.train import device
from eval.model_columns import evaluate, save

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="model_small.pt")
ap.add_argument("--cells", type=int, default=512)
ap.add_argument("--out", default="model_columns.json")
a = ap.parse_args()

ck = torch.load(f"out/{a.ckpt}", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
dev = device()
m = GeneTransformer(ck["n_genes"], ck["d"], ck["layers"]).to(dev)
m.load_state_dict(ck["state"])
m.eval()
print(f"модель: панель {ck['panel']}, {ck['n_genes']} генів, {ck['layers']} шари, d={ck['d']}")
print(f"валід MSE {min(h[2] for h in ck['hist']):.4f}   лінійна стеля {ck['ceiling']:.4f}\n")

expr, idx, pos, neg, _ = get_panel(ck["panel"], ck["seed"], ck["size"], link=ck["link"])
res, _ = evaluate(expr, idx, pos, neg, model=m, n_cells=a.cells, seed=ck["seed"])

print(f"{'метод':16}{'проти закладеної':>20}{'контроль':>12}{'збіг з істиною моделі':>24}")
print("-"*72)
order = [n for n in res["methods"] if n.startswith("увага")] + ["градієнт", "проба"]
for nm in order:
    v = res["methods"].get(nm)
    if v is None:
        continue
    print(f"{nm:16}{v['vs_planted']:>20.3f}{v['control']:>12.3f}"
          f"{v['spearman_vs_model_truth']:>24.3f}")
mt = res["model_truth"]
print(f"\n{'ІСТИНА МОДЕЛІ':16}{mt['vs_planted']:>20.3f}{mt['control']:>12.3f}")
print(f"\nсередні ефекти втручання: " +
      "  ".join(f"{t} {v:.4f}" for t, v in mt["means"].items()))
save({ck["panel"]: res}, a.out)
print(f"\nout/{a.out}")
