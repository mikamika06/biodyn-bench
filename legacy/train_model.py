import argparse, sys, pathlib, time
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
import numpy as np, torch
from eval.panels import get as get_panel
from model.train import train, linear_ceiling

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--genes", default="small", choices=["full", "small"])
ap.add_argument("--panel", default="confounder")
ap.add_argument("--link", default="linear")
ap.add_argument("--lib", type=float, default=None,
                help="глибина бібліотеки; без неї ідеальні лічильники")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default=None)
ap.add_argument("--out", default="model_conf.pt")
a = ap.parse_args()

ckw = None if a.lib is None else {"lib_loc": a.lib}
expr, idx, pos, neg, cfg = get_panel(a.panel, a.seed, a.genes, counts_kw=ckw, link=a.link)
ceil = linear_ceiling(expr)
print(f"панель {a.panel} {expr.shape}  звʼязок {a.link}  нулі {float((expr==0).mean()):.3f}", flush=True)
print(f"лінійна стеля MSE {ceil:.4f}   база 1.0", flush=True)
dev = torch.device(a.device) if a.device else None
t = time.time()
m, hist, norm = train(expr, steps=a.steps, d=a.d, n_layers=a.layers, lr=a.lr,
                      batch=a.batch, seed=a.seed, dev=dev)
print(f"час {time.time()-t:.0f} с   найкраща валід MSE {min(h[2] for h in hist):.4f}")
torch.save({"state": m.state_dict(), "hist": hist, "seed": a.seed, "cfg": cfg,
            "panel": a.panel, "size": a.genes, "link": a.link, "lib": a.lib,
            "n_genes": expr.shape[1], "d": a.d, "layers": a.layers,
            "norm": norm, "ceiling": ceil}, f"out/{a.out}")
print(f"out/{a.out}")
