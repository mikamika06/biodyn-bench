"""Еталонний прогін arboreto в ізольованому середовищі. Пише матриці у .npy."""
import sys, json, time
import numpy as np, pandas as pd
from distributed import Client, LocalCluster
from arboreto.algo import genie3, grnboost2

expr = np.load("out/ref_expr.npy")
p = expr.shape[1]
names = [f"g{i}" for i in range(p)]
df = pd.DataFrame(expr, columns=names)


def to_matrix(net):
    w = np.zeros((p, p))
    ix = {n: i for i, n in enumerate(names)}
    for tf, tgt, imp in net[["TF", "target", "importance"]].itertuples(index=False):
        w[ix[tf], ix[tgt]] = imp
    return np.maximum(w, w.T)


def main():
    cluster = LocalCluster(n_workers=1, threads_per_worker=4, processes=False,
                           dashboard_address=None)
    client = Client(cluster)
    meta = {}
    try:
        for nm, fn in (("GENIE3", genie3), ("GRNBoost2", grnboost2)):
            t = time.time()
            net = fn(expression_data=df, tf_names=names, client_or_address=client,
                     seed=0, verbose=False)
            np.save(f"out/ref_{nm}.npy", to_matrix(net))
            meta[nm] = round(time.time() - t, 1)
            print(f"{nm} {meta[nm]} с", flush=True)
    finally:
        client.close(); cluster.close()
    json.dump(meta, open("out/ref_trees_meta.json", "w"))


if __name__ == "__main__":
    main()
