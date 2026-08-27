"""Звірка нашої реалізації GENIE3/GRNBoost2 з еталонним arboreto.

Кластер на ПОТОКАХ, не процесах: дерева sklearn відпускають GIL, а процеси
на macOS перезапускають модуль і ламають запуск.
"""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import numpy as np, pandas as pd
from eval.panels import get as get_panel
from eval.metrics import auroc
from methods.trees import genie3 as our_genie3, grnboost2 as our_grnboost2
from distributed import Client, LocalCluster
from arboreto.algo import genie3 as ref_genie3, grnboost2 as ref_grnboost2


def main():
    seed = 0
    expr, idx, pos, neg, cfg = get_panel("confounder", seed, "small")
    p = expr.shape[1]
    names = [f"g{i}" for i in range(p)]
    df = pd.DataFrame(expr, columns=names)
    print(f"панель {expr.shape}", flush=True)

    def to_matrix(net):
        w = np.zeros((p, p))
        ix = {n: i for i, n in enumerate(names)}
        for tf, tgt, imp in net[["TF", "target", "importance"]].itertuples(index=False):
            w[ix[tf], ix[tgt]] = imp
        return np.maximum(w, w.T)

    def score(w, tag):
        g = lambda t: np.array([w[i, j] for i, j in idx[t]])
        return {"tag": tag, "main": auroc(g(pos), g(neg)), "control": auroc(g("D"), g("N"))}

    cluster = LocalCluster(n_workers=1, threads_per_worker=8, processes=False,
                           dashboard_address=None)
    client = Client(cluster)
    res = []
    try:
        for nm, ref_fn, our_fn in (("GENIE3", ref_genie3, our_genie3),
                                   ("GRNBoost2", ref_grnboost2, our_grnboost2)):
            t = time.time()
            net = ref_fn(expression_data=df, tf_names=names,
                         client_or_address=client, seed=seed, verbose=False)
            wr, tr = to_matrix(net), time.time() - t
            t = time.time()
            wo, to = our_fn(expr, seed=1000 * seed), time.time() - t
            r1, r2 = score(wr, f"{nm} еталон"), score(wo, f"{nm} наш")
            r1["sec"], r2["sec"] = round(tr, 1), round(to, 1)
            iu = np.triu_indices(p, 1)
            rank = lambda x: np.argsort(np.argsort(x))
            a, b = wr[iu], wo[iu]
            r1["spearman_with_other"] = float(np.corrcoef(rank(a), rank(b))[0, 1])
            top = lambda x, k=100: set(np.argsort(-x)[:k])
            r1["top100_overlap"] = len(top(a) & top(b)) / 100
            res += [r1, r2]
            print(f"{nm:11} еталон {r1['main']:.3f}/{r1['control']:.3f} ({tr:.0f}с) | "
                  f"наш {r2['main']:.3f}/{r2['control']:.3f} ({to:.0f}с) | "
                  f"spearman {r1['spearman_with_other']:.3f} | "
                  f"топ-100 {r1['top100_overlap']:.2f}", flush=True)
    finally:
        client.close()
        cluster.close()

    pathlib.Path("out").mkdir(exist_ok=True)
    json.dump(res, open("out/validate_trees.json", "w"), ensure_ascii=False, indent=1)
    print("out/validate_trees.json")


if __name__ == "__main__":
    main()
