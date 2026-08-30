import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/Users/macbook/biodyn-bench")
TRRUST = Path("/Users/macbook/claimbase/out/realdata/bmf/trrust_human.tsv")
VOCAB = ROOT / "data/scgpt_whole_human/scGPT_human/vocab.json"
OUT = ROOT / "data/scgpt_collider"


def load_trrust():
    tr = pd.read_csv(TRRUST, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    return tr[["tf", "target"]].drop_duplicates().reset_index(drop=True)


def bin_row(v, n_bins=51):
    from scgpt.preprocess import binning
    return binning(v.astype(np.float64), n_bins).astype(np.float32)


def main(n_hvg=1200, seed=0):
    vocab = json.load(open(VOCAB))
    tr = load_trrust()
    a = sc.read_h5ad(ROOT / "data/pbmc3k_raw.h5ad")
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    a = a[:, a.var_names.isin(set(vocab))].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts")
    tfs = set(tr.tf)
    keep = a.var["highly_variable"].to_numpy() | a.var_names.isin(tfs)
    a = a[:, keep].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    a.layers["X_log1p"] = X.astype(np.float32)
    a.var["is_tf"] = a.var_names.isin(tfs)
    a.var["vocab_id"] = [vocab[g] for g in a.var_names]
    np.random.seed(seed)
    a.layers["X_binned"] = np.stack([bin_row(r) for r in X]).astype(np.float32)
    rng = np.random.default_rng(seed)
    a.obs["order"] = rng.permutation(a.n_obs)
    a = a[np.argsort(a.obs["order"].to_numpy())].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(OUT / "pbmc3k_prepped.h5ad")
    genes = list(a.var_names)
    tf_in = [g for g in genes if g in tfs]
    tgt_of = tr.groupby("tf").target.apply(set).to_dict()
    gset = set(genes)
    info = {
        "n_cells": int(a.n_obs), "n_genes": len(genes), "n_hvg": int(a.var["highly_variable"].sum()),
        "n_tf": len(tf_in), "n_tf_trrust_total": len(tfs),
        "n_tf_with_target_in_geneset": sum(1 for t in tf_in if tgt_of[t] & gset),
        "mean_nonzero_per_cell": float((X > 0).sum(1).mean()),
        "tf_mean_expr_median": float(np.median(X[:, a.var["is_tf"].to_numpy()].mean(0))),
        "tf_frac_cells_nonzero_median": float(np.median((X[:, a.var["is_tf"].to_numpy()] > 0).mean(0))),
    }
    json.dump(info, open(OUT / "prep_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:]])
