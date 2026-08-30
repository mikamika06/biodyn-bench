import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.model import load_data
from scgpt_collider.pairs import pair_table
from scgpt_collider.prep import load_trrust
from scgpt_collider.stats import evaluate

OUT = Path("/Users/macbook/biodyn-bench/out/scgpt_collider")


def main(min_co=10, n_rewire=200, tag="attn"):
    rng = np.random.default_rng(0)
    genes, gene_ids, xb, xl, is_tf = load_data()
    present = xb > 0
    df, tfs, tgt, tgt_in = pair_table(genes, present, min_co=min_co)
    ref = load_trrust()
    A = np.load(OUT / "attn_layers.npy")
    AL = np.load(OUT / "attn_layers_lennorm.npy")
    mean_expr = xl.mean(0)
    df["expr"] = np.log(mean_expr[df.ia.values] + 1e-3) + np.log(mean_expr[df.ib.values] + 1e-3)
    ia, ib = df.ia.values, df.ib.values
    M = A.mean(0)
    ML = AL.mean(0)
    df["attn_ab"] = M[ia, ib]
    df["attn_ba"] = M[ib, ia]
    df["attn_sym"] = 0.5 * (df.attn_ab + df.attn_ba)
    df["attn_max"] = np.maximum(df.attn_ab, df.attn_ba)
    df["attn_len_sym"] = 0.5 * (ML[ia, ib] + ML[ib, ia])
    for li in range(A.shape[0]):
        df[f"attn_L{li}"] = 0.5 * (A[li][ia, ib] + A[li][ib, ia])
    df.to_csv(OUT / f"test1_pairs_{tag}.csv", index=False)
    res = {"min_co": min_co, "n_pairs": int(len(df)), "n_tfs": len(tfs)}
    res["main_sym"] = evaluate(df, "attn_sym", ref, rng, n_rewire=n_rewire)
    res["ordered_ab"] = evaluate(df, "attn_ab", ref, rng, n_rewire=0, n_perm=300)
    res["max"] = evaluate(df, "attn_max", ref, rng, n_rewire=0, n_perm=300)
    res["len_norm_sym"] = evaluate(df, "attn_len_sym", ref, rng, n_rewire=0, n_perm=300)
    res["shared_in_geneset_label"] = evaluate(df, "attn_sym", None, rng, n_rewire=0, n_perm=300, label="shared_in")
    res["layers"] = {}
    for li in range(A.shape[0]):
        r = evaluate(df, f"attn_L{li}", None, rng, n_rewire=0, n_perm=300)
        res["layers"][li] = {k: r.get(k) for k in ("auroc", "auroc_ci95", "strat_null_mean", "strat_p_ge", "median_A", "median_B")}
    hi = df[df.n_co >= 50]
    res["high_cooccurrence_ge50"] = evaluate(hi, "attn_sym", ref, rng, n_rewire=100, n_perm=500)
    gi = {g: i for i, g in enumerate(genes)}
    cnt = np.load(OUT / "attn_count.npy")
    tf_idx = np.array([gi[t] for t in tfs])
    targets = sorted({c for t in tfs for c in tgt_in[t]})
    tg_idx = np.array([gi[c] for c in targets])
    rows = []
    for t in tfs:
        for c in targets:
            if c == t or cnt[gi[t], gi[c]] < min_co:
                continue
            rows.append({"a": t, "b": c, "edge": int(c in tgt[t]), "attn_tf_to_tgt": M[gi[t], gi[c]], "attn_tgt_to_tf": M[gi[c], gi[t]], "attn_sym": 0.5 * (M[gi[t], gi[c]] + M[gi[c], gi[t]]), "expr": np.log(mean_expr[gi[t]] + 1e-3) + np.log(mean_expr[gi[c]] + 1e-3), "deg": len(tgt[t]), "n_co": int(cnt[gi[t], gi[c]])})
    ed = pd.DataFrame(rows)
    ed.to_csv(OUT / f"test1_edges_{tag}.csv", index=False)
    res["positive_control_direct_edges"] = {k: evaluate(ed, k, None, rng, n_rewire=0, n_perm=300, label="edge") for k in ("attn_sym", "attn_tf_to_tgt", "attn_tgt_to_tf")}
    json.dump(res, open(OUT / f"test1_{tag}.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in v if kk in ("auroc", "auroc_ci95", "n", "nA", "nB", "mwu_p", "rewire_null_mean", "rewire_null_ci95", "rewire_p_ge", "strat_null_mean", "strat_p_ge", "median_A", "median_B", "auroc_deg_predicts_A", "ols", "dose", "auroc_direct_vs_rest", "n_direct", "spearman_score_nshared")}) for k, v in res.items() if k not in ("layers", "positive_control_direct_edges")}, indent=1))
    print("layers", {li: (round(v["auroc"], 3), round(v["strat_null_mean"], 3), v["strat_p_ge"]) for li, v in res["layers"].items()})
    print("pos control", {k: (round(v["auroc"], 3), v["n"], v["nA"], round(v["strat_null_mean"], 3), v["strat_p_ge"]) for k, v in res["positive_control_direct_edges"].items()})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
