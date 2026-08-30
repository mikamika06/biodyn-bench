import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.engine import Engine
from scgpt_collider.pairs import pair_table
from scgpt_collider.prep import load_trrust
from scgpt_collider.stats import evaluate, strata

OUT = Path("/Users/macbook/biodyn-bench/out/scgpt_collider")


def sample_matched(df, n_a, rng):
    key = strata(np.log(df.deg.values), df.expr.values, np.log(df.n_co.values), q=4)
    df = df.assign(key=key)
    A = df[df.shared == 1]
    A = A.sample(min(n_a, len(A)), random_state=int(rng.integers(1 << 30)))
    B = df[df.shared == 0]
    used = set()
    picks = []
    for k, cnt in A.key.value_counts().items():
        cand = B[(B.key == k) & (~B.index.isin(used))]
        take = cand.sample(min(cnt, len(cand)), random_state=int(rng.integers(1 << 30)))
        used |= set(take.index)
        picks.append(take)
    return pd.concat([A] + picks)


def build_jobs(eng, pairs, n_cells, rng, mask_frac=0.25):
    rows, meta = [], []
    for r in pairs.itertuples():
        for a, b in ((r.ia, r.ib), (r.ib, r.ia)):
            cells = np.where(eng.present[:, a] & eng.present[:, b])[0]
            cells = rng.choice(cells, min(n_cells, len(cells)), replace=False)
            for c in cells:
                idx = np.where(eng.present[c])[0]
                others = idx[(idx != a) & (idx != b)]
                k = int(mask_frac * len(idx))
                extra = set(rng.choice(others, min(k, len(others)), replace=False).tolist())
                mask = extra | {a}
                pool = np.where(eng.present[:, b] & (eng.xb[:, b] != eng.xb[c, b]))[0]
                if len(pool) == 0:
                    continue
                d = int(rng.choice(pool))
                rows.append((c, a, b, mask, {}))
                rows.append((c, a, b, mask, {b: eng.xb[d, b]}))
                meta.append({"pair": r.Index, "a": a, "b": b, "cell": int(c), "donor": d, "v_b": float(eng.xb[c, b]), "v_b_donor": float(eng.xb[d, b]), "v_a": float(eng.xb[c, a])})
    return rows, pd.DataFrame(meta)


def summarize(meta, preds, att_rp, att_pr, nl):
    p0, p1 = preds[0::2], preds[1::2]
    meta = meta.copy()
    meta["p0"] = p0
    meta["p1"] = p1
    meta["eff"] = np.abs(p1 - p0)
    meta["eff_rel"] = np.abs(p1 - p0) / np.abs(meta.v_b_donor - meta.v_b)
    for li in range(nl):
        meta[f"att_ab_L{li}"] = att_rp[0::2, li]
    meta["att_ab"] = att_rp[0::2].mean(1)
    meta["att_ba"] = att_pr[0::2].mean(1)
    return meta


def stats(tag="int", n_edges=400, n_cells=20, seed=0, fname="pbmc3k_prepped.h5ad", min_co=20, do_edges=True):
    rng = np.random.default_rng(seed)
    eng = Engine(want_attn=True, fname=fname)
    df, tfs, tgt, tgt_in = pair_table(eng.genes, eng.present, min_co=min_co)
    mean_expr = eng.xl.mean(0)
    res_df = pd.read_csv(OUT / f"test2_pairs_{tag}.csv")
    ref = load_trrust()
    res = {"n_pairs": int(len(res_df)), "mask_frac_extra": 0.25, "n_rows": int(res_df.n_cells.sum() * 2)}
    res["main_eff"] = evaluate(res_df, "eff", ref, rng, n_rewire=200)
    res["eff_median_cells"] = evaluate(res_df, "eff_med", None, rng, n_rewire=0, n_perm=500)
    res["eff_rel"] = evaluate(res_df, "eff_rel", None, rng, n_rewire=0, n_perm=500)
    res["eff_max_dir"] = evaluate(res_df, "eff_max", None, rng, n_rewire=0, n_perm=500)
    res["attention_same_cells_masked_A"] = evaluate(res_df, "att", ref, rng, n_rewire=100, n_perm=500)
    res["shared_in_geneset_label"] = evaluate(res_df, "eff", None, rng, n_rewire=0, n_perm=500, label="shared_in")
    json.dump(res, open(OUT / f"test2_{tag}.json", "w"), indent=1)
    print("main stats saved", flush=True)
    if not do_edges:
        return
    gi = {g: i for i, g in enumerate(eng.genes)}
    targets = sorted({c for t in tfs for c in tgt_in[t]})
    tgt_idx = np.array([gi[c] for c in targets])
    tf_idx = np.array([gi[t] for t in tfs])
    P = eng.present.astype(np.float32)
    co = P[:, tf_idx].T @ P[:, tgt_idx]
    erows = []
    for i, t in enumerate(tfs):
        for j, c in enumerate(targets):
            if c != t and co[i, j] >= 20:
                erows.append({"a": c, "b": t, "ia": gi[c], "ib": gi[t], "edge": int(c in tgt[t]), "n_co": int(co[i, j]), "deg": len(tgt[t]), "expr": np.log(mean_expr[gi[t]] + 1e-3) + np.log(mean_expr[gi[c]] + 1e-3)})
    ed = pd.DataFrame(erows)
    ed["key"] = strata(np.log(ed.deg.values), ed.expr.values, np.log(ed.n_co.values), q=4)
    E1 = ed[ed.edge == 1].sample(min(n_edges, int(ed.edge.sum())), random_state=1)
    picks = []
    for k, cnt in E1.key.value_counts().items():
        cand = ed[(ed.edge == 0) & (ed.key == k)]
        picks.append(cand.sample(min(cnt, len(cand)), random_state=2))
    E = pd.concat([E1] + picks)
    erows2, emeta = [], []
    for r in E.itertuples():
        a, b = r.ia, r.ib
        cells = np.where(eng.present[:, a] & eng.present[:, b])[0]
        cells = rng.choice(cells, min(n_cells, len(cells)), replace=False)
        for c in cells:
            idx = np.where(eng.present[c])[0]
            others = idx[(idx != a) & (idx != b)]
            extra = set(rng.choice(others, min(int(0.25 * len(idx)), len(others)), replace=False).tolist())
            pool = np.where(eng.present[:, b] & (eng.xb[:, b] != eng.xb[c, b]))[0]
            if len(pool) == 0:
                continue
            d = int(rng.choice(pool))
            erows2.append((c, a, b, extra | {a}, {}))
            erows2.append((c, a, b, extra | {a}, {b: eng.xb[d, b]}))
            emeta.append({"pair": r.Index, "cell": int(c), "v_b": float(eng.xb[c, b]), "v_b_donor": float(eng.xb[d, b])})
    print("edge rows", len(erows2), flush=True)
    ep, ea, eb = eng.run(erows2)
    em = pd.DataFrame(emeta)
    em["eff"] = np.abs(ep[1::2] - ep[0::2])
    em["att"] = ea[0::2].mean(1)
    eg = em.groupby("pair").agg(eff=("eff", "mean"), att=("att", "mean")).reset_index().set_index("pair")
    E2 = E.join(eg, how="inner")
    E2.to_csv(OUT / f"test2_edges_{tag}.csv", index=False)
    res["positive_control_direct_edges"] = {"n": int(len(E2)), "n_edges": int(E2.edge.sum()), "n_cells": n_cells, "eff": evaluate(E2, "eff", None, rng, n_rewire=0, n_perm=500, label="edge"), "att": evaluate(E2, "att", None, rng, n_rewire=0, n_perm=500, label="edge")}
    json.dump(res, open(OUT / f"test2_{tag}.json", "w"), indent=1)
    keys = ("auroc", "auroc_ci95", "n", "nA", "nB", "mwu_p", "median_A", "median_B", "mean_A", "mean_B", "rewire_null_mean", "rewire_null_ci95", "rewire_p_ge", "strat_null_mean", "strat_p_ge", "ols", "dose", "spearman_score_nshared", "auroc_direct_vs_rest", "n_direct")
    for k, v in res.items():
        if isinstance(v, dict) and "auroc" in v:
            print(k, json.dumps({kk: v[kk] for kk in keys if kk in v}))
    pc = res["positive_control_direct_edges"]
    print("pos control eff", json.dumps({kk: pc["eff"][kk] for kk in keys if kk in pc["eff"]}))
    print("pos control att", json.dumps({kk: pc["att"][kk] for kk in keys if kk in pc["att"]}))


def main(n_a=1500, n_cells=40, tag="int", seed=0, fname="pbmc3k_prepped.h5ad", min_co=20):
    rng = np.random.default_rng(seed)
    eng = Engine(want_attn=True, fname=fname)
    df, tfs, tgt, tgt_in = pair_table(eng.genes, eng.present, min_co=min_co)
    mean_expr = eng.xl.mean(0)
    df["expr"] = np.log(mean_expr[df.ia.values] + 1e-3) + np.log(mean_expr[df.ib.values] + 1e-3)
    pairs = sample_matched(df, n_a, rng)
    print("pairs", len(pairs), "A", int(pairs.shared.sum()), "B", int((1 - pairs.shared).sum()), flush=True)
    rows, meta = build_jobs(eng, pairs, n_cells, rng)
    print("rows", len(rows), flush=True)
    preds, att_rp, att_pr = eng.run(rows)
    meta = summarize(meta, preds, att_rp, att_pr, eng.nl)
    meta.to_csv(OUT / f"test2_cells_{tag}.csv", index=False)
    g = meta.groupby(["pair", "a", "b"]).agg(eff=("eff", "mean"), eff_med=("eff", "median"), eff_rel=("eff_rel", "mean"), att_ab=("att_ab", "mean"), att_ba=("att_ba", "mean"), n_cells=("eff", "size")).reset_index()
    per_pair = g.groupby("pair").agg(eff=("eff", "mean"), eff_med=("eff_med", "mean"), eff_rel=("eff_rel", "mean"), eff_max=("eff", "max"), att=("att_ab", "mean"), n_cells=("n_cells", "sum")).reset_index().set_index("pair")
    res_df = pairs.join(per_pair, how="inner")
    res_df.to_csv(OUT / f"test2_pairs_{tag}.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "stats":
        stats(a[1] if len(a) > 1 else "int", fname=a[2] if len(a) > 2 else "pbmc3k_prepped.h5ad", min_co=int(a[3]) if len(a) > 3 else 20, do_edges=len(a) < 5)
    else:
        main(int(a[0]) if a else 1500, int(a[1]) if len(a) > 1 else 40, a[2] if len(a) > 2 else "int", fname=a[3] if len(a) > 3 else "pbmc3k_prepped.h5ad", min_co=int(a[4]) if len(a) > 4 else 20)
