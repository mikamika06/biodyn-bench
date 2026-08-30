import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.engine import Engine
from scgpt_collider.pairs import pair_table
from scgpt_collider.stats import auroc_ci, strata
from scgpt_collider.test2 import sample_matched

OUT = Path("/Users/macbook/biodyn-bench/out/scgpt_collider")


def build(eng, pairs, tgt_in, gi, n_cells, rng, mask_frac=0.25):
    rows, meta = [], []
    ks = []
    for r in pairs.itertuples():
        cs = np.array(sorted(gi[c] for c in (tgt_in[r.a] & tgt_in[r.b]))) if r.shared_in else np.array([], dtype=int)
        for a, b in ((r.ia, r.ib), (r.ib, r.ia)):
            both = eng.present[:, a] & eng.present[:, b]
            if len(cs):
                both &= eng.present[:, cs].any(1)
            cells = np.where(both)[0]
            cells = rng.choice(cells, min(n_cells, len(cells)), replace=False)
            for c in cells:
                idx = np.where(eng.present[c])[0]
                cset = set(cs[eng.present[c, cs]].tolist()) if len(cs) else set()
                k = len(cset) if cset else int(rng.choice(ks)) if ks else 1
                others = idx[(idx != a) & (idx != b) & ~np.isin(idx, list(cset))]
                if len(others) < k + 2:
                    continue
                extra = set(rng.choice(others, min(int(mask_frac * len(idx)), len(others) - k), replace=False).tolist())
                rest = [g for g in others if g not in extra]
                rnd = set(rng.choice(rest, k, replace=False).tolist())
                pool = np.where(eng.present[:, b] & (eng.xb[:, b] != eng.xb[c, b]))[0]
                if len(pool) == 0:
                    continue
                d = int(rng.choice(pool))
                base = extra | {a}
                for arm, extra_mask in (("base", set()), ("maskC", cset), ("maskR", rnd)):
                    if arm == "maskC" and not cset:
                        continue
                    rows.append((c, a, b, base | extra_mask, {}))
                    rows.append((c, a, b, base | extra_mask, {b: eng.xb[d, b]}))
                    meta.append({"pair": r.Index, "a": a, "b": b, "cell": int(c), "arm": arm, "k": k, "shared": int(r.shared), "v_b": float(eng.xb[c, b]), "v_b_donor": float(eng.xb[d, b])})
                if cset:
                    ks.append(len(cset))
    return rows, pd.DataFrame(meta)


def main(n_a=600, n_cells=20, tag="mor", seed=0):
    rng = np.random.default_rng(seed)
    eng = Engine(want_attn=True)
    gi = {g: i for i, g in enumerate(eng.genes)}
    df, tfs, tgt, tgt_in = pair_table(eng.genes, eng.present, min_co=20)
    mean_expr = eng.xl.mean(0)
    df["expr"] = np.log(mean_expr[df.ia.values] + 1e-3) + np.log(mean_expr[df.ib.values] + 1e-3)
    P = eng.present
    ok = []
    for r in df.itertuples():
        if not r.shared_in:
            ok.append(True); continue
        cs = np.array([gi[c] for c in (tgt_in[r.a] & tgt_in[r.b])])
        ok.append(int((P[:, r.ia] & P[:, r.ib] & P[:, cs].any(1)).sum()) >= 10)
    df = df[np.array(ok)]
    df = df[(df.shared_in == 1) | (df.shared == 0)]
    df = df.assign(shared=df.shared_in)
    pairs = sample_matched(df, n_a, rng)
    print("pairs", len(pairs), "A", int(pairs.shared.sum()), "B", int((1 - pairs.shared).sum()), flush=True)
    pairs = pairs.sort_values("shared", ascending=False)
    rows, meta = build(eng, pairs, tgt_in, gi, n_cells, rng)
    print("rows", len(rows), flush=True)
    preds, att_rp, att_pr = eng.run(rows)
    meta["p0"] = preds[0::2]
    meta["p1"] = preds[1::2]
    meta["eff"] = np.abs(meta.p1 - meta.p0)
    meta["att"] = att_rp[0::2].mean(1)
    for li in range(eng.nl):
        meta[f"att_L{li}"] = att_rp[0::2, li]
    meta.to_csv(OUT / f"test3_cells_{tag}.csv", index=False)
    cols = ["eff", "att"] + [f"att_L{li}" for li in range(eng.nl)]
    w = meta.pivot_table(index=["pair", "a", "b", "cell", "shared", "k"], columns="arm", values=cols).reset_index()
    w.columns = ["_".join(c).strip("_") if c[1] else c[0] for c in w.columns]
    per_pair = w.groupby(["pair", "shared"]).mean(numeric_only=True).reset_index()
    per_pair.to_csv(OUT / f"test3_pairs_{tag}.csv", index=False)
    res = {"n_pairs": int(len(per_pair)), "nA": int(per_pair.shared.sum()), "nB": int((1 - per_pair.shared).sum()), "n_rows": len(rows), "n_cells_per_direction": n_cells}
    A = per_pair[per_pair.shared == 1]
    B = per_pair[per_pair.shared == 0]

    def block(m, prefix):
        out = {}
        eps = 1e-9
        rC = (A[f"{m}_maskC"] / (A[f"{m}_base"] + eps)).values
        rRA = (A[f"{m}_maskR"] / (A[f"{m}_base"] + eps)).values
        rRB = (B[f"{m}_maskR"] / (B[f"{m}_base"] + eps)).values
        dC = (A[f"{m}_maskC"] - A[f"{m}_base"]).values
        dR = (A[f"{m}_maskR"] - A[f"{m}_base"]).values
        out["A_median_base"] = float(A[f"{m}_base"].median())
        out["A_median_maskC"] = float(A[f"{m}_maskC"].median())
        out["A_median_maskR"] = float(A[f"{m}_maskR"].median())
        out["B_median_base"] = float(B[f"{m}_base"].median())
        out["B_median_maskR"] = float(B[f"{m}_maskR"].median())
        out["A_ratio_maskC_over_base_median"] = float(np.median(rC))
        out["A_ratio_maskR_over_base_median"] = float(np.median(rRA))
        out["B_ratio_maskR_over_base_median"] = float(np.median(rRB))
        out["A_frac_pairs_maskC_lt_base"] = float((dC < 0).mean())
        out["A_frac_pairs_maskR_lt_base"] = float((dR < 0).mean())
        out["A_paired_wilcoxon_maskC_vs_maskR_p"] = float(wilcoxon(A[f"{m}_maskC"], A[f"{m}_maskR"]).pvalue)
        out["A_paired_wilcoxon_maskC_vs_base_p"] = float(wilcoxon(A[f"{m}_maskC"], A[f"{m}_base"]).pvalue)
        out["A_frac_pairs_maskC_lt_maskR"] = float((A[f"{m}_maskC"] < A[f"{m}_maskR"]).mean())
        out["A_ratioC_vs_B_ratioR_mwu_p"] = float(mannwhitneyu(rC, rRB).pvalue)
        y = np.r_[np.ones(len(rC)), np.zeros(len(rRB))]
        s = -np.r_[rC, rRB]
        out["auroc_A_ratioC_lower_than_B_ratioR"] = float(roc_auc_score(y, s))
        out["auroc_ci95"] = auroc_ci(y, s, rng, 500)
        y2 = np.r_[np.ones(len(rC)), np.zeros(len(rRA))]
        s2 = -np.r_[rC, rRA]
        out["auroc_A_ratioC_lower_than_A_ratioR"] = float(roc_auc_score(y2, s2))
        yb = np.r_[np.ones(len(A)), np.zeros(len(B))]
        out["auroc_A_vs_B_base"] = float(roc_auc_score(yb, np.r_[A[f"{m}_base"].values, B[f"{m}_base"].values]))
        out["auroc_A_vs_B_after_maskR"] = float(roc_auc_score(yb, np.r_[A[f"{m}_maskR"].values, B[f"{m}_maskR"].values]))
        out["auroc_A_maskC_vs_B_maskR"] = float(roc_auc_score(yb, np.r_[A[f"{m}_maskC"].values, B[f"{m}_maskR"].values]))
        return out

    res["intervention"] = block("eff", "eff")
    res["attention"] = block("att", "att")
    res["attention_layers"] = {li: {k: v for k, v in block(f"att_L{li}", "").items() if k in ("A_ratio_maskC_over_base_median", "A_ratio_maskR_over_base_median", "A_paired_wilcoxon_maskC_vs_maskR_p", "A_frac_pairs_maskC_lt_maskR")} for li in range(eng.nl)}
    res["k_shared_present_median"] = float(meta[meta.shared == 1].k.median())
    json.dump(res, open(OUT / f"test3_{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "attention_layers"}, indent=1))
    print("layers", {li: (round(v["A_ratio_maskC_over_base_median"], 3), round(v["A_ratio_maskR_over_base_median"], 3), round(v["A_paired_wilcoxon_maskC_vs_maskR_p"], 4)) for li, v in res["attention_layers"].items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 600, int(a[1]) if len(a) > 1 else 20, a[2] if len(a) > 2 else "mor")
