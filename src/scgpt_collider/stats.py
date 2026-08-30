import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from scipy.stats import t as tdist
from sklearn.metrics import roc_auc_score


def rewire(edges, rng, mult=2):
    e = edges.reset_index(drop=True)
    tf = e.tf.values.copy()
    tg = e.target.values.copy()
    S = set(zip(tf, tg))
    n = len(e)
    done = tries = 0
    while done < mult * n and tries < 50 * mult * n:
        i, j = rng.integers(0, n, 2)
        tries += 1
        a, b, c, d = tf[i], tg[i], tf[j], tg[j]
        if a == c or b == d or (a, d) in S or (c, b) in S or a == d or c == b:
            continue
        S.discard((a, b)); S.discard((c, d)); S.add((a, d)); S.add((c, b))
        tg[i] = d; tg[j] = b
        done += 1
    return pd.DataFrame({"tf": tf, "target": tg})


def shared_labels(edges, pairs):
    tgt = edges.groupby("tf").target.apply(set).to_dict()
    return np.array([len(tgt.get(x, set()) & tgt.get(y, set())) > 0 for x, y in pairs], dtype=int)


def auroc_ci(y, s, rng, n_boot=1000):
    idx = np.arange(len(s))
    bs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx))
        if 0 < y[b].sum() < len(b):
            bs.append(roc_auc_score(y[b], s[b]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def strata(*cols, q=5):
    key = np.zeros(len(cols[0]), dtype=int)
    for c in cols:
        r = pd.Series(c).rank(method="first")
        b = pd.qcut(r, min(q, len(set(c))), labels=False).values
        key = key * q + b
    return key


def evaluate(df, score_col, ref, rng, n_rewire=200, n_perm=1000, label="shared", strat_cols=("deg", "expr")):
    d = df.dropna(subset=[score_col]).reset_index(drop=True)
    s = d[score_col].values.astype(float)
    y = d[label].values.astype(int)
    out = {"score": score_col, "n": int(len(d)), "nA": int(y.sum()), "nB": int((1 - y).sum())}
    if y.sum() < 3 or (1 - y).sum() < 3:
        return out
    auc = roc_auc_score(y, s)
    out["auroc"] = float(auc)
    out["auroc_ci95"] = auroc_ci(y, s, rng)
    out["mwu_p"] = float(mannwhitneyu(s[y == 1], s[y == 0]).pvalue)
    out["median_A"] = float(np.median(s[y == 1]))
    out["median_B"] = float(np.median(s[y == 0]))
    out["mean_A"] = float(np.mean(s[y == 1]))
    out["mean_B"] = float(np.mean(s[y == 0]))
    pairs = list(zip(d.a, d.b))
    if n_rewire and ref is not None:
        null = []
        for _ in range(n_rewire):
            yr = shared_labels(rewire(ref, rng), pairs)
            if 0 < yr.sum() < len(yr):
                null.append(roc_auc_score(yr, s))
        null = np.array(null)
        out["rewire_null_mean"] = float(null.mean())
        out["rewire_null_ci95"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
        out["rewire_p_ge"] = float((null >= auc).mean())
        out["n_rewire"] = int(len(null))
    cols = [d[c].values for c in strat_cols if c in d]
    if cols and n_perm:
        key = strata(*cols)
        pn = []
        for _ in range(n_perm):
            lab = y.copy()
            for k in np.unique(key):
                mm = key == k
                lab[mm] = rng.permutation(y[mm])
            if 0 < lab.sum() < len(lab):
                pn.append(roc_auc_score(lab, s))
        pn = np.array(pn)
        out["strat_null_mean"] = float(pn.mean())
        out["strat_null_ci95"] = [float(np.percentile(pn, 2.5)), float(np.percentile(pn, 97.5))]
        out["strat_p_ge"] = float((pn >= auc).mean())
        out["strat_cols"] = list(strat_cols)
    if "deg" in d:
        out["auroc_deg_predicts_A"] = float(roc_auc_score(y, d.deg.values))
        out["spearman_score_deg"] = float(spearmanr(s, d.deg.values).correlation)
    if "expr" in d:
        out["spearman_score_expr"] = float(spearmanr(s, d.expr.values).correlation)
    X = [np.ones(len(s)), y]
    names = ["shared"]
    for c in ("deg", "n_co"):
        if c in d:
            X.append(np.log(d[c].values + 1e-3)); names.append("log_" + c)
    if "expr" in d:
        X.append(d.expr.values.astype(float)); names.append("expr")
    X = np.column_stack(X)
    yr = pd.Series(s).rank().values / len(s)
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]
    resid = yr - X @ beta
    cov = (resid @ resid / (len(yr) - X.shape[1])) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    out["ols"] = {nm: {"beta": float(beta[i + 1]), "p": float(2 * tdist.sf(abs(beta[i + 1] / se[i + 1]), len(yr) - X.shape[1]))} for i, nm in enumerate(names)}
    if "direct" in d:
        dr = d.direct.values.astype(int)
        m = (dr == 0) | (y == 0)
        if y[m].sum() > 2:
            out["auroc_A_nodirect_vs_B"] = float(roc_auc_score(y[m], s[m]))
            out["nA_nodirect"] = int(y[m].sum())
        if dr.sum() >= 3:
            out["n_direct"] = int(dr.sum())
            out["median_direct"] = float(np.median(s[dr == 1]))
            out["auroc_direct_vs_rest"] = float(roc_auc_score(dr, s))
    if "n_shared" in d:
        ns = d.n_shared.values
        dose = {}
        for lo, hi, lab in [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10 ** 6, ">=5")]:
            mm = (ns >= lo) & (ns <= hi)
            if mm.sum():
                dose[lab] = {"n": int(mm.sum()), "median": float(np.median(s[mm])), "mean": float(np.mean(s[mm]))}
        out["dose"] = dose
        out["spearman_score_nshared"] = float(spearmanr(s, ns).correlation)
    return out
