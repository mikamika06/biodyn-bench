import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from methods.marginal import _standardize

N_JOBS = -1


def _fit_one(x, j, cols, model_factory, seed):
    others = cols[cols != j]
    m = model_factory(seed + j)
    m.fit(x[:, others], x[:, j])
    imp = m.feature_importances_
    s = imp.sum()
    return others, (imp / s if s > 0 else imp)


def _tree_importances(expr, model_factory, seed):
    p = expr.shape[1]
    x = _standardize(expr)
    cols = np.arange(p)
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_fit_one)(x, j, cols, model_factory, seed) for j in range(p))
    w = np.zeros((p, p))
    for j, (others, imp) in enumerate(res):
        w[others, j] = imp
    return np.maximum(w, w.T)


def genie3(expr, seed=0, n_estimators=64):
    return _tree_importances(
        expr,
        lambda rs: ExtraTreesRegressor(n_estimators=n_estimators, max_features="sqrt",
                                       random_state=rs, n_jobs=1),
        seed)


def grnboost2(expr, seed=0, n_estimators=200):
    return _tree_importances(
        expr,
        lambda rs: GradientBoostingRegressor(n_estimators=n_estimators, max_depth=3,
                                             learning_rate=0.01, subsample=0.9,
                                             max_features="sqrt", random_state=rs,
                                             n_iter_no_change=25, validation_fraction=0.1,
                                             tol=1e-4),
        seed)
