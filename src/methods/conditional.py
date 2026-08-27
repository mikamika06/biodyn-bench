import numpy as np
from methods.marginal import _standardize


def partial_corr_matrix(expr, ridge=1e-2):
    x = _standardize(expr)
    cov = np.cov(x, rowvar=False)
    prec = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
    d = np.sqrt(np.diag(prec))
    return -prec / np.outer(d, d)


def glasso_matrix(expr, alpha=0.1, max_iter=100):
    """Розріджена матриця точності (graphical lasso).

    Звичайна часткова кореляція обертає повну матрицю коваріацій і розсипається,
    коли генів стільки ж, скільки клітин. Лассо-штраф зануляє слабкі елементи
    точності й має тримати високе p/n. Основа SpiecEasi та GeneNet.
    """
    from sklearn.covariance import GraphicalLasso
    x = _standardize(expr)
    g = GraphicalLasso(alpha=alpha, max_iter=max_iter, assume_centered=True)
    try:
        g.fit(x)
    except Exception:
        return np.abs(partial_corr_matrix(expr))
    prec = g.precision_
    d = np.sqrt(np.abs(np.diag(prec)))
    d[d == 0] = 1.0
    return np.abs(-prec / np.outer(d, d))
