"""Парна міра напрямку через негаусовість (Hyvärinen & Smith, JMLR 14:111).

Усі чотири класичні методи симетричні за побудовою: вони бачать звʼязок, але
не його напрямок. Під ГАУСОВИМ шумом напрямок і не ідентифіковний. Але наші
спостережувані дані негаусові — лічильники проходять через exp, Пуассона й
log1p — тому асиметрія в принципі доступна.

Міра:  R = mean[x·g(y)] − mean[g(x)·y],  g = tanh
Знак R вказує напрямок, |R| — силу свідчення про однобічність.
"""
import numpy as np


def _std(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def pair_direction(expr, pairs):
    """Знакова міра для кожної пари: >0 означає i -> j."""
    out = []
    for i, j in pairs:
        x, y = _std(expr[:, i]), _std(expr[:, j])
        rho = float((x * y).mean())
        out.append(rho * float((x * np.tanh(y)).mean() - (np.tanh(x) * y).mean()))
    return np.array(out)


def asymmetry_matrix(expr):
    """|R| для всіх пар. Високе значення = свідчення однобічного ребра."""
    x = expr - expr.mean(axis=0)
    sd = x.std(axis=0)
    x = x / np.where(sd == 0, 1.0, sd)
    t = np.tanh(x)
    n = x.shape[0]
    a = (x.T @ t) / n
    rho = (x.T @ x) / n
    r = rho * (a - a.T)
    return np.abs(r)
