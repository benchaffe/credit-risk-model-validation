"""Shared ranking metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    y = np.asarray(y); o = np.argsort(p)
    ys = y[o]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))
