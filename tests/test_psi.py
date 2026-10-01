import numpy as np
import pandas as pd
from src.validate.psi import psi


def test_identical_is_zero():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(x, x) < 1e-9


def test_shift_is_large():
    r = np.random.default_rng(0)
    assert psi(pd.Series(r.normal(size=5000)), pd.Series(r.normal(2, 1, 5000))) > 0.25


def test_categorical():
    a = pd.Series(["x"] * 90 + ["y"] * 10); b = pd.Series(["x"] * 50 + ["y"] * 50)
    assert psi(a, b) > 0.25
