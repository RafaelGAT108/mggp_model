import numpy as np
import pytest
import random

@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
    random.seed(42)

@pytest.fixture
def siso_data():
    """
    Sistema linear simples:
    y[k] = 0.5*y[k-1] + 0.2*u[k-1]
    """
    n = 200
    u = np.random.randn(n, 1)
    y = np.zeros((n, 1))

    for k in range(1, n):
        y[k] = 0.5*y[k-1] + 0.2*u[k-1]

    return u, y

@pytest.fixture
def miso_data():
    n = 200
    u = np.random.randn(n, 2)
    y = np.zeros((n, 1))

    for k in range(1, n):
        y[k] = 0.3*y[k-1] + 0.1*u[k-1, 0] - 0.2*u[k-1, 1]

    return u, y
