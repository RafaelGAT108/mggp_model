import numpy as np
import pytest
import random
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

@pytest.fixture
def mimo_data():
    """
    Sistema MIMO simples:

    y1[k] = 0.5*y1[k-1] + 0.2*u1[k-1] - 0.1*u2[k-1]
    y2[k] = -0.3*y2[k-1] + 0.4*u1[k-1] + 0.1*u2[k-1]
    """

    n = 200

    u = np.random.randn(n, 2)
    y = np.zeros((n, 2))

    for k in range(1, n):

        y[k, 0] = (0.5 * y[k-1, 0] + 0.2 * u[k-1, 0] - 0.1 * u[k-1, 1])
        y[k, 1] = (-0.3 * y[k-1, 1] + 0.4 * u[k-1, 0] + 0.1 * u[k-1, 1])

    return u, y

@pytest.fixture
def hysteresis_siso_data():
    """
    Sistema histerético simples SISO.

    Entrada:
        seno

    Histerese:
        ganhos diferentes para subida e descida.
    """

    n = 500
    t = np.linspace(0, 10*np.pi, n)

    u = np.sin(t).reshape(-1, 1)
    y = np.zeros((n, 1))

    for k in range(1, n):

        du = u[k] - u[k-1]

        # ramo de subida
        if du >= 0:
            a = 0.92
            b = 0.25

        # ramo de descida
        else:
            a = 0.85
            b = 0.10

        y[k] = (a * y[k-1] + b * u[k])

    return u, y


@pytest.fixture
def classification_data():
    """
    Problema de classificação multiclasse sintético.

    Classe 0 -> x1 + x2 < -0.5
    Classe 1 -> -0.5 <= x1 + x2 <= 0.5
    Classe 2 -> x1 + x2 > 0.5
    """

    rng = np.random.default_rng(42)

    X = rng.uniform(-1, 1, size=(300, 2))

    s = X[:, 0] + X[:, 1]

    y = np.zeros(len(s), dtype=int)
    y[s > 0.5] = 2
    y[(s >= -0.5) & (s <= 0.5)] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    return X_train, X_test, y_train, y_test