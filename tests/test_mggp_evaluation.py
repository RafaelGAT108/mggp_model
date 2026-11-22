import numpy as np
from mggp import MGGP

def test_evaluation_returns_scalar(siso_data):
    u, y = siso_data

    model = MGGP(inputs=u, outputs=y, generations=1, populationSize=8, nTerms=2, nDelays=2, maxHeight=2)
    model.initPop()

    ind = model._hof[0]

    fitness = model.evaluation(ind)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert np.isfinite(fitness[0])


def test_mimo_evaluation_returns_scalar(mimo_data):
    u, y = mimo_data

    model = MGGP(inputs=u, outputs=y, generations=1, populationSize=8, nTerms=2, nDelays=2, maxHeight=2)
    model.initPop()

    ind = model._hof[0]

    fitness = model.evaluation(ind)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert np.isfinite(fitness[0])
