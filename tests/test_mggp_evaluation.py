import numpy as np
from mggp import MGGP

def test_evaluation_returns_scalar(siso_data):
    u, y = siso_data

    model = MGGP(inputs=u, outputs=y, generations=1, populationSize=5)
    model.initPop()

    ind = model._pop[0]

    fitness = model.evaluation(ind)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
