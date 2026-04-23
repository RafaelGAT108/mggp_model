import numpy as np
from mggp import MGGP

def test_population_initialization(siso_data):
    u, y = siso_data

    model = MGGP(inputs=u, outputs=y, generations=2, populationSize=10)
    model.initPop()

    assert len(model._pop) == 10
    assert len(model._hof) > 0


def test_step_execution(siso_data):
    u, y = siso_data

    model = MGGP(inputs=u, outputs=y, generations=2, populationSize=10)
    model.initPop()

    model.step(1)

    assert len(model._pop) == 10
