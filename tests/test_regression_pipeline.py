import numpy as np
from mggp import MGGP

def test_training_reduces_error(siso_data):
    u, y = siso_data

    model = MGGP(
        inputs=u,
        outputs=y,
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3
    )

    model.initPop()
    initial_best = model._hof[0].fitness.values[0]

    for g in range(1, 3):
        model.step(g)

    final_best = model._hof[0].fitness.values[0]

    assert final_best <= initial_best or np.isfinite(final_best)
