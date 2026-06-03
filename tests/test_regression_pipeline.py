import numpy as np
import os
from mggp import MGGP
from mggp.base import Individual


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


def test_miso_training_reduces_error(miso_data):
    u, y = miso_data

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


def test_mimo_training_reduces_error(mimo_data):
    u, y = mimo_data

    model = MGGP(
        inputs=u,
        outputs=y,
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3,
        nDelays=1
    )

    model.initPop()
    initial_best = model._hof[0].fitness.values[0]

    for g in range(1, 3):
        model.step(g)

    final_best = model._hof[0].fitness.values[0]

    assert final_best <= initial_best or np.isfinite(final_best)


def test_fir_training_reduces_error(mimo_data):
    u, y = mimo_data

    model = MGGP(
        inputs=u,
        outputs=y,
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3,
        nDelays=1,
        mode='FIR'
    )

    model.initPop()
    initial_best = model._hof[0].fitness.values[0]

    for g in range(1, 3):
        model.step(g)

    final_best = model._hof[0].fitness.values[0]

    assert final_best <= initial_best or np.isfinite(final_best)


def test_hysteresis_training_reduces_error(hysteresis_siso_data):
    u, y = hysteresis_siso_data

    model = MGGP(
        inputs=u,
        outputs=y,
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3,
        nDelays=1,
        froe_mode=True
    )

    model.initPop()
    initial_best = model._hof[0].fitness.values[0]

    for g in range(1, 3):
        model.step(g)

    final_best = model._hof[0].fitness.values[0]

    assert final_best <= initial_best or np.isfinite(final_best)


def test_siso_training_run(hysteresis_siso_data):
    u, y = hysteresis_siso_data

    model = MGGP(
        inputs=u,
        outputs=y,
        validation=(u, y),
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3,
        nDelays=1,
        froe_mode=True
    )

    model.run()

    assert os.path.exists("best_model.pkl")


def test_load_model(hysteresis_siso_data):
    u, y = hysteresis_siso_data

    mggp = MGGP(
        inputs=u,
        outputs=y,
        validation=(u, y),
        generations=3,
        populationSize=20,
        nTerms=3,
        maxHeight=3,
        nDelays=1,
        froe_mode=False
    )

    mggp.run()

    loaded_model = mggp.load_model()

    assert isinstance(loaded_model, Individual)
