import numpy as np
from mggp.base import Element

def test_osa_shapes(siso_data):
    u, y = siso_data

    el = Element(nInputs=1, nOutputs=1, nTerms=3, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = np.ones((len(model)+1, 1))
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape == yd.shape
    np.testing.assert_array_equal(yd, y[model.lagMax:])


def test_miso_osa_shapes(miso_data):
    u, y = miso_data

    el = Element(nInputs=2, nOutputs=1, nTerms=3, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = np.ones((len(model)+1, 1))
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape[0] == yd.shape[0]


def test_mimo_osa_shapes(mimo_data):
    u, y = mimo_data

    el = Element(nInputs=2, nOutputs=2, nTerms=3, nDelays=2, mode='MIMO')
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = [np.ones(len(genes) + 1) for genes in model]
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape == yd.shape == (len(y) - model.lagMax, 2)
    np.testing.assert_array_equal(yd, y[model.lagMax:])



def test_freerun_stability(siso_data):
    u, y = siso_data

    el = Element(nInputs=1, nOutputs=1, nTerms=2, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    model.theta = np.ones(len(model)+1)

    yp, yd = model.predict("FreeRun", y, u)

    assert not np.isnan(yp).any()
    assert not np.isinf(yp).any()


def test_miso_freerun_stability(miso_data):
    u, y = miso_data

    el = Element(nInputs=2, nOutputs=1, nTerms=2, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    model.theta = np.ones(len(model)+1)

    yp, yd = model.predict("FreeRun", y, u)

    assert not np.isnan(yp).any()
    assert not np.isinf(yp).any()


def test_mimo_freerun_stability(mimo_data):
    u, y = mimo_data

    el = Element(nInputs=2, nOutputs=2, nTerms=2, nDelays=2, mode='MIMO')
    el.renameArguments({"ARG0": "y1", "ARG1": "y2", "ARG2": "u1", "ARG3": "u2"})
    model = el.buildModelFromList([["y1"], ["y2"]])
    el.compileModel(model)
    model.theta = [[0., .5], [0., -.25]]
    y = y.copy()
    y[0] = [2., -4.]
    yp, yd = model.predict("FreeRun", y, u)
    k = np.arange(1, len(y))
    np.testing.assert_allclose(yp, np.column_stack([2*.5**k, -4*(-.25)**k]))
    np.testing.assert_array_equal(yd, y[1:])


def test_fir_freerun_stability(mimo_data):
    u, y = mimo_data

    el = Element(nInputs=2, nOutputs=2, nTerms=2, nDelays=2, mode='FIR')
    model = el.buildRandomModel()
    el.compileModel(model)

    model.theta = np.ones([2, len(model)+1])

    yp, yd = model.predict("FreeRun", y, u)

    assert not np.isnan(yp).any()
    assert not np.isinf(yp).any()
