import numpy as np
from mggp.base import Element

def test_osa_shapes(siso_data):
    u, y = siso_data

    el = Element(nInputs=1, nOutputs=1, nTerms=3, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = np.ones(len(model)+1)
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape[0] -1 == yd.shape[0]


def test_miso_osa_shapes(miso_data):
    u, y = miso_data

    el = Element(nInputs=2, nOutputs=1, nTerms=3, nDelays=2)
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = np.ones(len(model)+1)
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape[0] == yd.shape[0]


def test_mimo_osa_shapes(mimo_data):
    u, y = mimo_data

    el = Element(nInputs=2, nOutputs=2, nTerms=3, nDelays=2, mode='MIMO')
    model = el.buildRandomModel()
    el.compileModel(model)

    theta = np.ones(len(model)+1)
    model.theta = theta

    yp, yd = model.predict("OSA", y, u)

    assert yp.shape[-1] == yd.shape[-1]
    assert yp.shape[1] - 1 == yd.shape[0]



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
    model = el.buildRandomModel()
    el.compileModel(model)

    model.theta = np.ones([2, len(model)+1])

    yp, yd = model.predict("FreeRun", y, u)

    assert not np.isnan(yp).any()
    assert not np.isinf(yp).any()


def test_fir_freerun_stability(mimo_data):
    u, y = mimo_data

    el = Element(nInputs=2, nOutputs=2, nTerms=2, nDelays=2, mode='FIR')
    model = el.buildRandomModel()
    el.compileModel(model)

    model.theta = np.ones([2, len(model)+1])

    yp, yd = model.predict("FreeRun", y, u)

    assert not np.isnan(yp).any()
    assert not np.isinf(yp).any()
