import numpy as np
from mggp.base import Element

def test_build_and_compile_model(siso_data):
    u, y = siso_data

    el = Element(nInputs=1, nOutputs=1, nTerms=3, nDelays=2)
    
    model = el.buildRandomModel()
    el.compileModel(model)

    assert hasattr(model, "_funcs")
    assert model.lagMax >= 0


def test_lag_computation_consistency(siso_data):
    u, y = siso_data

    el = Element(nInputs=1, nOutputs=1, nTerms=3, nDelays=3)
    model = el.buildRandomModel()
    el.compileModel(model)

    # lagMax nunca pode ser negativo
    assert model.lagMax >= 0


def test_parse_tree_output():
    el = Element(nInputs=1, nOutputs=1)
    model = el.buildRandomModel()
    el.compileModel(model)

    tree = model[0]
    expr, _ = model.parse_tree(tree)

    assert isinstance(expr, str)
    assert "k" in expr
