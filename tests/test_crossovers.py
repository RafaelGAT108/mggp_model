from collections import Counter

import numpy as np
import pytest

from mggp.base import Element
from mggp.crossings import CrossHighOnePoint, CrossHighUniform, CrossLowOnePoint, CrossLowUniform


@pytest.mark.parametrize("operator", [CrossLowOnePoint, CrossLowUniform, CrossHighOnePoint, CrossHighUniform])
@pytest.mark.parametrize("outputs", [1, 2])
def test_crossover_conserves_parental_material_per_output(operator, outputs):
    element = Element(nInputs=2, nOutputs=outputs, mode="FIR", nTerms=4,
                      nDelays=2, maxHeight=2, operators=["mul"])
    parents = [element.buildRandomModel() for _ in range(2)]
    def inventory(pair):
        groups = [list(ind) if ind.multi_output else [ind] for ind in pair]
        return [Counter(node.name for genes in (a, b) for tree in genes for node in tree)
                for a, b in zip(*groups)]
    initial = inventory(parents)
    children = operator(element).cross(*parents)
    assert inventory(children) == initial
    for child in children:
        element.compileModel(child)
        equations = child if child.multi_output else [child]
        assert all(len(eq) == 4 and all(tree.height <= 2 for tree in eq) for eq in equations)
        u = np.linspace(.1, 1., 32).reshape(16, 2)
        matrices = child.makeRegressors(np.zeros((16, outputs)), u, align="INSTANT")
        for matrix in (matrices if child.multi_output else [matrices]):
            assert matrix.shape == (16-child.lagMax, 5)
            assert np.isfinite(matrix).all()
