import numpy as np
import pytest

from mggp.base import Element
from mggp.mutations import MutGPOneTree, MutGPUniform, MutGPReplace


@pytest.mark.parametrize("op", [MutGPOneTree, MutGPUniform, MutGPReplace])
@pytest.mark.parametrize("outputs", [1, 2])
def test_deap_mutations_generate_executable_variants_with_bounded_trees(op, outputs):
    el = Element(nInputs=2, nOutputs=outputs, mode="FIR", nDelays=2, nTerms=3,
                 maxHeight=2, operators=["mul"])
    el.renameArguments({"ARG0": "u1", "ARG1": "u2"})
    genes = ["u1", "u2", "q1(u1)"]
    structures = genes if outputs == 1 else [genes, genes]
    candidate = el.buildModelFromList(structures)
    strategy = op(el)
    u = np.random.default_rng(6).normal(size=(16, 2))
    y = np.zeros((16, outputs))
    changed = False
    for _ in range(6):
        candidate, = strategy.mutate(candidate)
        changed |= candidate.model2List() != structures
        el.compileModel(candidate)
        equations = candidate if candidate.multi_output else [candidate]
        assert all(len(eq) == 3 and max(g.height for g in eq) <= 2 for eq in equations)
        candidate.leastSquares(y, u, align="INSTANT")
        yp, yd = candidate.predict("INSTANT", y, u)
        np.testing.assert_array_equal(yp, np.zeros_like(yd))
    assert changed
