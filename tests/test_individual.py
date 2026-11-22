import numpy as np
import pytest

from mggp.base import Element


@pytest.mark.parametrize("mode,outputs", [("SISO", 1), ("MISO", 1), ("MIMO", 2), ("FIR", 1), ("FIR", 2)])
def test_tree_listing_and_compiled_design_agree_across_output_layouts(mode, outputs):
    n_inputs = 1 if mode == "SISO" else 2
    el = Element(nInputs=n_inputs, nOutputs=outputs, mode=mode, nTerms=2, nDelays=2)
    args = ([] if mode == "FIR" else [f"y{i+1}" for i in range(outputs)]) + [f"u{i+1}" for i in range(n_inputs)]
    el.renameArguments({f"ARG{i}": arg for i, arg in enumerate(args)})
    genes = ["q2(u1)", "mul(u1, q1(u1))"]
    layout = genes if outputs == 1 else [genes, list(reversed(genes))]
    model = el.buildModelFromList(layout)
    el.compileModel(model)
    assert model.model2List() == layout
    listing = str(model).splitlines()
    assert [line for line in listing if line in genes] == (genes if outputs == 1 else genes + genes[::-1])
    assert model.lagMax == 2
    u = np.arange(9*n_inputs, dtype=float).reshape(9, n_inputs) + .5
    design = model.makeRegressors(np.zeros((9, outputs)), u, align="INSTANT")
    ref = np.column_stack([np.ones(7), u[:-2, 0], u[2:, 0]*u[1:-1, 0]])
    np.testing.assert_allclose(design if outputs == 1 else design[0], ref)
    if outputs > 1:
        np.testing.assert_allclose(design[1], ref[:, [0, 2, 1]])


def test_nested_delays_follow_each_branch_and_recompilation_resets_history():
    el = Element(nDelays=3, operators=["mul"])
    el.renameArguments({"ARG0": "y1", "ARG1": "u1"})
    model = el.buildModelFromList(["q2(mul(q1(y1), q2(u1)))"])
    el.compileModel(model)
    assert model.lagMax == 4
    expression, consumed = model.parse_tree(model[0])
    assert (expression, consumed) == ("y1[k-4] * u1[k-4]", len(model[0]))
    model[:] = el.buildModelFromList(["u1"])
    el.compileModel(model)
    assert model.lagMax == 0
    assert "y1" not in model.terminals
