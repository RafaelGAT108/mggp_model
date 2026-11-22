import numpy as np
import pytest

from mggp import MGGP


def test_constrained_least_squares_matches_independent_optimum_and_singular_case():
    u = np.arange(5.)[:, None]
    engine = MGGP(inputs=u, outputs=u)
    model = engine.element.buildModelFromList(["u1"])
    engine.element.compileModel(model)
    x = np.arange(-2., 3.)
    p = np.column_stack([np.ones(5), x])
    y = (4 + 3*x)[:, None]
    np.testing.assert_allclose(model.constrained_least_squares(y, p).ravel(), [4, 3])
    c = {"S": np.array([[1., 1.]]), "c": np.array([0.])}
    # Minimize ||a(1-x)-y|| with b=-a.
    a = ((1-x) @ y[:, 0]) / np.sum((1-x)**2)
    np.testing.assert_allclose(model.constrained_least_squares(y, p, c).ravel(), [a, -a])
    singular = np.column_stack([x, x])
    coef = model.constrained_least_squares((5*x)[:, None], singular,
                                           {"S": np.array([[1., -1.]]), "c": np.array([1.])})
    np.testing.assert_allclose(coef.ravel(), [3, 2], atol=1e-12)
    with pytest.raises(ValueError, match="Inconsistent"):
        model.constrained_least_squares(y, p, {"S": [[1, 0], [1, 0]], "c": [1, 0]})


def test_hysteretic_fit_classifies_trees_and_preserves_equilibria():
    rng = np.random.default_rng(22)
    u, y = rng.normal(size=(2, 70, 1))
    engine = MGGP(inputs=u, outputs=y, operators=["mul", "sign", "subtraction"],
                  nDelays=2, evaluationType="OSA")
    genes = ["y1", "q1(y1)", "u1", "q2(u1)", "mul(y1, y1)", "mul(u1, u1)",
             "mul(u1, y1)", "sign(u1, q1(u1))", "subtraction(u1, q2(u1))"]
    model = engine.element.buildModelFromList(genes)
    fitness, = engine.evaluation(model)
    yp, yd = model.predict("OSA", y, u)
    assert fitness == pytest.approx(np.sqrt(np.mean((yp-yd)**2)))
    clusters = model.identify_term_clusters(y, u)
    assert clusters == dict(bias=[0], linear_output=[1, 2], linear_input=[3, 4],
                            nonlinear_y=[5], nonlinear_u=[6], cross_terms=[7], phi_terms=[8, 9])
    assert engine._check_hysteretic_constraints(model)
    for u0, y0 in [(-3, .75), (2, -1.6)]:
        yp, _ = model.predict("FreeRun", np.full((20, 1), y0), np.full((20, 1), u0))
        np.testing.assert_allclose(yp, y0, atol=1e-10)
    for index in [0, 1, 3, 5, 6, 7]:
        model.theta[index] += .01
        assert not engine._check_hysteretic_constraints(model)
        model.theta[index] -= .01
    invalid = engine.element.buildModelFromList(["mul(y1, y1)", "q1(u1)"])
    assert engine.evaluation(invalid) == (np.inf,)


@pytest.mark.parametrize("multi", [False, True])
def test_phi_repair_changes_only_illegal_arguments(multi):
    u = np.ones((12, 2))
    engine = MGGP(inputs=u, outputs=u if multi else u[:, :1], nDelays=2,
                  operators=["mul", "sign", "subtraction"])
    genes = ["sign(mul(y1, u1), y1)", "subtraction(q2(u2), q1(u1))", "mul(y1, u2)"]
    expected = ["sign(u1, q1(u1))", genes[1], genes[2]]
    model = engine.element.buildModelFromList([genes, genes] if multi else genes)
    engine._constrain_phi_functions(model)
    assert model.model2List() == ([expected, expected] if multi else expected)
    engine.element.compileModel(model)
    assert engine.get_terminal_by_name("u2").value == "u2"
    assert engine.get_terminal_by_name("u9") is None


@pytest.mark.parametrize("multi", [False, True])
def test_pruning_removes_orthogonal_terms_then_refits_remaining_genes(multi):
    # The three Walsh columns have zero mean and are pairwise orthogonal.
    u = np.tile([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], (8, 1))
    y = (2*u[:, :1] if not multi else np.column_stack([2*u[:, 0], -3*u[:, 1]]))
    engine = MGGP(inputs=u, outputs=y, mode="FIR", evaluationType="INSTANT",
                  froe_mode=True, pruning_probability=1, pruning_tolerance=.1, nDelays=1)
    genes = ["u1", "u2", "mul(u1, u2)"]
    model = engine.element.buildModelFromList([genes, genes] if multi else genes)
    assert engine.evaluation(model)[0] < 1e-12
    assert model.model2List() == ([["u1"], ["u2"]] if multi else ["u1"])
    xp = u[::-1].copy()
    yp, _ = model.predict("INSTANT", y, xp)
    np.testing.assert_allclose(yp, y[::-1], atol=1e-12)


def test_pruning_all_zero_energy_terms_leaves_valid_bias_model():
    u = np.zeros((20, 1))
    y = np.full_like(u, 2.5)
    engine = MGGP(inputs=u, outputs=y, evaluationType="OSA", froe_mode=True,
                  pruning_probability=1, pruning_tolerance=.01)
    model = engine.element.buildModelFromList(["u1", "q1(u1)"])
    assert engine.evaluation(model)[0] < 1e-12
    assert model.model2List() == []
    assert model.lagMax == 0
    np.testing.assert_allclose(model.predict("FreeRun", y, u)[0], y)
