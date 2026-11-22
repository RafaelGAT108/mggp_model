import operator
import numpy as np
import pytest

from mggp import MGGP
from mggp.base import Element
from mggp.utils import make_cost


def test_expression_round_trip_keeps_primitive_semantics_and_per_output_coefficients(tmp_path):
    x = np.array([[.2, .4], [.7, -.8], [-.3, .25], [.6, -.5], [-.4, -.3]])
    y = np.zeros_like(x)
    engine = MGGP(inputs=x, outputs=y, mode="FIR", evaluationType="INSTANT", nDelays=2,
                  operators=["sign", "subtraction", "tanh", "sin", "div", "add"])
    genes = [["q1(sign(u1, u2))", "subtraction(u1, u2)", "subtraction(u1, u2)"],
             ["sin(u1)", "div(u1, u2)", "tanh(u1, u2)"]]
    model = engine.element.buildModelFromList(genes)
    engine.element.compileModel(model)
    model.theta = np.array([[1, 2, 3, -2], [-1, .25, .5, .75]])
    expected = np.column_stack([
        1 + 2*np.sign(x[:-1, 0]-x[:-1, 1]) + x[1:, 0]-x[1:, 1],
        -1 + .25*np.sin(x[1:, 0]) + .5*x[1:, 0]/x[1:, 1] + .75*np.tanh(x[1:, 0]-x[1:, 1])])
    # The simplified equation must be valid arithmetic, including a standalone intercept.
    equation = engine.simplify_model(model)
    assert equation.count("(u1[k] - u2[k])") == 1
    assert "sign(u1[k-1] - u2[k-1])" in equation
    for expression in equation.splitlines():
        assert not expression.rstrip().endswith("*")
    path = tmp_path / "equations.pkl"
    engine.save_model(model, path)
    restored = engine.load_model(path)
    assert restored.model2List() == model.model2List()
    assert engine.simplify_model(restored) == equation
    np.testing.assert_allclose(restored.predict("INSTANT", y, x)[0], expected, atol=1e-12)


def test_extension_primitive_and_serialized_gene_list_restore_numerical_mapping(tmp_path):
    el = Element(nInputs=2, nOutputs=1, mode="FIR", nDelays=[1, 3], operators=["mul"])
    el.renameArguments({"ARG0": "u1", "ARG1": "u2"})
    el.addPrimitive(operator.add, 2, "custom_sum")
    model = el.buildModelFromList(["q3(custom_sum(u1, u2))"])
    el.compileModel(model)
    u = np.column_stack([np.arange(10.), np.arange(10.)**2])
    y = np.zeros((10, 1))
    y[3:, 0] = -1 + .6*u[:-3].sum(axis=1)
    model.leastSquares(y, u, align="INSTANT")
    path = tmp_path / "custom.pkl"
    el.save(path, {"genes": model.model2List(), "theta": model.theta})
    saved = el.load(path)
    rebuilt = el.buildModelFromList(saved["genes"])
    el.compileModel(rebuilt)
    rebuilt.theta = saved["theta"]
    np.testing.assert_allclose(rebuilt.predict("INSTANT", y, u)[0], y[3:], atol=1e-12)


@pytest.mark.parametrize("form,outputs", [("NARX", 1), ("NARX", 2), ("FIR", 1), ("FIR", 2)])
def test_shape_and_method_errors_are_detected_at_prediction(form, outputs):
    x = np.ones((12, 2))
    y = np.ones((12, outputs))
    e = MGGP(inputs=x, outputs=y, mode=form)
    m = e.element.buildModelFromList(["u1"] if outputs == 1 else [["u1"], ["u2"]])
    e.element.compileModel(m)
    for mode, args, message in [
        ("OSA", (y, x[:-1]), "samples"), ("FreeRun", (y, x[:-1]), "samples"),
        ("MShooting", (0, y, x), "horizon"), ("bad-mode", (y, x), "mode")]:
        with pytest.raises(ValueError, match=message):
            m.predict(mode, *args)
    wrong_y = np.ones((12, 3))
    with pytest.raises(ValueError, match="output"):
        m.makeRegressors(wrong_y, x)
    with pytest.raises(ValueError, match="alignment"):
        m.makeRegressors(y, x, align="future")
    with pytest.raises(ValueError, match="trained"):
        m.predict_proba("INSTANT", y, x)
    with pytest.raises(ValueError, match="metric"):
        m.score(y, y, "unknown")
    with pytest.raises(ValueError, match="metric"):
        m.score_classification(y, y, "unknown")


def test_shooting_classifier_returns_rowwise_probabilities_and_two_objectives():
    u = np.tile([[1., 0.], [-1., 0.]], (5, 1))
    y = np.eye(2)[[0, 1]*5]
    e = MGGP(inputs=u, outputs=y, mode="NARX", problem_type="classification",
             classification_metric="log_loss", evaluationType="MShooting", k=4,
             new_evaluation=make_cost)
    m = e.element.buildModelFromList([["u1"], ["u1"]])
    loss, cost = e.evaluation(m)
    probs, yd = m.predict_proba("MShooting", 4, y, u)
    expected = np.where(y[:8] == 1, np.e/(np.e+1), 1/(np.e+1))
    np.testing.assert_allclose(probs, expected, atol=1e-12)
    np.testing.assert_array_equal(yd, y[:8])
    assert loss == pytest.approx(np.log1p(1/np.e)) and cost == 4
    with pytest.raises(ValueError, match="classification"):
        m.predict_proba("OSA", y, u)


def test_population_restart_reevaluates_seeds_and_resets_the_archive():
    x = np.linspace(-1, 1, 20)[:, None]
    e = MGGP(inputs=x, outputs=2*x, evaluationType="OSA", populationSize=1, generations=1)
    seed = e.element.buildModelFromList(["u1"])
    first, = e.run(seed=[seed])
    assert first.fitness.values[0] < 1e-12
    e.outputs = x**2
    second, = e.run(seed=[seed])
    yp, yd = second.predict("OSA", e.outputs, x)
    assert second.fitness.values[0] == pytest.approx(np.sqrt(np.mean((yp-yd)**2)))
    assert second.fitness.values[0] > .1
    assert len(e._logbook) == 1


def test_zero_error_stagnation_with_terminal_only_genes():
    x = np.arange(12.)[:, None]
    engine = MGGP(inputs=x, outputs=0*x, mode="FIR", evaluationType="INSTANT",
                  generations=10, populationSize=4, nTerms=1, maxHeight=0,
                  mutationRate=.8, crossoverRate=.8)
    best, = engine.run()
    assert best.fitness.values == (0.,)
    assert engine.mutationRate == pytest.approx(.9)
    assert len(engine._logbook) == 10


def test_invalid_second_objective_does_not_enter_a_valid_front():
    x = np.arange(12.)[:, None]
    engine = MGGP(inputs=x, outputs=x, evaluationType="OSA", populationSize=2,
                  generations=1, new_evaluation=lambda ind: np.nan if len(ind) == 2 else 2.)
    bad = engine.element.buildModelFromList(["u1", "u1"])
    good = engine.element.buildModelFromList(["y1"])
    assert engine.evaluation(bad) == (np.inf, np.inf)
    front = engine.run(seed=[bad, good])
    assert [i.model2List() for i in front] == [["y1"]]


def test_repeated_structures_retain_dominating_representative():
    x = np.arange(9.)[:, None]
    engine = MGGP(inputs=x, outputs=x, new_evaluation=lambda ind: len(ind))
    larger = engine.element.buildModelFromList(["u1", "u1"])
    smaller = engine.element.buildModelFromList(["u1"])
    larger.fitness.values = 0., 2
    smaller.fitness.values = 0., 1
    assert engine._survive([larger, smaller], 1)[0] is smaller
