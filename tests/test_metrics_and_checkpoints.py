import pickle
import numpy as np
import pytest

from mggp import MGGP
from mggp.base import Individual, accuracy_score, log_loss, mean_squared_error
from mggp.utils import make_cost


@pytest.mark.parametrize("mode,expected", [("MSE", 6.25), ("RMSE", (np.sqrt(2.5) + np.sqrt(10))/2),
                                          ("NMSE", (2.5/4 + 10)/2), ("MAPE", 37.5)])
def test_output_metrics_have_known_scales(mode, expected):
    # The second output is constant. NMSE falls back to its MSE and the
    # range-normalized MAPE omits it.
    yd = np.array([[2., 4.], [6., 4.]])
    yp = yd + [[1., 2.], [-2., 4.]]
    assert Individual().score(yd, yp, mode) == pytest.approx(expected)


def test_classification_metrics_with_labels_one_hot_and_extreme_probabilities():
    y = np.array([0, 1, 2])
    scores = np.array([[.1, .65, .25], [.8, .05, .15], [.02, .08, .9]])
    expected = -np.log([.1, .05, .9]).mean()
    assert log_loss(y, scores) == pytest.approx(expected)
    assert log_loss(np.eye(3), scores) == pytest.approx(expected)
    assert accuracy_score(np.eye(3), np.eye(3)[[1, 0, 2]]) == pytest.approx(1/3)
    assert accuracy_score(y, scores.argmax(axis=1)) == pytest.approx(1/3)
    assert log_loss([0, 1], [.25, .5]) == pytest.approx(-np.log([.75, .5]).mean())
    assert log_loss([1], [0]) == pytest.approx(-np.log(1e-15))


@pytest.mark.parametrize("metric,y,p", [
    (mean_squared_error, [1], [1, 2]), (accuracy_score, [0], [0, 1]),
    (log_loss, [0], [.2, .3]), (log_loss, [0], [[.5, .5], [.2, .8]]),
    (log_loss, [[0, 1]], [[.2, .3, .5]]), (log_loss, [[[1]]], [[1]]),
    (log_loss, [1], [[[1]]])
])
def test_metrics_reject_incompatible_shapes(metric, y, p):
    with pytest.raises(ValueError):
        metric(y, p)


@pytest.mark.parametrize("metric", ["accuracy", "log_loss", "f1_macro"])
def test_classifier_fit_predict_score_and_reload(metric):
    # Three non-collinear vertices let affine class scores interpolate one-hot
    # targets exactly. Softmax([1,0,0]) still has fractional probabilities.
    vertices = np.array([[0., 0.], [2., 0.], [0., 3.]])
    x = np.tile(vertices, (5, 1))
    y = np.tile(np.eye(3), (5, 1))
    engine = MGGP(inputs=x, outputs=y, mode="FIR", problem_type="classification",
                  classification_metric=metric, evaluationType="INSTANT", evaluationTypeTest="INSTANT",
                  populationSize=1, generations=1, validation=(x, y), filename="vertices")
    seed = engine.element.buildModelFromList([["u1", "u2"] for _ in range(3)])
    model, = engine.run(seed=[seed])
    test = np.array([[1., 0.], [0., 2.7], [.1, .3]])
    targets = np.eye(3)[[0, 2, 0]]
    logits = np.column_stack([1-test[:, 0]/2-test[:, 1]/3, test[:, 0]/2, test[:, 1]/3])
    exp = np.exp(logits)
    expected = exp / exp.sum(axis=1, keepdims=True)
    for candidate in [model, engine.load_model()]:
        proba, truth = candidate.predict_proba("INSTANT", targets, test)
        np.testing.assert_allclose(proba, expected, atol=1e-12)
        np.testing.assert_array_equal(truth, targets)
        # Test unambiguous points independently of floating-point tie breaking.
        labels, _ = candidate.predict_classes("INSTANT", targets[1:], test[1:])
        np.testing.assert_array_equal(labels, targets[1:])
        loss = np.log(np.e + 2) - 1 if metric == "log_loss" else 0.
        assert candidate.fitness.values[0] == pytest.approx(loss, abs=1e-12)


def test_loaded_old_checkpoint_rebuilds_predictions_from_saved_genes(tmp_path):
    x = np.arange(12.)[:, None]
    y = np.zeros_like(x)
    engine = MGGP(inputs=x, outputs=y, nDelays=1)
    # Legacy schema deliberately omits mode, weights, classification state and fitness.
    record = dict(model_structure=["q1(u1)"], theta=np.array([[.25], [-2.]]),
                  nInputs=1, nOutputs=1, nTerms=1, maxHeight=1, nDelays=1,
                  arguments={"ARG0": "y1", "ARG1": "u1"}, operators=["mul"])
    path = tmp_path / "previous.pkl"
    with path.open("wb") as handle:
        pickle.dump(record, handle)
    model = engine.load_model(path)
    yp, yd = model.predict("OSA", y, x)
    np.testing.assert_allclose(yp, .25 - 2*x[:-1])
    np.testing.assert_array_equal(yd, y[1:])
    assert not model.fitness.valid


def test_separate_engines_keep_their_own_model_type_and_fitness():
    u = np.arange(12.)[:, None]
    scalar = MGGP(inputs=u, outputs=u, new_evaluation=make_cost)
    other = MGGP(inputs=u, outputs=np.column_stack([u, u]), mode="FIR")
    first = scalar.element.buildModelFromList(["u1"])
    second = other.element.buildModelFromList([["u1"], ["u1"]])
    assert not first.multi_output and first.fitness.weights == (-1, -1)
    assert second.multi_output and second.input_only and second.fitness.weights == (-1,)


def test_validation_scores_each_dataset_and_checks_channel_counts(capsys):
    u = np.arange(10.)[:, None]
    engine = MGGP(inputs=u, outputs=u, mode="FIR", evaluationType="INSTANT",
                  evaluationTypeTest="INSTANT", validation=[(u, u), (u, u+3)])
    model = engine.element.buildModelFromList(["u1"])
    assert engine.evaluation(model)[0] < 1e-12
    engine.validation_all(model)
    out = capsys.readouterr().out
    assert "validation dataset 1: 0.0" in out and "validation dataset 2: 3.0" in out
    for data, message in [((np.tile(u, (1, 2)), u), "inputs"),
                          ((u, np.tile(u, (1, 2))), "outputs"),
                          ((u, None), "Missing"), ("invalid", "dataset type")]:
        engine.validation = data
        with pytest.raises(Exception, match=message):
            engine.validation_all(model)


def test_invalid_individual_is_penalized_per_objective_and_all_invalid_run_fails():
    u = np.ones((8, 1))
    for cost, penalty in [(None, (np.inf,)), (make_cost, (np.inf, np.inf))]:
        engine = MGGP(inputs=u, outputs=u, nDelays=[20], evaluationType="OSA",
                      new_evaluation=cost, populationSize=1, generations=1)
        model = engine.element.buildModelFromList(["q20(u1)"])
        assert engine.evaluation(model) == penalty
        with pytest.raises(RuntimeError, match="No finite fitness"):
            engine.run(seed=[model])


@pytest.mark.parametrize("outputs,stored_inputs_only", [(2, True), (1, False)])
def test_legacy_loader_infers_input_only_or_feedback_from_saved_arguments(tmp_path, outputs, stored_inputs_only):
    x = np.arange(12.)[:, None]
    y = np.tile(x, (1, outputs))
    # Use the opposite caller configuration: the legacy file identifies its own terminals.
    engine = MGGP(inputs=x, outputs=y, mode="NARX" if stored_inputs_only else "FIR")
    if stored_inputs_only:
        args = {"ARG0": "u1"}
        genes, theta = [["u1"], ["u1"]], [[0., 2.], [1., -1.]]
        expected = np.column_stack([2*x[:, 0], 1-x[:, 0]])
        prediction_mode = "INSTANT"
    else:
        args = {"ARG0": "y1", "ARG1": "u1"}
        genes, theta = ["sin(y1)"], [[0.], [1.]]
        expected = np.sin(y[:-1])
        prediction_mode = "OSA"
    record = dict(arguments=args, model_structure=genes, theta=theta, nInputs=1,
                  nOutputs=outputs, maxHeight=2, nTerms=1, nDelays=1)
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(record, handle)
    loaded = engine.load_model(path)
    yp, _ = loaded.predict(prediction_mode, y, x)
    np.testing.assert_allclose(yp, expected)


def test_constant_output_range_leaves_mape_undefined():
    y = np.ones((5, 2))
    assert np.isnan(Individual().score(y, 2*y, "MAPE"))


@pytest.mark.parametrize("failure", ["transposed_data", "unknown_metric", "step_without_population", "too_many_seeds"])
def test_invalid_run_configuration_produces_an_actionable_error(failure):
    x = np.arange(12.)[:, None]
    if failure == "transposed_data":
        with pytest.raises(ValueError, match="sample counts"):
            MGGP(inputs=x.T, outputs=x)
    elif failure == "unknown_metric":
        with pytest.raises(ValueError, match="RMSE"):
            MGGP(inputs=x, outputs=x, evaluationMode="unknown")
    else:
        engine = MGGP(inputs=x, outputs=x, populationSize=1)
        if failure == "step_without_population":
            with pytest.raises(RuntimeError, match="initialized"):
                engine.step(1)
        else:
            seed = engine.element.buildModelFromList(["u1"])
            with pytest.raises(ValueError, match="population size"):
                engine.run(seed=[seed, seed])


def test_large_common_logit_offset_preserves_calibrated_output_distribution():
    u = np.array([[1.], [-1.], [.5]])
    y = np.eye(2)[[0, 1, 0]]
    e = MGGP(inputs=u, outputs=y, mode="FIR")
    m = e.element.buildModelFromList([["u1"], ["u1"]])
    e.element.compileModel(m)
    m.logistic_model = True
    for offset in [0., 10000., -10000.]:
        m.theta = [[offset, 1.], [offset, -1.]]
        p, _ = m.predict_proba("INSTANT", y, u)
        first = 1 / (1 + np.exp(-2*u[:, 0]))
        np.testing.assert_allclose(p, np.column_stack([first, 1-first]), atol=1e-12)


def test_undefined_regressor_cannot_win_a_population():
    u = np.column_stack([np.arange(1., 9.), np.zeros(8)])
    y = u[:, :1] + .25
    e = MGGP(inputs=u, outputs=y, operators=["div"], evaluationType="FreeRun",
             generations=1, populationSize=2)
    bad = e.element.buildModelFromList(["div(u1, u2)"])
    good = e.element.buildModelFromList(["u1"])
    assert e.evaluation(bad) == (np.inf,)
    best, = e.run(seed=[bad, good])
    assert best.model2List() == ["u1"] and best.fitness.values[0] < 1e-12
