import numpy as np
import pytest

from mggp import MGGP


@pytest.mark.parametrize(("mode", "evaluation", "metric"), [
    ("FIR", "INSTANT", "accuracy"),
    ("FIR", "MShooting", "log_loss"),
    ("NARX", "OSA", "log_loss"),
    ("NARX", "FreeRun", "f1_macro"),
])
def test_label_training_and_reload(mode, evaluation, metric):
    classes = np.array([-7, 10, 40])
    codes = np.tile([2, 0, 1], 24)
    u = codes.astype(float)[:, None]
    labels = classes[np.roll(codes, 1) if mode == "FIR" and evaluation != "INSTANT" else codes]
    mggp = MGGP(inputs=u[:48], outputs=labels[:48], validation=(u[48:], labels[48:, None]),
                problem_type="classification", classification_encoding="label", mode=mode,
                evaluationType=evaluation, evaluationTypeTest=evaluation, classification_metric=metric,
                k=3, nDelays=1, nTerms=1, populationSize=1, generations=1)
    seed = mggp.element.buildModelFromList(["u1"])
    model, = mggp.run(seed=[seed])
    restored = MGGP().load_model("models_saved/best_model_0.pkl")

    assert mggp.nOutputs == 1
    for classifier in (model, restored):
        assert not classifier.multi_output
        assert classifier.theta.shape == (2, 1)
        np.testing.assert_array_equal(classifier.classes_, classes)
        args = (labels, u) if evaluation != "MShooting" else (3, labels, u)
        pred, ref = classifier.predict_classes(evaluation, *args)
        proba, targets = classifier.predict_proba(evaluation, *args)
        expected = labels.reshape(-1, 4)[:, 1:].ravel() if evaluation == "MShooting" else labels
        np.testing.assert_array_equal(pred, expected)
        np.testing.assert_array_equal(ref, expected)
        np.testing.assert_array_equal(targets, ref)
        np.testing.assert_array_equal(classes[proba.argmax(axis=1)], pred)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_allclose(classifier.score_classification(ref, proba, "log_loss"),
                                   -np.log(proba[np.arange(len(ref)), np.searchsorted(classes, ref)]).mean())


def test_narx_feedback_uses_discrete_codes():
    classes = np.array([10, 30, 90])
    codes = np.array([0, 1, 0, 2, 0, 1, 2, 1, 2, 2, 1])
    y = classes[codes]
    u = np.zeros((len(y), 2))
    mggp = MGGP(inputs=u, outputs=y[:, None], mode="NARX", problem_type="classification",
                classification_encoding="label", nDelays=1)
    model = mggp.element.buildModelFromList(["q1(y1)"])
    mggp._fit_model(model)
    model.theta = np.array([[1.6], [-0.8]])  # Code 0 selects 2, code 1 selects 1, code 2 selects 0.
    assert model.lagMax == 2
    cases = [
        ("OSA", (y, u), codes[:-2], y[2:]),
        ("INSTANT", (y, u), codes[:-2], y[2:]),
        ("FreeRun", (y, u), [0, 1, 2, 1, 0, 1, 2, 1, 0], y[2:]),
        ("MShooting", (3, y, u), [0, 1, 2, 1, 2, 1], y[[2, 3, 4, 7, 8, 9]]),
    ]
    for mode, args, past_codes, expected_targets in cases:
        scores = 1.6 - 0.8 * np.asarray(past_codes)[:, None]
        weights = np.exp(-(scores - np.arange(3)) ** 2)
        proba, ref = model.predict_proba(mode, *args)
        np.testing.assert_allclose(proba, weights / weights.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(ref, expected_targets)
        pred, _ = model.predict_classes(mode, *args)
        np.testing.assert_array_equal(pred, classes[weights.argmax(axis=1)])

    unknown_future = np.full(len(y), np.nan)
    unknown_future[:2] = y[:2]
    np.testing.assert_array_equal(model.predict_classes("FreeRun", unknown_future, u)[0],
                                  model.predict_classes("FreeRun", y, u)[0])


def test_binary_string_labels_and_validation_subsets():
    u = np.random.default_rng(42).integers(0, 2, (72, 1))
    names = np.array(["off", "on"])
    labels = names[np.bitwise_xor.accumulate(u[:, 0])]
    mggp = MGGP(inputs=u[:48], outputs=labels[:48], validation=[(u[48:], labels[48:])],
                mode="NARX", problem_type="classification", classification_encoding="label",
                evaluationType="MShooting", evaluationTypeTest="FreeRun", k=3,
                nTerms=3, nDelays=1, populationSize=1, generations=1)
    seed = mggp.element.buildModelFromList(["y1", "u1", "mul(y1, u1)"])
    model, = mggp.run(seed=[seed])
    np.testing.assert_array_equal(model.predict_classes("FreeRun", labels[48:], u[48:])[0], labels[49:])
    assert model.score_classification(["on"], ["on"], "f1_macro") == 0.5
    assert model.score_classification(["on"], [[0.2, 0.8]], "log_loss") == pytest.approx(-np.log(0.8))
    with pytest.raises(ValueError, match="probability column"):
        model.score_classification(["on"], [[0.8]], "log_loss")
    with pytest.raises(ValueError, match="absent"):
        model.score_classification(["unknown"], [[0.5, 0.5]], "log_loss")


def test_midpoints_and_scores_outside_the_code_range():
    labels = np.array(["a", "b", "c"])
    mggp = MGGP(inputs=np.arange(3)[:, None], outputs=labels, mode="FIR", problem_type="classification",
                classification_encoding="label", evaluationType="INSTANT")
    model = mggp.element.buildModelFromList(["u1"])
    mggp._fit_model(model)
    model.theta = np.array([[0.0], [1.0]])
    u = np.array([-1e308, 0.5, 1.5, 1e308])[:, None]
    y = np.repeat("a", len(u))
    with np.errstate(over="raise", invalid="raise"):
        p, _ = model.predict_proba("INSTANT", y, u)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(labels[p.argmax(axis=1)], ["a", "a", "b", "c"])


@pytest.mark.parametrize(("targets", "message"), [
    (np.eye(3), "single column"),
    (np.repeat(1, 3), "at least two"),
    (np.array([0, 1, np.nan]), "finite"),
])
def test_invalid_label_targets(targets, message):
    with pytest.raises(ValueError, match=message):
        MGGP(inputs=np.zeros((3, 1)), outputs=targets, problem_type="classification",
             classification_encoding="label")
