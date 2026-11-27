from pathlib import Path

import numpy as np
import pytest

from mggp import MGGP


@pytest.mark.parametrize("metric", ["accuracy", "log_loss"])
def test_classifier_generalizes_and_checkpoint_reproduces_held_out_scores(classification_data, metric):
    x, xv, y, yv = classification_data
    engine = MGGP(inputs=x, outputs=y, validation=(xv, yv), mode="FIR",
                  problem_type="classification", classification_metric=metric,
                  evaluationType="INSTANT", evaluationTypeTest="INSTANT",
                  generations=2, populationSize=20, nTerms=10, maxHeight=2,
                  nDelays=1, mutationRate=.2, crossoverRate=.9, filename="classifier")
    best, = engine.run()
    assert Path("models_saved/classifier_0.pkl").is_file()
    prediction, target = best.predict_proba("INSTANT", yv, xv)
    reloaded, _ = engine.load_model().predict_proba("INSTANT", yv, xv)
    np.testing.assert_allclose(prediction, reloaded)
    true_labels = target.argmax(axis=1)
    if metric == "accuracy":
        score = np.mean(true_labels == prediction.argmax(axis=1))
        majority = np.max(np.bincount(true_labels)) / len(true_labels)
        assert score > majority
    else:
        likelihood = prediction[np.arange(len(target)), true_labels]
        training_priors = y.mean(axis=0)
        assert -np.log(likelihood).mean() < -np.log(training_priors[true_labels]).mean()


def test_narx_classification_feedback():
    labels = np.array([0, 2, 1, 1, 0, 1, 1, 2, 1, 0, 2])
    y = np.eye(3)[labels]
    u = np.zeros((len(y), 1))
    mggp = MGGP(inputs=u, outputs=y, mode="NARX", problem_type="classification", nDelays=1)
    model = mggp.element.buildModelFromList([["q1(y3)"], ["q1(y1)"], ["q1(y2)"]])
    mggp._fit_model(model)
    # The winner is the successor of the class at k-2. Its score is 0.8, the others are 0.1.
    model.theta = np.tile([0.1, 0.7], (3, 1))
    assert model.lagMax == 2

    cases = [
        ("OSA", (y, u), (labels[:-2] + 1) % 3, y[2:]),
        ("INSTANT", (y, u), (labels[:-2] + 1) % 3, y[2:]),
        ("FreeRun", (y, u), [1, 0, 2, 1, 0, 2, 1, 0, 2], y[2:]),
        ("MShooting", (3, y, u), [1, 0, 2, 2, 2, 0], y[[2, 3, 4, 7, 8, 9]]),
    ]
    for mode, args, expected_labels, references in cases:
        expected = np.eye(3)[expected_labels]
        classes, targets = model.predict_classes(mode, *args)
        np.testing.assert_array_equal(classes, expected)
        np.testing.assert_array_equal(targets, references)
        probabilities, targets = model.predict_proba(mode, *args)
        exp_scores = np.exp(0.1 + 0.7 * expected)
        np.testing.assert_allclose(probabilities, exp_scores / exp_scores.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(targets, references)


@pytest.mark.parametrize(("metric", "evaluation"), [
    ("accuracy", "OSA"), ("log_loss", "MShooting"), ("f1_macro", "FreeRun"),
])
def test_binary_narx_classification_pipeline(metric, evaluation):
    rng = np.random.default_rng(42)
    u = rng.integers(0, 2, size=(72, 1))
    # The class changes whenever u[k] is one.
    y = np.eye(2)[np.bitwise_xor.accumulate(u[:, 0])]
    mggp = MGGP(inputs=u[:48], outputs=y[:48], validation=(u[48:], y[48:]),
                mode="NARX", problem_type="classification", classification_metric=metric,
                evaluationType=evaluation, evaluationTypeTest="FreeRun", k=3,
                nTerms=3, nDelays=1, populationSize=1, generations=1)
    genes = ["y2", "u1", "mul(y2, u1)"]
    seed = mggp.element.buildModelFromList([genes, genes])
    model, = mggp.run(seed=[seed])

    expected_loss = np.log1p(np.exp(-1)) if metric == "log_loss" else 0.0
    np.testing.assert_allclose(model.fitness.values, [expected_loss], atol=1e-12)
    for classifier in (model, mggp.load_model()):
        classes, targets = classifier.predict_classes("FreeRun", y[48:], u[48:])
        np.testing.assert_array_equal(classes, y[49:])
        np.testing.assert_array_equal(targets, y[49:])


def test_narx_classification_requires_one_column_per_class():
    with pytest.raises(ValueError, match="one-hot"):
        MGGP(inputs=np.zeros((8, 1)), outputs=np.zeros((8, 1)),
             mode="NARX", problem_type="classification")


@pytest.mark.parametrize("encoding", ["one_hot", "label"])
def test_fir_shooting_scores_only_forecasts_and_ignores_observed_classes(encoding):
    codes = np.tile([0, 1], 7)
    u = codes[:, None].astype(float)
    classes = np.eye(2) if encoding == "one_hot" else np.array([10, 30])
    y = classes[np.roll(codes, 1)]  # The true class at t is determined by u[t-1].
    engine = MGGP(inputs=u, outputs=y, mode="FIR", problem_type="classification",
                  classification_encoding=encoding, evaluationType="MShooting", k=3)
    genes = [["u1"], ["u1"]] if encoding == "one_hot" else ["u1"]
    model = engine.element.buildModelFromList(genes)
    engine._fit_model(model)

    # Each full window has one history sample and three forecasts. Two samples remain.
    indices = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    pred, ref = model.predict_classes("MShooting", 3, y, u)
    np.testing.assert_array_equal(ref, y[indices])
    np.testing.assert_array_equal(pred, y[indices])
    proba, _ = model.predict_proba("MShooting", 3, y, u)

    flipped = classes[1 - np.roll(codes, 1)]
    new_proba, new_ref = model.predict_proba("MShooting", 3, flipped, u)
    np.testing.assert_allclose(new_proba, proba)
    np.testing.assert_array_equal(new_ref, flipped[indices])


@pytest.mark.parametrize("encoding", ["one_hot", "label"])
def test_narx_fault_persistence_uses_the_history_available_in_each_mode(encoding):
    classes = np.eye(2) if encoding == "one_hot" else np.array(["healthy", "fault"])
    y_train = classes[np.repeat([0, 1, 0, 1], 8)]
    engine = MGGP(inputs=np.zeros((len(y_train), 1)), outputs=y_train, mode="NARX",
                  problem_type="classification", classification_encoding=encoding)
    genes = [["y1"], ["y2"]] if encoding == "one_hot" else ["y1"]
    model = engine.element.buildModelFromList(genes)
    engine._fit_model(model)

    # A persistence classifier can only detect a change after observing the new state.
    codes = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    y = classes[codes]
    u = np.zeros((len(y), 1))
    osa, ref = model.predict_classes("OSA", y, u)
    np.testing.assert_array_equal(osa, y[:-1])
    np.testing.assert_array_equal(ref, y[1:])
    free, ref = model.predict_classes("FreeRun", y, u)
    np.testing.assert_array_equal(free, classes[np.zeros(len(y) - 1, dtype=int)])
    np.testing.assert_array_equal(ref, y[1:])
    shooting, ref = model.predict_classes("MShooting", 3, y, u)
    np.testing.assert_array_equal(shooting, classes[np.repeat(codes[::4], 3)])
    np.testing.assert_array_equal(ref, classes[codes.reshape(-1, 4)[:, 1:].ravel()])
