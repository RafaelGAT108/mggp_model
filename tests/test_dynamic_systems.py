"""Use scalar time-domain recurrences as references for compiled MGGP trees."""
import numpy as np
import pytest

from mggp import MGGP


def generate_system(n_inputs, n_outputs, seed, n=53):
    u = np.random.default_rng(seed).uniform(-1, 1, (n, n_inputs))
    y = np.zeros((n, n_outputs))
    y[:3] = np.random.default_rng(seed + 1).normal(size=(3, n_outputs))
    for k in range(3, n):
        previous = y[k - 1, -1]
        y[k, 0] = (0.15 + 0.4 * previous - 0.12 * y[k - 3, 0]
                   + 0.7 * u[k, 0] * previous - 0.2 * u[k - 2, -1])
        if n_outputs == 2:
            y[k, 1] = -0.1 + 0.15 * y[k - 1, 0] + 0.3 * u[k, -1]
    return u, y


@pytest.mark.parametrize("n_inputs,n_outputs", [(1, 1), (2, 1), (2, 2), (1, 2)],
                         ids=["siso", "miso", "mimo", "one-input-two-outputs"])
def test_fit_and_forecast_coupled_recurrence(n_inputs, n_outputs):
    u, y = generate_system(n_inputs, n_outputs, seed=10)
    mggp = MGGP(inputs=u, outputs=y, nDelays=[1, 2], evaluationType="OSA")
    last_y = f"y{n_outputs}"
    genes = [last_y, "q2(y1)", f"mul(u1, {last_y})", f"q2(u{n_inputs})"]
    if n_outputs == 2:
        genes = [genes, ["y1", f"u{n_inputs}"]]
    model = mggp.element.buildModelFromList(genes)
    assert mggp.evaluation(model)[0] < 1e-12
    assert model.lagMax == 3
    theta = model.theta if n_outputs == 2 else [model.theta]
    np.testing.assert_allclose(np.ravel(theta[0]), [.15, .4, -.12, .7, -.2], atol=1e-12)
    if n_outputs == 2:
        np.testing.assert_allclose(theta[1], [-.1, .15, .3], atol=1e-12)

    u, y = generate_system(n_inputs, n_outputs, seed=140)
    for mode in ["OSA", "FreeRun", "MShooting"]:
        horizon = 5
        if mode == "MShooting":
            args = (horizon, y, u)
            width = 3 + horizon
            expected = y[:len(y) // width * width]
        else:
            args = (y, u)
            expected = y[3:]
        predicted, measured = model.predict(mode, *args)
        np.testing.assert_array_equal(measured, expected)
        np.testing.assert_allclose(predicted, expected, atol=1e-12)


def test_feedback_uses_only_measured_initial_conditions_per_shoot():
    u = np.zeros((19, 1))
    y = (np.arange(19.) * 7 + 3)[:, None]
    mggp = MGGP(inputs=u, outputs=y)
    model = mggp.element.buildModelFromList(["y1"])
    mggp.element.compileModel(model)
    model.theta = np.array([[1.], [.5]])

    osa, measured = model.predict("OSA", y, u)
    np.testing.assert_allclose(osa[:, 0], 1 + .5 * y[:-1, 0])
    np.testing.assert_array_equal(measured, y[1:])
    # The recurrence tends to 2. Each shooting block restarts with its own y[0].
    free_run, _ = model.predict("FreeRun", y[:, 0], u[:, 0])
    np.testing.assert_allclose(free_run[:, 0], 2 + (y[0, 0] - 2) * .5 ** np.arange(1, 19))
    for horizon in [1, 3, 6]:
        width = 1 + horizon
        yp, yd = model.predict("MShooting", horizon, y, u)
        n_complete = len(y) // width * width
        initial = y[:n_complete:width]
        expected = 2 + (initial - 2) * .5 ** np.arange(width)
        np.testing.assert_allclose(yp[:, 0], expected.ravel())
        np.testing.assert_array_equal(yd, y[:n_complete])


def test_future_input_cannot_change_earlier_predictions_or_design_rows():
    u, y = generate_system(1, 1, 7)
    engine = MGGP(inputs=u, outputs=y)
    model = engine.element.buildModelFromList(["q2(u1)", "q1(q2(y1))"])
    engine.element.compileModel(model)
    model.theta = [[0.], [2.], [-.5]]
    expected = 2*u[2:-2] - .5*y[:-4]
    yp, yd = model.predict("OSA", y, u)
    np.testing.assert_allclose(yp, expected)
    np.testing.assert_array_equal(yd, y[4:])
    u[-1] += 1e6
    y[-1] -= 1e6
    np.testing.assert_allclose(model.predict("OSA", y, u)[0], expected)


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("training_mode,offset", [("INSTANT", 0), ("OSA", 1)])
def test_fir_alignment_recovers_coefficients_without_feedback(outputs, training_mode, offset):
    u = np.random.default_rng(17).normal(size=(43, 2))
    k = np.arange(2 + offset, len(u))
    y = np.zeros((len(u), outputs))
    y[k, 0] = 1.2 + 2*u[k-offset, 0] - .5*u[k-offset-2, 1]
    if outputs == 2:
        y[k, 1] = -.3 + .8*u[k-offset, 1]
    engine = MGGP(inputs=u, outputs=y, mode="FIR", evaluationType=training_mode, nDelays=2)
    genes = ["u1", "q2(u2)"]
    model = engine.element.buildModelFromList(genes if outputs == 1 else [genes, ["u2"]])
    assert engine.evaluation(model)[0] < 1e-12
    first = model.theta if outputs == 1 else model.theta[0]
    np.testing.assert_allclose(np.ravel(first), [1.2, 2., -.5], atol=1e-12)
    changed_targets = y + 42
    yp, yd = model.predict(training_mode, changed_targets, u)
    np.testing.assert_array_equal(yd, changed_targets[k])
    np.testing.assert_allclose(yp, y[k], atol=1e-12)
    if training_mode == "OSA":
        free, _ = model.predict("FreeRun", changed_targets, u)
        np.testing.assert_allclose(free, y[k], atol=1e-12)
        shoot, ref = model.predict("MShooting", 4, changed_targets, u)
        width = 3 + 4
        expected = y[:len(y)//width*width].copy().reshape(-1, width, outputs)
        expected[:, :3] += 42
        np.testing.assert_allclose(shoot, expected.reshape(-1, outputs), atol=1e-12)
        np.testing.assert_array_equal(ref, changed_targets[:len(shoot)])


def test_static_scalar_fir_and_rank_deficient_fit():
    u = np.linspace(-2, 3, 21)[:, None]
    y = 2 + 3*u
    engine = MGGP(inputs=u, outputs=y, mode="FIR", evaluationType="INSTANT")
    model = engine.element.buildModelFromList(["u1", "u1"])
    assert engine.evaluation(model)[0] < 1e-12
    assert model.input_only and model.lagMax == 0
    np.testing.assert_allclose(np.ravel(model.theta), [2, 1.5, 1.5], atol=1e-12)
    np.testing.assert_allclose(model.predict("INSTANT", y[:, 0], u[:, 0])[0], y, atol=1e-12)


@pytest.mark.parametrize("mode,extra", [("OSA", ()), ("FreeRun", ()), ("MShooting", (2,))])
def test_insufficient_history_raises_before_numerical_scoring(mode, extra):
    u, y = generate_system(1, 1, 3)
    mggp = MGGP(inputs=u, outputs=y)
    model = mggp.element.buildModelFromList(["q2(y1)"])
    mggp.element.compileModel(model)
    model.theta = np.array([[0.], [1.]])
    with pytest.raises(ValueError, match="samples"):
        model.predict(mode, *extra, y[:2], u[:2])
