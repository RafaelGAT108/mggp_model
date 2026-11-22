import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mggp import MGGP
from utils import utils as signals
from utils import utils_classification as classification
from utils import utils_hysteresis as hysteresis


@pytest.mark.parametrize("mod", [signals, hysteresis, classification])
def test_tire_csv_filter_keeps_feature_order_and_force_units(mod, tmp_path):
    path = tmp_path / "tire.csv"
    df = pd.DataFrame({
        "Initial_Time": [0, 1, 2], "Final_Time": [1, 2, 3], "Ax": [1]*3, "Ay": [2]*3,
        "Pneu": ["FL", "FR", "FL"], "AccY": [3, 100, -2], "AccZ": [8, 110, 12],
        "Fx": [1000, 3000, -500], "Fy": [-2400, 4500, 2000],
        "Fz": [5000, 6100, 4700], "unused": [11, 12, 13],
    })
    df.to_csv(path, index=False)
    args = (path, "slalom") if mod is hysteresis else (path, ["Fx", "Fy", "Fz"], "slalom")
    x, y, *_ = mod.load_data(*args, filter=r"^(Acc|F[xyz]$)", tire="FL")
    np.testing.assert_array_equal(x, [[3, 8], [-2, 12]])
    np.testing.assert_allclose(y, [[1, -2.4, 5], [-.5, 2, 4.7]])
    # Already-clean extracts do not need metadata or a tire selector.
    df[["AccZ", "Fx", "Fy", "Fz"]].to_csv(path, index=False)
    x, y, *_ = mod.load_data(*args)
    np.testing.assert_equal(x[:, 0], [8, 110, 12])
    np.testing.assert_allclose(y[1], [3, 4.5, 6.1])


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_cross_correlation_finds_delayed_inverted_impulse_and_limits_lags(mod):
    x, y = np.zeros(22), np.zeros(22)
    x[6], y[10] = 2., -5.
    inputs = pd.DataFrame({"ax": x, "quiet": 0*x})
    outputs = pd.DataFrame({"fy": y})
    peaks = mod.analyze_cross_correlations(inputs, outputs, max_lag=6)
    cross = peaks["Output_fy_Input_ax"]
    assert (cross["peak_lag"], cross["peak_value"]) == (4, -1)
    np.testing.assert_array_equal(cross["lags"], np.arange(-6, 7))
    np.testing.assert_array_equal(peaks["Output_fy_Input_quiet"]["correlation"], np.zeros(13))
    narrow = mod.analyze_cross_correlations(inputs, outputs, max_lag=2)
    assert narrow["Output_fy_Input_ax"]["peak_value"] == 0.
    try:
        mod.plot_top_correlations(peaks, inputs, outputs, top_n=1)
        bars, heatmap = plt.gcf().axes[:2]
        assert [bar.get_height() for bar in bars.patches] == [-1.]
        np.testing.assert_array_equal(heatmap.images[0].get_array(), [[-1., 0.]])
    finally:
        plt.close("all")


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_filter_preserves_slow_sine_while_attenuating_fast_component(mod):
    t = np.arange(800) / 200
    slow = np.sin(2*np.pi*2*t)
    signal = slow + .7*np.sin(2*np.pi*50*t)
    filtered = mod.butter_lowpass_filter(signal, cutoff=8, fs=200, order=4)
    np.testing.assert_allclose(filtered[60:-60], slow[60:-60], atol=.001)


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_runge_kutta_and_bouc_wen_simulation(mod):
    # One RK4 step for x'=2x has the fourth-order Taylor polynomial as its amplification.
    z = .2
    expected = sum([1, z, z**2/2, z**3/6, z**4/24])
    assert mod.rungeKutta(3., .1, lambda x, rate: rate*x, 2.) == pytest.approx(3*expected)
    drive = np.ones(50)
    t = np.arange(50)*.001
    # Monotonic loading with udot=1 gives h'=0.9-0.016*h and h(0)=0.
    h = .9 / .016 * (-np.expm1(-.016*t))
    np.testing.assert_allclose(mod.simulate(t, drive, dp=2), 2*t-h, atol=1e-12)


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_causal_rolling_mean_by_channel_retains_fractional_values(mod):
    data = np.array([[1, 8], [2, 5], [6, -1], [0, 2]])
    expected = np.array([[np.nan, np.nan], [1.5, 6.5], [3., 4.], [8/3, 2.]])
    np.testing.assert_allclose(mod.media_movel_pandas(data, 3, min_periods=2), expected)
    np.testing.assert_allclose(mod.media_movel_pandas(data[:, 0], 3, min_periods=2), expected[:, 0])


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_excludes_union_of_flagged_front_and_rear_windows(mod, monkeypatch):
    def list_images(path):
        if "_front" in path:
            return ["window_1.png", "window_4.png", "notes.txt", "window_bad.png"]
        assert "_rear" in path
        return ["window_1.png", "window_3.png", "preview_window_0.png"]
    monkeypatch.setattr(mod.os, "listdir", list_images)
    df = pd.DataFrame({"force": np.arange(6)})
    pd.testing.assert_frame_equal(mod.remove_window_with_noise(df, "trial"), df.iloc[[0, 2, 5]])


def test_sequence_windows_and_constant_segments_preserve_chronology():
    u = np.column_stack([np.arange(7.), np.arange(7.)**2])
    y = (20 + 2*u[:, :1])
    x, target = hysteresis.create_sequences(u, y, time_steps=3)
    for k in range(4):
        np.testing.assert_array_equal(x[k], np.column_stack([u[k:k+3], y[k:k+3]]))
    np.testing.assert_array_equal(target, y[3:])
    samples = np.arange(10)[:, None]
    extended = hysteresis.insert_constant_segments(samples, n_const=3, len_const=4, seed=7)
    np.testing.assert_array_equal(extended, hysteresis.insert_constant_segments(samples, 3, 4, seed=7))
    values, counts = np.unique(extended, return_counts=True)
    np.testing.assert_array_equal(values, samples.ravel())
    np.testing.assert_equal(np.sort(counts), [1]*7 + [5]*3)
    assert np.all(np.diff(extended.ravel()) >= 0)


def test_regression_report_and_pickled_results(tmp_path):
    y = np.array([1., 2., 3.])
    yp = np.array([1., 3., 3.])
    values = hysteresis.calculate_metrics(y, yp, "NARX")
    np.testing.assert_allclose(values, [100/6, 1/3, .5])
    assert hysteresis.r_squared(y, yp) == pytest.approx(.5)
    path = tmp_path / "results.pkl"
    hysteresis.save(path, dict(metrics=values))
    with path.open("rb") as handle:
        np.testing.assert_allclose(pickle.load(handle)["metrics"], values)


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_prediction_helper_uses_frozen_checkpoint_without_refitting_on_validation(mod, tmp_path):
    u = np.arange(8.)[:, None]
    y = (1+2*u)
    engine = MGGP(inputs=u, outputs=y, evaluationType="OSA")
    model = engine.element.buildModelFromList(["u1"])
    assert engine.evaluation(model)[0] < 1e-12
    path = tmp_path / "trained.pkl"
    engine.save_model(model, path)
    xp, arbitrary_y = -u, np.full_like(y, -99)
    yp, yd = mod.mggp_predict(xp, arbitrary_y, evaluationTypeTest="FreeRun", save_model=str(path))
    np.testing.assert_allclose(yp, 1+2*xp)
    np.testing.assert_equal(yd, arbitrary_y)


@pytest.mark.parametrize("mod", [signals, hysteresis])
def test_unrestricted_and_invalid_correlation_windows(mod):
    u = np.array([0, 1., 0])
    y = np.array([1., 0., 0., 0.])
    full = mod.analyze_cross_correlations(u, y, max_lag=None)["Output_0_Input_0"]
    assert full["peak_lag"] == -1
    np.testing.assert_array_equal(full["lags"], [-2, -1, 0, 1, 2, 3])
    with pytest.raises(ValueError, match="nonnegative"):
        mod.analyze_cross_correlations(u, y, max_lag=-1)
