import numpy as np
import pytest

from utils import utils as signals
from utils import utils_hysteresis as h


def test_published_cases_osa_and_free_run_match_when_given_simulated_history():
    rng = np.random.default_rng(62)
    u = rng.uniform(-.1, .1, (36, 1))
    initial = rng.normal(0, .1, (36, 1))
    for free, osa, history in [(h.mcase, h.mcaseOSA, 11), (h.mcase2, h.mcase2OSA, 10)]:
        y = free(u, initial)
        np.testing.assert_allclose(y[:history], initial[:history, 0])
        np.testing.assert_allclose(osa(u, y[:, None])[history:], y[history:])
        altered = u.copy()
        altered[-1, 0] += 5
        np.testing.assert_allclose(free(altered, initial)[:-1], y[:-1])
    free = h.mcase(u, initial)
    combined, terms = h.mcase_free_run_terms(u, initial)
    np.testing.assert_allclose(combined, free[11:])
    np.testing.assert_allclose(terms.sum(axis=0), free[11:])


@pytest.mark.parametrize("fn,history,first", [
    (h.mcase30_free_run_terms, 2, lambda a: -1.34210 - 9.48303*a),
    (h.mcase24_free_run_terms, 1,
     lambda a: 17.989*a - 1.13020 - 5.58052 - .0302835*a**3 - .769282*a**2),
    (h.mcase_butterfly_free_run_terms, 3,
     lambda a: .000187658*a**3 - .831818*a + .161398 - .0183973 - .0243298*a**2 - 2.74938*a),
])
def test_term_decompositions_predict_known_response_to_first_step(fn, history, first):
    a = .25
    u = np.zeros((18, 1))
    u[history:] = a
    y0 = np.zeros_like(u)
    yp, terms = fn(u, y0)
    assert yp[0] == pytest.approx(first(a), abs=1e-12)
    np.testing.assert_allclose(yp, terms.sum(axis=0))
    y0[history:] = 100.
    np.testing.assert_allclose(fn(u, y0)[0], yp)


def test_case21_predicts_known_response_to_input_reversal():
    u = np.zeros((12, 1))
    u[3] = .25
    a = u[3, 0]
    expected = -2.14570*a + .111516 + 5.29324*a + (.767350+.0749283)*a**2
    assert h.mcase_21(u, np.zeros_like(u))[0] == pytest.approx(expected)


def test_comparison_case_and_delayed_osa_reference_values():
    u = np.zeros((24, 1))
    y = np.full_like(u, .2)
    u[2] = .5
    yp, terms = h.comparecase_free_run(u, y)
    assert yp[0] == pytest.approx(.2 + .77*.5)
    np.testing.assert_allclose(terms.sum(axis=0), yp)
    u[:] = 1.
    y[:, 0] = np.arange(len(u))*.03
    for func in (h.mcase1, signals.mcase1):
        # All input differences are zero, leaving only theta_1*y[k-1].
        np.testing.assert_allclose(func(u, y)[15:], y[14:-1, 0])
