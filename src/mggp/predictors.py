"""Predictors return (predictions, references), with one row per time step."""
import numpy as np


def _columns(values):
    values = np.asarray(values)
    return values[:, None] if values.ndim == 1 else values


def _fitted_values(ind, y, u, align="OSA"):
    y = _columns(y)
    if ind.input_only:
        regressors = ind.makeRegressors(y, u, align=align)
    else:
        regressors = ind.makeRegressors(y, u)
    if ind.multi_output:
        predictions = np.column_stack([p @ theta for p, theta in zip(regressors, ind.theta)])
    else:
        predictions = _columns(regressors @ ind.theta)
    start = ind.lagMax + int(ind.input_only and align == "OSA")
    return predictions, y[start:]


def miso_OSA(ind, y_true, u):
    return _fitted_values(ind, y_true, u)


def mimo_OSA(ind, y_true, u):
    return _fitted_values(ind, y_true, u)


def miso_FIR_INSTANT(ind, y_true, u):
    return _fitted_values(ind, y_true, u, align="INSTANT")


def mimo_FIR_INSTANT(ind, y_true, u):
    return _fitted_values(ind, y_true, u, align="INSTANT")


def mimo_INSTANT(ind, y_true, u):
    return _fitted_values(ind, y_true, u, align="INSTANT")


def mimo_CLASSIFY(ind, y_true, u):
    return _fitted_values(ind, y_true, u, align="INSTANT")


def miso_CLASSIFY(ind, y_true, u):
    return _fitted_values(ind, y_true, u, align="INSTANT")


def _simulate(ind, y, u, horizon=None):
    """Simulate independent windows with measured initial conditions for each."""
    y, u = _columns(y), _columns(u)
    if len(y) != len(u):
        raise ValueError("Inputs and outputs must have the same number of samples")
    warmup = ind.lagMax + int(ind.input_only)
    if horizon is not None and (not isinstance(horizon, (int, np.integer)) or horizon < 1):
        raise ValueError("MShooting requires a positive integer horizon")
    width = len(y) if horizon is None else warmup + horizon
    if len(y) <= warmup or width > len(y):
        raise ValueError("Not enough samples for the history and prediction horizon")

    batch_count = len(y) // width
    end = batch_count * width
    observed = y[:end].reshape(batch_count, width, y.shape[1])
    inputs = u[:end].reshape(batch_count, width, u.shape[1])
    # The leading slot aligns y[t-1] with u[t]. The valid lag bound guarantees
    # that a gene cannot read this placeholder in the first forecast.
    predicted = np.zeros((batch_count, width + 1, y.shape[1]))
    predicted[:, 1:warmup + 1] = observed[:, :warmup]
    equations = ind.funcs if ind.multi_output else [ind.funcs]
    coefficients = ind.theta if ind.multi_output else [ind.theta]
    for t in range(warmup, width):
        first = t - ind.lagMax
        if ind.input_only:
            window = inputs[:, first - 1:t]
            variables = []
        else:
            window = inputs[:, first:t + 1]
            variables = [predicted[:, first:t + 1, j:j + 1] for j in range(y.shape[1])]
        variables.extend(window[:, :, j:j + 1] for j in range(u.shape[1]))
        for j, (terms, theta) in enumerate(zip(equations, coefficients)):
            regressors = [np.ones(batch_count)]
            regressors.extend(term(*variables)[:, -1, 0] for term in terms)
            predicted[:, t + 1, j] = np.column_stack(regressors) @ np.asarray(theta).ravel()
    start = warmup if horizon is None else 0
    return (np.nan_to_num(predicted[:, start + 1:].reshape(-1, y.shape[1])),
            observed[:, start:].reshape(-1, y.shape[1]))


def miso_FreeRun(ind, y_true, u):
    return _simulate(ind, y_true, u)


def mimo_FreeRun(ind, y_true, u):
    return _simulate(ind, y_true, u)


def mimo_FIR_FreeRun(ind, y_true, u):
    # An FIR model has no output feedback. A single matrix product evaluates
    # the entire free run with the one-step input alignment.
    return _fitted_values(ind, y_true, u)


def miso_MShooting(ind, k, y, u):
    return _simulate(ind, y, u, k)


def mimo_MShooting(ind, k, y_true, u):
    return _simulate(ind, y_true, u, k)


def mimo_FIR_MShooting(ind, k, y_true, u):
    return _simulate(ind, y_true, u, k)
