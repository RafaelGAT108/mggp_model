from tqdm import tqdm
import numpy as np

def miso_FIR_INSTANT(ind, y_true, u):
    """
    FIR instantâneo (mesmo instante):
      y_pred[k] alinhado com y_true[k], iniciando em k = lagMax
    """
    regressors = ind.makeRegressors(y_true, u, align="INSTANT")
    return np.dot(regressors, ind.theta), y_true[ind.lagMax:]


def mimo_FIR_INSTANT(ind, y_true, u):
    """
    FIR instantâneo (MIMO):
      y_pred[k,:] alinhado com y_true[k,:], iniciando em k = lagMax
    """
    regressors = ind.makeRegressors(y_true, u, align="INSTANT")
    y_pred = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind._theta))]
    return np.array(y_pred).T, y_true[ind.lagMax:]


def mimo_INSTANT(ind, y_true, u):
    regressors = ind.makeRegressors(y, u, align="instant")
    yp = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind._theta))]
    y_pred = np.array(yp).T
    y_true = y_true[ind.lagMax:]          # mesmo instante
    return y_pred, y_true


def mimo_CLASSIFY(ind, y_true, u):
    """
    Preditor "mesmo instante" para classificação (MIMO).
    Alinha y_pred[k] com y_true[k], iniciando em k = lagMax.
    """
    regressors = ind.makeRegressors(y_true, u)
    yp = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind._theta))]
    return np.array(yp).T, y_true[ind.lagMax:]


def miso_CLASSIFY(ind, y_true, u):
    """
    Preditor "mesmo instante" para classificação (MIMO).
    Alinha y_pred[k] com y_true[k], iniciando em k = lagMax.
    """
    regressors = ind.makeRegressors(y_true, u)
    
    y_pred = np.dot(regressors, ind.theta)
    return np.array(y_pred).T, y_true[ind.lagMax:]


def miso_OSA(ind, y_true, u):
    """
    Implements the One-Step_Ahead predictor for MISO models
    Arguments:
        ind = C_Individual object
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    regressors = ind.makeRegressors(y_true, u)
    y_pred = np.dot(regressors, ind.theta)
    return y_pred, y_true[ind.lagMax + 1:]


def mimo_OSA(ind, y_true, u):
    """
    Implements the One-Step_Ahead predictor for MIMO models
    Arguments:
        ind = C_Individual object
        y   = n-dimensional array with output data
        u   = m-dimensional array with input data
    """
    regressors = ind.makeRegressors(y_true, u)
    y_pred = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind._theta))]

    return np.array(y_pred).T, y_true[ind.lagMax + 1:]


def miso_FreeRun(ind, y_true, u):
    """
    Implements the Free-Run predictor for MISO models
    Arguments:
        ind = C_Individual object
        y0  = 1-dimensional array with initial conditions
        u   = n-dimensional array with input data
    """
    y_true = y_true.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    y = y_true[:ind.lagMax + 1].reshape(-1, 1)

    for i in range(u.shape[0] - ind.lagMax):

        listV = [y[i:i + ind.lagMax + 1].reshape(-1, 1)]

        for v in u.T:
            listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))

        regressors = [np.ones((ind.lagMax + 1))]

        for i in range(len(ind)):

            func = ind._funcs[i]
            out = func(*listV)
            regressors.append(out.reshape(-1))

        regressors = np.array(regressors).T[ind.lagMax:]
        y = np.vstack((y, np.dot(regressors, ind.theta)))
    return np.nan_to_num(y[:-1], nan=0), np.nan_to_num(y_true, nan=0)


def mimo_FreeRun(ind, y_true, u):
    """
    Implements the Free-Run predictor for MIMO models.
    Args:
        ind: C_Individual object (MIMO).
        y0: Initial conditions (n_outputs x history).
        u: Input data (n_samples x n_inputs).
    Returns:
        y_pred: Predicted outputs (n_samples x n_outputs).
        y_true: Ground truth (trimmed to match y_pred).
    """
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    n_samples = u.shape[0]
    n_outputs = y_true.shape[1]
    initial_conditions_size = ind.lagMax
    
    y_pred = np.zeros((n_samples - initial_conditions_size, n_outputs))
    y_history = np.ones((initial_conditions_size + 1, n_outputs))

    for step in tqdm(range(n_samples - ind.lagMax), desc="Processing iterations in FreeRun"):

        listV = []
        
        for v in y_history.T:
            listV.append(v[step:step + initial_conditions_size + 1].reshape(-1, 1))

        for v in u.T:
            listV.append(v[step:step + initial_conditions_size + 1].reshape(-1, 1))

        for idx_output in range(n_outputs):
            regressors = [1.0]  # bias

            for idx_equation_tree in range(len(ind[idx_output])):

                genetic_programming_term = ind._funcs[idx_output][idx_equation_tree]
                out = genetic_programming_term(*listV)
                regressors.append(float(out[-1])) 
            
            y_pred[step, idx_output] = np.dot(regressors, ind._theta[idx_output])

        y_history = np.column_stack([y_history.T, y_pred[step, :]]).T

    y_true = y_true[ind.lagMax:, :]
    return np.nan_to_num(y_pred, nan=-100_000), np.nan_to_num(y_true, nan=-100_000)


def mimo_FIR_FreeRun(ind, y_true, u):
    """
    Implements the Free-Run predictor for MIMO models.
    Args:
        ind: C_Individual object (MIMO).
        y0: Initial conditions (n_outputs x history).
        u: Input data (n_samples x n_inputs).
    Returns:
        y_pred: Predicted outputs (n_samples x n_outputs).
        y_true: Ground truth (trimmed to match y_pred).
    """
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    n_samples = u.shape[0]
    n_outputs = y_true.shape[1]
    initial_conditions_size = ind.lagMax
    y_pred = np.zeros((n_samples - initial_conditions_size, n_outputs))
    
    y_history = y_true[:initial_conditions_size + 1 , :].copy()

    for step in tqdm(range(n_samples - initial_conditions_size), desc="Processing iterations in FreeRun"):

        listV = []
        for v in u.T:
            listV.append(v[step:step + initial_conditions_size + 1].reshape(-1, 1))

        for idx_output in range(n_outputs):
            regressors = [1.0]  # bias

            for idx_equation_tree in range(len(ind[idx_output])):

                genetic_programming_term = ind._funcs[idx_output][idx_equation_tree]
                out = genetic_programming_term(*listV)
                regressors.append(float(out[-1])) 
            
            y_pred[step, idx_output] = np.dot(regressors, ind._theta[idx_output])

        y_history = np.column_stack([y_history.T, y_pred[step, :]]).T

    y_true = y_true[initial_conditions_size:, :]
    return y_pred, y_true


def miso_MShooting(ind, k, y, u):
    """
    Implements the Multiple-Shooting predictor for MISO models
    Arguments:
        ind = C_Individual object
        k   = steps ahead prediction for each 'shooting'
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    initial_conditions_size = ind.lagMax 
    batch_size = initial_conditions_size + 1 + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))
    newshape = (n_batchs, batch_size, 1)

    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))

    y_true = np.resize(y, newshape)
    y_pred = y_true[:, :initial_conditions_size + 1, :]

    for shooting in range(k):

        regressors = []
        out = np.ones((n_batchs, 1, 1))
        regressors.append(out)

        for idx_output_equation in range(len(ind)):

            genetic_programming_term = ind._funcs[idx_output_equation]
            listV = [y_pred[:, shooting:shooting + initial_conditions_size + 1, :]]

            for v in listU:

                listV.append(v[:, shooting:shooting + initial_conditions_size + 1, :])
            
            out = genetic_programming_term(*listV)
            out = out[:, initial_conditions_size:, :]
            regressors.append(out)

        regressors = np.concatenate(regressors, axis=2)
        y_pred = np.concatenate((y_pred, np.dot(regressors, ind.theta)), axis=1)
    
    return np.nan_to_num(y_pred.reshape(-1, 1), nan=0), np.nan_to_num(y_true.reshape(-1, 1), nan=0)


def mimo_MShooting(ind, k, y, u):
    """
    Implements the Multiple-Shooting predictor for MIMO models
    Arguments:
        ind = C_Individual object
        k   = steps ahead prediction for each 'shooting'
        y   = n-dimensional array with output data
        u   = m-dimensional array with input data
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    
    initial_conditions_size = ind.lagMax 
    batch_size = initial_conditions_size + 1 + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))
    newshape = (n_batchs, batch_size, 1)

    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))

    y_true = np.resize(y, (n_batchs, batch_size, y.shape[1]))
    y_pred = y_true[:, :initial_conditions_size + 1, :]

    for shooting in range(k):

        listV = []
        for v in y_pred.T:
            listV.append(v.T[:, shooting:shooting + initial_conditions_size + 1].reshape(n_batchs, -1, 1))
        
        for v in listU:
            listV.append(v[:, shooting:shooting + initial_conditions_size + 1, 0].reshape(n_batchs, -1, 1))

        predictions_for_output_equation = []
        for idx_output_equation in range(len(ind)): # for output_equation in ind

            regressors = []
            out = np.ones((n_batchs, 1, 1))
            regressors.append(out)

            for idx_equation_tree in range(len(ind[idx_output_equation])): # for equation_tree in output_equation

                genetic_programming_term = ind._funcs[idx_output_equation][idx_equation_tree] # genetic_programming_term = equation_tree._funcs
                out = genetic_programming_term(*listV)
                out = out[:, initial_conditions_size:, :]
                regressors.append(out)

            regressors = np.concatenate(regressors, axis=2)
            output_pred = np.dot(regressors, ind._theta[idx_output_equation].T).reshape(-1, 1, 1)
            predictions_for_output_equation.append(output_pred)

        predictions_for_output_equation = np.concatenate(predictions_for_output_equation, axis=2)
        y_pred = np.concatenate((y_pred, predictions_for_output_equation), axis=1)

    return np.nan_to_num(y_pred.reshape(-1, y.shape[1]), nan=0), np.nan_to_num(y_true.reshape(-1, y.shape[1]), nan=0)


def mimo_FIR_MShooting(ind, k, y, u):

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    
    initial_conditions_size = ind.lagMax 
    batch_size = initial_conditions_size + 1 + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))
    newshape = (n_batchs, batch_size, 1)

    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))

    y_true = np.resize(y, (n_batchs, batch_size, y.shape[1]))
    y_pred = y_true[:, :initial_conditions_size + 1, :]

    for shooting in range(k):

        listV = []
        for v in listU:
            listV.append(v[:, shooting:shooting + initial_conditions_size + 1, 0].reshape(n_batchs, -1, 1))

        predictions_for_output_equation = []
        for idx_output_equation in range(len(ind)):

            regressors = []
            out = np.ones((n_batchs, 1, 1))
            regressors.append(out)

            for idx_equation_tree in range(len(ind[idx_output_equation])):

                func = ind._funcs[idx_output_equation][idx_equation_tree]
                out = func(*listV)
                out = out[:, initial_conditions_size:, :]
                regressors.append(out)

            regressors = np.concatenate(regressors, axis=2)
            output_pred = np.dot(regressors, ind._theta[idx_output_equation].T).reshape(-1, 1, 1)
            predictions_for_output_equation.append(output_pred)
        
        predictions_for_output_equation = np.concatenate(predictions_for_output_equation, axis=2)
        y_pred = np.concatenate((y_pred, predictions_for_output_equation), axis=1)

    return np.nan_to_num(y_pred.reshape(-1, y.shape[1]), nan=0), np.nan_to_num(y_true.reshape(-1, y.shape[1]), nan=0)
