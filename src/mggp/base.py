from __future__ import annotations
from numpy.typing import NDArray
import pickle
from functools import partial
from abc import abstractmethod
from typing import List
import sys
import random
from deap import gp, creator, base, tools
import operator
import numpy as np
import re
from .predictors import miso_OSA, miso_FreeRun, miso_MShooting, mimo_CLASSIFY, miso_FIR_INSTANT, mimo_FIR_INSTANT, miso_CLASSIFY
from .predictors import mimo_OSA, mimo_FreeRun, mimo_MShooting, mimo_FIR_MShooting, mimo_FIR_FreeRun
from .predictors import _class_probabilities, _encode_labels, _nearest_codes
import warnings
from threading import Lock

_CREATOR_LOCK = Lock()

def _roll(*args, i):
    signal, = args
    axis = 0 if np.ndim(signal) == 1 else -2
    return np.roll(signal, shift=i, axis=axis)

def sign(X1, X2):
    return np.sign(X1-X2)

def tanh(X1, X2):
    return np.tanh(X1-X2)

def sin(X1):
    return np.sin(X1)

def div(X1, X2):
    return X1 / X2

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.mean((y_true - y_pred) ** 2)


def accuracy_score(y_true, y_pred):
    """
    Accuracy Score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))
    return np.mean(y_true == y_pred)


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Log Loss (Cross-Entropy)

    Suporta:
    - Binário: y_true em {0,1}, y_pred probabilidade da classe 1
    - Multiclasse: y_true como labels inteiros ou one-hot, y_pred como probs
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clipping para evitar log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Caso binário
    if y_pred.ndim == 1:
        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch for binary case")

        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    # Caso multiclasse
    elif y_pred.ndim == 2:
        n_samples, n_classes = y_pred.shape

        # Se y_true está como labels (inteiros)
        if y_true.ndim == 1:
            if len(y_true) != n_samples:
                raise ValueError("y_true length mismatch")

            return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))

        # Se y_true é one-hot
        elif y_true.ndim == 2:
            if y_true.shape != y_pred.shape:
                raise ValueError("Shape mismatch for one-hot case")

            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        else:
            raise ValueError("Invalid y_true format")

    else:
        raise ValueError("Invalid y_pred dimensions")



class Individual(list):
    input_only = False
    multi_output = False
    classes_ = None

    def __init__(self, data: List = []):
        super().__init__(data)
        self.funcs = []
        self.lagMax = None
        self.theta = np.array([])
        self.terminals = ''
        self.nTerms = 0
        self.logistic_model = None 

    def makeRegressors(self, y: NDArray, u: NDArray, align: str = "OSA"):
        """Evaluate genes on full signals, then discard the required history."""
        y = np.asarray(y)
        u = np.asarray(u)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        u = u.reshape(-1, 1) if u.ndim == 1 else u

        if len(y) != len(u):
            raise ValueError("Inputs and outputs must have the same number of samples")

        if self.multi_output and y.shape[1] != len(self):
            raise ValueError("Wrong number of output channels")

        if not self.multi_output and y.shape[1] != 1:
            raise ValueError("Expected a single output channel")

        if align not in ("OSA", "INSTANT"):
            raise ValueError("Unknown regressor alignment")

        input_shift = int(self.input_only and align == "OSA")
        start = self.lagMax + input_shift

        if start >= len(y):
            raise ValueError("Not enough samples for the chosen lag or alignment")
        variables = []

        if not self.input_only:
            # At row k an output terminal means y[k-1].

            if self.classes_ is not None:
                previous = np.zeros((len(y), 1))
                previous[1:, 0] = _encode_labels(y[:-1], self.classes_)
                variables.append(previous)

            else:
                variables.extend(np.roll(y[:, j:j+1], 1, axis=0) for j in range(y.shape[1]))

        inputs = np.roll(u, input_shift, axis=0) if input_shift else u
        variables.extend(inputs[:, j:j+1] for j in range(inputs.shape[1]))

        def design(genes):
            columns = [np.ones(len(y) - start)]
            columns.extend(gene(*variables).reshape(-1)[start:] for gene in genes)
            return np.column_stack(columns)

        if self.multi_output:
            return [design(genes) for genes in self.funcs]
        return design(self.funcs)


    def leastSquares(self, y: NDArray, u: NDArray, align: str = "OSA"):

        matrices = self.makeRegressors(y, u, align=align)
        self.input_offset = int(self.input_only and align == "OSA")
        target = np.asarray(y)[self.lagMax + self.input_offset:]

        if self.classes_ is not None:
            target = _encode_labels(target, self.classes_)

        if not self.multi_output:

            solver = theta_fir if self.input_only else theta_miso
            self.theta = solver(matrices, target).reshape(-1, 1)

        else:

            coefficients = [theta_mimo(p, target[:, j])
                            for j, p in enumerate(matrices)]

            if len({len(theta) for theta in coefficients}) == 1:
                self.theta = np.stack(coefficients)

            else:
                self.theta = coefficients

        return self.theta


    def constrained_least_squares(self, y: NDArray, p: NDArray, constraints=None) -> NDArray:

        yd = np.asarray(y).reshape(-1, 1)[self.lagMax:]
        if constraints is None:
            self.theta = np.linalg.lstsq(p, yd, rcond=None)[0]
            return self.theta

        # Solve in the null space of S. This also enforces the equalities when
        # repeated genes make the regressor matrix rank deficient.
        S = np.atleast_2d(constraints["S"])
        c = np.asarray(constraints["c"]).reshape(-1, 1)
        particular = np.linalg.lstsq(S, c, rcond=None)[0]

        if not np.allclose(S @ particular, c):
            raise ValueError("Inconsistent coefficient constraints")

        _, _, vt = np.linalg.svd(S, full_matrices=True)
        null_space = vt[np.linalg.matrix_rank(S):].T
        adjustment = np.linalg.lstsq(p @ null_space, yd - p @ particular, rcond=None)[0]

        self.theta = particular + null_space @ adjustment

        return self.theta

    def identify_term_clusters(self, y: NDArray, u: NDArray) -> dict:
        """Identify regressor term clusters used by the hysteretic constraints (Property 1).

        The paper groups parameters into clusters and enforces:
        - sum of *linear output* parameters = 1
        - sum of all other (non-ϕ) clusters = 0
        - terms containing ϕ-functions are ignored in steady-state constraints

        Important: `makeRegressors` in this codebase includes a leading bias column of ones.
        The paper's model form (Eq. 10) has no explicit intercept. If we keep a bias term,
        to preserve a *continuum of equilibria* we must constrain its parameter to 0.
        """
        p = self.makeRegressors(y, u)
        if not isinstance(p, np.ndarray):
            raise TypeError(
                "identify_term_clusters currently supports scalar-output regressors matrices (SISO/MISO/FIR). "
                "For MIMO, run this per-output (one Ψ per output) or refactor makeRegressors accordingly."
            )

        n_terms = p.shape[1]

        clusters = {
            "bias": [],
            "linear_output": [],
            "linear_input": [],
            "nonlinear_y": [],
            "nonlinear_u": [],
            "cross_terms": [],
            "phi_terms": [],
        }

        for term_index in range(n_terms):
            clusters[self._classify_term(term_index)].append(term_index)

        return clusters


    @staticmethod
    def _node_is_q(node) -> bool:
        return isinstance(node, gp.Primitive) and re.fullmatch(r"q\d+", getattr(node, "name", "")) is not None


    @staticmethod
    def _node_is_phi(node) -> bool:
        return isinstance(node, gp.Primitive) and getattr(node, "name", "") in ("subtraction", "sign")


    def _classify_term(self, term_index: int) -> str:

        if term_index == 0:
            return "bias"

        tree = self[term_index - 1]

        if any(self._node_is_phi(node) for node in tree):
            return "phi_terms"

        variables = {str(node.value)[0] for node in tree if isinstance(node, gp.Terminal)}

        if variables == {"u", "y"}:
            return "cross_terms"

        variable, = variables

        if all(isinstance(node, gp.Terminal) or self._node_is_q(node) for node in tree):
            return "linear_output" if variable == "y" else "linear_input"

        return "nonlinear_y" if variable == "y" else "nonlinear_u"

    def hysteretic_constrained_ls(self, y: NDArray, u: NDArray) -> NDArray:
        """Constrained LS enforcing the hysteretic equilibrium conditions (Property 1).

        Enforced constraints (paper Section 3):
        - sum(theta_linear_output) = 1
        - sum(theta_cluster) = 0 for all other non-ϕ clusters

        Additionally (because this codebase includes a bias term):
        - theta_bias = 0
        """
        clusters = self.identify_term_clusters(y, u)

        constraints = []
        p = self.makeRegressors(y, u)
        length_model = p.shape[1]

        if clusters["bias"]:
            s = np.zeros(length_model)
            s[clusters["bias"]] = 1.0
            constraints.append((s, 0.0))

        if not clusters["linear_output"]:
            raise ValueError("No linear output term (pure delay of y) found. Property-1 constraints cannot be enforced.")

        s = np.zeros(length_model)
        s[clusters["linear_output"]] = 1.0
        constraints.append((s, 1.0))

        for cluster_name in ("linear_input", "cross_terms", "nonlinear_y", "nonlinear_u"):
            idxs = clusters[cluster_name]
            if not idxs:
                continue
            s = np.zeros(length_model)
            s[idxs] = 1.0
            constraints.append((s, 0.0))

        # for cluster_name in ("linear_input",):
        #     idxs = clusters[cluster_name]
        #     if not idxs:
        #         continue
        #     s = np.zeros(length_model)
        #     s[idxs] = 1.0
        #     constraints.append((s, 1.0))

        S = np.vstack([c[0] for c in constraints])
        c = np.array([c[1] for c in constraints])

        return self.constrained_least_squares(y, p, {"S": S, "c": c})

    def _classification_scores(self, mode, *args):

        if not self.logistic_model:
            raise ValueError("Classification model has not been trained")

        if mode == "INSTANT":
            scores, targets = mimo_CLASSIFY(self, *args)

        elif mode in ("OSA", "FreeRun", "MShooting"):
            scores, targets = self.predict(mode, *args)

        else:
            raise ValueError("Choose INSTANT, OSA, FreeRun or MShooting for classification")

        if self.classes_ is not None:
            targets = targets[:, 0]

        return scores, targets

    def predict_proba(self, mode: str = "INSTANT", *args: tuple):
        """Return one probability per class and the aligned targets in their original encoding."""

        scores, targets = self._classification_scores(mode, *args)

        return _class_probabilities(scores, self.classes_), targets

    def predict_classes(self, mode: str = "INSTANT", *args: tuple):

        if self.classes_ is not None:
            scores, targets = self._classification_scores(mode, *args)
            return self.classes_[_nearest_codes(scores, len(self.classes_))], targets

        probabilities, targets = self.predict_proba(mode, *args)
        labels = probabilities.argmax(axis=1)

        return np.eye(probabilities.shape[1])[labels], targets

    def score_classification(self, yd: NDArray, yp: NDArray, mode: str ="accuracy") -> float:
        """Métricas de avaliação para classificação"""
        if self.classes_ is not None:

            yd = _encode_labels(yd, self.classes_)

            if mode == "log_loss":
                if np.shape(yp) != (len(yd), len(self.classes_)):
                    raise ValueError("Expected one probability column per training class")

            else:
                yp = _encode_labels(yp, self.classes_)
        
        if mode == "accuracy":
            return accuracy_score(yd, yp)
        
        elif mode == "log_loss":
            # Para log_loss, precisamos das probabilidades
            return log_loss(yd, yp)
        
        elif mode == "f1_macro":

            from sklearn.metrics import f1_score

            if self.classes_ is not None:
                return f1_score(yd, yp, labels=np.arange(len(self.classes_)), average='macro', zero_division=0)

            return f1_score(yd, yp, average='macro')
        
        else:
            raise ValueError("Choose a valid metric: accuracy, log_loss, f1_macro")

    
    def predict(self, mode: str = "OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return mimo_OSA(self, *args)
        
        if mode == "FreeRun":
            return mimo_FreeRun(self, *args)
        
        if mode == "MShooting":
            return mimo_MShooting(self, *args)
        
        else:
            raise ValueError("Choose a mode between: OSA, FreeRun, MShooting")

    
    def _mape(self, yd: NDArray, yp: NDArray) -> float:
        """MAPE Calculate in the array form."""
        diff = np.abs(yd - yp)
        denominator = np.abs(np.max(yd, axis=0) - np.min(yd, axis=0))
        
        varying = denominator > 0
        if not np.any(varying):
            return np.nan
        per_output = np.nanmean(diff[:, varying] / denominator[varying], axis=0)
        return 100 * np.nanmean(per_output)

    def _compute_metric_per_output(self, yd: NDArray, yp: NDArray, metric_func: callable) -> float:
        """Apply a metric (MSE, RMSE) in each output and return the mean"""
        return np.mean([metric_func(yd[:, i], yp[:, i]) for i in range(yd.shape[1])])
        # return max([metric_func(yd[:, i], yp[:, i]) for i in range(yd.shape[1])])

    
    def score(self, yd: NDArray, yp: NDArray, mode: str ="MSE"):
        """Calculate the error metric choosed (MSE, MAPE, RMSE)."""
        if mode not in ["MSE", "NMSE", "MAPE", "RMSE"]:
            raise ValueError("Choose a valid metric: MSE, NMSE, MAPE or RMSE")
        
        if mode == "MSE":
            return self._compute_metric_per_output(yd, yp, mean_squared_error)
        
        if mode == "MAPE":
            return self._mape(yd, yp)
        
        if mode == "RMSE":
            return self._compute_metric_per_output(yd, yp, lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
        
        if mode == "NMSE":
            return self._compute_metric_per_output(yd, yp, self._nmse_variance)
        
    
    def _nmse_variance(self, y_true, y_pred) -> float:

        mse = mean_squared_error(y_true, y_pred)
        variance = np.var(y_true)
        return mse / variance if variance != 0 else mse


    @abstractmethod
    def model2List(self):
        pass


    def parse_tree(self, tree: gp.PrimitiveTree, i: int = 0, lag_acc: int = 0) -> tuple[str, int]:
        node = tree[i]

        if self._node_is_q(node):
            k = int(node.name[1:])
            return self.parse_tree(tree, i + 1, lag_acc + k)

        elif isinstance(node, gp.Primitive):
            args = []
            next_i = i + 1

            for _ in range(node.arity):
                arg_str, next_i = self.parse_tree(tree, next_i, lag_acc)
                args.append(arg_str)

            if node.name == "mul":
                return " * ".join(args), next_i

            elif node.name == "subtraction":
                return f"({args[0]} - {args[1]})", next_i

            elif node.name == "sign":
                return f"sign({args[0]} - {args[1]})", next_i

            else:
                return f"{node.name}({', '.join(args)})", next_i

        elif isinstance(node, gp.Terminal):
            var = node.value           
            base = var

            if base.startswith('u'):
                lag = lag_acc + getattr(self, "input_offset", 0)
            elif base.startswith('y'):
                lag = lag_acc + 1
            else:
                lag = lag_acc

            return f"{base}[k-{lag}]" if lag > 0 else f"{base}[k]", i + 1

        else:
            return "", i + 1
        

    def to_equation(self, precision: int = 5) -> str:

        outputs = self if self.multi_output else [self]
        coefficients = self.theta if self.multi_output else [self.theta]
        lines = []
        for k, (genes, theta) in enumerate(zip(outputs, coefficients)):
            if self.multi_output:
                lines.append(f"Output {k+1}:")
            bias, *weights = np.asarray(theta).ravel()
            lines.append(f"{bias:.{precision}e} + ")
            for tree, weight in zip(genes, weights):
                expression, _ = self.parse_tree(tree)
                lines.append(f"{weight:.{precision}e} * {expression} + ")
        return "\n".join(lines)

class Element(object):
    def __init__(self,
                 weights: tuple = (-1,),
                 nDelays: int | List[int] = 3,
                 nInputs: int = 1,
                 nOutputs: int = 1,
                 nTerms: int = 10,
                 maxHeight: int = 5,
                 mode: str = "MISO",
                 single_delay_only: bool = False,
                 operators: List[str] = ['mul', 'subtraction', 'sign', 'add', 'tanh', 'div', 'sin']):
        
        self._pset: gp.PrimitiveSet = None
        self._toolbox: base.Toolbox = None
        self._mode = mode.upper()
        self._single_delay_only = single_delay_only

        if self._mode != "FIR" and nInputs == 1 and nOutputs == 1:
            self._mode = "SISO"
        
        if self._mode == "FIR":
            self._nVar = nInputs

        elif self._mode == "SISO":
            self._nVar = 2 

        else:
            self._nVar = nInputs + nOutputs
        
        if type(nDelays) is int:
            self._delays = np.arange(1, nDelays + 1)
        elif type(nDelays) is list:
            self._delays = nDelays

        self.operators = operators
        self.primitives_sets = {
            "mul": (operator.mul, 2),
            "sign": (sign, 2),
            "subtraction": (operator.sub, 2),
            "add": (operator.add, 2),
            "tanh": (tanh, 2),
            "div": (div, 2),
            "sin": (sin, 1)
        }

        self.iniciatePrimitivesSets()

        self._weights = weights
        self.nTerms = nTerms
        self._nOutputs = nOutputs
        self._maxHeight = maxHeight     

        types = {"MISO": IndividualMISO, "MIMO": IndividualMIMO, "SISO": IndividualSISO,
                 "FIR": IndividualFIR if nOutputs == 1 else IndividualFIRMIMO}
        if self._mode not in types:
            raise ValueError("Choose MISO, MIMO, FIR or SISO")
        # DEAP publishes classes under global names. Bind this element's classes
        # together before another engine can register different weights or terminals.
        with _CREATOR_LOCK, warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"deap\.creator",
                                    message=r"A class named '(Program|FitnessMin|Individual)' has already been created.*")
            creator.create("Program", gp.PrimitiveTree, fitness=None, pset=self.pset)
            creator.create("FitnessMin", base.Fitness, weights=self._weights)
            creator.create("Individual", types[self._mode], fitness=creator.FitnessMin)
            self._program_type = creator.Program
            self._individual_type = creator.Individual
        self.iniciateToolbox()


    def iniciatePrimitivesSets(self) -> None:
        
        self._pset = gp.PrimitiveSet("main", self._nVar)
        
        for operator in self.operators:

            if operator in self.primitives_sets.keys():
                func, arity = self.primitives_sets[operator]
                self._pset.addPrimitive(func, arity, name=operator)
            
            else:
                raise ValueError(f"Operator '{operator}' is not defined in primitives_sets.")
        
        if not self._single_delay_only:
            delays = [partial(_roll, i=i) for i in self._delays]

            for i, roll in zip(self._delays, delays):
                self._pset.addPrimitive(roll, 1, name=f'q{i}')
        
        else:
            self._pset.addPrimitive(partial(_roll, i=1), 1, name=f'q1')


    def iniciateToolbox(self) -> None:

        self._toolbox = base.Toolbox()
        # self._toolbox.register("_expr", generate_terminal_only)
        self._toolbox.register("_expr", self.genGrowLimitedNodes, pset=self._pset, min_depth=0, max_depth=self._maxHeight, max_nodes=20)
        # self._toolbox.register("_expr", gp.genHalfAndHalf, pset=self._pset, min_=0, max_=self._maxHeight)
        self._toolbox.register("_program", tools.initIterate, self._program_type, self._toolbox._expr)

        if self._mode == "SISO":
            
            self._toolbox.register("individual", tools.initRepeat, self._individual_type, self._toolbox._program, self.nTerms)
        
        elif self._mode == "MISO" or (self._mode == "FIR" and self._nOutputs == 1):
        # Para MISO ou FIR SISO (1 saída)
            self._toolbox.register("individual", tools.initRepeat, self._individual_type, self._toolbox._program, self.nTerms)
        
        elif self._mode == "MIMO" or (self._mode == "FIR" and self._nOutputs > 1):
            self._toolbox.register("_outputs", tools.initRepeat, list, self._toolbox._program, self.nTerms)
            self._toolbox.register("individual", tools.initRepeat, self._individual_type, self._toolbox._outputs, self._nOutputs)
        
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)


    def genGrowLimitedNodes(self, pset: gp.PrimitiveSet, min_depth: int, max_depth: int, max_nodes: int) -> List:
        """Versão do genGrow com limite de nós"""
        while True:
            expr = gp.genHalfAndHalf(pset, min_depth, max_depth)
            tree = gp.PrimitiveTree(expr)
            if len(tree) <= max_nodes:
                return expr


    @property
    def pset(self) -> gp.PrimitiveSet:
        return self._pset
    

    @property
    def toolbox(self) -> base.Toolbox:
        return self._toolbox


    def renameArguments(self, dictionary: dict = {'ARG0': 'y', 'ARG1': 'u'}) -> None:
        self._pset.renameArguments(**dictionary)


    def addPrimitive(self, *args: tuple) -> None:
        self._pset.addPrimitive(*args)


    def buildModelFromList(self, listString: List[str]) -> Individual:
        model = self._individual_type()

        if self._mode == "SISO":
            for string in listString:
                model.append(gp.PrimitiveTree.from_string(string, self.pset))
    
        if self._mode == "MISO" or self._mode == "FIR" and self._nOutputs == 1:
            for string in listString:
                model.append(gp.PrimitiveTree.from_string(string, self.pset))
        
        if self._mode == "MIMO" or (self._mode == "FIR" and self._nOutputs > 1):
            for out in listString:
                aux = [gp.PrimitiveTree.from_string(string, self.pset) for string in out]
                model.append(aux)
        return model


    def buildRandomModel(self) -> Individual:
        return self._toolbox.individual()


    def compileModel(self, model) -> None:
        if self._mode == 'SISO':
            model.funcs = [gp.compile(tree, self.pset) for tree in model]

        elif self._mode == 'MISO' or self._mode == "FIR" and self._nOutputs == 1:
            model.funcs = [gp.compile(tree, self.pset) for tree in model]

        elif self._mode == 'MIMO' or (self._mode == "FIR" and self._nOutputs > 1):
            model.funcs = [[gp.compile(tree, self.pset) for tree in out] for out in model]
            # model.funcs = [[self._compile_to_function(tree, self.pset) for tree in out] for out in model]

        self._setModelLagMax(model)


    def _compile_to_function(self, expr, pset: gp.PrimitiveSet) -> dict:
        code = str(expr)

        if len(pset.arguments) > 0:
            args = ",".join(arg for arg in pset.arguments)
            code = f"def generated_function({args}):\n    return {code}"

        local_scope = {}
        try:
            exec(code, pset.context, local_scope)
            return local_scope["generated_function"]
        
        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError("DEAP: Error in tree evaluation: Python cannot evaluate a tree higher than 90. "
                "To avoid this problem, you should use bloat control on your operators. "
                "See the DEAP documentation for more information. DEAP will now abort."
            ).with_traceback(traceback)


    def _setModelLagMax(self, model):
        def history(tree):
            # Walk the prefix expression with the cumulative shift at each node.
            pending = [0]
            lags = []
            for node in tree:
                lag = pending.pop()
                if node.arity:
                    if model._node_is_q(node):
                        lag += int(node.name[1:])
                    pending.extend([lag] * node.arity)
                else:
                    # Raw Element objects can still have DEAP's ARG names.
                    is_output = (not model.input_only
                                 and node.value in self.pset.arguments[:self._nOutputs])
                    lag += int(is_output)
                    lags.append(lag)
                    terminals.append(f"{node.value}[i-{lag}]")
            return max(lags, default=0)

        terminals = []
        equations = model if model.multi_output else [model]
        model.lagMax = max((history(tree) for genes in equations for tree in genes), default=0)
        model.terminals = " * ".join(terminals)

    def save(self, filename: str, dictionary: dict) -> None:
        
        with open(filename, 'wb') as f:
            pickle.dump(dictionary, f)
            f.close()

    def load(self, filename: str) -> dict:
        
        with open(filename, 'rb') as f:
            o = pickle.load(f)
            f.close()
            return o



def theta_miso(p, yd):
    return np.linalg.lstsq(p, yd, rcond=None)[0]
    # return np.linalg.lstsq(p, yd, rcond=None)[0]


def theta_mimo(p, yd):
    # return np.dot(np.dot(np.linalg.inv(np.dot(p.T, p)), p.T), yd)
    return np.linalg.lstsq(p, yd, rcond=None)[0]
    # return cp.linalg.lstsq(cp.asarray(p), cp.asarray(yd), rcond=None)[0]


def theta_fir(p, yd):
    return np.linalg.lstsq(p, yd, rcond=None)[0]


class IndividualSISO(Individual):
    
    def __init__(self, data: str = []):
        super().__init__(data)
    
    def predict(self, mode: str = "OSA", *args)  -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise ValueError("Choose a mode between: OSA, FreeRun, MShooting")
    

    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string
    

    def model2List(self) -> List[str]:
        return [str(tree) for tree in self]


class IndividualMISO(Individual):

    def __init__(self, data: str = []):
        super().__init__(data)

    def predict(self, mode: str = "OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise ValueError("Choose a mode between: OSA, FreeRun, MShooting")

    # def predict_proba(self, mode="INSTANT", *args):
    #     """Predição de probabilidades para classificação"""
    #     def one_hot_argmax(x):

    #         result = np.zeros_like(x)
    #         result[np.argmax(x)] = 1
            
    #         return result

    #     def softmax(x):
    #         """Calcula softmax para cada linha do vetor de entrada x."""
    #         # Subtrair o máximo melhora a estabilidade numérica (evita overflow)
    #         e_x = np.exp(x - np.max(x))
    #         return e_x / e_x.sum()
        
    #     if mode == "INSTANT":
    #         X_regressors, y_true = miso_CLASSIFY(self, *args)
    #         # y_true = y_true[:len(y_true)-1] # para a classificação FIR precisa desse termo. Já para a MIMO não.
    #     # elif mode == "FreeRun":
    #     #     X_regressors, y_true = mimo_FreeRun(self, *args)
        
    #     elif mode == "MShooting":
    #         X_regressors, y_true = miso_MShooting(self, *args)
        
    #     else:
    #         raise ValueError("Choose a mode between: INSTANT")
        
    #     if self.logistic_model is not None:            
    #         # probabilities = np.array([one_hot_argmax(x) for x in X_regressors])
    #         pred_one_hot_encode = np.array([one_hot_argmax(softmax(x)) for x in X_regressors])
    #         # pred_labels = np.array([np.argmax(prob) for prob in pred_one_hot_encode]).astype(np.float64)
    #         return pred_one_hot_encode, y_true
        
    #     else:
    #         raise Exception("Logistic regression model not trained!")



    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string


    def model2List(self) -> List[str]:
        listString = []
        for tree in self:
            listString.append(str(tree))
        # return listString
        return [str(tree) for tree in self]


class IndividualMIMO(Individual):
    multi_output = True

    def __init__(self, data: str = []):
        super().__init__(data)

    
    def __str__(self) -> str:
        string = ''
        i = 1
        for out in self:
            string += 'Output %d:\n\n1\n' % (i)
            for tree in out:
                string += str(tree) + '\n'
            i += 1
            string += '\n'
        return string


    def model2List(self) -> List[List[str]]:
        return [[str(tree) for tree in out] for out in self]


class IndividualFIR(Individual):

    input_only = True

    def __init__(self, data: str = []):
        super().__init__(data)

    # def predict(self, mode="OSA", *args):
    #     if mode == "OSA":
    #         return miso_OSA(self, *args)
    #     if mode == "FreeRun":
    #         return miso_FreeRun(self, *args)
    #     if mode == "MShooting":
    #         return miso_MShooting(self, *args)
    #     else:
    #         raise ValueError("Choose a mode between: OSA, FreeRun, MShooting")

    # def makeRegressors(self, y, u):
    #     if len(u.shape) == 1:
    #         u = u.reshape(-1, 1)

    #     listV = []
    #     for v in u.T:
    #         listV.append(v[:-1].reshape(-1, 1))

    #     p = np.ones((u.shape[0] - self.lagMax - 1, len(self) + 1))

    #     for i in range(len(self)):
    #         func = self.funcs[i]
    #         out = func(*listV)
    #         p[:, i + 1] = out.reshape(-1)[self.lagMax:]
    #     return p

    # def leastSquares(self, y, u):
    #     '''
    #     The leastSquare(y,u) function implements the Least Squares method
    #     for parameter estimation.
        
    #     The arguments are the output y and the inputs u, in which each entry 
    #     must be in column formm.
    #     '''
    #     p = self.makeRegressors(y, u)
    #     if np.linalg.cond(p, -2) < 1e-10:
    #         raise np.linalg.LinAlgError(
    #             'Ill conditioned regressors matrix!')
    #     yd = y[self.lagMax + 1:]
    #     # self.theta = np.linalg.inv(p.T @ p) @ p.T @ yd
    #     self.theta = theta_fir(p, yd)
    #     if len(self.theta.shape) == 1:
    #         self.theta = self.theta.reshape(-1, 1)
    #     return self.theta

    def predict(self, mode: str ="OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "INSTANT":
            return miso_FIR_INSTANT(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise ValueError("Choose a mode between: OSA, INSTANT, FreeRun, MShooting")


    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string

    def model2List(self) -> List[str]:
        return [str(tree) for tree in self]


class IndividualFIRMIMO(Individual):
    input_only = True

    multi_output = True

    def __init__(self, data: list = []):
        super().__init__(data)

    # def makeRegressors(self, y, u):
    #     if len(u.shape) == 1:
    #         u = u.reshape(-1, 1)

    #     listV = []
    #     for v in u.T:
    #         listV.append(v[:-1].reshape(-1, 1))

    #     P = []
    #     for o in range(len(self)):  # Para cada saída
    #         p = np.ones((u.shape[0] - self.lagMax - 1, len(self[o]) + 1))
    #         for i in range(len(self[o])):  # Para cada termo da saída
    #             func = self.funcs[o][i]
    #             out = func(*listV)
    #             p[:, i + 1] = out.reshape(-1)[self.lagMax:]
    #         P.append(p)
    #     return P

    # def leastSquares(self, y, u):
    #     P = self.makeRegressors(y, u)
    #     self.theta = np.array([theta_mimo(P[o], y[self.lagMax + 1:, o]) for o in range(len(P))])
    #     return self.theta
    
    # def predict(self, mode="OSA", *args):
    #     if mode == "OSA":
    #         return mimo_OSA(self, *args)
    #     if mode == "FreeRun":
    #         return mimo_FIR_FreeRun(self, *args)
    #     if mode == "MShooting":
    #         return mimo_FIR_MShooting(self, *args)
    #     else:
    #         raise ValueError("Choose a mode between: OSA, FreeRun, MShooting")


    def predict(self, mode: str ="OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return mimo_OSA(self, *args)
        if mode == "INSTANT":
            return mimo_FIR_INSTANT(self, *args)
        if mode == "FreeRun":
            return mimo_FIR_FreeRun(self, *args)
        if mode == "MShooting":
            return mimo_FIR_MShooting(self, *args)
        else:
            raise ValueError("Choose a mode between: OSA, INSTANT, FreeRun, MShooting")


    def __str__(self) -> str:
        string = ''
        i = 1
        for out in self:
            string += 'Output %d:\n\n1\n' % (i)
            for tree in out:
                string += str(tree) + '\n'
            i += 1
            string += '\n'
        return string

    def model2List(self) -> List[List[str]]:
        return [[str(tree) for tree in out] for out in self]

