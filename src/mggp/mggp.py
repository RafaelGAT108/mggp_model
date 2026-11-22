from __future__ import annotations
from numpy.typing import NDArray

# import multiprocessing
from typing import Literal, Tuple, Optional, List
import numpy as np
from .base import Element, Individual
import time
import os
import re
from deap import tools
from deap.tools import sortNondominated
from copy import deepcopy
from .mutations import *
from .crossings import *
from .utils import *
from tqdm.auto import tqdm
from IPython.display import clear_output
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict

class MGGP:

    def __init__(self,
                 inputs: NDArray = np.array([[0, 0]]),
                 outputs: NDArray = np.array([[0, 0]]),
                 generations: int = 100,
                 validation: Optional[Tuple[NDArray, NDArray]] = (None, None),
                 evaluationMode: Literal['RMSE', 'MSE', 'MAPE'] = 'RMSE',
                 evaluationType: Literal['OSA', 'MShooting', 'FreeRun', 'INSTANT'] = 'MShooting',
                 evaluationTypeTest: Literal['OSA', 'MShooting', 'FreeRun', 'INSTANT'] = 'FreeRun',
                 k: int = 5,
                 nTerms: int = 15,
                 maxHeight: int = 15,
                 weights: tuple = (-1,),
                 nDelays: int | List[int] = 15,
                 crossoverRate: float = 0.8,
                 mutationRate: float = 0.1,
                 populationSize: int = 100,
                 elitePercentage: int = 10,
                 filename: str = "best_model",
                 mode: Literal['NARX', 'FIR'] = 'NARX',
                 problem_type: Literal['regression', 'classification'] = 'regression',
                 classification_metric: Literal['accuracy', 'log_loss', 'f1_macro'] = 'accuracy',
                 froe_mode: bool = False, 
                 pruning_probability: float = 0.5,     
                 pruning_tolerance: float = 1e-5,      
                 phi_functions: List[str] = ['subtraction', 'sign'],  
                 operators: List[str] = ['mul'],   
                 new_evaluation: callable = None,
                 **kwargs):
        """
        Args:
            inputs (ndarray): The inputs in the system. Each column represent an input.
            outputs (ndarray): The outputs in the system. Each column represent an output.
            generations (int): Number of generations to train the model.
            validation (Optional[Tuple[NDArray, NDArray]]): Inputs and outputs to validate the model. Must be a tuple (inputs, outputs).
            evaluationMode (Literal['RMSE', 'MSE', 'MAPE']):  Mode to evaluate the models and ranking the better.
            evaluationType (Literal['OSA', 'MShooting', 'FreeRun']): One-Step-Ahead, Multiple-Shooting and Free-Run predictors.
            evaluationTypeTest (Literal['OSA', 'MShooting', 'FreeRun']): One-Step-Ahead, Multiple-Shooting and Free-Run predictors for Test.
            k (int): Used with Multiple-Shooting predictor. Define the number of shooting.
            nTerms (int): Number of terms each output model will possess.
            maxHeight (int): Maximum height of Genetic Program individual.
            weights (tuple): Defines the type of optimization (-1 for minimization, 1 for maximization). It must be a tuple.
            nDelays (float | Literal['fixed']): The number that will define the backshift operators q^{-n}, for n in delays.
            crossoverRate (float): Crossover probability.
            mutationRate (float): Mutation probability.
            populationSize (int): Population size.
            elitePercentage (int): Percentile of population to be kept in the next generation.
            problem_type (Literal['regression', 'classification']): problem type ('regression' or 'classification')
            classification_metric (Literal['accuracy', 'log_loss', 'f1_macro']): Metric for evaluation in classification problem type
        """

        self.inputs = inputs
        self.outputs = outputs
        self.validation = validation
        self.generations = generations
        self.evaluationMode = evaluationMode
        self.evaluationType = evaluationType
        self.evaluationTypeTest = evaluationTypeTest
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.elitePercentage = elitePercentage
        self.k = k
        self.nTerms = nTerms
        self.new_evaluation = new_evaluation
        self.fronts = []

        os.makedirs('models_saved', exist_ok=True)
        if self.new_evaluation is not None:
            self.weights = (-1, -1)

        else:
            self.weights = weights

        self.nDelays = nDelays
        self.single_delay_only = True if isinstance(self.nDelays, int) and self.nDelays == 1 else False

        self.maxHeight = maxHeight
        self.filename = filename
        self.problem_type = problem_type
        self.classification_metric = classification_metric
        self.nInputs = self.inputs.shape[1]
        self.nOutputs = self.outputs.shape[1]
        self.froe_mode = froe_mode
        self.pruning_probability = pruning_probability
        self.pruning_tolerance = pruning_tolerance
        self.phi_functions = phi_functions 
        self.operators = operators
        if self.evaluationMode not in ["MSE", "NMSE", "MAPE", "RMSE"]:
            raise ValueError("Choose a measure between:\n" +
                            "MSE, NMSE, MAPE, or RMSE")

        if self.nInputs == 0 or self.nOutputs == 0 or len(inputs) != len(outputs):
            raise ValueError("Expected input and output channels with matching sample counts")
        if mode == "FIR":
            self.mode = "FIR"
        elif self.nOutputs > 1:
            self.mode = "MIMO"
        else:
            self.mode = "SISO" if self.nInputs == 1 else "MISO"

        self.element = Element(weights=self.weights,
                               nDelays=self.nDelays,
                               nInputs=self.nInputs,
                               nOutputs=self.nOutputs,
                               nTerms=self.nTerms,
                               maxHeight=self.maxHeight,
                               mode=self.mode,
                               single_delay_only=self.single_delay_only,
                               operators=self.operators)

        self.element.renameArguments(self.buildArgumentsDict())

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", self.evaluation)

        self._mutList = []
        self._crossList = []
        self._stats = self._createStatistics()

        self._hofSize = int(round(self.populationSize * (self.elitePercentage / 100)))
        self._hof = tools.HallOfFame(max(1, self._hofSize))

        if self.new_evaluation is not None:
            self._toolbox.register("select", tools.selTournamentDCD)
        
        else:
            self._toolbox.register("select", tools.selTournament, tournsize=2)

        self.addMutation(MutGPOneTree)
        self.addMutation(MutGPUniform)
        self.addMutation(MutGPReplace)

        self.addCrossOver(CrossHighUniform)
        self.addCrossOver(CrossLowUniform)

    def addMutation(self, mutation):
        self._mutList.append(mutation(self.element))

    def addCrossOver(self, crossover):
        self._crossList.append(crossover(self.element))

    def _fir_align(self, eval_type: str) -> str:
        """
        - OSA: usa u[k-1] -> y[k] (seu alinhamento antigo)
        - INSTANT: usa u[k] -> y[k]
        """
        if self.mode == "FIR" and eval_type == "INSTANT":
            return "INSTANT"
        return "OSA"


    def _fit_model(self, ind):
        self.element.compileModel(ind)
        if self.problem_type == "classification":
            ind.logistic_model = True
        if self.mode == "FIR":
            ind.leastSquares(self.outputs, self.inputs, align=self._fir_align(self.evaluationType))
        elif self.problem_type == "regression" and self.mode in ("SISO", "MISO") and "sign" in self.operators:
            ind.hysteretic_constrained_ls(self.outputs, self.inputs)
        else:
            ind.leastSquares(self.outputs, self.inputs)

    def _delAttr(self, ind: Individual) -> None:
        """Remove atributos do indivíduo de forma segura"""
        
        attrs_to_remove = ['fitness.values', 'funcs', 'lagMax', 'kfuncs']
        
        for attr in attrs_to_remove:
            try:
                if '.' in attr:
                    parts = attr.split('.')
                    obj = ind
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            break
                    else:
                        if hasattr(obj, parts[-1]):
                            delattr(obj, parts[-1])
                else:

                    if hasattr(ind, attr):
                        delattr(ind, attr)
            except (AttributeError, TypeError):
                continue


    def stream(self) -> None:
        print(self._logbook.stream)

    def initPop(self, seed: Optional[List] = None) -> None:
        seed = [] if seed is None else seed
        if len(seed) > self.populationSize:
            raise ValueError('Seed exceeds population size!')

        self._hof.clear()
        self.fronts = []
        self._logbook = tools.Logbook()
        self._logbook.header = "gen", "evals", "fitness"
        self._logbook.chapters["fitness"].header = "min", "avg", "max"
        if not seed:
            self._pop = self.element._toolbox.population(self.populationSize)

        else:
            self._pop = self.element._toolbox.population(self.populationSize - len(seed))
            for candidate in seed:
                individual = self.element.buildModelFromList(candidate.model2List())
                self._pop.append(individual)

        invalid_ind = [ind for ind in self._pop if not ind.fitness.valid]

        if self.evaluationType == 'OSA':
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

        else:
            fitnesses = list(tqdm(self._toolbox.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))
            # fitnesses = list(self._toolbox.map(self.evaluation, invalid_ind))
            # with ProcessPoolExecutor(max_workers=10) as executor:
            #     fitnesses = list(tqdm(executor.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

                
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = {'fitness': self._stats.compile(self._pop)}
        self._logbook.record(gen=1, evals=len(invalid_ind), **record)
        valid_ind = [ind for ind in self._pop if finite(ind)]
        if not valid_ind:
            raise RuntimeError("No finite fitness in the initial population")
        self._hof.update(valid_ind)
        if self.new_evaluation is not None:
            self._rank_population()

    def _rank_population(self):
        valid = []
        for individual in self._pop:
            if finite(individual):
                valid.append(individual)
            else:
                individual.fitness.crowding_dist = -np.inf
        self.fronts = sortNondominated(valid, k=len(valid))
        for front in self.fronts:
            tools.emo.assignCrowdingDist(front)

    def get_fitness_value(self, individual: Individual):
        return individual.fitness.values[0]


    def _createStatistics(self) -> tools.Statistics:
        stats = tools.Statistics(self.get_fitness_value)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        return stats


    def step(self, gen_number: int) -> None:
        if not getattr(self, "_pop", None):
            raise RuntimeError('Population must be initialized!')


        count = self.populationSize - self._hofSize
        if self.new_evaluation is None:
            parents = self._toolbox.select(self._pop, count)
        else:
            # DEAP draws DCD tournaments in blocks of four, even when k is smaller.
            bulk = count - count % 4
            parents = self._toolbox.select(self._pop, bulk)
            if count > bulk:
                pool = self._pop
                if len(pool) < 4:
                    pool = pool + random.choices(pool, k=4-len(pool))
                parents += self._toolbox.select(pool, 4)[:count - bulk]
        offspring = [deepcopy(ind) for ind in parents]

        for i in range(0, len(offspring) - 1, 2):
            if np.random.random() < self.crossoverRate:
                cross = random.choice(self._crossList)
                offspring[i], offspring[i + 1] = cross.cross(offspring[i], offspring[i + 1])
                self._delAttr(offspring[i])
                self._delAttr(offspring[i + 1])

        for i in range(len(offspring)):
            if np.random.random() < self.mutationRate:
                mut = random.choice(self._mutList)
                offspring[i], = mut.mutate(offspring[i])
                self._delAttr(offspring[i])


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Population"))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.new_evaluation is not None:
            self._pop = self._survive(self._pop + offspring, self.populationSize)

        else:
            elites = self._hof.items[:self._hofSize]
            # HallOfFame stores distinct genotypes and can be smaller than the quota.
            missing = self.populationSize - len(offspring) - len(elites)
            self._pop = elites + tools.selBest(self._pop, missing) + offspring
        
        self._hof.update([ind for ind in self._pop if finite(ind)])

        if self.new_evaluation is not None:
            self._rank_population()
            models = self.fronts[0]

        else:
            models = [deepcopy(self._hof[0])]

        for i, model in enumerate(models):
            self._fit_model(model)
            os.makedirs("models_saved", exist_ok=True)
            self.save_model(model, filename=f"models_saved/{self.filename}_{i}.pkl")
            
        record = {'fitness': self._stats.compile(self._pop)}

        self._logbook.record(gen=gen_number+1, evals=len(invalid_ind), **record)

    
    def buildArgumentsDict(self) -> dict:
        arguments = dict()
        
        if self.mode == "FIR":
            # Modo FIR: apenas entradas
            arguments.update({f'ARG{i}': f'u{i + 1}' for i in range(self.nInputs)})

        elif self.mode == "SISO":
            # Modo SISO: y1 e u1
            arguments.update({'ARG0': 'y1', 'ARG1': 'u1'})
        
        else:
            # Modos MISO/MIMO: saídas + entradas
            arguments.update({f'ARG{i}': f'y{i + 1}' for i in range(self.nOutputs)})
            arguments.update({f'ARG{self.nOutputs + i}': f'u{i + 1}' for i in range(self.nInputs)})
        
        return arguments


    @np.errstate(over="raise", divide="raise", invalid="raise")
    def evaluation(self, ind: Individual) -> tuple[float, ...]:
        try:
            if self.problem_type == "regression" and self.froe_mode and np.random.random() < self.pruning_probability:
                self.element.compileModel(ind)
                self._apply_froe_pruning(ind)
            self._fit_model(ind)
            if self.problem_type == 'regression':
                args = (self.outputs, self.inputs) if self.evaluationType != "MShooting" else (self.k, self.outputs, self.inputs)
                yp, yd = ind.predict(self.evaluationType, *args)
                error = ind.score(yd, yp, self.evaluationMode)
                score = error

            elif self.problem_type == 'classification':
                args = (self.outputs, self.inputs) if self.evaluationType != "MShooting" else (self.k, self.outputs, self.inputs)
                
                if self.classification_metric == 'log_loss':
                    yp_proba, yd = ind.predict_proba(self.evaluationType, *args)
                    score = ind.score_classification(yd, yp_proba, 'log_loss')
                
                else:
                    yp_classes, yd = ind.predict_classes(self.evaluationType, *args)
                    score = ind.score_classification(yd, yp_classes, self.classification_metric)
                
                if self.classification_metric in ['accuracy', 'f1_macro']:
                    score = 1 - score
                
            
            else:
                raise ValueError("problem_type must be 'regression' or 'classification'")

            fitness = (score,) if self.new_evaluation is None else (score, self.new_evaluation(ind))
            if all(np.isfinite(value) for value in fitness):
                return fitness
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, IndexError):
            pass
        return (np.inf,) * len(self.weights)

    def run(self, seed: Optional[List] = None) -> List[Individual]:

        print(f"System Mode: {self.mode}. N° Inputs: {self.nInputs}. N° Outputs: {self.nOutputs}")
        print(f"Input Samples: {len(self.inputs)}. Output Samples: {len(self.outputs)}\n")
        
        self._toolbox.register("map", map)    

        init = time.time()
        self.initPop(seed=seed)

        
        self.stream()
        
        err_before = self._hof[0].fitness.values[0]
        err_stop_before = err_before
        counts_err = 0
        for g in range(1, self.generations):
            
            self.step(g)
            self.stream()

            # clear progressbar tqdm after 50 progressbar
            if g % 50 == 0:
                clear_output(wait=True)

            err_current = self._hof[0].fitness.values[0]
            err_stop_current = err_current

            if g%9 == 0:
                # A cada 10 gerações, verifico se melhorou em 5%, se não, aumenta a taxa de mutação em 10% até o maximo de 90%
                if err_before - err_current < 0.05 * err_current or err_before == err_current:
                    self.mutationRate = min(self.mutationRate+0.1, 0.9)

                err_before = err_current
            
            # if g%20 == 0:
                
            #     if (err_stop_before - err_stop_current) < 0.0001:
            #         # EarlyStop: Se depois de 20 épocas não teve melhoria significativa, para o treinamento
            #         break

            #     err_stop_before = err_stop_current
            
        #    if (err_stop_before - err_stop_current) < 0.0001:
        #        counts_err +=1
                
        #        if counts_err == 20:
                    # EarlyStop: Se depois de 20 épocas não teve melhoria significativa, interrompe o treinamento
        #            break
        #    else:
        #        counts_err = 0
            
        #    err_stop_before = err_stop_current

        if self.new_evaluation is not None:
            models = self.fronts[0]

        else:
            models = [self._hof[0]]

        for i, model in enumerate(models):
            self._fit_model(model)
            os.makedirs("models_saved", exist_ok=True)
            self.save_model(model, filename=f"models_saved/{self.filename}_{i}.pkl")
            print(self.simplify_model(model))
            self.validation_all(model=model)

        end = time.time()
        print(f"Executed in: {round(end - init, 3)} seg")
        return models


    def validation_all(self, model):
        if self.validation is None or (isinstance(self.validation, tuple) and
                                       all(item is None for item in self.validation)):
            return

        def make_validation(dataset_validation, dataset_value=0):
            if all([value is not None for value in dataset_validation]):
                u_val, y_val = dataset_validation

                if u_val.shape[1] != self.nInputs:
                    raise Exception("The number of inputs to validate and to train must be the same")

                if y_val.shape[1] != self.nOutputs:
                    raise Exception("The number of outputs to validate and to train must be the same")

                args = (y_val, u_val) if self.evaluationTypeTest != "MShooting" else (self.k, y_val, u_val)
                
                if self.problem_type == 'regression':
                    yp, yd = model.predict(self.evaluationTypeTest, *args)
                    error = round(model.score(yd, yp, self.evaluationMode), 6)
                    print(f"{self.evaluationMode} in validation dataset {dataset_value+1}: {error}")
                
                elif self.problem_type == 'classification':
                    if self.classification_metric == 'log_loss':
                        yp_proba, yd = model.predict_proba(self.evaluationTypeTest, *args)
                        score = model.score_classification(yd, yp_proba, 'log_loss')
                        print(f"Log Loss in validation dataset {dataset_value+1}: {score:.6f}")
                    else:
                        yp_classes, yd = model.predict_classes(self.evaluationTypeTest, *args)
                        accuracy = model.score_classification(yd, yp_classes, 'accuracy')
                        f1 = model.score_classification(yd, yp_classes, 'f1_macro')
                        print(f"Accuracy in validation dataset {dataset_value+1}: {accuracy:.6f}")
                        print(f"F1-Score (macro) in validation dataset {dataset_value+1}: {f1:.6f}")

            else:
                raise Exception("Missing value of Y or U in the validation dataset")
            
        if type(self.validation) == tuple:
            make_validation(self.validation)

        elif type(self.validation) == list:
            for i, dataset_validation in enumerate(self.validation):    
                make_validation(dataset_validation, dataset_value=i)
                
        else:
            raise Exception("Choose a valuation dataset type between list or tuple.")


    def simplify_model(self, model) -> str:
        equation = model.to_equation(precision=17) if isinstance(model, Individual) else model.to_equation()

        lines = equation.split('\n')
        output_lines = []
        current_output = []

        for line in lines:
            if line.startswith('Output'):
                if current_output:
                    output_lines.append(self.simplify_terms(current_output))
                    current_output = []
                output_lines.append(line)
            elif line.strip() and not line.startswith((' ', '\t')):
                current_output.append(line)

        if current_output:
            output_lines.append(self.simplify_terms(current_output))

        return '\n'.join(output_lines)


    def simplify_terms(self, terms) -> str:
      
        coefficients = {}
        for term in terms:
            if term.strip():
                coefficient, _, expression = term.strip().rstrip("+ ").partition(" * ")
                factors = expression.split(" * ")
                if all(re.fullmatch(r"[uy]\d+\[k(?:-\d+)?\]", factor) for factor in factors):
                    expression = " * ".join(sorted(factors))
                coefficients[expression] = coefficients.get(expression, 0.0) + float(coefficient)

        pieces = []
        for expression, value in coefficients.items():
            piece = f"{abs(value):.5e}"
            if expression:
                piece += f" * {expression}"
            sign = ("- " if value < 0 else "+ ") if pieces else ("-" if value < 0 else "")
            pieces.append(sign + piece)
        return "\n".join(pieces)

    def load_model(self, path = None):
        """
        Load a saved model to use as seed
        Args:
            filename: Name of the file containing the saved model
        Returns:
            A model instance ready to be used as seed
        """
        import pickle
        if path is None:
            with open(f"models_saved/{self.filename}_0.pkl", 'rb') as f:
                model_data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

        mode = model_data.get("mode")
        if mode is None:
            arguments = model_data["arguments"].values()
            if not any(argument.startswith("y") for argument in arguments):
                mode = "FIR"
            elif model_data["nOutputs"] > 1:
                mode = "MIMO"
            else:
                mode = "MISO" if model_data["nInputs"] > 1 else "SISO"
        primitives = {"operators": model_data["operators"]} if "operators" in model_data else {}
        element = Element(
            weights=model_data.get("weights", self.weights),
            nDelays=model_data['nDelays'],
            nInputs=model_data['nInputs'],
            nOutputs=model_data['nOutputs'],
            nTerms=model_data['nTerms'],
            maxHeight=model_data['maxHeight'],
            mode=mode,
            **primitives,
        )
        element.renameArguments(model_data['arguments'])
        
        model = element.buildModelFromList(model_data['model_structure'])
        element.compileModel(model)
        model.theta = model_data['theta']
        model.logistic_model = model_data.get("logistic_model", self.problem_type == "classification")
        model.input_offset = model_data.get("input_offset", 0)
        if model_data.get("fitness") is not None:
            model.fitness.values = model_data["fitness"]
        
        return model
    

    def save_model(self, model, filename) -> None:
        """
        Save the best model to a file for later use as seed
        Args:
            model: The model to be saved
            filename: Name of the file to save the model
        """
        import pickle
        
        model_data = {
            'model_structure': model.model2List(),
            'theta': model.theta,
            'nInputs': self.nInputs,
            'nOutputs': self.nOutputs,
            'nTerms': self.nTerms,
            'maxHeight': self.maxHeight,
            'nDelays': self.nDelays,
            'arguments': self.buildArgumentsDict(),
            'operators': self.operators,
            'mode': self.mode,
            'weights': model.fitness.weights,
            'fitness': model.fitness.values if model.fitness.valid else None,
            'logistic_model': model.logistic_model,
            'input_offset': getattr(model, "input_offset", 0),
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)


    def _check_hysteretic_constraints(self, ind: Individual, tol: float=1e-10) -> bool:
        """
        Verifica se o modelo atende às restrições de continuum de equilíbrio
        """
        try:
            clusters = ind.identify_term_clusters(self.outputs, self.inputs)
            theta = np.asarray(ind.theta).flatten() 
            
            if abs(theta[0]) > tol or not np.all(np.isfinite(theta)):
                return False
            linear_output_sum = sum(theta[idx] for idx in clusters['linear_output'])
            if abs(linear_output_sum - 1.0) > tol:
                return False
                
            for cluster_type in ['linear_input', 'cross_terms', 'nonlinear_y', 'nonlinear_u']:
                cluster_sum = sum(theta[idx] for idx in clusters[cluster_type])
                if abs(cluster_sum) > tol:
                    return False

            # for term in ind:
            #     # sign_u = str(term).count('sign(u') + str(term).count('sign(q')
            #     # if sign_u >= 2:
            #     #     return False
            #
            #     if str(term) == "mul(sign(u1, q1(u1)), sign(u1, q1(u1)))" or str(term) == 'mul(sign(q1(u1), u1), sign(u1, q1(u1)))'\
            #             or str(term) == "mul(sign(u1, q1(u1)), sign(q1(u1), u1))" or str(term) == "mul(sign(q1(u1), u1), sign(q1(u1), u1))":
            #         return False

            # if any([f'k-{i}' in self.simplify_model(ind) for i in range(3, 10)]):
            #     return False
            
            # if len(clusters['cross_terms']) > 0: # Evitar termos cruzados
            #     return False

            # if len(clusters['linear_input']) > 0: # proibir termos de entrada isolados
            #     return False
            #
            # for idx in clusters['linear_input']:
            #     if ind[idx].height > 1:
            #         return False

            # if len(clusters['linear_output']) > 1: # apenas um termo de saída
            #     return False

            # for idx in clusters['linear_output']: # apenas termos de saída k-1
            #     if ind[idx].height > 1:
            #         return False
                    
            return True
        except:
            return False
        

    def get_terminal_by_name(self, name):

        if isinstance(self.element._pset.terminals, defaultdict):

            for term in self.element._pset.terminals[object]:
                if hasattr(term, 'value') and term.value == name:
                    return term
        else:
            for term in self.element._pset.terminals:
                if hasattr(term, 'value') and term.value == name:
                    return term
        return None
    

    def _constrain_phi_functions(self, ind: Individual):
        """
        Garante que as funções φ (subtraction e sign) recebam apenas 
        variáveis de entrada defasadas.
        - Primeira substituição dentro de φ: u1
        - Substituições subsequentes dentro da mesma φ: q1(u1)
        """
        
        PHI_NAMES = {"subtraction", "sign"}

        def _is_q_chain_to_u(tree: gp.PrimitiveTree, idx: int) -> bool:
            """True se a subárvore em idx é q*(u#) (uma cadeia de q's terminando em u)."""
            node = tree[idx]

            if isinstance(node, gp.Terminal):
                return isinstance(node.value, str) and node.value.startswith("u")

            if not isinstance(node, gp.Primitive):
                return False
            if not node.name.startswith("q"):
                return False
            if node.arity != 1:
                return False

            child_idx = idx + 1
            return _is_q_chain_to_u(tree, child_idx)


        def _make_q1_u1_tree(pset: gp.PrimitiveSet) -> gp.PrimitiveTree:
            q1 = next(p for p in pset.primitives[pset.ret] if p.name == "q1")
            u1 = next(t for t in pset.terminals[pset.ret] if getattr(t, "value", None) == "u1")
            return gp.PrimitiveTree([q1, u1])
        
        
        def _make_u1_tree(pset: gp.PrimitiveSet) -> gp.PrimitiveTree:
            u1 = next(t for t in pset.terminals[pset.ret] if getattr(t, "value", None) == "u1")
            return gp.PrimitiveTree([u1])


        def constrain_phi_tree(tree: gp.PrimitiveTree, pset) -> gp.PrimitiveTree:
            """Garante que cada argumento de φ é q*(u#). Se não for, vira q1(u1)."""
            i = 0
            repl = _make_q1_u1_tree(pset)
            rep_u1 = _make_u1_tree(pset)

            while i < len(tree):
                node = tree[i]
                if isinstance(node, gp.Primitive) and node.name in PHI_NAMES:
                    
                    arg_idx = i + 1
                    first_substitution = True
                    for _ in range(node.arity):
                        arg_slice = tree.searchSubtree(arg_idx)
                        if not _is_q_chain_to_u(tree, arg_idx):
                            if first_substitution:
                                tree[arg_slice] = rep_u1
                                first_substitution = False

                                arg_idx = arg_slice.start + len(rep_u1)
                            else:
                                tree[arg_slice] = repl
                                
                                arg_idx = arg_slice.start + len(repl)
                        else:
                            arg_idx = arg_slice.stop
                i += 1
            return tree

        if self.mode in ["SISO", "MISO"] or (self.mode == "FIR" and self.nOutputs == 1):
            for i, tree in enumerate(ind):
                ind[i] = constrain_phi_tree(tree, self.element._pset)

        else:  # MIMO
            for o in range(len(ind)):
                for i, tree in enumerate(ind[o]):
                    ind[o][i] = constrain_phi_tree(tree, self.element._pset)


    def _is_lagged_input(self, node) -> bool:
        """Verifica se o nó é uma variável de entrada defasada"""
        if isinstance(node, gp.Terminal) and node.value.startswith('u'):
            return True
        return False


    def _apply_froe_pruning(self, ind: Individual):
        """
        Aplica o algoritmo FROE para remover termos com ERR baixo
        """

        align = self._fir_align(self.evaluationType)
        matrices = ind.makeRegressors(self.outputs, self.inputs, align=align)
        offset = int(ind.input_only and align == "OSA")
        target = self.outputs[ind.lagMax + offset:]
        if ind.multi_output:
            for j, genes in enumerate(ind):
                self._froe_pruning_miso(genes, matrices[j], target[:, j])
        else:
            self._froe_pruning_miso(ind, matrices, target[:, 0])

    def _froe_pruning_miso(self, genes, P: NDArray, yd: NDArray) -> None:
        """Filter genes by their existing marginal error-reduction ratio."""
        y = np.asarray(yd).ravel()
        terms = P[:, 1:]
        denominator = (y @ y) * np.sum(terms * terms, axis=0)
        err = np.divide((terms.T @ y) ** 2, denominator,
                        out=np.zeros(len(genes)), where=denominator > 0)
        genes[:] = [gene for gene, score in zip(genes, err) if score >= self.pruning_tolerance]

    def _survive(self, pool, n):
        valid = [ind for ind in pool if finite(ind)]
        unique, repeated, positions = [], [], {}
        for ind in valid:
            try:
                key = canonical_terms(ind)
            except ValueError:
                # A custom objective can use primitives outside the polynomial grammar.
                genes = ind.model2List()
                key = tuple(tuple(eq) for eq in genes) if ind.multi_output else tuple(genes)
            if key not in positions:
                positions[key] = len(unique)
                unique.append(ind)
            else:
                i = positions[key]
                if ind.fitness.dominates(unique[i].fitness):
                    unique[i], ind = ind, unique[i]
                repeated.append(ind)
        if len(unique) >= n:
            return tools.selNSGA2(unique, n)
        need = n - len(unique)
        fill = tools.selNSGA2(repeated, min(need, len(repeated))) if repeated else []
        invalid = [ind for ind in pool if not finite(ind)]
        return unique + fill + invalid[:need - len(fill)]
