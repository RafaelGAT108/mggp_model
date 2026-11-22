import numpy as np
import pytest

from mggp import MGGP
from mggp.utils import arithmetic_cost, canonical_terms, count_times, finite, gene_monomial, make_cost


def test_polynomial_cost_merges_equivalent_lagged_monomials():
    u = np.arange(10.)[:, None]
    engine = MGGP(inputs=u, outputs=u, nDelays=[1, 2], operators=["mul", "sin"])
    build = engine.element.buildModelFromList
    model = build(["q1(mul(q2(u1), y1))", "mul(q1(y1), q2(q1(u1)))", "mul(u1, u1)"])
    expected = ((("u1", 3), 1), (("y1", 1), 1))
    assert gene_monomial(model[0]) == gene_monomial(model[1]) == expected
    assert canonical_terms(model) == frozenset((expected, ((("u1", 0), 2),)))
    assert make_cost(model) == 2 + 4
    assert arithmetic_cost(model, rho=2.5) == 2 + 2.5 * 4
    assert count_times(model, times_weight=.5) == 1.5
    with pytest.raises(ValueError, match="sin"):
        make_cost(build(["sin(u1)"]))


def test_survival_prioritizes_unique_finite_structures_and_pareto_extremes():
    u = np.arange(14.)[:, None]
    engine = MGGP(inputs=u, outputs=u, new_evaluation=make_cost)
    def candidate(gene, error, cost):
        ind = engine.element.buildModelFromList([gene])
        ind.fitness.values = (error, cost)
        return ind
    accurate = candidate("mul(u1, y1)", .1, 8)
    simple = candidate("u1", .5, 2)
    dominated = candidate("q1(u1)", .8, 5)
    copy = candidate("mul(y1, u1)", .1, 8)
    invalid = candidate("q2(u1)", np.inf, 1)
    pool = [invalid, accurate, copy, dominated, simple]
    assert not finite(invalid)
    for n, expected in [(2, [accurate, simple]), (3, [accurate, simple, dominated]),
                        (4, [accurate, simple, dominated, copy]), (5, pool)]:
        chosen = engine._survive(pool, n)
        assert {id(ind) for ind in chosen} == {id(ind) for ind in expected}


@pytest.mark.parametrize("pop_size", [3, 9, 12])
def test_pareto_run_preserves_tradeoffs_and_checkpointed_fitness(pop_size):
    u = np.random.default_rng(8).uniform(-1, 1, (40, 1))
    y = .2 + u + .75*u**2
    engine = MGGP(inputs=u, outputs=y, mode="FIR", evaluationType="INSTANT",
                  evaluationTypeTest="INSTANT", new_evaluation=make_cost,
                  generations=3, populationSize=pop_size, elitePercentage=0,
                  nTerms=2, maxHeight=2, nDelays=1)
    seeds = [engine.element.buildModelFromList(genes) for genes in
             [["u1"], ["u1", "mul(u1, u1)"]]]
    # Fitness from another experiment must never be reused as training error.
    seeds[0].fitness.values = (-100., -100.)
    front = engine.run(seed=seeds)
    assert len(engine._pop) == pop_size
    assert [ind.fitness.values for ind in front] == [ind.fitness.values for ind in engine.fronts[0]]
    fitness = np.array([ind.fitness.values for ind in engine._pop])
    nondominated = {tuple(v) for v in fitness
                    if not np.any(np.all(fitness <= v, axis=1) & np.any(fitness < v, axis=1))}
    assert {ind.fitness.values for ind in front} == nondominated
    assert seeds[0].fitness.values == (-100., -100.)
    assert min(ind.fitness.values[0] for ind in front) < 1e-12
    assert min(ind.fitness.values[1] for ind in front) == 2
    for i, ind in enumerate(front):
        yp, yd = ind.predict("INSTANT", y, u)
        rmse = np.sqrt(np.mean((yp-yd)**2))
        np.testing.assert_allclose(ind.fitness.values, (rmse, make_cost(ind)), atol=1e-12)
        loaded = engine.load_model(f"models_saved/best_model_{i}.pkl")
        assert loaded.fitness.values == ind.fitness.values
        np.testing.assert_allclose(loaded.predict("INSTANT", y, u)[0], yp)


def test_custom_nonpolynomial_objective_and_multioutput_cost():
    u = np.arange(10.)[:, None]
    engine = MGGP(inputs=u, outputs=np.column_stack([u, u]), operators=["sin", "mul"],
                  new_evaluation=lambda ind: sum(map(len, ind)))
    build = engine.element.buildModelFromList
    a, b = (build(g) for g in [[["sin(u1)"], ["y2"]], [["y2"], ["sin(u1)"]]])
    a.fitness.values, b.fitness.values = (.1, 2), (.1, 2)
    assert {id(i) for i in engine._survive([a, b], 2)} == {id(a), id(b)}
    poly = build([["u1", "u1"], ["mul(u1, y2)"]])
    assert make_cost(poly) == 2 + 3


def test_printed_polynomial_and_arithmetic_cost_agree_for_duplicate_products():
    u = np.arange(12.)[:, None]
    engine = MGGP(inputs=u, outputs=u, nDelays=2)
    model = engine.element.buildModelFromList(["mul(y1, q1(u1))", "mul(q1(u1), y1)"])
    engine.element.compileModel(model)
    model.theta = np.array([[.5], [.125], [.375]])
    assert make_cost(model) == 3
    equation = engine.simplify_model(model)
    assert equation.splitlines() == ["5.00000e-01", "+ 5.00000e-01 * u1[k-1] * y1[k-1]"]
    yp, _ = model.predict("OSA", u, u)
    np.testing.assert_allclose(yp, .5 + .5*u[:-1]**2)


def test_mating_tournaments_receive_crowding_distances_within_each_front(monkeypatch):
    u = np.arange(16.)[:, None]
    engine = MGGP(inputs=u, outputs=u, evaluationType="OSA", populationSize=6,
                  crossoverRate=0, mutationRate=0, new_evaluation=make_cost)
    genes = ["u1", "y1", "q1(u1)", "q1(y1)", "q2(u1)", "q2(y1)"]
    # Two translated fronts have the same normalized spacings within each front.
    objectives = dict(zip(genes, [(0, 4), (2, 2), (4, 0), (1, 5), (3, 3), (5, 1)]))
    engine._toolbox.register("evaluate", lambda m: objectives[str(m[0])])
    seeds = [engine.element.buildModelFromList([gene]) for gene in genes]
    engine.initPop(seed=seeds)
    observed = []
    select = engine._toolbox.select
    def capture_then_select(pop, k):
        observed.append([model.fitness.crowding_dist for model in pop])
        return select(pop, k)
    monkeypatch.setattr(engine._toolbox, "select", capture_then_select)
    engine.step(1)
    np.testing.assert_allclose(observed[0], [np.inf, 1, np.inf, np.inf, 1, np.inf])
