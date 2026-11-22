import random
import numpy as np
import pytest
from deap import gp

from mggp import MGGP
from mggp.crossings import CrossHighOnePoint, CrossHighUniform, CrossLowOnePoint, CrossLowUniform
from mggp.mutations import MutGPOneTree, MutGPUniform, MutGPReplace


def configure(multi=False, max_height=2):
    u = np.arange(8.)[:, None]
    return MGGP(inputs=u, outputs=np.column_stack([u, u]) if multi else u,
                mode="FIR", maxHeight=max_height, nDelays=2)


@pytest.mark.parametrize("multi", [False, True])
def test_whole_gene_crossovers_exchange_only_selected_genes(multi, monkeypatch):
    engine = configure(multi)
    build = engine.element.buildModelFromList
    left = ["u1", "q1(u1)", "q2(u1)"]
    right = ["mul(u1, u1)", "q2(q1(u1))"]
    def pair():
        return (build([left, right] if multi else left),
                build([right, left] if multi else right))
    a, b = pair()
    monkeypatch.setattr(random, "randint", lambda lower, upper: 1)
    CrossHighOnePoint(engine.element).cross(a, b)
    x, y = left[:1] + right[1:], right[:1] + left[1:]
    assert a.model2List() == ([x, y] if multi else x)
    assert b.model2List() == ([y, x] if multi else y)

    a, b = pair()
    draws = iter([.1, .9, .1, .9])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    CrossHighUniform(engine.element).cross(a, b)
    x, y = right[:1]+left[1:], left[:1]+right[1:]
    assert a.model2List() == ([x, y] if multi else x)
    assert b.model2List() == ([y, x] if multi else y)


@pytest.mark.parametrize("operator", [CrossLowOnePoint, CrossLowUniform])
def test_subtree_crossover_preserves_roots_and_exchanges_terminal(operator, monkeypatch):
    u = np.column_stack([np.arange(12.), np.arange(12.)**2])
    engine = MGGP(inputs=u, outputs=u[:, :1], mode="FIR", nDelays=2)
    a = engine.element.buildModelFromList(["q2(u1)"])
    b = engine.element.buildModelFromList(["q1(u2)"])
    monkeypatch.setattr(random, "random", lambda: 0.)
    operator(engine.element).cross(a, b)
    assert a.model2List() == ["q2(u2)"]
    assert b.model2List() == ["q1(u1)"]
    engine.element.compileModel(a)
    a.theta = [[0.], [1.]]
    np.testing.assert_array_equal(a.predict("INSTANT", u[:, :1], u)[0][:, 0], u[:-2, 1])


@pytest.mark.parametrize("operator", [MutGPOneTree, MutGPUniform, MutGPReplace])
@pytest.mark.parametrize("multi", [False, True])
def test_mutation_changes_genes_without_mixing_output_equations(operator, multi, monkeypatch):
    engine = configure(multi)
    genes = ["q1(u1)"]
    model = engine.element.buildModelFromList([genes, genes] if multi else genes)
    mutation = operator(engine.element)
    replacement = gp.PrimitiveTree.from_string("q2(u1)", engine.element.pset)
    # Fix the proposed gene and the draw; mutation still selects/updates the actual genome.
    if operator is MutGPReplace:
        monkeypatch.setattr(engine.element.toolbox, "_program", lambda: gp.PrimitiveTree(replacement))
    else:
        monkeypatch.setattr(mutation._toolbox, "mutateGP", lambda tree: (gp.PrimitiveTree(replacement),))
        monkeypatch.setattr(random, "random", lambda: .1)
    child, = mutation.mutate(model)
    assert child.model2List() == ([["q2(u1)"], ["q2(u1)"]] if multi else ["q2(u1)"])
    engine.element.compileModel(child)
    assert child.lagMax == 2


def test_excessive_subtree_height_rolls_back_in_place_edit(monkeypatch):
    engine = configure(max_height=1)
    a = engine.element.buildModelFromList(["q1(u1)"])
    b = engine.element.buildModelFromList(["q2(u1)"])
    too_tall = gp.PrimitiveTree.from_string("q1(q2(u1))", engine.element.pset)
    def mutate_in_place(tree):
        tree[tree.searchSubtree(0)] = too_tall
        return tree,
    def cross_in_place(left, right):
        mutate_in_place(left)
        right[right.searchSubtree(0)] = gp.PrimitiveTree.from_string("u1", engine.element.pset)
        return left, right
    op = MutGPOneTree(engine.element)
    monkeypatch.setattr(op._toolbox, "mutateGP", mutate_in_place)
    op.mutate(a)
    assert a.model2List() == ["q1(u1)"]
    monkeypatch.setattr(gp, "cxOnePoint", cross_in_place)
    CrossLowOnePoint(engine.element).cross(a, b)
    assert (a.model2List(), b.model2List()) == (["q1(u1)"], ["q2(u1)"])


def test_variation_after_pruning_handles_empty_and_single_gene_outputs(monkeypatch):
    e = configure(multi=True)
    a = e.element.buildModelFromList([[], ["q1(u1)"]])
    b = e.element.buildModelFromList([["u1"], ["q2(u1)"]])
    CrossHighOnePoint(e.element).cross(a, b)
    assert a.model2List() == [[], ["q2(u1)"]]
    CrossLowOnePoint(e.element).cross(a, b)
    MutGPOneTree(e.element).mutate(a)
    MutGPReplace(e.element).mutate(a)
    e.element.compileModel(a)
    assert all(len(eq) == 1 and eq[0].height <= 2 for eq in a)
