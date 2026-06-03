from mggp.base import Element
from mggp.mutations import MutGPOneTree, MutGPUniform, MutGPReplace

def test_mutgponetree_preserves_structure():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)
    ind = el.buildRandomModel()

    mut = MutGPOneTree(el)
    new_ind, = mut.mutate(ind)

    assert len(new_ind) == len(ind)



def test_mutgpuniform_preserves_structure():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)
    ind = el.buildRandomModel()

    mut = MutGPUniform(el)
    new_ind, = mut.mutate(ind)

    assert len(new_ind) == len(ind)


def test_mutgpuniform_preserves_structure_mimo():
    el = Element(nInputs=2, nOutputs=2, nTerms=3, mode='MIMO')
    ind = el.buildRandomModel()

    mut = MutGPUniform(el)
    new_ind, = mut.mutate(ind)

    assert len(new_ind) == len(ind)



def test_mutgpraplace_preserves_structure():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)
    ind = el.buildRandomModel()

    mut = MutGPReplace(el)
    new_ind, = mut.mutate(ind)

    assert len(new_ind) == len(ind)
