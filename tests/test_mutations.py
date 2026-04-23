from mggp.base import Element
from mggp.mutations import MutGPOneTree

def test_mutation_preserves_structure():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)
    ind = el.buildRandomModel()

    mut = MutGPOneTree(el)
    new_ind, = mut.mutate(ind)

    assert len(new_ind) == len(ind)
