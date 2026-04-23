from mggp.base import Element
from mggp.crossings import CrossLowUniform

def test_crossover_execution():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)

    ind1 = el.buildRandomModel()
    ind2 = el.buildRandomModel()

    cross = CrossLowUniform(el)

    new1, new2 = cross.cross(ind1, ind2)

    assert len(new1) == len(ind1)
    assert len(new2) == len(ind2)
