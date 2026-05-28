from mggp.base import Element
from mggp.crossings import CrossLowUniform, CrossLowOnePoint, CrossHighUniform, CrossHighOnePoint

def test_crosslowuniform_execution():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)

    ind1 = el.buildRandomModel()
    ind2 = el.buildRandomModel()

    cross = CrossLowUniform(el)

    new1, new2 = cross.cross(ind1, ind2)

    assert len(new1) == len(ind1)
    assert len(new2) == len(ind2)


def test_croslowonepoint_execution():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)

    ind1 = el.buildRandomModel()
    ind2 = el.buildRandomModel()

    cross = CrossLowOnePoint(el)

    new1, new2 = cross.cross(ind1, ind2)

    assert len(new1) == len(ind1)
    assert len(new2) == len(ind2)


def test_croshighuniform_execution():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)

    ind1 = el.buildRandomModel()
    ind2 = el.buildRandomModel()

    cross = CrossHighUniform(el)

    new1, new2 = cross.cross(ind1, ind2)

    assert len(new1) == len(ind1)
    assert len(new2) == len(ind2)


def test_crosshighonepoint_execution():
    el = Element(nInputs=1, nOutputs=1, nTerms=3)

    ind1 = el.buildRandomModel()
    ind2 = el.buildRandomModel()

    cross = CrossHighOnePoint(el)

    new1, new2 = cross.cross(ind1, ind2)

    assert len(new1) == len(ind1)
    assert len(new2) == len(ind2)
