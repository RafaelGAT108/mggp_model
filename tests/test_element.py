import pytest
from mggp.base import Element

def test_element_initialization_siso(siso_data):
    u, y = siso_data

    el = Element(
        nInputs=1,
        nOutputs=1,
        nTerms=5,
        nDelays=2,
        maxHeight=3
    )

    assert el._mode == "SISO"
    assert el._nVar == 2
    assert len(el._delays) == 2


def test_invalid_operator():
    with pytest.raises(ValueError):
        Element(operators=["invalid_op"])
