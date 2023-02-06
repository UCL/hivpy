import math

import pandas as pd
import pytest

from hivpy.common import between


@pytest.mark.parametrize(("limits", "expected"), [
    ((-10, 50), [True, True, True, True, True]),
    ((5, 50), [True, False, True, True, True]),  # lower bound inclusive
    ((5, 20), [True, False, True, False, True]),  # upper bound exclusive
    ((40, 50), [False, False, False, False, False]),  # no match
    ((12, math.inf), [False, False, False, True, True])  # infinite bound
])
def test_between(limits, expected):
    """
    Check the between helper behaves as expected for numerical Series.
    """
    s = pd.Series([5, 2.5, 10, 20, 15])
    assert list(between(s, limits)) == expected
