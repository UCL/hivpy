import numpy as np
import pytest

from hivpy.demographics import DemographicsModule, FEMALE_RATIO

@pytest.fixture(scope="module")
def default_module():
    return DemographicsModule()

def test_sex_distribution(default_module):
    count = 100000
    sex = default_module.initialize_sex(count)
    female = np.sum(sex == "female")
    assert pytest.approx(female/count, rel=0.05) == FEMALE_RATIO
