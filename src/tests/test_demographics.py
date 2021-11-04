import numpy as np
import pytest
import scipy.integrate

from hivpy.demographics import (FEMALE_RATIO, ContinuousAgeDistribution,
                                DemographicsModule)


@pytest.fixture(scope="module")
def default_module():
    return DemographicsModule()


def test_sex_distribution(default_module):
    count = 100000
    sex = default_module.initialize_sex(count)
    female = np.sum(sex == "female")
    assert pytest.approx(female/count, rel=0.05) == FEMALE_RATIO


def test_age_distribution(default_module):
    count = 100000
    ages = default_module.initialise_age(count)
    min_age, max_age = (-68, 65)
    boundaries = np.linspace(min_age, max_age, 11)
    m, c, A, B = ContinuousAgeDistribution.modelParams1

    def prob(x):
        return (m*x + c)*np.exp(A*(x-B))

    norm = 1 / (scipy.integrate.quad(prob, min_age, max_age)[0])
    for i in range(0, boundaries.size - 1):
        num_pop = np.count_nonzero((boundaries[i] < ages) & (ages < boundaries[i+1]))
        expectation = scipy.integrate.quad(prob, boundaries[i], boundaries[i+1])[0]*norm
        assert pytest.approx(num_pop, rel=0.1) == expectation*count
