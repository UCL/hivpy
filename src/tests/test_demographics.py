import logging

import numpy as np
import pytest
import scipy.integrate

from hivpy.demographics import (FEMALE_RATIO, ContinuousAgeDistribution,
                                DemographicsModule, StepwiseAgeDistribution)


@pytest.fixture(scope="module")
def default_module():
    return DemographicsModule()


@pytest.fixture(scope="module")
def stepwise_age_module():
    return DemographicsModule(use_stepwise_ages=True)


def test_sex_distribution(default_module):
    count = 100000
    sex = default_module.initialize_sex(count)
    female = np.sum(sex == "female")
    assert pytest.approx(female/count, rel=0.05) == FEMALE_RATIO


def test_continuous_age_distribution(default_module):
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


def test_continous_age_logging(caplog):
    caplog.set_level(logging.WARNING)
    ContinuousAgeDistribution(-65, 200, ContinuousAgeDistribution.modelParams1)
    assert "Max age exceeds the maximum age limit" in caplog.text


def test_stepwise_age_distribution(stepwise_age_module):
    count = 100000
    ages = stepwise_age_module.initialise_age(count)
    age_dist = StepwiseAgeDistribution.stepwise_model1
    age_boundaries = StepwiseAgeDistribution.stepwise_boundaries
    assert age_boundaries.size == (age_dist.size + 1)
    N = age_dist.size
    for i in range(0, N):
        age_count = sum((ages >= age_boundaries[i]) & (ages < age_boundaries[i+1]))
        assert pytest.approx(age_count, rel=0.1) == age_dist[i]*count
