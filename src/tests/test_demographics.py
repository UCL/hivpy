import logging
from datetime import date, datetime, timedelta

import numpy as np
import pytest
import scipy.integrate

from hivpy.common import SexType
from hivpy.demographics import (ContinuousAgeDistribution, DemographicsModule,
                                StepwiseAgeDistribution)
from hivpy.demographics_data import DemographicsData
from hivpy.population import Population


@pytest.fixture(scope="module")
def default_module():
    return DemographicsModule()


@pytest.fixture(scope="module")
def stepwise_age_module():
    return DemographicsModule(use_stepwise_ages=True)


@pytest.mark.parametrize("ratio", [0.4, 0.52, 0.8])
def test_sex_distribution(ratio):
    module = DemographicsModule(female_ratio=ratio)
    count = 100000
    sex = module.initialize_sex(count)
    female = np.sum(sex == SexType.Female)
    assert female/count == pytest.approx(ratio, rel=0.01)


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


def test_date_permanent(default_module):
    """Check that that death is not recorded again in future steps."""
    pop = Population(size=3, start_date=date(1989, 1, 1))

    pop.data['date_of_death'] = [None, datetime.today(), datetime.today() - timedelta(days=1)]
    pop.data['age'] = [30, 20, 50]
    pop.data['sex'] = [SexType.Female, SexType.Female, SexType.Male]

    new_deaths = default_module.determine_deaths(pop.data)
    # Find who already had a date of death, and check that they are not marked
    # as having died in this time step.
    assert not new_deaths[pop.data.date_of_death.notnull()].any()


def test_death_rate():
    """Check that we record the expected number of deaths."""
    # Set up the population to have equal-sized age groups and balanced sexes
    data_path = "src/tests/test_data/demographics_testing.yaml"
    data = DemographicsData(data_path)
    module = DemographicsModule(death_rates=data.death_rates)
    N = 300000
    ages_to_try = [10, 23, 33]
    age_groups = [0, 2, 4]
    group_size = N // 2 // len(ages_to_try)
    ages = sum(([age] * group_size for age in ages_to_try), []) * 2
    sexes = [SexType.Female] * (N // 2) + [SexType.Male] * (N // 2)
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data['date_of_death'] = None
    pop.data['age'] = ages
    pop.data['sex'] = sexes

    # The rates in the data file are annualised
    expected_annual_deaths = {
        (sex, age_group): group_size * data.death_rates[sex][age_group]
        for sex in SexType
        for age_group in age_groups
    }
    # Simulate for a year
    n_steps = 4  # currently death determination assumes 3-month step
    for _ in range(n_steps):
        deaths = module.determine_deaths(pop.data)
        # We only care about recording the death here, not its date
        pop.data.loc[deaths, "date_of_death"] = datetime.today()
    recorded_deaths = pop.data.groupby(
        ["sex", "age_group"]
        ).date_of_death.count().to_dict()
    assert recorded_deaths == pytest.approx(expected_annual_deaths, rel=0.1)
