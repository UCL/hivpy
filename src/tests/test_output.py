from math import isclose

import pytest

import hivpy.column_names as col
from hivpy.common import Date, SexType, TimeDelta, rng
from hivpy.output import SimulationOutput
from hivpy.population import Population


@pytest.fixture(autouse=True)
def resetRandomState():
    rng.set_seed(42)


# age boundaries
age_min = 15
age_max_active = 65
age_step = 10


def test_HIV_prevalence():

    # build population
    N = 1000
    pop = Population(size=N, start_date=Date(1990, 1, 1))
    pop.set_variable_range(col.SEX, SexType.Female, 0, int(N / 2) - 1)
    pop.set_variable_range(col.SEX, SexType.Male, int(N / 2))
    pop.set_present_variable(col.AGE, 25)
    pop.set_present_variable(col.SEX_WORKER, False)
    pop.set_variable_range(col.SEX_WORKER, True, 0, int(N / 4) - 1)
    pop.set_variable_range(col.HIV_STATUS, True, 0, int(N / 4) - 1)

    out = SimulationOutput(Date(1990, 1, 1), Date(1990, 3, 1), TimeDelta(days=90))
    out._update_HIV_prevalence(pop)

    # a quarter of all people have HIV
    assert isclose(out.output_stats["HIV prevalence (tot)"], 0.25)
    # all are in the 25-35 age bracket
    assert isclose(out.output_stats["HIV prevalence (25-34)"], 0.25)
    # all HIV positive people are women
    assert isclose(out.output_stats["HIV prevalence (female)"], 0.5)
    assert isclose(out.output_stats["HIV prevalence (male)"], 0)
    # all HIV positive people are sex workers
    assert isclose(out.output_stats["HIV prevalence (sex worker)"], 1)


def test_HIV_incidence():

    # build population
    N = 1000
    pop = Population(size=N, start_date=Date(1990, 1, 1))
    pop.set_present_variable(col.SEX, SexType.Female)
    pop.set_variable_range(col.AGE, 20, 0, int(N * 0.2) - 1)
    pop.set_variable_range(col.AGE, 30, int(N * 0.2), int(N * 0.4) - 1)
    pop.set_variable_range(col.AGE, 40, int(N * 0.4), int(N * 0.6) - 1)
    pop.set_variable_range(col.AGE, 50, int(N * 0.6), int(N * 0.8) - 1)
    pop.set_variable_range(col.AGE, 60, int(N * 0.8))

    pop.set_present_variable(col.HIV_STATUS, False)
    pop.set_present_variable(col.IN_PRIMARY_INFECTION, False)
    pop.set_variable_range(col.IN_PRIMARY_INFECTION, True, 0, int(N * 0.2) - 1)
    pop.set_variable_range(
        col.IN_PRIMARY_INFECTION, True, int(N * 0.2), int(N * 0.35) - 1
    )
    pop.set_variable_range(
        col.IN_PRIMARY_INFECTION, True, int(N * 0.4), int(N * 0.5) - 1
    )
    pop.set_variable_range(
        col.IN_PRIMARY_INFECTION, True, int(N * 0.6), int(N * 0.65) - 1
    )

    out = SimulationOutput(Date(1990, 1, 1), Date(1990, 3, 1), TimeDelta(days=90))
    out._update_HIV_incidence(pop)

    # get age stats
    for age_bound in range(age_min, age_max_active, age_step):
        age_group = int(age_bound / 10) - 1
        key = f"HIV incidence ({age_bound}-{age_bound+(age_step-1)}, female)"
        assert isclose(out.output_stats[key], 1 - age_group * 0.25)
