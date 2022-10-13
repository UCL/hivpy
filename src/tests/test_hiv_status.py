import operator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import hivpy.column_names as col
from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population
from hivpy.sexual_behaviour import selector


@pytest.fixture
def pop_with_hiv():
    pop_size = 100000
    pop = Population(size=pop_size, start_date=date(1989, 1, 1))
    # Wait a few months so more people are positive
    for _ in range(5):
        pop.evolve(timedelta(days=30))
    print("Num HIV+ = ", sum(pop.data["HIV_status"]))
    return pop


@pytest.mark.parametrize("pop_percentage", [0.4, 0])
def test_initial_hiv_threshold(pop_percentage):
    """Check that HIV is initially introduced only to those with high enough newp."""
    pop = Population(size=1000, start_date=date(1989, 1, 1)).data
    HIV_module = HIVStatusModule()
    threshold = HIV_module.initial_hiv_newp_threshold
    # select a proportion of the population to have high enough newp
    newp = np.random.default_rng().choice(
        [threshold, threshold - 1],
        p=[pop_percentage, 1 - pop_percentage],
        size=len(pop)
    )
    pop[col.NUM_PARTNERS] = newp
    initial_status = HIV_module.introduce_HIV(pop)
    assert not any(initial_status & (pop[col.NUM_PARTNERS] < threshold))


def test_initial_hiv_probability():
    """Check that HIV is initially assigned with the specified probability."""
    pop = Population(size=1000, start_date=date(1989, 1, 1)).data
    HIV_module = HIVStatusModule()
    # Have everyone be a candidate for initial introduction
    pop[col.NUM_PARTNERS] = HIV_module.initial_hiv_newp_threshold
    expected_infections = HIV_module.initial_hiv_prob * len(pop)
    initial_status = HIV_module.introduce_HIV(pop)
    # TODO Could check against variance of binomial distribution, see issue #45
    assert initial_status.sum() == pytest.approx(expected_infections, rel=0.05)


def test_HIV_introduced_only_once(mocker):
    """Check that we do not initialise HIV status repeatedly."""
    pop = Population(size=1000, start_date=date(1988, 12, 1))
    spy = mocker.spy(pop.hiv_status, "introduce_HIV")
    pop.evolve(timedelta(days=31))
    spy.assert_not_called()
    # Starting from 1989/1/1, so HIV should now be introduced...
    pop.evolve(timedelta(days=31))
    spy.assert_called_once()
    # ...but should not be repeated at the next time step
    pop.evolve(timedelta(days=31))
    spy.assert_called_once()


def test_hiv_initial_ages(pop_with_hiv):
    """Check that HIV is not introduced to anyone <= 15 or > 65."""
    under_15s = selector(pop_with_hiv.data, HIV_status=(operator.eq, True), age=(operator.le, 15))
    over_65s = selector(pop_with_hiv.data, HIV_status=(operator.eq, True), age=(operator.gt, 65))
    assert not any(under_15s)
    assert not any(over_65s)


def test_hiv_update(pop_with_hiv):
    pd.set_option('display.max_columns', None)
    data = pop_with_hiv.data
    prev_status = data["HIV_status"].copy()
    for i in range(10):
        pop_with_hiv.hiv_status.update_HIV_status(pop_with_hiv.data)

    new_cases = data["HIV_status"] & (~ prev_status)
    print("Num new HIV+ = ", sum(new_cases))
    miracles = (~ data["HIV_status"]) & (prev_status)
    under_15s_idx = selector(data, HIV_status=(operator.eq, True), age=(operator.le, 15))
    assert not any(miracles)
    assert any(new_cases)
    assert not any(under_15s_idx)
