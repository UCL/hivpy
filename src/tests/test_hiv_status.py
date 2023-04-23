import operator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import hivpy.column_names as col
from hivpy.common import SexType, rng
from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population


@pytest.fixture
def pop_with_initial_hiv():
    pop_size = 100000
    pop = Population(size=pop_size, start_date=date(1989, 1, 1))
    partners = pop.get_variable(col.NUM_PARTNERS) > 0
    overThresh = pop.get_variable(col.NUM_PARTNERS) > pop.hiv_status.initial_hiv_newp_threshold
    print("total partnered", sum(partners))
    print("total over threshold", sum(overThresh))
    print("Num HIV+ = ", sum(pop.get_variable(col.HIV_STATUS)))
    return pop


@pytest.mark.parametrize("pop_percentage", [0.4, 0.0])
def test_initial_hiv_threshold(pop_percentage):
    """
    Check that HIV is initially introduced only to those with high enough newp.
    """
    # Start before 1989 to avoid having HIV introduced when creating population
    pop = Population(size=1000, start_date=date(1988, 1, 1))
    HIV_module = HIVStatusModule()
    threshold = HIV_module.initial_hiv_newp_threshold
    # select a proportion of the population to have high enough newp
    newp = rng.choice(
        [threshold, threshold - 1],
        p=[pop_percentage, 1 - pop_percentage],
        size=pop.size
    )
    print("newp = ", newp)
    print("sum newp = ", sum(newp))
    pop.set_present_variable(col.NUM_PARTNERS, newp)
    initial_status = HIV_module.introduce_HIV(pop)
    print(pop.get_variable(col.NUM_PARTNERS))
    print(sum(pop.get_variable(col.NUM_PARTNERS)))
    print(initial_status)
    assert not any(initial_status & (pop.get_variable(col.NUM_PARTNERS) < threshold))


def test_initial_hiv_probability():
    """
    Check that HIV is initially assigned with the specified probability.
    """
    # Start before 1989 to avoid having HIV introduced when creating population
    pop = Population(size=1000, start_date=date(1988, 1, 1))
    HIV_module = HIVStatusModule()
    # Have everyone be a candidate for initial introduction
    pop.set_present_variable(col.NUM_PARTNERS, HIV_module.initial_hiv_newp_threshold)
    expected_infections = HIV_module.initial_hiv_prob * pop.size
    initial_status = HIV_module.introduce_HIV(pop)
    # TODO Could check against variance of binomial distribution, see issue #45
    assert initial_status.sum() == pytest.approx(expected_infections, rel=0.05)


def test_hiv_introduced_only_once(mocker):
    """
    Check that we do not initialise HIV status repeatedly.
    """
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


def test_hiv_not_reintroduced_after_1989(mocker):
    """
    Check that we do not initialise HIV status repeatedly.
    """
    pop = Population(size=1000, start_date=date(1989, 4, 1))
    assert pop.HIV_introduced
    # HIV has been introduced, so that should not be called again
    spy = mocker.spy(pop.hiv_status, "introduce_HIV")
    for _ in range(10):
        pop.evolve(timedelta(days=31))
    spy.assert_not_called()


def test_hiv_initial_ages(pop_with_initial_hiv: Population):
    """
    Check that HIV is not introduced to anyone <= 15 or > 65.
    """
    under_15s = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, operator.eq, True),
                                                  (col.AGE, operator.le, 15)])
    over_65s = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, operator.eq, True),
                                                 (col.AGE, operator.gt, 65)])
    HIV_pos = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, operator.eq, True)])
    assert not any(under_15s)
    assert not any(over_65s)
    assert any(HIV_pos)


def test_hiv_update(pop_with_initial_hiv: Population):
    pd.set_option('display.max_columns', None)
    prev_status = pop_with_initial_hiv.get_variable(col.HIV_STATUS).copy()

    for i in range(10):
        pop_with_initial_hiv.hiv_status.update_HIV_status(pop_with_initial_hiv)

    current_status = np.array(pop_with_initial_hiv.get_variable(col.HIV_STATUS))

    new_cases = current_status & (~ prev_status)
    print("Num new HIV+ = ", sum(new_cases))
    miracles = (~current_status) & (prev_status)
    under_15s_idx = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, operator.eq, True),
                                                      (col.AGE, operator.le, 15)])
    assert not any(miracles)
    assert any(new_cases)
    assert not any(under_15s_idx)


def test_HIV_risk_vector():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    hiv_module = pop.hiv_status
    # Test probability of partnering with someone with HIV by sex and age group
    # 5 age groups (15-25, 25-35, 35-45, 45-55, 55-65) and 2 sexes = 10 groups
    N_group = N // 10  # number of people we will put in each group
    sex_list = []
    age_group_list = []
    HIV_list = []
    HIV_ratio = 10  # mark 1 in 10 people as HIV positive
    for sex in SexType:
        for age_group in range(5):
            sex_list += [sex] * N_group
            age_group_list += [age_group] * N_group
            HIV_list += [True] * (N_group // HIV_ratio) + [False] * (N_group - N_group//HIV_ratio)
    pop.data[col.SEX] = np.array(sex_list)
    pop.data[col.SEX_MIX_AGE_GROUP] = np.array(age_group_list)
    pop.data[col.HIV_STATUS] = np.array(HIV_list)
    pop.data[col.NUM_PARTNERS] = 1  # give everyone a single stp to start with

    # if everyone has the same number of partners,
    # probability of being with someone with HIV should be = HIV prevalence
    hiv_module.update_partner_risk_vectors(pop)
    expectation = np.array([0.1]*5)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Male], expectation)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Female], expectation)

    # Check for differences in male and female rate correctly
    # change HIV rate in men to double
    males = pop.data.index[pop.data[col.SEX] == SexType.Male]
    # transform group fails when only grouped by one field
    # appears to change the type of the object passed to the function!
    male_HIV_status = pop.transform_group([col.SEX_MIX_AGE_GROUP, col.SEX], lambda x, y: np.array(
        [True] * (2 * N_group // HIV_ratio) +
        [False] * (N_group - 2*N_group // HIV_ratio)), False, males)
    pop.data.loc[males, col.HIV_STATUS] = male_HIV_status
    hiv_module.update_partner_risk_vectors(pop)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Male], 2*expectation)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Female], expectation)

    # Check for difference when changing number of partners between HIV + / - people
    HIV_positive = pop.data.index[pop.data[col.HIV_STATUS]]
    # 2 partners for each HIV+ person, one for each HIV- person.
    pop.data.loc[HIV_positive, col.NUM_PARTNERS] = 2
    expectation_male = (2 * 0.2) / (2*0.2 + 0.8)
    expectation_female = (2 * 0.1) / (2*0.1 + 0.9)
    hiv_module.update_partner_risk_vectors(pop)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Male], expectation_male)
    assert np.allclose(hiv_module.stp_HIV_rate[SexType.Female], expectation_female)


def test_viral_group_risk_vector():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    hiv_module = pop.hiv_status
    # Test probability of partnering with someone with HIV by sex and age group
    # 5 age groups (15-25, 25-35, 35-45, 45-55, 55-65) and 2 sexes = 10 groups
    N_group = N // 10  # number of people we will put in each group
    sex_list = []
    age_group_list = []
    HIV_list = []
    HIV_ratio = 10  # mark 1 in 10 people as HIV positive
    for sex in SexType:
        for age_group in range(5):
            sex_list += [sex] * N_group
            age_group_list += [age_group] * N_group
            HIV_list += [True] * (N_group // HIV_ratio) + [False] * (N_group - N_group//HIV_ratio)
    pop.data[col.SEX] = np.array(sex_list)
    pop.data[col.SEX_MIX_AGE_GROUP] = np.array(age_group_list)
    pop.data[col.NUM_PARTNERS] = 1  # give everyone a single stp to start with]
    pop.data[col.VIRAL_LOAD_GROUP] = 1  # put everyone in the same viral load group to begin with
    hiv_module.update_partner_risk_vectors(pop)  # probability of group 1 should be 100%
    expectation = np.array([0., 1., 0., 0., 0., 0., 0.])
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Male], expectation)
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Female], expectation)
    pop.data[col.VIRAL_LOAD_GROUP] = np.array([1, 2] * (N // 2))  # alternate groups 1 & 2
    pop.data.loc[pop.data[col.VIRAL_LOAD_GROUP] == 1, col.NUM_PARTNERS] = 2
    hiv_module.update_partner_risk_vectors(pop)
    expectation = np.array([0., 2/3, 1/3, 0., 0., 0., 0.])
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Male], expectation)
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Female], expectation)
    # check for appropriate sex differences
    pop.data.loc[(pop.data[col.VIRAL_LOAD_GROUP] == 1) & (
        pop.data[col.SEX] == SexType.Female), col.VIRAL_LOAD_GROUP] = 3
    hiv_module.update_partner_risk_vectors(pop)
    expecation_female = np.array([0., 0., 1/3, 2/3, 0., 0., 0.])
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Male], expectation)
    assert np.allclose(hiv_module.stp_viral_group_rate[SexType.Female], expecation_female)
