import operator
from datetime import date

import pandas as pd
import numpy as np
from hivpy.common import SexType

from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population
from hivpy.sexual_behaviour import selector
import hivpy.column_names as col


def test_initial_hiv():
    """Check no one under 15 or over 65 has HIV initially, but some people do"""
    pop = Population(size=1000, start_date=date(1989, 1, 1)).data
    HIV_module = HIVStatusModule()
    pop["HIV_status"] = HIV_module.initial_HIV_status(pop)
    index_young = selector(pop, HIV_status=(operator.eq, True), age=(operator.le, 15))
    assert not any(index_young)
    index_old = selector(pop, HIV_status=(operator.eq, True), age=(operator.ge, 65))
    assert not any(index_old)
    index_pos = selector(pop, HIV_status=(operator.eq, True))
    assert any(index_pos)


def test_hiv_update():
    pd.set_option('display.max_columns', None)
    pop_size = 100000
    pop = Population(size=pop_size, start_date=date(1989, 1, 1))
    data = pop.data
    prev_status = data["HIV_status"].copy()

    for i in range(10):
        pop.hiv_status.update_HIV_status(pop)

    new_cases = data["HIV_status"] & (~ prev_status)
    miracles = (~ data["HIV_status"]) & (prev_status)
    under_15s_idx = selector(data, HIV_status=(operator.eq, True), age=(operator.le, 15))
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


def test_hiv_update2():
    N = 100000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    hiv_module = pop.hiv_status
    # create some easy to calculate scenarios
    # let everyone have one partner so P(HIV) is just the fraction with HIV
    pop.data[col.SEX] = np.array([SexType.Male, SexType.Female] * (N//2))
    pop.data[col.AGE] = 30
    pop.data[col.HIV_STATUS] = False
    pop.data[col.HIV_STATUS][:10000] = np.array([True] * 10000)  # 10% HIV+
    pop.data[col.NUM_PARTNERS] = 1
    pop.sexual_behaviour.assign_stp_ages(pop)
    pop.data[col.VIRAL_LOAD_GROUP] = 5  # set everyone to same viral load group
    prev_hiv_status = pop.data[col.HIV_STATUS].copy()
    hiv_module.update_HIV_status(pop)
    # 10% probability of HIV+ and 10% probability of transmission given viral loa
    expected_transmission_prob = 0.1 * 0.1
    pop.data["delta_HIV"] = pop.data[col.HIV_STATUS] ^ prev_hiv_status
    # count number of new transmissions for men
    n_new_male = sum(pop.data.loc[pop.data[col.SEX] == SexType.Male, "delta_HIV"])
    exp_new_male = 45000 * expected_transmission_prob
    sig = np.sqrt(45000 * expected_transmission_prob * (1 - expected_transmission_prob))
    print(n_new_male, exp_new_male, sig)
