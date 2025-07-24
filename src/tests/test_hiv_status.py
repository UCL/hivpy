import operator as op

import numpy as np
import pandas as pd
import pytest

import hivpy.column_names as col
from hivpy.common import AND, COND, SexType, date, rng, timedelta
from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population


@pytest.fixture(autouse=True)
def resetRandomState():
    rng.set_seed(42)


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
    HIV_module.introduce_HIV(pop)
    initial_status = pop.get_variable(col.HIV_STATUS)
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
    HIV_module.introduce_HIV(pop)
    initial_status = pop.get_variable(col.HIV_STATUS)
    # TODO Could check against variance of binomial distribution, see issue #45
    assert initial_status.sum() == pytest.approx(expected_infections, rel=0.05)


def test_hiv_introduced_only_once(mocker):
    """
    Check that we do not initialise HIV status repeatedly.
    """
    pop = Population(size=10000, start_date=date(1988, 12, 1))
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
    under_15s = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                  (col.AGE, op.le, 15)])
    over_65s = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                 (col.AGE, op.gt, 65)])
    HIV_pos = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, op.eq, True)])
    assert not any(under_15s)
    assert not any(over_65s)
    assert any(HIV_pos)


def test_hiv_update(pop_with_initial_hiv: Population):
    pd.set_option('display.max_columns', None)
    prev_status = pop_with_initial_hiv.get_variable(col.HIV_STATUS).copy()
    initial_infected = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, op.eq, True)])
    for i in range(10):
        pop_with_initial_hiv.date += timedelta(days=30)
        pop_with_initial_hiv.hiv_status.set_viral_load_groups(pop_with_initial_hiv)
        pop_with_initial_hiv.hiv_status.update_HIV_status(pop_with_initial_hiv)

    current_status = np.array(pop_with_initial_hiv.get_variable(col.HIV_STATUS))

    new_cases = current_status & (~ prev_status)
    assert not any(pop_with_initial_hiv.get_variable(col.DATE_HIV_INFECTION, initial_infected)
                   == pop_with_initial_hiv.date)
    print("Num new HIV+ = ", sum(new_cases))
    miracles = (~current_status) & (prev_status)
    under_15s_idx = pop_with_initial_hiv.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                      (col.AGE, op.le, 15)])
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
    pop.set_present_variable(col.SEX, np.array(sex_list))
    pop.set_present_variable(col.SEX_MIX_AGE_GROUP, np.array(age_group_list))
    pop.set_present_variable(col.HIV_STATUS, np.array(HIV_list))
    pop.set_present_variable(col.NUM_PARTNERS, 1)  # give everyone a single stp to start with

    # if everyone has the same number of partners,
    # probability of being with someone with HIV should be = HIV prevalence
    hiv_module.update_partner_risk_vectors(pop)
    expectation = np.array([0.1]*5)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Male], expectation)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Female], expectation)

    # Check for differences in male and female rate correctly
    # change HIV rate in men to double
    males = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Male))
    # transform group fails when only grouped by one field
    # appears to change the type of the object passed to the function!
    male_HIV_status = pop.transform_group([col.SEX_MIX_AGE_GROUP, col.SEX], lambda x, y: np.array(
        [True] * (2 * N_group // HIV_ratio) +
        [False] * (N_group - 2*N_group // HIV_ratio)), False, males)
    pop.set_present_variable(col.HIV_STATUS, male_HIV_status, males)
    hiv_module.update_partner_risk_vectors(pop)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Male], 2*expectation)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Female], expectation)

    # Check for difference when changing number of partners between HIV + / - people
    HIV_positive = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
    # 2 partners for each HIV+ person, one for each HIV- person.
    pop.set_present_variable(col.NUM_PARTNERS, 2, HIV_positive)
    expectation_male = (2 * 0.2) / (2*0.2 + 0.8)
    expectation_female = (2 * 0.1) / (2*0.1 + 0.9)
    hiv_module.update_partner_risk_vectors(pop)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Male], expectation_male)
    assert np.allclose(hiv_module.ratio_infected_stp[SexType.Female], expectation_female)


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
    pop.set_present_variable(col.DATE_HIV_INFECTION, pop.date)
    pop.date += timedelta(days=180)  # no one in primary infection
    hiv_module.set_primary_infection(pop)
    pop.set_present_variable(col.HIV_STATUS, HIV_list)
    pop.set_present_variable(col.SEX, np.array(sex_list))
    pop.set_present_variable(col.SEX_MIX_AGE_GROUP, np.array(age_group_list))
    pop.set_present_variable(col.NUM_PARTNERS, 1)  # give everyone a single stp to start with
    pop.set_present_variable(col.VIRAL_LOAD, 3.0)  # put everyone in the same viral load group to begin with
    hiv_module.set_viral_load_groups(pop)
    hiv_module.update_partner_risk_vectors(pop)  # probability of group 1 should be 100%
    expectation = np.array([0., 1., 0., 0., 0., 0.])
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Male], expectation)
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Female], expectation)
    pop.set_present_variable(col.VIRAL_LOAD, np.array([3.0, 4.0] * (N // 2)))  # alternate groups 1 & 2
    pop.set_present_variable(col.NUM_PARTNERS, 2, pop.get_sub_pop(COND(col.VIRAL_LOAD, op.eq, 3.0)))  # 2 partners for anyone with viral load of 3
    hiv_module.set_viral_load_groups(pop)
    hiv_module.update_partner_risk_vectors(pop)
    expectation = np.array([0., 2/3, 1/3, 0., 0., 0.])
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Male], expectation)
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Female], expectation)
    # check for appropriate sex differences
    women_vl_3 = pop.get_sub_pop(AND(COND(col.VIRAL_LOAD, op.eq, 3.0),
                                     COND(col.SEX, op.eq, SexType.Female)))
    pop.set_present_variable(col.VIRAL_LOAD, 5.0, women_vl_3)
    hiv_module.set_viral_load_groups(pop)
    hiv_module.update_partner_risk_vectors(pop)
    expecation_female = np.array([0., 0., 1/3, 2/3, 0., 0.])
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Male], expectation)
    assert np.allclose(hiv_module.ratio_vl_stp[SexType.Female], expecation_female)


def test_initial_vl():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.HIV_STATUS, [True, False] * 500)
    # Reset viral load for testing (original values affected by intro of HIV)
    pop.set_present_variable(col.VIRAL_LOAD, 0.0)
    hivpos_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
    hivneg_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, False))
    hiv_module = pop.hiv_status
    assert (np.allclose(pop.get_variable(col.VIRAL_LOAD, hivneg_subpop), 0.0))
    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Male)
    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Female)

    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Male, age_diff=10)
    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Female, age_diff=10)

    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Male, age_diff=-10)
    check_initial_vl_by_sex(pop, hiv_module, hivpos_subpop, hivneg_subpop, SexType.Female, age_diff=-10)


def check_initial_vl_by_sex(pop: Population, hiv_module, hivpos_subpop, hivneg_subpop, sex, age_diff=0):
    pop.set_present_variable(col.AGE, 35 + age_diff)
    hiv_module.initialise_HIV_progression(pop, hivpos_subpop)
    hiv_pos_by_sex = pop.get_sub_pop_intersection(hivpos_subpop,
                                                  pop.get_sub_pop([(col.SEX, op.eq, sex)]))
    assert np.allclose(pop.get_variable(col.VIRAL_LOAD, hivneg_subpop), 0)
    new_vls = pop.get_variable(col.VIRAL_LOAD, hiv_pos_by_sex)
    average_vl = np.average(new_vls)
    vl_sigma = np.std(new_vls)
    expected_average = 4.075 if sex == SexType.Male else 3.875
    expected_average += age_diff*0.005
    expected_sigma = 0.5
    assert (np.isclose(average_vl, expected_average, atol=0.1))
    assert (np.isclose(vl_sigma, expected_sigma, atol=0.1))
    assert (np.all(new_vls <= 6.5))


def test_naive_vl_progression():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    init_vl = 5
    pop.set_present_variable(col.AGE, 35)
    pop.set_present_variable(col.VIRAL_LOAD, init_vl)
    pop.set_present_variable(col.HIV_STATUS, [True, False] * 500)
    # Reset viral load for testing (original values affected by intro of HIV)
    pop.set_present_variable(col.VIRAL_LOAD, 0.0)
    pop.set_present_variable(col.ART_NAIVE, True)
    hiv_module = pop.hiv_status
    hiv_module.vl_base_change = 1.5
    hivpos_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
    hivneg_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, False))
    pop.set_present_variable(col.CD4, 1000.0, hivpos_subpop)
    # Check general formula and age variance
    check_vl_update(pop, init_vl, hiv_module, hivpos_subpop, hivneg_subpop, 0)

    check_vl_update(pop, init_vl, hiv_module, hivpos_subpop, hivneg_subpop, 10)

    check_vl_update(pop, init_vl, hiv_module, hivpos_subpop, hivneg_subpop, -10)

    # Check that we can't exceed maximum viral load
    pop.set_present_variable(col.AGE, 35)
    pop.set_present_variable(col.VIRAL_LOAD, 6.5)
    hiv_module.update_HIV_progression(pop, hivpos_subpop)
    new_vls = pop.get_variable(col.VIRAL_LOAD, hivpos_subpop)
    assert (all(new_vls <= 6.5))


def check_vl_update(pop, init_vl, hiv_module, hivpos_subpop, hivneg_subpop, age_diff):
    pop.set_present_variable(col.AGE, 35 + age_diff)
    pop.set_present_variable(col.VIRAL_LOAD, init_vl)
    hiv_module.update_HIV_progression(pop, hivpos_subpop)
    assert (all(pop.get_variable(col.VIRAL_LOAD, hivneg_subpop) == 5))
    new_vls = pop.get_variable(col.VIRAL_LOAD, hivpos_subpop)
    average_new_vl = np.average(new_vls)
    new_vl_sigma = np.std(new_vls)
    expected_average = init_vl + 0.02275*1.5 + age_diff*0.00075
    expected_sigma = 0.05
    # Tolerances are basically arbitrary
    assert (np.isclose(average_new_vl, expected_average, atol=0.01))
    assert (np.isclose(new_vl_sigma, expected_sigma, atol=0.2))


def test_initial_cd4():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.HIV_STATUS, [True, False] * 5000)
    pop.set_present_variable(col.CD4, 0.0)
    pop.set_present_variable(col.VIRAL_LOAD, 0.0)
    hivpos_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
    hivneg_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, False))
    hiv_module = pop.hiv_status

    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Male, 0)
    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Male, 10)
    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Male, -10)
    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Female, 0)
    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Female, 10)
    check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, SexType.Female, -10)


def check_init_cd4_by_sex_age(pop, hivpos_subpop, hivneg_subpop, hiv_module, sex, age_diff):
    pop.set_present_variable(col.AGE, 35 + age_diff)
    pop.set_present_variable(col.CD4, 0.0)
    pop.set_present_variable(col.VIRAL_LOAD, 0.0)
    hiv_module.initialise_HIV_progression(pop, hivpos_subpop)
    neg_cd4_counts = pop.get_variable(col.CD4, hivneg_subpop)
    assert np.allclose(neg_cd4_counts, 0.0)
    hiv_pop_by_sex = pop.get_sub_pop_intersection(
        pop.get_sub_pop([(col.SEX, op.eq, sex)]),
        hivpos_subpop
    )
    cd4_counts = pop.get_variable(col.CD4, hiv_pop_by_sex)
    sqrt_cd4_counts = np.sqrt(cd4_counts)
    average_sqrt_cd4 = np.average(sqrt_cd4_counts)
    sigma_sqrt_cd4 = np.std(sqrt_cd4_counts)
    expected_VL = 4.075 if sex == SexType.Male else 3.875
    expected_VL += age_diff*0.005
    expected_sigma_VL = 0.5
    expected_sqrt_cd4 = 27.5 - 1.5*expected_VL - age_diff*0.05
    expected_sigma_sqrt_cd4 = np.sqrt(4 + expected_sigma_VL**2)
    assert np.isclose(average_sqrt_cd4, expected_sqrt_cd4, rtol=0.01)
    assert np.isclose(sigma_sqrt_cd4, expected_sigma_sqrt_cd4, rtol=0.1)
    assert np.all(cd4_counts >= 324)
    assert np.all(cd4_counts <= 1500)


def test_naive_cd4_progression():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.AGE, 35)
    pop.set_present_variable(col.HIV_STATUS, [True, False] * 5000)
    # Reset viral load for testing (original values affected by intro of HIV)
    pop.set_present_variable(col.ART_NAIVE, True)
    hiv_module = pop.hiv_status
    hiv_module.cd4_base_change = 1.5  # Fix to avoid sampling
    hivpos_subpop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))

    pop.set_present_variable(col.VIRAL_LOAD, 5.25)  # in the middle of a VL group
    check_cd4_progression(pop, hiv_module, hivpos_subpop, 0.85)
    pop.set_present_variable(col.VIRAL_LOAD, 5.75)  # in the middle of a VL group
    check_cd4_progression(pop, hiv_module, hivpos_subpop, 1.3)
    pop.set_present_variable(col.VIRAL_LOAD, 4.75)  # in the middle of a VL group
    check_cd4_progression(pop, hiv_module, hivpos_subpop, 0.4)


def check_cd4_progression(pop: Population, hiv_module, hivpos_subpop, change_factor):
    pop.set_present_variable(col.CD4, 1000.0)
    hiv_module.update_HIV_progression(pop, hivpos_subpop)
    cd4_counts = pop.get_variable(col.CD4, hivpos_subpop)
    assert np.all(cd4_counts <= 1500)
    assert np.all(cd4_counts > 324)
    sqrt_cd4 = np.sqrt(cd4_counts)
    average_sqrt_cd4 = np.average(sqrt_cd4)
    sigma_sqrt_cd4 = np.std(sqrt_cd4)
    expected_sqrt_cd4 = np.sqrt(1000) - change_factor * rng.normal(1.5, 1.2)
    expected_sigma = 1.2
    assert np.isclose(average_sqrt_cd4, expected_sqrt_cd4, rtol=0.1)
    assert np.isclose(sigma_sqrt_cd4, expected_sigma, rtol=0.1)


def test_who3_tb():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.AGE, 35)
    pop.set_present_variable(col.HIV_STATUS, True)

    pop.hiv_status.HIV_related_disease_risk(pop, timedelta(0, 3, 0))
    tb_infected = pop.get_sub_pop(COND(col.TB, op.eq, True))
    assert len(tb_infected) > 0  # FIXME: figure out correct TB probability
    tb_infected_dates = pop.get_variable(col.TB_INFECTION_DATE, tb_infected)
    assert all(tb_infected_dates == date(1989, 1, 1))


def test_no_infection_in_nonmonogamous_ltp():
    """
    Check that the non-monogamous ltp transmission does not occur if noone has HIV.
    """
    N = 1000
    pop = Population(size=N, start_date=date(1990, 1, 1))

    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_MONOGAMOUS, False)
    pop.set_present_variable(col.HIV_STATUS, False)
    pop.set_present_variable(col.IN_PRIMARY_INFECTION, False)
    pop.hiv_status.update_ltp_HIV(pop)

    assert len(pop.get_sub_pop([(col.LTP_STATUS, op.eq, True)])) == 0


def test_ltp_infected_by_third_party():
    """
    Tests conditions under which non monogamous long term partners are infected
    by some third party
    """
    N = 10000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    HIVM = pop.hiv_status

    pop.set_present_variable(col.AGE, 25)
    pop.set_present_variable(col.NUM_PARTNERS, 0)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    # set half ot he population monogamous; these should be unaffected
    pop.set_present_variable(col.LTP_MONOGAMOUS, [True]*(N//2) + [False]*(N//2))
    monogamous_partners = pop.get_sub_pop_from_array([True]*(N//2) + [False]*(N//2))

    # incidence factor depends on numbers of sero-discordant couples
    pop.set_present_variable(col.SEX, [SexType.Male, SexType.Female]*(N//2))
    # no transmission if no HIV present
    pop.set_present_variable(col.HIV_STATUS, False)
    HIVM.non_monogamous_ltp_transmission(pop)
    assert not any(pop.get_variable(col.LTP_STATUS))

    men = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Male))
    women = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Female))

    # half of men are HIV+, half of which are in primary infection, and a quarter (1250) are HIV+ with a negative LTP
    pop.set_present_variable(col.HIV_STATUS, [True, False]*(N//4), men)
    pop.set_present_variable(col.IN_PRIMARY_INFECTION, ([True] + [False]*7)*(N//16), men)
    pop.set_present_variable(col.NUM_PARTNERS, [1, 0] * (N//4), men)  # half of these men are non-monogamous
    pop.set_present_variable(col.LTP_STATUS, [True, True, False, False]*(N//8), men)
    # half of women HIV-, and 1150 are HIV- with a positive LTP
    pop.set_present_variable(col.HIV_STATUS, [True, False]*(N//4), women)
    pop.set_present_variable(col.LTP_STATUS, [True, True] * 1240 + [False, False] * 1260, women)

    # this creates a difference of -10 out of 10000 => incidence factor of 3.5 for men
    HIVM.update_LTP_risk_vectors(pop)
    assert (HIVM.incidence_factor[SexType.Male] == 3.5)
    # neg men w/pos partners = 1250, pos women w/neg partners = 1350, difference = -10
    assert (HIVM.incidence_factor[SexType.Female] == 3.5)

    prev_status_of_monogamous_partners = pop.get_variable(col.LTP_STATUS, monogamous_partners)
    prev_status_ltp_of_women = pop.get_variable(col.LTP_STATUS, women)
    men_with_non_monog_ltp = pop.get_sub_pop(AND(COND(col.SEX, op.eq, SexType.Male),
                                                 COND(col.LTP_MONOGAMOUS, op.eq, False)))
    prev_status_non_monog_ltp_of_men = pop.get_variable(col.LTP_STATUS,
                                                        men_with_non_monog_ltp)
    num_pos_ltp_of_men = sum(prev_status_non_monog_ltp_of_men)
    num_neg_ltp_of_men = len(prev_status_non_monog_ltp_of_men) - num_pos_ltp_of_men
    HIVM.non_monogamous_ltp_transmission(pop)

    # Monogamous long term partners cannot be infected by a third party
    assert all(pop.get_variable(col.LTP_STATUS, monogamous_partners) == prev_status_of_monogamous_partners)
    # No women are in primary infection, so LTPs of women (i.e. male partners) should not be infected
    assert all(pop.get_variable(col.LTP_STATUS, women) == prev_status_ltp_of_women)

    # Some non-monogamous partners of men should be infected because 1/4 non-monogamous men are in primary infection
    new_pos_ltp_of_men = sum(pop.get_variable(col.LTP_STATUS, men_with_non_monog_ltp)) - num_pos_ltp_of_men
    prob_infection = 3.5 / 4
    expectation = num_neg_ltp_of_men * prob_infection
    variance = num_neg_ltp_of_men * prob_infection * (1 - prob_infection)
    assert ((expectation - 3 * variance) < new_pos_ltp_of_men < (expectation + 3 * variance))


@pytest.mark.parametrize("vl_group", [0, 1, 2, 3, 4, 5])
def test_ltp_infection_by_subject(vl_group):
    N = 10000
    pop = Population(size=N, start_date=date(1990, 1, 1))

    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_MONOGAMOUS, True)
    pop.set_present_variable(col.HIV_STATUS, True)
    pop.set_present_variable(col.AGE, 25)
    pop.set_present_variable(col.SEX, ([SexType.Male] * (N//2)) + ([SexType.Female] * (N//2)))
    pop.set_present_variable(col.LTP_STATUS, False)

    # 50% of men are monogamous and 75% of women are monogamous
    pop.set_present_variable(col.NUM_PARTNERS, [1]*(N//4) + [0] * (N//4) + [1] * (N//8) + [0] * (3 * N // 8))
    pop.set_present_variable(col.VIRAL_LOAD_GROUP, vl_group)

    HIVM = pop.hiv_status
    HIVM.set_ltp_age_groups(pop)
    HIVM.set_monogamous_ltp(pop)
    expected_men_monog = N//4
    expected_women_monog = 3*(N//8)
    men = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Male))
    women = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Female))
    num_women_monog = sum(pop.get_variable(col.LTP_MONOGAMOUS, men))
    num_men_monog = sum(pop.get_variable(col.LTP_MONOGAMOUS, women))
    assert (expected_men_monog * 0.75 < num_men_monog < expected_men_monog * 1.25)
    assert (expected_women_monog * 0.75 < num_women_monog < expected_women_monog * 1.25)

    expected_women_infected = num_women_monog * HIVM.transmission_rate_means[vl_group] * HIVM.women_transmission_factor
    sigma_women_infected = num_women_monog * HIVM.transmission_rate_sigmas[vl_group]
    expected_men_infected = num_men_monog * HIVM.transmission_rate_means[vl_group]
    sigma_men_infected = num_men_monog * HIVM.transmission_rate_sigmas[vl_group]

    HIVM.monogamous_ltp_transmission(pop)
    women_infected = sum(pop.get_variable(col.LTP_STATUS, men))
    men_infected = sum(pop.get_variable(col.LTP_STATUS, women))
    assert (np.floor(expected_men_infected - 4 * sigma_men_infected) <= men_infected)
    assert (men_infected <= np.ceil(expected_men_infected + 4 * sigma_men_infected))
    assert (np.floor(expected_women_infected - 4 * sigma_women_infected) <= women_infected)
    assert (women_infected <= np.ceil(expected_women_infected + 4 * sigma_women_infected))


@pytest.mark.parametrize("risk_factors", zip([0, 1, 2], [False, True], [False, True]))
def test_prob_infection_from_infected_ltp(risk_factors):
    """
    Check the expected infection risk is assigned to the subjects.
    """
    vl = risk_factors[0]
    sti = risk_factors[1]
    circumcised = risk_factors[2]

    N = 10000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.set_present_variable(col.SEX, [SexType.Male]*(N//2) + [SexType.Female]*(N//2))
    men = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Male))
    women = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Female))

    pop.set_present_variable(col.AGE, 25)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_STATUS, True)
    pop.set_present_variable(col.HIV_STATUS, False)
    pop.set_present_variable(col.STI, sti)
    if (circumcised):
        pop.set_present_variable(col.CIRCUMCISED, True, men)
    if (vl == 0):
        pop.set_present_variable(col.LTP_VIRAL_SUPPRESSED, True)
    elif (vl == 2):
        pop.set_present_variable(col.LTP_INFECTION_DATE, date(1990, 1, 1))

    HIVM = pop.hiv_status
    HIVM.set_infection_from_infected_ltp(pop)
    base_transmission = HIVM.tr_rate_undetectable_vl if vl == 0 else HIVM.tr_rate_primary if vl == 2 else 0.05*HIVM.transmission_factor
    expected_male_infections = (N//2) * base_transmission
    if (sti):
        expected_male_infections *= HIVM.sti_transmission_factor
    expected_female_infections = expected_male_infections * HIVM.women_transmission_factor
    sigma_female_infections = np.sqrt(expected_female_infections * (1-base_transmission))
    if (circumcised):
        expected_male_infections *= HIVM.circumcision_risk_reduction
    sigma_male_infections = np.sqrt(expected_male_infections * (1-base_transmission))

    male_infections = sum(pop.get_variable(col.HIV_STATUS, men))
    female_infections = sum(pop.get_variable(col.HIV_STATUS, women))
    assert (expected_male_infections - 3 * sigma_male_infections <= male_infections <= expected_male_infections + 3 * sigma_male_infections)
    assert (expected_female_infections - 3 * sigma_female_infections <= female_infections <= expected_female_infections + 3 * sigma_female_infections)
