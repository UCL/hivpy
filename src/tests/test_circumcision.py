from datetime import date, timedelta
from math import isclose, sqrt

import hivpy.column_names as col
from hivpy.common import SexType
from hivpy.population import Population


def reset_pop_circ(pop):
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    pop.data[col.VMMC] = False


def set_covid(circ_module, truth_val):
    circ_module.covid_disrup_affected = truth_val
    circ_module.vmmc_disrup_covid = truth_val


def set_vmmc_default_dates(circ_module):
    circ_module.vmmc_start_year = 2008
    circ_module.circ_rate_change_year = 2013
    circ_module.prob_circ_calc_cutoff_year = 2019
    circ_module.policy_intervention_year = 2022


def general_circumcision_checks(mean, stdev, no_circumcised, data):

    # no female is marked as circumcised
    assert sum((data[col.SEX] == SexType.Female)
               & data[col.CIRCUMCISED]) == 0

    # no uncircumcised people have circumcision dates and
    # all circumcised people have circumcision dates
    assert (data[col.CIRCUMCISED] == data[col.CIRCUMCISION_DATE].notnull()).all()

    # check circumcised value is within 3 standard deviations
    assert mean - 3 * stdev <= no_circumcised <= mean + 3 * stdev


def get_birth_circ_stats(pop, no_male):

    no_circumcised = sum(pop.data[col.CIRCUMCISED])
    mean = no_male * pop.circumcision.prob_birth_circ
    stdev = sqrt(mean * (1 - pop.circumcision.prob_birth_circ))

    return no_circumcised, mean, stdev


def get_vmmc_stats(pop, prob_circ):

    no_vmmc = sum(pop.data[col.VMMC])
    no_male = sum((pop.data[col.SEX] == SexType.Male)
                  & ~pop.data[col.HARD_REACH]
                  & (pop.data[col.AGE] >= 10)
                  & (pop.data[col.AGE] < 50))
    mean = no_male * prob_circ
    stdev = sqrt(mean * (1 - prob_circ))

    return no_vmmc, mean, stdev


def test_birth_circumcision_atonce():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    reset_pop_circ(pop)
    pop.circumcision.init_birth_circumcision_all(pop.data, pop.date)

    # get stats
    no_male = sum(pop.data[col.SEX] == SexType.Male)
    no_circumcised, mean, stdev = get_birth_circ_stats(pop, no_male)
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)
    # check that no circumcised people have undergone VMMC
    assert sum(pop.data[col.CIRCUMCISED] & pop.data[col.VMMC]) == 0


def test_birth_circumcision_stages():

    N = 100000
    start_date = date(2000, 1, 1)
    stop_date = date(2020, 1, 1)
    time_step = timedelta(days=90)

    # build population
    pop = Population(size=N, start_date=start_date)
    reset_pop_circ(pop)
    pop.circumcision.init_birth_circumcision_born(pop.data, pop.date)

    # get stats
    no_male = sum((pop.data[col.SEX] == SexType.Male)
                  & (pop.data[col.AGE] > 0.25))
    no_circumcised, mean, stdev = get_birth_circ_stats(pop, no_male)
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)

    # evolve population
    while pop.date <= stop_date:
        # advance ages and birth circumcision
        pop.data.age += time_step.days / 365
        pop.circumcision.update_birth_circumcision(pop.data, time_step, pop.date)
        pop.date += time_step

    # get stats
    no_male = sum((pop.data[col.SEX] == SexType.Male)
                  & (pop.data[col.AGE] > 0.25))
    no_circumcised, mean, stdev = get_birth_circ_stats(pop, no_male)
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)
    # check that no circumcised people have undergone VMMC
    assert sum(pop.data[col.CIRCUMCISED] & pop.data[col.VMMC]) == 0


def test_calc_prob_circ():

    # setup
    pop = Population(size=1, start_date=date(2010, 1, 1))
    pop.circumcision.date = pop.date
    pop.circumcision.circ_policy_scenario = 0
    set_covid(pop.circumcision, False)
    set_vmmc_default_dates(pop.circumcision)
    # fixing some probabilities
    pop.circumcision.circ_increase_rate = 0.1
    pop.circumcision.circ_rate_change_post_2013 = 1.5
    pop.circumcision.circ_rate_change_15_19 = 2
    pop.circumcision.circ_rate_change_20_29 = 0.5
    pop.circumcision.circ_rate_change_30_49 = 0.25

    # check basic case
    # (2 * 0.1) = 0.2
    assert isclose(pop.circumcision.calc_prob_circ(1), 0.2)
    # (2 * 0.1 * 0.5) = 0.1
    assert isclose(pop.circumcision.calc_prob_circ(2), 0.1)
    # (2 * 0.1 * 0.25) = 0.05
    assert isclose(pop.circumcision.calc_prob_circ(3), 0.05)

    # re-fixing some probabilities for easier math
    pop.circumcision.circ_rate_change_15_19 = 0.5
    pop.circumcision.circ_rate_change_20_29 = 0.2
    pop.circumcision.circ_rate_change_30_49 = 0.1

    pop.circumcision.date = date(2019, 1, 1)
    # check basic case post 2013
    # ((5 + 6 * 1.5) * 0.1) = 1.4
    assert isclose(pop.circumcision.calc_prob_circ(1), 1)
    # ((5 + 6 * 1.5) * 0.1 * 0.2) = 0.28
    assert isclose(pop.circumcision.calc_prob_circ(2), 0.28)
    # ((5 + 6 * 1.5) * 0.1 * 0.1) = 14
    assert isclose(pop.circumcision.calc_prob_circ(3), 0.14)

    pop.circumcision.date = date(2022, 1, 1)
    # check the year is capped at 2019 as expected
    # (repeat last assert)
    assert isclose(pop.circumcision.calc_prob_circ(3), 0.14)

    pop.circumcision.circ_policy_scenario = 1
    # check special group 1 case
    # ((5 + 6 * 1.5 * 0.5) * 0.1) = 0.95
    assert isclose(pop.circumcision.calc_prob_circ(1), 0.95)


def test_vmmc_case_0():

    test_ages = [18, 25, 40]
    for age in test_ages:

        N = 100000
        start_date = date(2007, 1, 1)
        stop_date = date(2010, 1, 1)
        time_step = timedelta(days=90)

        # build artificial population
        pop = Population(size=N, start_date=start_date)
        pop.data[col.SEX] = SexType.Male
        pop.data[col.AGE] = age
        reset_pop_circ(pop)

        pop.circumcision.circ_policy_scenario = 0
        set_covid(pop.circumcision, False)
        set_vmmc_default_dates(pop.circumcision)

        # evolve population for a year
        for i in range(0, 5):
            pop.data.age += time_step.days / 365
            pop.circumcision.update_vmmc(pop)
            pop.date += time_step
        # check no VMMC occurs until vmmc_start_year
        assert sum(pop.data[col.VMMC]) == 0

        # evolve population during vmmc_start_year
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop)

        # get stats
        prob_circ = pop.circumcision.calc_prob_circ(test_ages.index(age)+1)
        no_vmmc, mean, stdev = get_vmmc_stats(pop, prob_circ)
        # basic checks
        general_circumcision_checks(mean, stdev, no_vmmc, pop.data)
        # no hard to reach people have undergone VMMC
        assert sum(pop.data[col.HARD_REACH] & pop.data[col.VMMC]) == 0
        # nobody over 50 has been circumcised
        assert sum((pop.data[col.AGE] >= 50) & (pop.data[col.VMMC])) == 0

        # evolve population for a few more years
        while pop.date <= stop_date:
            circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                        & pop.data[col.CIRCUMCISED]]
            # advance ages and vmmc
            pop.data.age += time_step.days / 365
            pop.circumcision.update_vmmc(pop)
            pop.date += time_step
            # check circumcisied people remain circumcised each step
            new_circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                            & pop.data[col.CIRCUMCISED]]
            assert circ_males.isin(new_circ_males).all()

        # nobody under 10 has been circumcised
        assert sum((pop.data[col.AGE] < 10) & (pop.data[col.VMMC])) == 0


def test_vmmc_case_1():

    test_ages = [14, 18, 25, 40]
    for age in test_ages:

        N = 100000
        start_date = date(2022, 1, 1)
        time_step = timedelta(days=90)

        # build population
        pop = Population(size=N, start_date=start_date)
        pop.data[col.SEX] = SexType.Male
        pop.data[col.AGE] = age
        reset_pop_circ(pop)

        # case 1
        # circumcision stops in 10-14 year olds and
        # increases in 15-19 year olds after policy_intervention_year
        pop.circumcision.circ_policy_scenario = 1
        set_covid(pop.circumcision, False)
        set_vmmc_default_dates(pop.circumcision)

        # evolve population
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop)
        # nobody under 15 has been circumcised
        if age < 15:
            assert sum((pop.data[col.AGE] < 15) & (pop.data[col.VMMC])) == 0

        if age >= 15:
            # get stats
            prob_circ = pop.circumcision.calc_prob_circ(test_ages.index(age))
            no_vmmc, mean, stdev = get_vmmc_stats(pop, prob_circ)
            # check circumcised value is within 3 standard deviations
            assert mean - 3 * stdev <= no_vmmc <= mean + 3 * stdev


def test_vmmc_case_2():

    N = 100000
    start_date = date(2021, 12, 1)
    time_step = timedelta(days=90)

    # build population
    pop = Population(size=N, start_date=start_date)
    reset_pop_circ(pop)

    # case 2
    # no further circumcision after policy_intervention_year
    pop.circumcision.circ_policy_scenario = 2
    set_covid(pop.circumcision, False)
    set_vmmc_default_dates(pop.circumcision)

    # evolve population
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop)
    circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male) & pop.data[col.CIRCUMCISED]]
    pop.date += time_step
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop)
    new_circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                    & pop.data[col.CIRCUMCISED]]
    # check that circumcision has stopped
    assert circ_males.tolist() == new_circ_males.tolist()


def test_vmmc_case_3():

    test_ages = [18, 25, 40]
    for age in test_ages:

        N = 100000
        start_date = date(2022, 1, 1)
        time_step = timedelta(days=90)

        # build population
        pop = Population(size=N, start_date=start_date)
        pop.data[col.SEX] = SexType.Male
        pop.data[col.AGE] = age
        reset_pop_circ(pop)

        # case 3
        # circumcision stops in 10-14 year olds and does not
        # increase in 15-19 year olds after policy_intervention_year
        pop.circumcision.circ_policy_scenario = 3
        set_covid(pop.circumcision, False)
        set_vmmc_default_dates(pop.circumcision)

        # evolve population
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop)
        # nobody under 15 has been circumcised
        assert sum((pop.data[col.AGE] < 15) & (pop.data[col.VMMC])) == 0

        # get stats
        prob_circ = pop.circumcision.calc_prob_circ(test_ages.index(age)+1)
        no_vmmc, mean, stdev = get_vmmc_stats(pop, prob_circ)
        # check circumcised value is within 3 standard deviations
        assert mean - 3 * stdev <= no_vmmc <= mean + 3 * stdev


def test_vmmc_case_4():

    test_ages = [18, 25, 40]
    for age in test_ages:

        N = 100000
        start_date = date(2026, 12, 1)
        time_step = timedelta(days=90)

        # build population
        pop = Population(size=N, start_date=start_date)
        pop.data[col.SEX] = SexType.Male
        pop.data[col.AGE] = age
        reset_pop_circ(pop)

        # case 4
        # after policy_intervention_year circumcision stops in 10-14 year olds,
        # does not increase in 15-19 year olds,
        # and VMMC stops after 5 years
        pop.circumcision.circ_policy_scenario = 4
        set_covid(pop.circumcision, False)
        set_vmmc_default_dates(pop.circumcision)

        # evolve population
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop)
        circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male) & pop.data[col.CIRCUMCISED]]
        # nobody under 15 has been circumcised
        assert sum((pop.data[col.AGE] < 15) & (pop.data[col.VMMC])) == 0

        # get stats
        prob_circ = pop.circumcision.calc_prob_circ(test_ages.index(age)+1)
        no_vmmc, mean, stdev = get_vmmc_stats(pop, prob_circ)
        # check circumcised value is within 3 standard deviations
        assert mean - 3 * stdev <= no_vmmc <= mean + 3 * stdev

        # evolve population
        pop.date += time_step
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop)
        new_circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                        & pop.data[col.CIRCUMCISED]]
        # check that circumcision has stopped
        assert circ_males.tolist() == new_circ_males.tolist()


def test_circ_covid():

    N = 100000
    start_date = date(2010, 1, 1)
    time_step = timedelta(days=90)

    # build population
    pop = Population(size=N, start_date=start_date)
    reset_pop_circ(pop)

    # covid disruption is in place
    pop.circumcision.circ_policy_scenario = 0
    set_covid(pop.circumcision, True)
    set_vmmc_default_dates(pop.circumcision)

    # evolve population
    pop.circumcision.update_birth_circumcision(pop.data, time_step, pop.date)
    pop.circumcision.update_vmmc(pop)

    # check there was no circumcision
    assert sum(pop.data[col.CIRCUMCISED]) == 0
