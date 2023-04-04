import operator as op
from datetime import date
from math import isclose, sqrt

import hivpy.column_names as col
from hivpy.common import rng
from hivpy.population import Population


def test_first_time_testers():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2010, 1, 1))
    pop.data[col.AGE] = 20
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.init_rate_first_test = 0.1
    pop.hiv_testing.date_test_rate_plateau = 2015.5
    pop.hiv_testing.an_lin_incr_test = 0.8
    pop.hiv_testing.no_test_if_np0 = False
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.update_hiv_testing(pop)

    # get stats
    testing_population = pop.get_sub_pop([(col.HARD_REACH, op.eq, False)])
    # all previously untested
    no_first_time_testers = len(pop.get_sub_pop([(col.EVER_TESTED, op.eq, True)]))
    prob_test = pop.hiv_testing.calc_prob_test(False, 0, 0)
    mean = len(testing_population) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= no_first_time_testers <= mean + 3 * stdev


def test_repeat_testers():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2010, 1, 1))
    pop.data[col.AGE] = 20
    pop.data[col.EVER_TESTED] = True
    pop.data[col.LAST_TEST_DATE] = date(2008, 1, 1)
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_test_rate_plateau = 2015.5
    pop.hiv_testing.an_lin_incr_test = 0.8
    pop.hiv_testing.no_test_if_np0 = False
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.update_hiv_testing(pop)

    # get stats
    testing_population = pop.get_sub_pop([(col.HARD_REACH, op.eq, False)])
    # all previously tested
    no_repeat_testers = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    prob_test = pop.hiv_testing.calc_prob_test(True, 0, 0)
    mean = len(testing_population) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= no_repeat_testers <= mean + 3 * stdev


def test_partner_reset_after_test():

    for ever_tested in [True, False]:

        # build population
        N = 100000
        pop = Population(size=N, start_date=date(2010, 1, 1))
        pop.data[col.AGE] = 20
        pop.data[col.EVER_TESTED] = ever_tested
        if ever_tested:
            pop.data[col.LAST_TEST_DATE] = date(2008, 1, 1)
        pop.data[col.NP_LAST_TEST] = 2
        pop.data[col.NSTP_LAST_TEST] = 1
        # fixing some values
        pop.hiv_testing.date_start_testing = 2003.5
        pop.hiv_testing.init_rate_first_test = 0.1
        pop.hiv_testing.date_test_rate_plateau = 2015.5
        pop.hiv_testing.an_lin_incr_test = 0.8
        pop.hiv_testing.no_test_if_np0 = False
        pop.hiv_testing.covid_disrup_affected = False
        pop.hiv_testing.testing_disrup_covid = False

        # diagnose roughly 20% of the population with HIV
        r = rng.uniform(size=len(pop.data))
        diagnosed = r < 0.2
        pop.data[col.HIV_STATUS] = diagnosed

        # evolve population
        pop.hiv_testing.update_hiv_testing(pop)

        # get people that were just tested
        tested_population = pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)])
        # check that partner numbers have been reset
        assert sum(pop.get_variable(col.NP_LAST_TEST, sub_pop=tested_population)) == 0
        assert sum(pop.get_variable(col.NSTP_LAST_TEST, sub_pop=tested_population)) == 0
        # check that no dead people were just tested
        assert (pop.get_variable(col.DATE_OF_DEATH, sub_pop=tested_population).isna()).all()
        # check that no people just tested were already diagnosed with HIV
        assert (~pop.get_variable(col.HIV_STATUS, sub_pop=tested_population)).all()


def test_calc_prob_test():

    # setup
    pop = Population(size=1, start_date=date(1990, 1, 1))
    # fixing some probabilities
    pop.hiv_testing.rate_first_test = 0.6
    pop.hiv_testing.rate_rep_test = 0.45
    pop.hiv_testing.eff_test_targeting = 1.5
    pop.hiv_testing.no_test_if_np0 = False

    # check first-time tester with no partners case
    # 0.6 / 1.5 = 0.4
    assert isclose(pop.hiv_testing.calc_prob_test(False, 0, 0), 0.4)
    # check first-time tester with ltp case
    # 0.6
    assert isclose(pop.hiv_testing.calc_prob_test(False, 1, 0), 0.6)
    # check first-time tester with stp case
    # 0.6 * 1.5 = 0.9
    assert isclose(pop.hiv_testing.calc_prob_test(False, 1, 1), 0.9)

    # check repeat tester with no partners case
    # 0.45 / 1.5 = 0.3
    assert isclose(pop.hiv_testing.calc_prob_test(True, 0, 0), 0.3)
    # check repeat tester with ltp case
    # 0.45
    assert isclose(pop.hiv_testing.calc_prob_test(True, 1, 0), 0.45)
    # check repeat tester with stp case
    # 0.45 * 1.5 = 0.675
    assert isclose(pop.hiv_testing.calc_prob_test(True, 1, 1), 0.675)

    pop.hiv_testing.no_test_if_np0 = True
    # check no_test_if_np0 and no partners case
    # 0
    assert isclose(pop.hiv_testing.calc_prob_test(False, 0, 0), 0)
