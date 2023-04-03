import operator as op
from datetime import date
from math import sqrt

import hivpy.column_names as col
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
