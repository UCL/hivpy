import operator as op
from math import isclose, sqrt

import hivpy.column_names as col
from hivpy.common import date, rng, timedelta
from hivpy.population import Population


def test_hiv_testing_covid():

    # build population
    N = 100
    pop = Population(size=N, start_date=date(2010, 1, 1))
    # covid disruption is in place
    pop.hiv_testing.covid_disrup_affected = True
    pop.hiv_testing.testing_disrup_covid = True

    # evolve population
    pop.hiv_testing.update_hiv_testing(pop, timedelta(days=30))
    # check that nobody was tested
    assert sum(pop.get_variable(col.EVER_TESTED)) == 0


def test_hiv_testing_before_start():

    # build population
    N = 100
    pop = Population(size=N, start_date=date(2003, 1, 1))
    pop.data[col.ADC] = True
    # start date is before testing begins
    pop.hiv_testing.date_start_testing = 2003.5

    # evolve population
    pop.hiv_testing.update_hiv_testing(pop, timedelta(days=30))
    # check that nobody was tested
    assert sum(pop.get_variable(col.EVER_TESTED)) == 0


def test_hiv_symptomatic_testing():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2003, 1, 1))
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.data[col.ADC] = True
    pop.data[col.TB_INITIAL_INFECTION] = False
    pop.data[col.NON_TB_WHO3] = False
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.prob_test_who4 = 0.6
    pop.hiv_testing.prob_test_tb = 0.5
    pop.hiv_testing.prob_test_non_tb_who3 = 0.4
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.test_mark_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)
    # check that nobody was tested before date_start_testing
    assert sum(pop.get_variable(col.EVER_TESTED)) == 0

    pop.date = date(2008, 1, 1)
    # re-evolve population
    pop.hiv_testing.test_mark_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # get stats
    tested_population = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    prob_test = pop.hiv_testing.prob_test_who4
    mean = len(pop.data) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= tested_population <= mean + 3 * stdev

    # reset some columns
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.data[col.ADC] = False
    pop.data[col.TB_INITIAL_INFECTION] = True

    # re-evolve population
    pop.hiv_testing.test_mark_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # get stats
    tested_population = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    prob_test = pop.hiv_testing.prob_test_tb
    mean = len(pop.data) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= tested_population <= mean + 3 * stdev

    # reset some columns
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.data[col.TB_INITIAL_INFECTION] = False
    pop.data[col.NON_TB_WHO3] = True

    # re-evolve population
    pop.hiv_testing.test_mark_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # get stats
    tested_population = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    prob_test = pop.hiv_testing.prob_test_non_tb_who3
    mean = len(pop.data) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= tested_population <= mean + 3 * stdev


def test_non_hiv_symptomatic_testing():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2003, 1, 1))
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.data[col.HIV_DIAGNOSED] = False
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.prob_test_non_hiv_symptoms = 1
    pop.hiv_testing.prob_test_who4 = 0.6
    pop.hiv_testing.prob_test_non_tb_who3 = 0.4
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.test_mark_non_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)
    # check that nobody was tested before date_start_testing
    assert sum(pop.get_variable(col.EVER_TESTED)) == 0

    pop.date = date(2008, 1, 1)
    # re-evolve population
    pop.hiv_testing.test_mark_non_hiv_symptomatic(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # get stats
    tested_population = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    prob_test = (pop.hiv_testing.prob_test_non_tb_who3 + pop.hiv_testing.prob_test_who4)/2
    mean = len(pop.data) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= tested_population <= mean + 3 * stdev


def test_general_sex_worker_testing():

    # build population
    N = 100
    start_date = date(2010, 1, 1)
    pop = Population(size=N, start_date=start_date)
    pop.data[col.AGE] = 20
    pop.data[col.SEX_WORKER] = True
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.LAST_TEST_DATE] = start_date - timedelta(days=150)
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.eff_max_freq_testing = 0
    pop.hiv_testing.sw_test_regularly = True
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # check that nobody has just been tested
    assert (pop.get_variable(col.LAST_TEST_DATE) != pop.date).all()

    # move date forward and evolve again
    pop.date += timedelta(days=30)
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # check that everyone has just been tested
    assert (pop.get_variable(col.LAST_TEST_DATE) == pop.date).all()


def test_general_testing_conditions():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2008, 1, 1))
    pop.data[col.AGE] = 20
    pop.data[col.HARD_REACH] = False
    pop.data[col.EVER_TESTED] = True
    pop.data[col.LAST_TEST_DATE] = date(2009, 1, 1)
    pop.data[col.NP_LAST_TEST] = 1
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.eff_max_freq_testing = 1
    pop.hiv_testing.date_general_testing_plateau = 2015.5
    pop.hiv_testing.an_lin_incr_test = 0.8
    pop.hiv_testing.no_test_if_np0 = False
    pop.hiv_testing.sw_test_regularly = False
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # diagnose roughly 20% of the population with HIV
    r = rng.uniform(size=len(pop.data))
    diagnosed = r < 0.2
    pop.set_present_variable(col.HIV_DIAGNOSED, diagnosed)

    # evolve population
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)
    # check that nobody was tested before date_rate_testing_incr
    assert all(pop.get_variable(col.LAST_TEST_DATE) > pop.date)

    pop.date = date(2010, 1, 1)
    # re-evolve population
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # get people that were just tested
    tested_population = pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)])
    # check that no people just tested were already diagnosed with HIV
    assert (~pop.get_variable(col.HIV_DIAGNOSED, sub_pop=tested_population)).all()

    # get stats
    testing_population = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False)])
    prob_test = pop.hiv_testing.calc_prob_test(True, 1, 0)
    mean = len(testing_population) * prob_test
    stdev = sqrt(mean * (1 - prob_test))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= len(tested_population) <= mean + 3 * stdev


def test_first_time_testers():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(2010, 1, 1))
    pop.data[col.AGE] = 20
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.init_rate_first_test = 0.1
    pop.hiv_testing.date_general_testing_plateau = 2015.5
    pop.hiv_testing.an_lin_incr_test = 0.8
    pop.hiv_testing.no_test_if_np0 = False
    pop.hiv_testing.sw_test_regularly = False
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

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
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    # fixing some values
    pop.hiv_testing.date_start_testing = 2003.5
    pop.hiv_testing.date_rate_testing_incr = 2009
    pop.hiv_testing.eff_max_freq_testing = 1
    pop.hiv_testing.date_general_testing_plateau = 2015.5
    pop.hiv_testing.an_lin_incr_test = 0.8
    pop.hiv_testing.no_test_if_np0 = False
    pop.hiv_testing.sw_test_regularly = False
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # evolve population
    pop.hiv_testing.test_mark_general_pop(pop)
    marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
    pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

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
        pop.hiv_testing.date_rate_testing_incr = 2009
        pop.hiv_testing.eff_max_freq_testing = 1
        pop.hiv_testing.init_rate_first_test = 0.1
        pop.hiv_testing.date_general_testing_plateau = 2015.5
        pop.hiv_testing.an_lin_incr_test = 0.8
        pop.hiv_testing.no_test_if_np0 = False
        pop.hiv_testing.sw_test_regularly = False
        pop.hiv_testing.covid_disrup_affected = False
        pop.hiv_testing.testing_disrup_covid = False

        # evolve population
        pop.hiv_testing.update_hiv_testing(pop, timedelta(days=30))

        # get people that were just tested
        tested_population = pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)])
        # check that partner numbers have been reset
        assert sum(pop.get_variable(col.NP_LAST_TEST, sub_pop=tested_population)) == 0
        assert sum(pop.get_variable(col.NSTP_LAST_TEST, sub_pop=tested_population)) == 0


def test_max_frequency_testing():

    start_date = date(2010, 1, 1)
    max_freq_testing = [0, 1, 2]
    for index in max_freq_testing:

        # build population
        N = 100
        pop = Population(size=N, start_date=start_date)
        pop.data[col.AGE] = 20
        pop.data[col.HIV_DIAGNOSED] = False
        pop.data[col.HARD_REACH] = False
        pop.data[col.EVER_TESTED] = True
        pop.data[col.LAST_TEST_DATE] = start_date - timedelta(days=pop.hiv_testing.days_to_wait[index]-30)
        pop.data[col.NP_LAST_TEST] = 1
        pop.data[col.NSTP_LAST_TEST] = 1
        # fixing some values
        pop.hiv_testing.date_start_testing = 2003.5
        pop.hiv_testing.date_rate_testing_incr = 2009
        pop.hiv_testing.eff_max_freq_testing = index
        pop.hiv_testing.date_general_testing_plateau = 2015.5
        # guaranteed testing
        pop.hiv_testing.an_lin_incr_test = 1
        pop.hiv_testing.no_test_if_np0 = False
        pop.hiv_testing.sw_test_regularly = False
        pop.hiv_testing.covid_disrup_affected = False
        pop.hiv_testing.testing_disrup_covid = False

        # evolve population
        pop.hiv_testing.test_mark_general_pop(pop)
        marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
        pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

        # check that nobody has just been tested
        assert (pop.get_variable(col.LAST_TEST_DATE) != pop.date).all()

        # move date forward and evolve again
        pop.date += timedelta(days=30)
        pop.hiv_testing.test_mark_general_pop(pop)
        marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
        pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

        # check that everyone has just been tested
        assert (pop.get_variable(col.LAST_TEST_DATE) == pop.date).all()


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
