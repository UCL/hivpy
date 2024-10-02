import operator as op
from math import ceil, isclose, sqrt

import hivpy.column_names as col
from hivpy.common import SexType, date, diff_years, rng, timedelta
from hivpy.population import Population


def test_fertility():

    # build population
    N = 10000
    pop = Population(size=N, start_date=date(1990, 1, 1))

    # get stats
    no_female = sum(pop.data[col.SEX] == SexType.Female)
    no_infertile = sum(pop.data[col.LOW_FERTILITY])
    mean = no_female * (1 - pop.pregnancy.can_be_pregnant)
    stdev = sqrt(mean * pop.pregnancy.can_be_pregnant)
    # check infertility value is within 3 standard deviations
    assert mean - 3 * stdev <= no_infertile <= mean + 3 * stdev


def test_num_children():

    test_ages = [20, 30, 40, 50]
    for age in test_ages:

        # build artificial population
        N = 10000
        pop = Population(size=N, start_date=date(1990, 1, 1))
        pop.data[col.SEX] = SexType.Female
        pop.data[col.AGE] = age
        pop.data[col.LAST_PREGNANCY_DATE] = None
        pop.data[col.NUM_CHILDREN] = 0
        pop.pregnancy.init_fertility(pop)
        pop.pregnancy.init_num_children(pop)

        # check that anyone with child has a pregnancy date
        assert ((pop.data[col.NUM_CHILDREN] > 0) == pop.data[col.LAST_PREGNANCY_DATE].notnull()).all()

        # probability distribution and values for initialising child numbers
        prob_children_dist = pop.pregnancy.init_num_children_distributions[test_ages.index(age)].probs
        num_children_values = pop.pregnancy.init_num_children_distributions[test_ages.index(age)].data
        prob_dist_size = len(prob_children_dist)

        # get stats
        no_female = sum(~pop.data[col.LOW_FERTILITY])
        # loop through distribution
        for i in range(0, prob_dist_size):
            no_children = sum((pop.data[col.NUM_CHILDREN] == num_children_values[i])
                              & (~pop.data[col.LOW_FERTILITY]))
            prob_children = prob_children_dist[i]
            mean = no_female * prob_children
            stdev = sqrt(mean * (1 - prob_children))
            # check pregnancy value is within 3 standard deviations
            assert mean - 3 * stdev <= no_children <= mean + 3 * stdev


def test_ltp_preg():

    test_ages = [10, 20, 30, 40, 50, 60]
    with rng.set_temp_seed(42):
        for age in test_ages:

            # build artificial population
            N = 10000
            pop = Population(size=N, start_date=date(1990, 1, 1))
            pop.data[col.SEX] = SexType.Female
            pop.data[col.AGE] = age
            pop.data[col.NUM_PARTNERS] = 0
            pop.data[col.LONG_TERM_PARTNER] = True
            pop.data[col.LAST_PREGNANCY_DATE] = [date(1980, 1, 1), None]*(N//2)
            pop.data[col.NUM_CHILDREN] = 0
            pop.pregnancy.prob_pregnancy_base = 0.4
            pop.pregnancy.init_fertility(pop)
            pop.pregnancy.update_pregnancy(pop)

            # age restrictions on those who can get pregnant
            if age < 15:
                assert sum((pop.data[col.AGE] < 15) & pop.data[col.PREGNANT]) == 0
            elif age >= 55:
                assert sum((pop.data[col.AGE] >= 55) & pop.data[col.PREGNANT]) == 0

            else:
                # get stats
                no_active_female = sum((~pop.data[col.LOW_FERTILITY])
                                       & ((pop.data[col.NUM_PARTNERS] > 0)
                                       | pop.data[col.LONG_TERM_PARTNER]))
                no_pregnant = sum(pop.data[col.PREGNANT])
                prob_preg = pop.pregnancy.prob_pregnancy_base * pop.pregnancy.fertility_factor[test_ages.index(age)-1]
                mean = no_active_female * prob_preg
                stdev = sqrt(mean * (1 - prob_preg))
                # check pregnancy value is within 3 standard deviations
                max = mean + 3*stdev
                min = mean - 3*stdev
                assert min <= no_pregnant
                assert no_pregnant <= max
                # low fertility individuals don't become pregnant
                assert sum(pop.data[col.LOW_FERTILITY] & pop.data[col.PREGNANT]) == 0


def test_stp_preg():

    test_ages = [20, 30, 40, 50]
    test_stp_partners = [1, 3, 5]
    for age in test_ages:
        for stp in test_stp_partners:

            # build artificial population
            N = 10000
            pop = Population(size=N, start_date=date(1990, 1, 1))
            pop.data[col.SEX] = SexType.Female
            pop.data[col.AGE] = age
            pop.data[col.NUM_PARTNERS] = stp
            pop.data[col.LONG_TERM_PARTNER] = False
            pop.data[col.LAST_PREGNANCY_DATE] = None
            pop.data[col.NUM_CHILDREN] = 0
            pop.data[col.WANT_NO_CHILDREN] = False
            pop.pregnancy.prob_pregnancy_base = 0.4
            pop.pregnancy.init_fertility(pop)
            pop.pregnancy.update_pregnancy(pop)

            # get stats
            no_active_female = sum((~pop.data[col.LOW_FERTILITY])
                                   & ((pop.data[col.NUM_PARTNERS] > 0)
                                      | pop.data[col.LONG_TERM_PARTNER]))
            no_pregnant = sum(pop.data[col.PREGNANT])
            prob_preg = pop.pregnancy.calc_prob_preg(test_ages.index(age)+1, False, stp, False)
            mean = no_active_female * prob_preg
            stdev = sqrt(mean * (1 - prob_preg))
            # check pregnancy value is within 3 standard deviations
            assert mean - 3 * stdev <= no_pregnant <= mean + 3 * stdev


def test_childbirth():

    N = 100
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1

    # evolve population
    for _ in range(0, ceil(timedelta(days=270)/time_step)):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    # final advancement into childbirth
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that nobody is pregnant anymore
    assert (~pop.data[col.PREGNANT]).all()
    # check that everyone now has one child
    assert (pop.data[col.NUM_CHILDREN] == 1).all()

    # test the pregnancy pause period
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that there are still no pregnancies
    assert (~pop.data[col.PREGNANT]).all()

    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that everyone is pregnant again
    assert pop.data[col.PREGNANT].all()


def test_child_cap():

    N = 100
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    # cap children
    pop.pregnancy.max_children = 1
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1

    # evolve population
    # get through pregnancy, childbirth, and pregnancy pause period
    for _ in range(0, ceil(timedelta(days=450)/time_step)):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    # past pregnancy pause period
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that there are no pregnancies due to reaching child cap
    assert (~pop.data[col.PREGNANT]).all()


def test_want_no_children():

    # build artificial population
    N = 10000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    pop.pregnancy.prob_pregnancy_base = 0.4

    # make half the population not want children
    pop.data[col.WANT_NO_CHILDREN] = [True, False] * (N//2)

    # advance pregnancy
    pop.pregnancy.update_pregnancy(pop)

    # get stats
    want_no_children = sum(pop.data[col.WANT_NO_CHILDREN])
    no_pregnant = sum(pop.data[col.WANT_NO_CHILDREN] & pop.data[col.PREGNANT])
    prob_preg = pop.pregnancy.prob_pregnancy_base * 2 * 0.2
    mean = (N - want_no_children) * prob_preg
    stdev = sqrt(mean * (1 - prob_preg))
    # check pregnancy value is within 3 standard deviations
    assert mean - 3 * stdev <= no_pregnant <= mean + 3 * stdev

    # get more stats
    want_children = sum(~pop.data[col.WANT_NO_CHILDREN])
    no_pregnant = sum(~pop.data[col.WANT_NO_CHILDREN] & pop.data[col.PREGNANT])
    prob_preg = pop.pregnancy.prob_pregnancy_base * 2
    mean = want_children * prob_preg
    stdev = sqrt(mean * (1 - prob_preg))
    # check pregnancy value is within 3 standard deviations
    assert mean - 3 * stdev <= no_pregnant <= mean + 3 * stdev


def test_anc_and_pmtct():

    # build artificial population
    N = 1000
    pop = Population(size=N, start_date=date(2010, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 20
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    pop.data[col.WANT_NO_CHILDREN] = False
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1
    pop.pregnancy.rate_test_anc_inc = 1
    pop.pregnancy.date_pmtct = date(2004)
    pop.pregnancy.pmtct_inc_rate = 1

    # advance pregnancy
    pop.pregnancy.update_pregnancy(pop)

    # get stats
    no_anc = sum(pop.data[col.ANC])
    mean = len(pop.data) * pop.pregnancy.prob_anc
    stdev = sqrt(mean * (1 - pop.pregnancy.prob_anc))
    # check anc attendance value is within 3 standard deviations
    assert mean - 3 * stdev <= no_anc <= mean + 3 * stdev

    # get stats
    no_pmtct = sum(pop.data[col.PMTCT])
    prob_pmtct = min(diff_years(pop.date, pop.pregnancy.date_pmtct) * pop.pregnancy.pmtct_inc_rate, 0.975)
    mean = no_anc * prob_pmtct
    stdev = sqrt(mean * (1 - prob_pmtct))
    # check pmtct value is within 3 standard deviations
    assert mean - 3 * stdev <= no_pmtct <= mean + 3 * stdev


def test_anc_testing():

    N = 10000
    time_step = timedelta(days=30)

    # build artificial population
    pop = Population(size=N, start_date=date(2000, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.EVER_TESTED] = False
    pop.data[col.LAST_TEST_DATE] = None
    pop.hiv_testing.covid_disrup_affected = False
    pop.hiv_testing.testing_disrup_covid = False

    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1
    # maximise anc chances
    pop.pregnancy.prob_anc = 1
    pop.pregnancy.rate_test_anc_inc = 1

    # get test outcomes
    def update_anc_testing_outcomes(pop, time_step):
        pop.hiv_testing.test_mark_anc(pop, time_step)
        marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
        pop.hiv_testing.apply_test_outcomes_to_sub_pop(pop, marked_population)

    # advance pregnancy to start of second trimester
    pop.pregnancy.update_pregnancy(pop)
    for _ in range(0, ceil(timedelta(days=90)/time_step)):
        pop.date += time_step
        pop.pregnancy.update_pregnancy(pop)
    update_anc_testing_outcomes(pop, time_step)

    # store people not in anc
    not_inc_anc = pop.get_sub_pop([(col.ANC, op.eq, False)])
    # store number of people in anc
    in_anc = len(pop.get_sub_pop([(col.ANC, op.eq, True)]))

    # get stats
    no_tested = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    test_prob = pop.hiv_testing.prob_anc_test_trim1
    mean = in_anc * test_prob
    stdev = sqrt(mean * (1 - test_prob))
    # check the correct proportion of the population has been tested
    assert mean - 3 * stdev <= no_tested <= mean + 3 * stdev

    # advance pregnancy to start of third trimester
    for _ in range(0, ceil(timedelta(days=90)/time_step)):
        pop.date += time_step
        pop.pregnancy.update_pregnancy(pop)
    update_anc_testing_outcomes(pop, time_step)

    # get stats
    no_tested = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    test_prob = pop.hiv_testing.prob_anc_test_trim2
    mean = in_anc * test_prob
    stdev = sqrt(mean * (1 - test_prob))
    # check the correct proportion of the population has been tested
    assert mean - 3 * stdev <= no_tested <= mean + 3 * stdev

    # final advancement into childbirth
    for _ in range(0, ceil(timedelta(days=90)/time_step)):
        pop.date += time_step
        pop.pregnancy.update_pregnancy(pop)
    update_anc_testing_outcomes(pop, time_step)
    pop.pregnancy.reset_anc_at_birth(pop)

    # get stats
    no_tested = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    test_prob = pop.hiv_testing.prob_anc_test_trim3
    mean = in_anc * test_prob
    stdev = sqrt(mean * (1 - test_prob))
    # check the correct proportion of the population has been tested
    assert mean - 3 * stdev <= no_tested <= mean + 3 * stdev

    # advance to post-delivery
    pop.date += time_step
    pop.pregnancy.update_pregnancy(pop)
    update_anc_testing_outcomes(pop, time_step)

    # get stats
    no_tested = len(pop.get_sub_pop([(col.LAST_TEST_DATE, op.eq, pop.date)]))
    test_prob = pop.hiv_testing.prob_anc_test_trim3 * pop.hiv_testing.prob_test_postdel
    mean = in_anc * test_prob
    stdev = sqrt(mean * (1 - test_prob))
    # check the correct proportion of the population has been tested
    assert mean - 3 * stdev <= no_tested <= mean + 3 * stdev

    # check that nobody not in ANC has been tested
    assert len(pop.get_sub_pop_intersection(
        not_inc_anc, pop.get_sub_pop([(col.EVER_TESTED, op.eq, True)]))) == 0


def test_infected_births():

    N = 100
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 0
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_PREGNANCY_DATE] = None
    pop.data[col.NUM_CHILDREN] = 0
    pop.data[col.NUM_HIV_CHILDREN] = 0
    pop.data[col.VIRAL_LOAD_GROUP] = 5
    pop.data[col.HIV_STATUS] = True
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1

    # evolve population
    for _ in range(0, ceil(timedelta(days=270)/time_step)):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    # final advancement into childbirth
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step

    # get stats
    no_infected_births = sum(pop.data[col.NUM_HIV_CHILDREN])
    mean = len(pop.data) * pop.pregnancy.prob_birth_with_infected_child
    stdev = sqrt(mean * (1 - pop.pregnancy.prob_birth_with_infected_child))
    # check infected birth value is within 3 standard deviations
    assert mean - 3 * stdev <= no_infected_births <= mean + 3 * stdev


def test_calc_prob_preg():

    # setup
    pop = Population(size=1, start_date=date(1990, 1, 1))
    # fixing some probabilities
    pop.pregnancy.can_be_pregnant = 0.95
    pop.pregnancy.fertility_factor = [2.0, 1.5, 1, 0.1]
    pop.pregnancy.prob_pregnancy_base = 0.1
    pop.pregnancy.stp_transmission_factor = 0.5

    # check basic ltp case
    # (0.1 * 2) = 0.2
    assert isclose(pop.pregnancy.calc_prob_preg(1, True, 0, False), 0.2)
    # (0.1 * 1.5) = 0.15
    assert isclose(pop.pregnancy.calc_prob_preg(2, True, 0, False), 0.15)
    # (0.1 * 1) = 0.1
    assert isclose(pop.pregnancy.calc_prob_preg(3, True, 0, False), 0.1)
    # (0.1 * 0.1) = 0.01
    assert isclose(pop.pregnancy.calc_prob_preg(4, True, 0, False), 0.01)

    # check basic stp case
    # (0.1 * 2 * 0.5) = 0.1
    assert isclose(pop.pregnancy.calc_prob_preg(1, False, 1, False), 0.1)
    # (0.1 * 1.5 * 0.5) = 0.075
    assert isclose(pop.pregnancy.calc_prob_preg(2, False, 1, False), 0.075)
    # (0.1 * 1 * 0.5) = 0.05
    assert isclose(pop.pregnancy.calc_prob_preg(3, False, 1, False), 0.05)
    # (0.1 * 0.1 * 0.5) = 0.005
    assert isclose(pop.pregnancy.calc_prob_preg(4, False, 1, False), 0.005)

    # check more stp partners case
    # (1 - (1 - 0.1 * 2 * 0.5)^2) = 0.19
    assert isclose(pop.pregnancy.calc_prob_preg(1, False, 2, False), 0.19)
    # (1 - (1 - 0.1 * 1.5 * 0.5)^2) = 0.144375
    assert isclose(pop.pregnancy.calc_prob_preg(2, False, 2, False), 0.144375)
    # (1 - (1 - 0.1 * 1 * 0.5)^2) = 0.0975
    assert isclose(pop.pregnancy.calc_prob_preg(3, False, 2, False), 0.0975)
    # (1 - (1 - 0.1 * 0.1 * 0.5)^2) = 0.009975
    assert isclose(pop.pregnancy.calc_prob_preg(4, False, 2, False), 0.009975)

    # check both ltp and stp case
    # (1 - (1 - 0.1 * 2) * (1 - 0.1 * 2 * 0.5)) = 0.28
    assert isclose(pop.pregnancy.calc_prob_preg(1, True, 1, False), 0.28)
    # (1 - (1 - 0.1 * 1.5) * (1 - 0.1 * 1.5 * 0.5)) = 0.21375
    assert isclose(pop.pregnancy.calc_prob_preg(2, True, 1, False), 0.21375)
    # (1 - (1 - 0.1 * 1) * (1 - 0.1 * 1 * 0.5)) = 0.145
    assert isclose(pop.pregnancy.calc_prob_preg(3, True, 1, False), 0.145)
    # (1 - (1 - 0.1 * 0.1) * (1 - 0.1 * 0.1 * 0.5)) = 0.01495
    assert isclose(pop.pregnancy.calc_prob_preg(4, True, 1, False), 0.01495)

    # check want no children (with ltp)
    # (0.1 * 2 * 0.2) = 0.2
    assert isclose(pop.pregnancy.calc_prob_preg(1, True, 0, True), 0.04)
    # (0.1 * 1.5 * 0.2) = 0.02
    assert isclose(pop.pregnancy.calc_prob_preg(2, True, 0, True), 0.03)
    # (0.1 * 1 * 0.2) = 0.02
    assert isclose(pop.pregnancy.calc_prob_preg(3, True, 0, True), 0.02)
    # (0.1 * 0.1 * 0.2) = 0.002
    assert isclose(pop.pregnancy.calc_prob_preg(4, True, 0, True), 0.002)
