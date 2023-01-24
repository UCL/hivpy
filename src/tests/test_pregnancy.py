from datetime import date, timedelta
from math import isclose, sqrt

import hivpy.column_names as col
from hivpy.common import SexType, rng
from hivpy.population import Population


def test_fertility():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(1990, 1, 1))

    # get stats
    no_female = sum(pop.data[col.SEX] == SexType.Female)
    no_infertile = sum(pop.data[col.LOW_FERTILITY])
    mean = no_female * (1 - pop.pregnancy.can_be_pregnant)
    stdev = sqrt(mean * pop.pregnancy.can_be_pregnant)
    # check infertility value is within 3 standard deviations
    assert mean - 3 * stdev <= no_infertile <= mean + 3 * stdev


def test_new_preg():

    test_ages = [10, 20, 30, 40, 50, 60]
    for age in test_ages:

        # build artificial population
        N = 100000
        pop = Population(size=N, start_date=date(1990, 1, 1))
        pop.data[col.SEX] = SexType.Female
        pop.data[col.AGE] = age
        pop.data[col.NUM_PARTNERS] = 1
        pop.pregnancy.prob_pregnancy_base = 0.1
        pop.pregnancy.init_fertility(pop)
        pop.pregnancy.update_pregnancy(pop)

        # age restrictions on those who can get pregnant
        if age < 15:
            assert sum((pop.data[col.AGE] < 15) & pop.data[col.PREGNANT]) == 0
        elif age >= 55:
            assert sum((pop.data[col.AGE] >= 55) & pop.data[col.PREGNANT]) == 0

        else:
            # get stats
            no_active_female = sum((pop.data[col.SEX] == SexType.Female)
                                   & (~pop.data[col.LOW_FERTILITY])
                                   & ((pop.data[col.NUM_PARTNERS] > 0)
                                      | pop.data[col.LONG_TERM_PARTNER]))
            no_pregnant = sum(pop.data[col.PREGNANT])
            prob_preg = pop.pregnancy.prob_pregnancy_base * pop.pregnancy.fold_preg[test_ages.index(age)-1]
            mean = no_active_female * prob_preg
            stdev = sqrt(mean * (1 - prob_preg))
            # check pregnancy value is within 3 standard deviations
            assert mean - 3 * stdev <= no_pregnant <= mean + 3 * stdev
            # low fertility individuals don't become pregnant
            assert sum(pop.data[col.LOW_FERTILITY] & pop.data[col.PREGNANT]) == 0


def test_childbirth():

    N = 100
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 1
    pop.data[col.NUM_CHILDREN] = 0
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1

    # evolve population
    # TODO: may need to account for including time step period when updating pregnancy
    for i in range(3):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    # final advancement into childbirth
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that nobody is pregnant anymore
    assert ~pop.data[col.PREGNANT].all()
    # check that everyone now has one child
    assert (pop.data[col.NUM_CHILDREN] == 1).all()

    # test the pregnancy pause period
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that there are still no pregnancies
    assert ~pop.data[col.PREGNANT].all()

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
    pop.data[col.NUM_PARTNERS] = 1
    pop.data[col.NUM_CHILDREN] = 0
    # cap children
    pop.pregnancy.max_children = 1
    # guaranteed pregnancy
    pop.pregnancy.prob_pregnancy_base = 1

    # evolve population
    # TODO: may need to account for including time step period when updating pregnancy
    # get through pregnancy, childbirth, and pregnancy pause period
    for i in range(5):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    # past pregnancy pause period
    pop.pregnancy.update_pregnancy(pop)
    pop.date += time_step
    # check that there are no pregnancies due to reaching child cap
    assert ~pop.data[col.PREGNANT].all()


def test_want_no_children():

    # build artificial population
    N = 100000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.LOW_FERTILITY] = False
    pop.data[col.NUM_PARTNERS] = 1
    pop.data[col.NUM_CHILDREN] = 0
    pop.pregnancy.prob_pregnancy_base = 0.1

    # make roughly half the population not want children
    r = rng.uniform(size=len(pop.data))
    want_no_children_outcomes = r < 0.5
    pop.data[col.WANT_NO_CHILDREN] = want_no_children_outcomes

    # advance pregnancy
    pop.pregnancy.update_pregnancy(pop)

    # get stats
    want_no_children = sum(pop.data[col.WANT_NO_CHILDREN])
    no_pregnant = sum(pop.data[col.WANT_NO_CHILDREN] & pop.data[col.PREGNANT])
    prob_preg = pop.pregnancy.prob_pregnancy_base * 2 * 0.2
    mean = want_no_children * prob_preg
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


def test_calc_prob_preg():

    # setup
    pop = Population(size=1, start_date=date(1990, 1, 1))
    # fixing some probabilities
    pop.pregnancy.can_be_pregnant = 0.95
    pop.pregnancy.fold_preg = [2.0, 1.5, 1, 0.1]
    pop.pregnancy.prob_pregnancy_base = 0.1

    # check basic case
    # (0.1 * 2) = 0.2
    assert isclose(pop.pregnancy.calc_prob_preg(1, False), 0.2)
    # (0.1 * 1.5) = 0.15
    assert isclose(pop.pregnancy.calc_prob_preg(2, False), 0.15)
    # (0.1 * 1) = 0.1
    assert isclose(pop.pregnancy.calc_prob_preg(3, False), 0.1)
    # (0.1 * 0.1) = 0.01
    assert isclose(pop.pregnancy.calc_prob_preg(4, False), 0.01)

    # check want no children
    # (0.1 * 2 * 0.2) = 0.2
    assert isclose(pop.pregnancy.calc_prob_preg(1, True), 0.04)
    # (0.1 * 1.5 * 0.2) = 0.02
    assert isclose(pop.pregnancy.calc_prob_preg(2, True), 0.03)
    # (0.1 * 1 * 0.2) = 0.02
    assert isclose(pop.pregnancy.calc_prob_preg(3, True), 0.02)
    # (0.1 * 0.1 * 0.2) = 0.002
    assert isclose(pop.pregnancy.calc_prob_preg(4, True), 0.002)
