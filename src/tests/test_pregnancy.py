from datetime import date, timedelta
from math import sqrt

import hivpy.column_names as col
from hivpy.common import SexType
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
            assert sum((pop.data[col.AGE] < 15) & pop.data[col.PREGNANCY_DATE].notnull()) == 0
        elif age >= 55:
            assert sum((pop.data[col.AGE] >= 55) & pop.data[col.PREGNANCY_DATE].notnull()) == 0

        else:
            # get stats
            no_active_female = sum((pop.data[col.SEX] == SexType.Female)
                                   & (~pop.data[col.LOW_FERTILITY])
                                   & ((pop.data[col.NUM_PARTNERS] > 0)
                                      | pop.data[col.LONG_TERM_PARTNER]))
            no_pregnant = sum(pop.data[col.PREGNANCY_DATE].notnull())
            prob_preg = pop.pregnancy.prob_pregnancy_base * pop.pregnancy.fold_preg[test_ages.index(age)-1]
            mean = no_active_female * prob_preg
            stdev = sqrt(mean * (1 - prob_preg))
            # check pregnancy value is within 3 standard deviations
            assert mean - 3 * stdev <= no_pregnant <= mean + 3 * stdev
            # low fertility individuals don't become pregnant
            assert sum(pop.data[col.LOW_FERTILITY] & pop.data[col.PREGNANCY_DATE].notnull()) == 0


def test_childbirth():

    N = 100000
    start_date = date(1990, 1, 1)
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=start_date)
    pop.data[col.SEX] = SexType.Female
    pop.data[col.AGE] = 18
    pop.data[col.NUM_PARTNERS] = 1
    pop.pregnancy.init_fertility(pop)

    # evolve population
    # TODO: may need to account for including time step period when updating pregnancy
    for i in range(3):
        # advance pregnancy
        pop.pregnancy.update_pregnancy(pop)
        pop.date += time_step

    pregnant_at_start = pop.data.index[pop.data[col.PREGNANCY_DATE] == start_date]
    # final advancement into childbirth
    pop.pregnancy.update_pregnancy(pop)
    # check that all who got pregnant during the start date are now no longer pregnant
    assert pop.data.loc[pregnant_at_start, col.PREGNANCY_DATE].isnull().all()
