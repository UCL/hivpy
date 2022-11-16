from datetime import date, timedelta
from math import sqrt
from time import perf_counter

import numpy as np

import hivpy.column_names as col
from hivpy.common import SexType
from hivpy.population import Population


# Find average birth circumcicion time: init all at once vs
# partial init followed by average time of updates per time step
def test_birth_circumcision_timing():

    runtime_all_init = []
    runtime_evolve_init = []
    runtime_evolve = []

    N = 100000
    start_date = date(2000, 1, 1)
    stop_date = date(2010, 1, 1)
    time_step = timedelta(days=90)

    iter_no = 100
    start = perf_counter()

    # iterate to find average
    for i in range(0, iter_no):

        # build population
        pop = Population(size=N, start_date=start_date)
        pop.data[col.CIRCUMCISED] = False
        pop.data[col.CIRCUMCISION_DATE] = None
        pop.data[col.VMMC] = False

        # time init all at once
        t_start = perf_counter()
        pop.circumcision.init_birth_circumcision_all(pop.data, pop.date)
        t_stop = perf_counter()
        runtime_all_init.append(t_stop - t_start)

        pop.data[col.CIRCUMCISED] = False
        pop.data[col.CIRCUMCISION_DATE] = None
        pop.data[col.VMMC] = False

        # time partial init
        t_start = perf_counter()
        pop.circumcision.init_birth_circumcision_born(pop.data, pop.date)
        t_stop = perf_counter()
        runtime_evolve_init.append(t_stop - t_start)

        # time update
        t_start = perf_counter()
        while pop.date <= stop_date:
            pop.data.age += time_step.days / 365
            pop.circumcision.update_birth_circumcision(pop.data, time_step, pop.date)
            pop.date += time_step
        t_stop = perf_counter()
        runtime_evolve.append(t_stop - t_start)

    stop = perf_counter()
    # get stats
    mean_rai = np.average(runtime_all_init) * 1000
    mean_rei = np.average(runtime_evolve_init) * 1000
    mean_re = np.average(runtime_evolve) * 1000
    tstep_no = (stop_date - start_date) / time_step

    print("\n>> Data from", iter_no, "iterations:")

    print("\nAverage all init elapsed time:", round(mean_rai, 3), "+/-",
          round(np.std(runtime_all_init) * 1000, 3), "ms")
    print("Average evolve init elapsed time:", round(mean_rei, 3), "+/-",
          round(np.std(runtime_evolve_init) * 1000, 3), "ms")

    print("\nAverage evolve elapsed time:", round(mean_re, 3), "ms (for",
          stop_date.year - start_date.year, "years)")
    print("Average evolve time per timestep:", round(mean_re/tstep_no, 3), "+/-",
          round((np.std(runtime_evolve) * 1000) / tstep_no, 3), "ms")
    print("Average evolve init + evolve elapsed time:", round(mean_rei + mean_re, 3), "ms")

    print("\nTotal elapsed test time:", round(stop - start, 3), "s")


def test_birth_circumcision_atonce():

    # build population
    N = 100000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    pop.data[col.VMMC] = False
    pop.circumcision.init_birth_circumcision_all(pop.data, pop.date)

    # get stats
    no_circumcised = len(pop.data[pop.data[col.CIRCUMCISED] == 1])
    no_male = len(pop.data[pop.data[col.SEX] == SexType.Male])
    mean = no_male * pop.circumcision.prob_birth_circ
    stdev = sqrt(mean * (1 - pop.circumcision.prob_birth_circ))
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)
    # check that no circumcised people have undergone VMMC
    assert len(pop.data[pop.data[col.CIRCUMCISED] & pop.data[col.VMMC]]) == 0


def test_birth_circumcision_stages():

    N = 100000
    start_date = date(2000, 1, 1)
    stop_date = date(2020, 1, 1)
    time_step = timedelta(days=90)

    # build population
    pop = Population(size=N, start_date=start_date)
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    pop.data[col.VMMC] = False
    pop.circumcision.init_birth_circumcision_born(pop.data, pop.date)

    # get stats
    no_circumcised = len(pop.data[pop.data[col.CIRCUMCISED]])
    male_population = pop.data[(pop.data[col.SEX] == SexType.Male)
                               & (pop.data[col.AGE] > 0.25)]
    mean = len(male_population) * pop.circumcision.prob_birth_circ
    stdev = sqrt(mean * (1 - pop.circumcision.prob_birth_circ))
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)

    # evolve population
    while pop.date <= stop_date:
        # advance ages and birth circumcision
        pop.data.age += time_step.days / 365
        pop.circumcision.update_birth_circumcision(pop.data, time_step, pop.date)
        pop.date += time_step

    # get stats
    no_circumcised = len(pop.data[pop.data[col.CIRCUMCISED]])
    male_population = pop.data[(pop.data[col.SEX] == SexType.Male)
                               & (pop.data[col.AGE] > 0.25)]
    mean = len(male_population) * pop.circumcision.prob_birth_circ
    stdev = sqrt(mean * (1 - pop.circumcision.prob_birth_circ))
    # basic checks
    general_circumcision_checks(mean, stdev, no_circumcised, pop.data)
    # check that no circumcised people have undergone VMMC
    assert len(pop.data[pop.data[col.CIRCUMCISED] & pop.data[col.VMMC]]) == 0


def general_circumcision_checks(mean, stdev, no_circumcised, data):

    # no female is marked as circumcised
    assert len(data[(data[col.SEX] == SexType.Female)
                    & data[col.CIRCUMCISED]]) == 0

    # no uncircumcised people have circumcision dates
    assert len(data[~data[col.CIRCUMCISED]
                    & data[col.CIRCUMCISION_DATE].notnull()]) == 0

    # all circumcised people have circumcision dates
    assert len(data[data[col.CIRCUMCISED]]) == \
           len(data[data[col.CIRCUMCISED] & data[col.CIRCUMCISION_DATE].notnull()])

    # check circumcised value is within 3 standard deviations
    assert mean - 3 * stdev < no_circumcised < mean + 3 * stdev


def test_vmmc():

    N = 100000
    start_date = date(2008, 12, 1)
    stop_date = date(2028, 12, 1)
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=start_date)
    pop.data[col.SEX] = SexType.Male
    pop.data[col.AGE] = 18
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    pop.data[col.VMMC] = False

    # evolve population
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop, 0)
    # check no VMMC occurs until after mc_int
    assert len(pop.data[pop.data[col.VMMC]]) == 0

    # change date to appropriate year
    pop.date += time_step
    # evolve population
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop, 0)

    # get stats
    no_vmmc = len(pop.data[pop.data[col.VMMC]])
    male_population = pop.data[(pop.data[col.SEX] == SexType.Male)
                               & (pop.data[col.AGE] >= 10)
                               & (pop.data[col.AGE] < 50)]
    prob_circ = (pop.date.year - pop.circumcision.mc_int) * pop.circumcision.circ_inc_rate
    mean = len(male_population) * prob_circ
    stdev = sqrt(mean * (1 - prob_circ))
    # basic checks
    general_circumcision_checks(mean, stdev, no_vmmc, pop.data)

    # evolve population
    while pop.date <= stop_date:
        circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male) & pop.data[col.CIRCUMCISED]]
        # advance ages and vmmc
        pop.data.age += time_step.days / 365
        pop.circumcision.update_vmmc(pop, 0)
        pop.date += time_step
        # check circumcisied people remain circumcised each step
        new_circ_males = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                        & pop.data[col.CIRCUMCISED]]
        assert circ_males.isin(new_circ_males).all()


# This "test" replicates an uncommon issue with
# population filtering in VMMC [currently fixed].
def test_uncommon_vmmc_bug():

    N = 5
    start_date = date(2009, 1, 1)
    time_step = timedelta(days=90)

    # build artificial population
    pop = Population(size=N, start_date=start_date)
    pop.data[col.SEX] = SexType.Male
    pop.data[col.AGE] = 12
    pop.data[col.CIRCUMCISED] = False
    pop.data[col.CIRCUMCISION_DATE] = None
    pop.data[col.VMMC] = False
    # set circumcision rate to 100%
    pop.circumcision.circ_inc_rate = 1

    print("\nNew Artificial Population:\n",
          pop.data[[col.SEX, col.AGE, col.CIRCUMCISED,
                   col.CIRCUMCISION_DATE, col.VMMC]])

    print("\n===== EVOLVE STEP #1 =====")
    print("\nNormal step. Behaviour as expected.\nCircumcises all males.")
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop, 1)
    pop.date += time_step
    print("\nPopulation After Evolve #1:\n",
          pop.data[[col.SEX, col.AGE, col.CIRCUMCISED,
                   col.CIRCUMCISION_DATE, col.VMMC]])

    print("\n===== EVOLVE STEP #2 =====")
    print("\nNormal step. No uncircumcised males are discovered.\nNobody is circumcised.")
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop, 1)
    pop.date += time_step
    print("\nPopulation After Evolve #2:\n",
          pop.data[[col.SEX, col.AGE, col.CIRCUMCISED,
                   col.CIRCUMCISION_DATE, col.VMMC]])

    print("\n===== EVOLVE STEP #3 =====")
    print("\nProblem step. Circumcised males are selected for the uncircumcised population.\
          \nEveryone is re-circumcised (though circumcision dates are not updated).")
    pop.data.age += time_step.days / 365
    pop.circumcision.update_vmmc(pop, 1)
    pop.date += time_step
    print("\nPopulation After Evolve #3:\n",
          pop.data[[col.SEX, col.AGE, col.CIRCUMCISED,
                   col.CIRCUMCISION_DATE, col.VMMC]])