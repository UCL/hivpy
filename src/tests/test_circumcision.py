from datetime import date
from time import perf_counter

import numpy as np

import hivpy.column_names as col
from hivpy.common import SexType
from hivpy.population import Population


def test_birth_circumcision():

    runtime = []
    percent_deviation = []
    start = perf_counter()

    for i in range(0, 100):
        # build and time population
        N = 100000
        pop = Population(size=N, start_date=date(1989, 1, 1))
        t_start = perf_counter()
        pop.circumcision.init_birth_circumcision(pop.data)
        t_stop = perf_counter()
        runtime.append(t_stop - t_start)

        no_circumcised = len(pop.data[pop.data[col.CIRCUMCISED] == 1])
        no_male = len(pop.data[pop.data[col.SEX] == SexType.Male])
        percent_deviation.append(abs(pop.circumcision.prob_birth_circ - no_circumcised/no_male))

        # no female is marked as circumcised
        assert len(pop.data[(pop.data[col.SEX] == SexType.Female)
                   & (pop.data[col.CIRCUMCISED] == 1)]) == 0

    stop = perf_counter()
    print("\nAverage elapsed time:", np.average(runtime) * 1000, "ms")
    print("Total elapsed test time:", stop - start, "s")

    # average deviation is less than half a percent
    assert np.average(percent_deviation) < 0.005
    # print("Average deviation:", np.average(percent_deviation) * 100, "%")
