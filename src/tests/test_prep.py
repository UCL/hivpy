import operator as op
from math import sqrt

import hivpy.column_names as col
from hivpy.common import SexType, date
from hivpy.population import Population


def test_at_risk_pop():
    N = 100
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # at_risk = num_stp >= 1 OR (ltp_diag AND not ltp_on_art)
    pop.data[col.NUM_PARTNERS] = [0, 1] * (N // 2)
    pop.data[col.LTP_HIV_DIAGNOSED] = [True, False] * (N // 2)
    pop.data[col.LTP_ON_ART] = False
    # everyone fulfills one of the conditions for being at risk
    assert len(pop.prep.get_at_risk_pop(pop)) == N


def test_risk_informed_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # risk_informed = ltp AND not ltp_on_art AND r < prob_risk_informed_prep
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = False
    pop.prep.prob_risk_informed_prep = 0.1

    # get stats
    no_risk_informed = len(pop.prep.get_risk_informed_pop(pop, pop.prep.prob_risk_informed_prep))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting ~10% of the population to be risk informed
    assert mean - 3 * stdev <= no_risk_informed <= mean + 3 * stdev


def test_suspect_risk_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # suspect_risk = ltp AND not ltp_on_art AND ltp_infected AND r < prob_suspect_risk_prep
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = True
    pop.prep.prob_suspect_risk_prep = 0.5

    # get stats
    no_suspect_risk = len(pop.prep.get_suspect_risk_pop(pop))
    mean = N * pop.prep.prob_suspect_risk_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_suspect_risk_prep))
    # expecting ~50% of the population to suspect they are at risk
    assert mean - 3 * stdev <= no_suspect_risk <= mean + 3 * stdev


def test_prep_eligibility_women_only():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.SEX] = SexType.Female
    pop.data[col.SEX_WORKER] = [False, True] * (N // 2)
    pop.data[col.AGE] = [20, 30] * (N // 2)
    pop.data[col.NUM_PARTNERS] = 1  # everyone is at risk

    # STRATEGY 1-3

    # fsw_agyw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 1
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # everyone fulfills the necessary eligibility conditions
    assert eligible == N

    pop.data[col.PREP_ELIGIBLE] = False
    # fsw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 2
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # only half of the population are sex workers
    assert eligible == N/2

    pop.data[col.PREP_ELIGIBLE] = False
    # agyw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 3
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # only half of the population are 15-25
    assert eligible == N/2

    # STRATEGY 6 & 10

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.NUM_PARTNERS] = 0  # nobody is inherently at risk
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = False
    pop.prep.prob_risk_informed_prep = 0.3
    pop.prep.prob_greater_risk_informed_prep = 0.6
    # gen_fem AND (at_risk OR (gen_age AND (risk_informed OR suspect_risk)))
    pop.prep.prep_strategy = 6  # same as 10 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    # gen_fem AND (at_risk OR (gen_age AND (risk_informed OR suspect_risk)))
    pop.prep.prep_strategy = 10  # same as 6 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    # STRATEGY 7 & 11

    pop.data[col.PREP_ELIGIBLE] = False
    # FIXME: include active people + add variety to stp activity timing
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 7  # same as 11 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 11  # same as 7 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev
