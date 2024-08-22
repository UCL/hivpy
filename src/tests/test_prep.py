import operator as op
from math import sqrt

import pytest

import hivpy.column_names as col
from hivpy.common import SexType, date, timedelta
from hivpy.population import Population


def test_prep_willingness():
    N = 100
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.data[col.VIRAL_LOAD] = 10000
    # make all prep types available
    pop.prep.date_prep_intro = [date(2000), date(2000), date(2000), date(2000)]
    # adjust chances of higher preference
    pop.prep.prep_oral_pref_beta = 3
    pop.prep.prep_inj_pref_beta = 3.3
    pop.prep.prep_vr_pref_beta = 2.9
    # vl prevalence accounted for
    pop.prep.vl_prevalence_affects_prep = True
    pop.prep.vl_prevalence_prep_threshold = 0.5

    pop.prep.prep_willingness(pop)

    # some willingness has been established
    assert sum(pop.data[col.PREP_ANY_WILLING]) > 0

    # reset willingness with low viral load prevalence
    pop.data[col.VIRAL_LOAD] = 100
    pop.prep.prep_willingness(pop)

    # no willingness remains
    assert sum(pop.data[col.PREP_ANY_WILLING]) == 0


def test_at_risk_pop():
    N = 100
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # at_risk = num_stp >= 1 OR (ltp_diag AND not ltp_on_art)
    pop.data[col.NUM_PARTNERS] = [0, 1] * (N // 2)
    pop.data[col.LTP_HIV_DIAGNOSED] = [True, False, False, False] * (N // 4)
    pop.data[col.LTP_ON_ART] = False
    # 3/4 of people fulfill one of the conditions for being at risk
    assert len(pop.prep.get_at_risk_pop(pop)) == N//4 * 3


def test_risk_informed_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # risk_informed = ltp AND not ltp_on_art AND r < prob_risk_informed_prep
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = False
    pop.prep.reroll_r_prep(pop)
    pop.prep.prob_risk_informed_prep = 0.1

    # get stats
    no_risk_informed = len(pop.prep.get_risk_informed_pop(pop, pop.prep.prob_risk_informed_prep))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting ~10% of the population to be risk informed
    assert mean - 3 * stdev <= no_risk_informed <= mean + 3 * stdev

    pop.data[col.LONG_TERM_PARTNER] = False
    no_risk_informed = len(pop.prep.get_risk_informed_pop(pop, pop.prep.prob_risk_informed_prep))
    assert no_risk_informed == 0


def test_suspect_risk_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # suspect_risk = ltp AND not ltp_on_art AND ltp_infected AND r < prob_suspect_risk_prep
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = [True, False] * (N//2)
    pop.data[col.LTP_HIV_STATUS] = True
    pop.prep.reroll_r_prep(pop)
    pop.prep.prob_suspect_risk_prep = 0.5

    # get stats
    no_suspect_risk = len(pop.prep.get_suspect_risk_pop(pop))
    mean = (N/2) * pop.prep.prob_suspect_risk_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_suspect_risk_prep))
    # expecting ~50% of the population with partners NOT on ART to suspect they are at risk
    assert mean - 3 * stdev <= no_suspect_risk <= mean + 3 * stdev


@pytest.mark.parametrize("prep_strategy", [i for i in range(1, 17)])
def test_prep_ineligible(prep_strategy):
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.data[col.HIV_DIAGNOSED] = [True, False] * (N // 2)
    pop.prep.prep_strategy = prep_strategy
    pop.prep.prep_eligibility(pop)

    # check that nobody diagnosed with HIV is eligible
    assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                (col.HIV_DIAGNOSED, op.eq, True)])) == 0

    # check that nobody under 15 or over 50 is eligible
    assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                (col.AGE, op.lt, 15),
                                (col.AGE, op.ge, 50)])) == 0

    # check that no men are eligible in women only strategies
    if prep_strategy not in [4, 5, 8, 9, 12, 14, 15]:
        assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                    (col.SEX, op.eq, SexType.Male)])) == 0


def test_prep_eligibility_continuity():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.data[col.AGE] = 30
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = [True, False] * (N // 2)
    pop.data[col.R_PREP] = 1.0

    pop.prep.prep_strategy = 9
    pop.prep.prep_eligibility(pop)
    # check all r_prep values were rerolled (nobody started out eligible)
    assert all(pop.data[col.R_PREP] < 1.0)

    # get initial eligible pop
    init_eligible = pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])
    # set eligibility again
    pop.prep.prep_eligibility(pop)
    new_eligible = pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])
    # check all previously eligible people are still eligible
    assert set(init_eligible).issubset(new_eligible)
    # check there are now more eligible people (demonstrates that r_prep is recalculated for those ineligible)
    assert len(new_eligible) > len(init_eligible)

    pop.data[col.AGE] = 50
    pop.prep.prep_eligibility(pop)
    # check that everyone has aged out of eligibility
    assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])) == 0


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
    # inflate probabilities to make test more sensitive with small test population
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
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 7  # same as 11 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = False
    pop.data[col.LAST_STP_DATE] = [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10)
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_STP_DATE] = None
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 11  # same as 7 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = False
    pop.data[col.LAST_STP_DATE] = [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10)
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 13

    pop.data[col.PREP_ELIGIBLE] = False
    # gen_fem AND active
    pop.prep.prep_strategy = 13
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 16

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.BREASTFEEDING] = [True, False] * (N // 2)
    # pregnant_or_lactating_women AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 16
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * 0.5 * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - 0.5 * pop.prep.prob_risk_informed_prep))
    # expecting base % of half of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev


def test_prep_eligibility_all():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.AGE] = 30
    pop.data[col.SEX] = [SexType.Female, SexType.Male] * (N // 2)
    pop.data[col.NUM_PARTNERS] = [0, 0, 0, 1] * (N // 4)  # half of all men are inherently at risk
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_ON_ART] = False
    pop.data[col.LTP_HIV_STATUS] = False
    pop.data[col.LTP_HIV_DIAGNOSED] = False
    pop.prep.prob_risk_informed_prep = 0.3
    pop.prep.prob_greater_risk_informed_prep = 0.6

    # STRATEGY 4 & 8

    # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 4  # same as 8 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible_men = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                        (col.SEX, op.eq, SexType.Male)]))
    # check no inactive men are eligible
    assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                (col.SEX, op.eq, SexType.Male),
                                (col.NUM_PARTNERS, op.eq, 0)])) == 0
    # half of all men (a quarter of the population) are eligible
    assert eligible_men == N/4

    eligible_women = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                          (col.SEX, op.eq, SexType.Female)]))
    mean = len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)])) * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 8  # same as 4 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible_women = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                          (col.SEX, op.eq, SexType.Female)]))
    mean = len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)])) * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    # STRATEGY 5 & 9

    pop.data[col.PREP_ELIGIBLE] = False
    # gen AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 5  # same as 9 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = False
    pop.data[col.LAST_STP_DATE] = [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10)
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_STP_DATE] = None
    # gen AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 9  # same as 5 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = False
    pop.data[col.LAST_STP_DATE] = [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10)
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 12

    pop.data[col.PREP_ELIGIBLE] = False
    # gen AND active
    pop.prep.prep_strategy = 12
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 14

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LAST_STP_DATE] = None
    # active_at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 14
    pop.prep.prep_eligibility(pop)

    eligible_women = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True),
                                          (col.SEX, op.eq, SexType.Female)]))
    mean = len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)])) * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = False
    pop.data[col.LAST_STP_DATE] = [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10)
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 50% of the population has recently (< 6 months) been sexually active
    assert eligible == N * 0.5

    # STRATEGY 15

    pop.data[col.PREP_ELIGIBLE] = False
    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_HIV_DIAGNOSED] = [True, False, False, True] * (N // 4)  # half of the population inherently at risk
    # at_risk_ltp OR gen_ltp
    pop.prep.prep_strategy = 15
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * 0.51
    stdev = sqrt(mean * (1 - 0.51))
    # expecting an additional 1% of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev
