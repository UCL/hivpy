import operator as op
from math import sqrt

import pytest

import hivpy.column_names as col
from hivpy.common import SexType, date, rng, timedelta
from hivpy.population import Population
from hivpy.prep import PrEPType


@pytest.fixture(autouse=True)
def resetRandomState():
    rng.set_seed(42)


def reset_prep_propensity_cols(pop: Population):
    pop.set_present_variable(col.PREP_ORAL_PREF, 0)
    pop.set_present_variable(col.PREP_CAB_PREF, 0)
    pop.set_present_variable(col.PREP_LEN_PREF, 0)
    pop.set_present_variable(col.PREP_VR_PREF, 0)
    pop.set_present_variable(col.PREP_ORAL_WILLING, False)
    pop.set_present_variable(col.PREP_CAB_WILLING, False)
    pop.set_present_variable(col.PREP_LEN_WILLING, False)
    pop.set_present_variable(col.PREP_VR_WILLING, False)
    pop.set_present_variable(col.PREP_ANY_WILLING, False)
    pop.set_present_variable(col.PREP_ORAL_RANK, 0)
    pop.set_present_variable(col.PREP_CAB_RANK, 0)
    pop.set_present_variable(col.PREP_LEN_RANK, 0)
    pop.set_present_variable(col.PREP_VR_RANK, 0)
    pop.set_present_variable(col.FAVOURED_PREP_TYPE, None)


def test_at_risk_pop():
    N = 100
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # at_risk = num_stp >= 1 OR (ltp_diag AND not ltp_on_art)
    pop.set_present_variable(col.NUM_PARTNERS, [0, 1] * (N // 2))
    pop.set_present_variable(col.LTP_DIAGNOSED, [True, False, False, False] * (N // 4))
    pop.set_present_variable(col.LTP_ON_ART, False)
    # 3/4 of people fulfill one of the conditions for being at risk
    assert len(pop.prep.get_at_risk_pop(pop)) == N // 4 * 3


def test_risk_informed_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # risk_informed = ltp AND not ltp_on_art AND r < prob_risk_informed_prep
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_ON_ART, False)
    pop.set_present_variable(col.LTP_STATUS, False)
    pop.prep.reroll_r_prep(pop)
    pop.prep.prob_risk_informed_prep = 0.1

    # get stats
    no_risk_informed = len(
        pop.prep.get_risk_informed_pop(pop, pop.prep.prob_risk_informed_prep)
    )
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting ~10% of the population to be risk informed
    assert mean - 3 * stdev <= no_risk_informed <= mean + 3 * stdev

    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    no_risk_informed = len(
        pop.prep.get_risk_informed_pop(pop, pop.prep.prob_risk_informed_prep)
    )
    assert no_risk_informed == 0


def test_suspect_risk_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    # suspect_risk = ltp AND not ltp_on_art AND ltp_infected AND r < prob_suspect_risk_prep
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_ON_ART, [True, False] * (N // 2))
    pop.set_present_variable(col.LTP_STATUS, True)
    pop.prep.reroll_r_prep(pop)
    pop.prep.prob_suspect_risk_prep = 0.5

    # get stats
    no_suspect_risk = len(pop.prep.get_suspect_risk_pop(pop))
    mean = (N / 2) * pop.prep.prob_suspect_risk_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_suspect_risk_prep))
    # expecting ~50% of the population with partners NOT on ART to suspect they are at risk
    assert mean - 3 * stdev <= no_suspect_risk <= mean + 3 * stdev


def test_presumed_hiv_neg_pop():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.set_present_variable(col.EVER_TESTED, True)
    pop.set_present_variable(col.HIV_DIAGNOSED, True)
    pop.set_present_variable(col.HIV_STATUS, True)
    pop.set_present_variable(col.DATE_HIV_INFECTION, date(2019, 12, 1))
    pop.hiv_diagnosis.init_prep_inj_na = True
    pop.hiv_diagnosis.test_sens_general = 0.8
    pop.hiv_diagnosis.test_sens_primary_ab = 0.5

    # nobody should be false negative because everyone is diagnosed
    assert len(pop.prep.get_presumed_hiv_neg_pop(pop)) == 0

    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    # get stats (general test sensitivity)
    no_presumed_hiv_neg = len(pop.prep.get_presumed_hiv_neg_pop(pop))
    mean = N * (1 - pop.hiv_diagnosis.test_sens_general)
    stdev = sqrt(mean * pop.hiv_diagnosis.test_sens_general)
    # expecting ~20% of the population to be false negative
    assert mean - 3 * stdev <= no_presumed_hiv_neg <= mean + 3 * stdev

    pop.hiv_diagnosis.init_prep_inj_na = False
    # get stats (primary test sensitivity)
    no_presumed_hiv_neg = len(pop.prep.get_presumed_hiv_neg_pop(pop))
    mean = N * (1 - pop.hiv_diagnosis.test_sens_primary_ab)
    stdev = sqrt(mean * pop.hiv_diagnosis.test_sens_primary_ab)
    # expecting ~50% of the population to be false negative
    assert mean - 3 * stdev <= no_presumed_hiv_neg <= mean + 3 * stdev


def test_prep_propensity():
    N = 100
    pop = Population(size=N, start_date=date(1999, 1, 1))
    pop.set_present_variable(col.AGE, [10, 20] * (N // 2))
    pop.set_present_variable(col.VIRAL_LOAD, 5.0)
    # all prep types have different intro dates
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(5000)]
    # adjust chances of higher preference
    pop.prep.prep_oral_pref_beta = 3
    pop.prep.prep_cab_pref_beta = 3.3
    pop.prep.prep_len_pref_beta = 3.3
    pop.prep.prep_vr_pref_beta = 2.9
    # vl prevalence accounted for
    pop.prep.vl_prevalence_affects_prep = True
    pop.prep.vl_prevalence_prep_threshold = 0.5

    # no willingness before prep intro date
    pop.prep.prep_propensity(pop)
    assert sum(pop.get_variable(col.PREP_ANY_WILLING)) == 0

    # find sub-pops
    under_15s = pop.get_sub_pop([(col.AGE, op.lt, 15)])
    over_15s = pop.get_sub_pop([(col.AGE, op.ge, 15)])
    # willingness calculated for under and over 15s
    pop.date = date(2000, 1, 1)
    pop.prep.prep_propensity(pop)

    # no willingness established for under 15s
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING, under_15s)) == 0
    assert sum(pop.get_variable(col.PREP_ANY_WILLING, under_15s)) == 0
    # some oral willingness established for over 15s
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING, over_15s)) > 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING, over_15s)) == 0
    # check oral is highest preference (or unassigned 0 ranks for under 15s)
    assert all(pop.get_variable(col.PREP_ORAL_RANK, under_15s) == 0)
    assert all(pop.get_variable(col.PREP_ORAL_RANK, over_15s) == 1)

    reset_prep_propensity_cols(pop)
    pop.date = date(3000, 1, 1)
    pop.prep.prep_propensity(pop)

    # no willingness established for under 15s
    assert sum(pop.get_variable(col.PREP_CAB_WILLING, under_15s)) == 0
    assert sum(pop.get_variable(col.PREP_ANY_WILLING, under_15s)) == 0
    # some cab willingness established for over 15s
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING, over_15s)) > 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING, over_15s)) == 0
    # check cab is highest preference (or unassigned 0 ranks for under 15s)
    assert all(pop.get_variable(col.PREP_CAB_RANK, under_15s) == 0)
    assert all(pop.get_variable(col.PREP_CAB_RANK, over_15s) == 1)

    reset_prep_propensity_cols(pop)
    pop.date = date(4000, 1, 1)
    pop.prep.prep_propensity(pop)

    # no willingness established for under 15s
    assert sum(pop.get_variable(col.PREP_LEN_WILLING, under_15s)) == 0
    assert sum(pop.get_variable(col.PREP_ANY_WILLING, under_15s)) == 0
    # some len willingness established for over 15s
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING, over_15s)) > 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING, over_15s)) == 0
    # check len is highest preference (or unassigned 0 ranks for under 15s)
    assert all(pop.get_variable(col.PREP_LEN_RANK, under_15s) == 0)
    assert all(pop.get_variable(col.PREP_LEN_RANK, over_15s) == 1)

    reset_prep_propensity_cols(pop)
    pop.date = date(5000, 1, 1)
    pop.prep.prep_propensity(pop)

    # no willingness established for under 15s
    assert sum(pop.get_variable(col.PREP_VR_WILLING, under_15s)) == 0
    assert sum(pop.get_variable(col.PREP_ANY_WILLING, under_15s)) == 0
    # some vr willingness established for over 15s
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING, over_15s)) == 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING, over_15s)) > 0
    # check vr is highest preference (or unassigned 0 ranks for men and under 15s)
    assert all(pop.get_variable(col.PREP_VR_RANK, under_15s) == 0)
    assert all(
        pop.get_variable(
            col.PREP_VR_RANK,
            pop.get_sub_pop_intersection(
                over_15s, pop.get_sub_pop([(col.SEX, op.eq, SexType.Male)])
            ),
        )
        == 0
    )
    assert all(
        pop.get_variable(
            col.PREP_VR_RANK,
            pop.get_sub_pop_intersection(
                over_15s, pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)])
            ),
        )
        == 1
    )

    # willingness calculated for those who turned 15 this time step
    reset_prep_propensity_cols(pop)
    pop.date = date(2020, 1, 1)
    pop.set_present_variable(col.AGE, 15)
    pop.prep.prep_propensity(pop)
    # some oral willingness established
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING)) == 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING)) == 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING)) == 0

    reset_prep_propensity_cols(pop)
    pop.date = date(3020, 1, 1)
    pop.prep.prep_propensity(pop)
    # some oral + cab willingness established
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING)) == 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING)) == 0

    reset_prep_propensity_cols(pop)
    pop.date = date(4020, 1, 1)
    pop.prep.prep_propensity(pop)
    # some oral + cab + len willingness established
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING)) == 0

    reset_prep_propensity_cols(pop)
    pop.date = date(5020, 1, 1)
    pop.prep.prep_propensity(pop)
    # some oral + cab + len + vr willingness established
    assert sum(pop.get_variable(col.PREP_ORAL_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_CAB_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_LEN_WILLING)) > 0
    assert sum(pop.get_variable(col.PREP_VR_WILLING)) > 0

    # reset willingness with low viral load prevalence
    pop.set_present_variable(col.VIRAL_LOAD, 2.0)
    pop.prep.prep_propensity(pop)
    # no willingness remains
    assert sum(pop.get_variable(col.PREP_ANY_WILLING)) == 0


@pytest.mark.parametrize("prep_strategy", [i for i in range(1, 17)])
def test_prep_ineligible(prep_strategy):
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.set_present_variable(col.HIV_DIAGNOSED, [True, False] * (N // 2))
    pop.prep.prep_strategy = prep_strategy
    pop.prep.prep_eligibility(pop)

    # check that nobody diagnosed with HIV is eligible
    assert (
        len(
            pop.get_sub_pop(
                [(col.PREP_ELIGIBLE, op.eq, True), (col.HIV_DIAGNOSED, op.eq, True)]
            )
        )
        == 0
    )

    # check that nobody under 15 or over 50 is eligible
    assert (
        len(
            pop.get_sub_pop(
                [
                    (col.PREP_ELIGIBLE, op.eq, True),
                    (col.AGE, op.lt, 15),
                    (col.AGE, op.ge, 50),
                ]
            )
        )
        == 0
    )

    # check that no men are eligible in women only strategies
    if prep_strategy not in [4, 5, 8, 9, 12, 14, 15]:
        assert (
            len(
                pop.get_sub_pop(
                    [(col.PREP_ELIGIBLE, op.eq, True), (col.SEX, op.eq, SexType.Male)]
                )
            )
            == 0
        )


def test_prep_eligibility_continuity():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))
    pop.set_present_variable(col.AGE, 30)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_ON_ART, False)
    pop.set_present_variable(col.LTP_STATUS, [True, False] * (N // 2))
    pop.set_present_variable(col.R_PREP, 1.0)

    pop.prep.prep_strategy = 9
    pop.prep.prep_eligibility(pop)
    # check all r_prep values were rerolled (nobody started out eligible)
    assert all(pop.get_variable(col.R_PREP) < 1.0)

    # get initial eligible pop
    init_eligible = pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])
    # set eligibility again
    pop.prep.prep_eligibility(pop)
    new_eligible = pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])
    # check all previously eligible people are still eligible
    assert set(init_eligible).issubset(new_eligible)
    # check there are now more eligible people (demonstrates that r_prep is recalculated for those ineligible)
    assert len(new_eligible) > len(init_eligible)

    pop.set_present_variable(col.AGE, 50)
    pop.prep.prep_eligibility(pop)
    # check that everyone has aged out of eligibility
    assert len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)])) == 0


def test_prep_eligibility_women_only():
    N = 1000
    pop = Population(size=N, start_date=date(2020, 1, 1))

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.SEX, SexType.Female)
    pop.set_present_variable(col.SEX_WORKER, [False, True] * (N // 2))
    pop.set_present_variable(col.AGE, [20, 30] * (N // 2))
    pop.set_present_variable(col.NUM_PARTNERS, 1)  # everyone is at risk

    # STRATEGY 1-3

    # fsw_agyw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 1
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # everyone fulfills the necessary eligibility conditions
    assert eligible == N

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # fsw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 2
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # only half of the population are sex workers
    assert eligible == N / 2

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # agyw AND (at_risk OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 3
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # only half of the population are 15-25
    assert eligible == N / 2

    # STRATEGY 6 & 10

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.NUM_PARTNERS, 0)  # nobody is inherently at risk
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_ON_ART, False)
    pop.set_present_variable(col.LTP_STATUS, False)
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

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # gen_fem AND (at_risk OR (gen_age AND (risk_informed OR suspect_risk)))
    pop.prep.prep_strategy = 10  # same as 6 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    # STRATEGY 7 & 11

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 7  # same as 11 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(
        col.LAST_STP_DATE,
        [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10),
    )
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LAST_STP_DATE, None)
    # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 11  # same as 7 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(
        col.LAST_STP_DATE,
        [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10),
    )
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 13

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # gen_fem AND active
    pop.prep.prep_strategy = 13
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 16

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.BREASTFEEDING, [True, False] * (N // 2))
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

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.AGE, 30)
    pop.set_present_variable(col.SEX, [SexType.Female, SexType.Male] * (N // 2))
    pop.set_present_variable(
        col.NUM_PARTNERS, [0, 0, 0, 1] * (N // 4)
    )  # half of all men are inherently at risk
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LTP_ON_ART, False)
    pop.set_present_variable(col.LTP_STATUS, False)
    pop.set_present_variable(col.LTP_DIAGNOSED, False)
    pop.prep.prob_risk_informed_prep = 0.3
    pop.prep.prob_greater_risk_informed_prep = 0.6

    # STRATEGY 4 & 8

    # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 4  # same as 8 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible_men = len(
        pop.get_sub_pop(
            [(col.PREP_ELIGIBLE, op.eq, True), (col.SEX, op.eq, SexType.Male)]
        )
    )
    # check no inactive men are eligible
    assert (
        len(
            pop.get_sub_pop(
                [
                    (col.PREP_ELIGIBLE, op.eq, True),
                    (col.SEX, op.eq, SexType.Male),
                    (col.NUM_PARTNERS, op.eq, 0),
                ]
            )
        )
        == 0
    )
    # half of all men (a quarter of the population) are eligible
    assert eligible_men == N / 4

    eligible_women = len(
        pop.get_sub_pop(
            [(col.PREP_ELIGIBLE, op.eq, True), (col.SEX, op.eq, SexType.Female)]
        )
    )
    mean = (
        len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))
        * pop.prep.prob_risk_informed_prep
    )
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 8  # same as 4 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible_women = len(
        pop.get_sub_pop(
            [(col.PREP_ELIGIBLE, op.eq, True), (col.SEX, op.eq, SexType.Female)]
        )
    )
    mean = (
        len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))
        * pop.prep.prob_greater_risk_informed_prep
    )
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    # STRATEGY 5 & 9

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # gen AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 5  # same as 9 but uses base risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_risk_informed_prep))
    # expecting base % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(
        col.LAST_STP_DATE,
        [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10),
    )
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LAST_STP_DATE, None)
    # gen AND (active_stp OR risk_informed OR suspect_risk)
    pop.prep.prep_strategy = 9  # same as 5 but uses greater risk informed prob
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * pop.prep.prob_greater_risk_informed_prep
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(
        col.LAST_STP_DATE,
        [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10),
    )
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 12

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    # gen AND active
    pop.prep.prep_strategy = 12
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 80% of the population has recently (< 9 months) been sexually active
    assert eligible == N * 0.8

    # STRATEGY 14

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(col.LAST_STP_DATE, None)
    # active_at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
    pop.prep.prep_strategy = 14
    pop.prep.prep_eligibility(pop)

    eligible_women = len(
        pop.get_sub_pop(
            [(col.PREP_ELIGIBLE, op.eq, True), (col.SEX, op.eq, SexType.Female)]
        )
    )
    mean = (
        len(pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))
        * pop.prep.prob_greater_risk_informed_prep
    )
    stdev = sqrt(mean * (1 - pop.prep.prob_greater_risk_informed_prep))
    # expecting greater % of women to be risk informed
    assert mean - 3 * stdev <= eligible_women <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(
        col.LAST_STP_DATE,
        [pop.date - timedelta(months=x) for x in range(1, 11)] * (N // 10),
    )
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    # 50% of the population has recently (< 6 months) been sexually active
    assert eligible == N * 0.5

    # STRATEGY 15

    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True)
    pop.set_present_variable(
        col.LTP_DIAGNOSED, [True, False, False, True] * (N // 4)
    )  # half of the population inherently at risk
    # at_risk_ltp OR gen_ltp
    pop.prep.prep_strategy = 15
    pop.prep.prep_eligibility(pop)

    eligible = len(pop.get_sub_pop([(col.PREP_ELIGIBLE, op.eq, True)]))
    mean = N * 0.51
    stdev = sqrt(mean * (1 - 0.51))
    # expecting an additional 1% of the population to be risk informed
    assert mean - 3 * stdev <= eligible <= mean + 3 * stdev


def test_starting_prep():
    N = 1000
    time_step = timedelta(months=1)
    pop = Population(size=N, start_date=date(5000, 1, 1))
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(5000)]
    pop.set_present_variable(col.HARD_REACH, False)
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.HIV_STATUS, False)
    pop.set_present_variable(col.PREP_ELIGIBLE, True)
    pop.set_present_variable(col.PREP_ANY_WILLING, True)
    pop.set_present_variable(col.EVER_PREP, False)
    pop.set_present_variable(col.ON_PREP, False)
    pop.set_present_variable(col.LAST_TEST_DATE, pop.date)
    pop.set_present_variable(col.CONT_ON_PREP, None)
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, None)
    pop.set_present_variable(col.CUMULATIVE_PREP_ORAL, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_CAB, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_LEN, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_VR, timedelta(months=0))
    # tested explicitly to start prep
    pop.set_present_variable(
        col.PREP_ORAL_TESTED, [True, False, False, False] * (N // 4)
    )
    pop.set_present_variable(
        col.PREP_CAB_TESTED, [False, True, False, False] * (N // 4)
    )
    pop.set_present_variable(
        col.PREP_LEN_TESTED, [False, False, True, False] * (N // 4)
    )
    pop.set_present_variable(col.PREP_VR_TESTED, [False, False, False, True] * (N // 4))

    pop.prep.start_prep(pop, time_step)
    assert all(pop.get_variable(col.ON_PREP))
    # prep types spread evenly among population
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Oral) == N / 4
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Cabotegravir) == N / 4
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir) == N / 4
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing) == N / 4
    # check continuous and cumulative prep usage
    assert all(pop.get_variable(col.CONT_ON_PREP) == time_step)
    assert all(pop.get_variable(col.CONT_ACTIVE_ON_PREP) == time_step)
    assert sum(pop.get_variable(col.CUMULATIVE_PREP_ORAL) == time_step) == N / 4
    assert sum(pop.get_variable(col.CUMULATIVE_PREP_CAB) == time_step) == N / 4
    assert sum(pop.get_variable(col.CUMULATIVE_PREP_LEN) == time_step) == N / 4
    assert sum(pop.get_variable(col.CUMULATIVE_PREP_VR) == time_step) == N / 4

    pop.set_present_variable(col.PREP_TYPE, PrEPType.NoPrEP)
    pop.set_present_variable(col.EVER_PREP, [True, False] * (N // 2))
    pop.set_present_variable(col.FIRST_ORAL_START_DATE, None)
    pop.set_present_variable(col.FIRST_CAB_START_DATE, None)
    pop.set_present_variable(col.FIRST_LEN_START_DATE, None)
    pop.set_present_variable(col.FIRST_VR_START_DATE, None)
    pop.set_present_variable(col.LAST_PREP_START_DATE, None)
    pop.prep.start_prep(pop, time_step)

    # only 50% eligible to start prep for the first time
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.NoPrEP) == N // 2
    # check that people who aren't on a specific type of prep don't have start dates
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.Oral)
        == (pop.get_variable(col.FIRST_ORAL_START_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.Cabotegravir)
        == (pop.get_variable(col.FIRST_CAB_START_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.Lenacapavir)
        == (pop.get_variable(col.FIRST_LEN_START_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.VaginalRing)
        == (pop.get_variable(col.FIRST_VR_START_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.NoPrEP)
        == (pop.get_variable(col.LAST_PREP_START_DATE) == pop.date)
    )

    pop.set_present_variable(col.PREP_TYPE, PrEPType.NoPrEP)
    pop.set_present_variable(col.EVER_PREP, False)
    # introduce different preference ranking distributions
    pop.set_present_variable(col.PREP_ORAL_RANK, [1, 2, 3, 4] * (N // 4))
    pop.set_present_variable(col.PREP_CAB_RANK, [2, 1, 2, 3] * (N // 4))
    pop.set_present_variable(col.PREP_LEN_RANK, [3, 3, 1, 2] * (N // 4))
    pop.set_present_variable(col.PREP_VR_RANK, [4, 4, 4, 1] * (N // 4))
    # all willing to take any prep
    pop.set_present_variable(col.PREP_ORAL_WILLING, True)
    pop.set_present_variable(col.PREP_CAB_WILLING, True)
    pop.set_present_variable(col.PREP_LEN_WILLING, True)
    pop.set_present_variable(col.PREP_VR_WILLING, True)
    # not tested explicitly to start prep
    pop.set_present_variable(col.PREP_ORAL_TESTED, False)
    pop.set_present_variable(col.PREP_CAB_TESTED, False)
    pop.set_present_variable(col.PREP_LEN_TESTED, False)
    pop.set_present_variable(col.PREP_VR_TESTED, False)
    # all prep types have different start probabilities
    pop.prep.prob_oral_prep_start = 0.9
    pop.prep.prob_cab_prep_start = 0.8
    pop.prep.prob_len_prep_start = 0.7
    pop.prep.prob_vr_prep_start = 0.6

    pop.prep.favoured_prep(pop, None)
    pop.prep.start_prep(pop, time_step)
    # test oral prep type start probability
    no_on_oral = sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
    mean = N / 4 * pop.prep.prob_oral_prep_start
    stdev = sqrt(mean * (1 - pop.prep.prob_oral_prep_start))
    assert mean - 3 * stdev <= no_on_oral <= mean + 3 * stdev
    # test cab prep type start probability
    no_on_cab = sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Cabotegravir)
    mean = N / 4 * pop.prep.prob_cab_prep_start
    stdev = sqrt(mean * (1 - pop.prep.prob_cab_prep_start))
    assert mean - 3 * stdev <= no_on_cab <= mean + 3 * stdev
    # test len prep type start probability
    no_on_len = sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
    mean = N / 4 * pop.prep.prob_len_prep_start
    stdev = sqrt(mean * (1 - pop.prep.prob_len_prep_start))
    assert mean - 3 * stdev <= no_on_len <= mean + 3 * stdev
    # test vr prep type start probability
    no_on_vr = sum(pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing)
    mean = N / 4 * pop.prep.prob_vr_prep_start
    stdev = sqrt(mean * (1 - pop.prep.prob_vr_prep_start))
    assert mean - 3 * stdev <= no_on_vr <= mean + 3 * stdev

    pop.set_present_variable(col.PREP_TYPE, PrEPType.NoPrEP)
    pop.set_present_variable(col.EVER_PREP, False)
    pop.set_present_variable(col.ON_PREP, False)
    pop.set_present_variable(col.CONT_ON_PREP, None)
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, None)
    pop.set_present_variable(col.CUMULATIVE_PREP_ORAL, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_CAB, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_LEN, timedelta(months=0))
    pop.set_present_variable(col.CUMULATIVE_PREP_VR, timedelta(months=0))
    # 100% chance to start prep
    pop.prep.prob_oral_prep_start = 1
    pop.prep.prob_cab_prep_start = 1
    pop.prep.prob_len_prep_start = 1
    pop.prep.prob_vr_prep_start = 1
    pop.prep.start_prep(pop, time_step)

    # everyone starts their most preferred prep type
    assert all(pop.get_variable(col.ON_PREP))
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
        == (pop.get_variable(col.PREP_ORAL_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Cabotegravir)
        == (pop.get_variable(col.PREP_CAB_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
        == (pop.get_variable(col.PREP_LEN_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing)
        == (pop.get_variable(col.PREP_VR_RANK) == 1)
    )
    # check continuous and cumulative prep usage
    assert all(pop.get_variable(col.CONT_ON_PREP) == time_step)
    assert all(pop.get_variable(col.CONT_ACTIVE_ON_PREP) == time_step)
    assert all(
        (pop.get_variable(col.CUMULATIVE_PREP_ORAL) == time_step)
        == (pop.get_variable(col.PREP_ORAL_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.CUMULATIVE_PREP_CAB) == time_step)
        == (pop.get_variable(col.PREP_CAB_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.CUMULATIVE_PREP_LEN) == time_step)
        == (pop.get_variable(col.PREP_LEN_RANK) == 1)
    )
    assert all(
        (pop.get_variable(col.CUMULATIVE_PREP_VR) == time_step)
        == (pop.get_variable(col.PREP_VR_RANK) == 1)
    )

    pop.set_present_variable(col.PREP_TYPE, PrEPType.NoPrEP)
    pop.set_present_variable(col.EVER_PREP, False)
    # nobody is willing to take oral or cab
    pop.set_present_variable(col.PREP_ORAL_WILLING, False)
    pop.set_present_variable(col.PREP_CAB_WILLING, False)
    pop.prep.favoured_prep(pop, None)
    pop.prep.start_prep(pop, time_step)

    # everyone is either on len or vr
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir) == N * 0.75
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing) == N * 0.25
    # check that people who aren't on a specific type of prep don't have start dates
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.Lenacapavir)
        == (pop.get_variable(col.FIRST_LEN_START_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.VaginalRing)
        == (pop.get_variable(col.FIRST_VR_START_DATE).isnull())
    )
    assert all(pop.get_variable(col.LAST_PREP_START_DATE) == pop.date)

    pop.set_present_variable(col.PREP_TYPE, PrEPType.NoPrEP)
    pop.set_present_variable(col.EVER_PREP, False)
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(6000)]
    pop.prep.favoured_prep(pop, None)
    pop.prep.start_prep(pop, time_step)
    # everyone is on len because vr is not yet available
    assert all(pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
    assert all(pop.get_variable(col.FIRST_LEN_START_DATE) == pop.date)


def test_continuing_prep():
    N = 100
    time_step = timedelta(months=1)
    pop = Population(size=N, start_date=date(5000, 1, 1))
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(5000)]
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.PREP_ELIGIBLE, True)
    pop.set_present_variable(col.EVER_PREP, True)
    pop.set_present_variable(col.ON_PREP, True)
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, None)
    pop.set_present_variable(col.PREP_JUST_STARTED, False)
    pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=3))
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.CUMULATIVE_PREP_ORAL, time_step)
    pop.set_present_variable(col.CUMULATIVE_PREP_CAB, time_step)
    pop.set_present_variable(col.CUMULATIVE_PREP_LEN, time_step)
    pop.set_present_variable(col.CUMULATIVE_PREP_VR, time_step)
    pop.set_present_variable(col.LAST_TEST_DATE, pop.date - timedelta(months=3))
    pop.set_present_variable(
        col.LAST_PREP_USE_DATE,
        [
            pop.date - time_step,
            pop.date - timedelta(months=3),
            pop.date - timedelta(months=6),
            pop.date - time_step,
        ]
        * (N // 4),
    )
    # prep types spread evenly among population
    pop.set_present_variable(
        col.PREP_TYPE,
        [
            PrEPType.Oral,
            PrEPType.Cabotegravir,
            PrEPType.Lenacapavir,
            PrEPType.VaginalRing,
        ]
        * (N // 4),
    )
    # everyone is taking their favoured prep type
    pop.set_present_variable(
        col.FAVOURED_PREP_TYPE,
        [
            PrEPType.Oral,
            PrEPType.Cabotegravir,
            PrEPType.Lenacapavir,
            PrEPType.VaginalRing,
        ]
        * (N // 4),
    )
    # 10% chance to stop prep
    prob_base_prep_stop = 0.1
    pop.prep.prob_oral_prep_stop = prob_base_prep_stop
    pop.prep.prob_cab_prep_stop = prob_base_prep_stop
    pop.prep.prob_len_prep_stop = prob_base_prep_stop
    pop.prep.prob_vr_prep_stop = prob_base_prep_stop

    pop.prep.continue_prep(pop, time_step)
    # expecting 90% of people to continue prep
    no_on_prep = sum(pop.get_variable(col.CONT_ACTIVE_ON_PREP) == timedelta(months=3))
    mean = N * (1 - prob_base_prep_stop)
    stdev = sqrt(mean * prob_base_prep_stop)
    assert mean - 3 * stdev <= no_on_prep <= mean + 3 * stdev
    # expecting 10% of people to stop prep
    no_off_prep = sum(pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
    mean = N * prob_base_prep_stop
    stdev = sqrt(mean * (1 - prob_base_prep_stop))
    assert mean - 3 * stdev <= no_off_prep <= mean + 3 * stdev

    # check cumulative prep usage (those that stopped should not be incremented)
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_ORAL) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_ORAL) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_CAB) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Cabotegravir)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_CAB) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_LEN) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_LEN) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_VR) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_VR) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )

    pop.set_present_variable(col.ON_PREP, True)
    pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=3))
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, None)
    pop.set_present_variable(
        col.LAST_PREP_USE_DATE,
        [
            pop.date - time_step,
            pop.date - timedelta(months=3),
            pop.date - timedelta(months=6),
            pop.date - time_step,
        ]
        * (N // 4),
    )
    # nobody is taking their favoured prep type
    pop.set_present_variable(
        col.FAVOURED_PREP_TYPE,
        [
            PrEPType.VaginalRing,
            PrEPType.Lenacapavir,
            PrEPType.Cabotegravir,
            PrEPType.Oral,
        ]
        * (N // 4),
    )

    pop.prep.continue_prep(pop, time_step)
    # expecting 90% of people to switch prep
    no_on_prep = sum(pop.get_variable(col.CONT_ACTIVE_ON_PREP) == time_step)
    mean = N * (1 - prob_base_prep_stop)
    stdev = sqrt(mean * prob_base_prep_stop)
    assert mean - 3 * stdev <= no_on_prep <= mean + 3 * stdev
    # expecting 10% of people to stop prep
    no_off_prep = sum(pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
    mean = N * prob_base_prep_stop
    stdev = sqrt(mean * (1 - prob_base_prep_stop))
    assert mean - 3 * stdev <= no_off_prep <= mean + 3 * stdev

    # check cumulative prep usage (those that stopped should not be incremented)
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_ORAL) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_ORAL) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_CAB) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Cabotegravir)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_CAB) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_LEN) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_LEN) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )
    assert all(
        (
            (pop.get_variable(col.CUMULATIVE_PREP_VR) == timedelta(months=2))
            == (pop.get_variable(col.PREP_TYPE) == PrEPType.VaginalRing)
        )
        | (
            (pop.get_variable(col.CUMULATIVE_PREP_VR) == time_step)
            == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
        )
    )

    pop.set_present_variable(col.ON_PREP, True)
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, None)
    pop.set_present_variable(col.LAST_PREP_USE_DATE, pop.date - timedelta(months=3))
    pop.set_present_variable(
        col.PREP_TYPE, [PrEPType.Cabotegravir, PrEPType.Lenacapavir] * (N // 2)
    )
    pop.set_present_variable(col.FAVOURED_PREP_TYPE, PrEPType.Oral)
    # nobody stops prep
    prob_base_prep_stop = 0
    pop.prep.prob_oral_prep_stop = prob_base_prep_stop
    pop.prep.prob_cab_prep_stop = prob_base_prep_stop
    pop.prep.prob_len_prep_stop = prob_base_prep_stop
    pop.prep.prob_vr_prep_stop = prob_base_prep_stop

    pop.prep.continue_prep(pop, time_step)
    # len prep not updated because last prep usage is too recent
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
        == (pop.get_variable(col.LAST_PREP_USE_DATE) == pop.date - timedelta(months=3))
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
        == (pop.get_variable(col.LAST_PREP_USE_DATE) == pop.date)
    )
    assert all(pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())

    pop.set_present_variable(col.LAST_PREP_USE_DATE, pop.date - time_step)
    pop.set_present_variable(
        col.PREP_TYPE,
        [
            PrEPType.Oral,
            PrEPType.Cabotegravir,
            PrEPType.Lenacapavir,
            PrEPType.VaginalRing,
        ]
        * (N // 4),
    )

    pop.prep.continue_prep(pop, time_step)
    # no injectable prep updated because last prep usage is too recent
    assert all(
        (pop.get_variable(col.PREP_TYPE) != PrEPType.Oral)
        == (pop.get_variable(col.LAST_PREP_USE_DATE) == pop.date - time_step)
    )
    assert all(
        (pop.get_variable(col.PREP_TYPE) == PrEPType.Oral)
        == (pop.get_variable(col.LAST_PREP_USE_DATE) == pop.date)
    )
    assert sum(pop.get_variable(col.PREP_TYPE) == PrEPType.Oral) == N / 2
    assert all(pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())

    pop.set_present_variable(col.LAST_PREP_USE_DATE, pop.date - time_step)
    pop.set_present_variable(col.PREP_TYPE, PrEPType.Lenacapavir)
    pop.set_present_variable(col.CONT_ON_PREP, time_step)
    pop.set_present_variable(col.CUMULATIVE_PREP_LEN, timedelta(months=2))

    pop.prep.continue_prep(pop, time_step)
    # everyone continues prep without choice
    assert all(pop.get_variable(col.PREP_TYPE) == PrEPType.Lenacapavir)
    assert all(pop.get_variable(col.LAST_PREP_USE_DATE) == pop.date - time_step)
    assert all((pop.get_variable(col.CONT_ON_PREP) == timedelta(months=2)))
    assert all((pop.get_variable(col.CUMULATIVE_PREP_LEN) == timedelta(months=3)))


def test_restarting_prep():
    N = 100
    time_step = timedelta(months=1)
    pop = Population(size=N, start_date=date(5000, 1, 1))
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(5000)]
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.PREP_ELIGIBLE, True)
    pop.set_present_variable(col.EVER_PREP, True)
    pop.set_present_variable(col.ON_PREP, False)
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, pop.date - time_step)
    pop.set_present_variable(col.PREP_PAUSED, False)
    pop.set_present_variable(col.PREP_JUST_STARTED, False)
    pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=0))
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=0))
    pop.set_present_variable(col.LAST_TEST_DATE, pop.date)
    pop.set_present_variable(col.PREP_TYPE, PrEPType.Oral)
    pop.set_present_variable(
        col.FAVOURED_PREP_TYPE,
        [
            PrEPType.Oral,
            PrEPType.Cabotegravir,
            PrEPType.Lenacapavir,
            PrEPType.VaginalRing,
        ]
        * (N // 4),
    )
    # 50% chance to restart prep
    pop.prep.prob_prep_restart = 0.5

    pop.prep.restart_prep(pop, time_step)
    # expecting 50% of people to restart prep
    no_on_prep = sum(pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())
    mean = N * pop.prep.prob_prep_restart
    stdev = sqrt(mean * (1 - pop.prep.prob_prep_restart))
    assert mean - 3 * stdev <= no_on_prep <= mean + 3 * stdev
    # check continuous prep usage
    assert all(
        (pop.get_variable(col.CONT_ON_PREP) == time_step)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())
    )
    assert all(
        (pop.get_variable(col.CONT_ACTIVE_ON_PREP) == time_step)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())
    )

    pop.set_present_variable(col.ON_PREP, False)
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, pop.date - time_step)
    pop.set_present_variable(col.PREP_PAUSED, True)
    pop.prep.restart_prep(pop, time_step)
    # expecting everyone to restart prep
    assert all(pop.get_variable(col.ON_PREP))
    assert all(pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())


def test_stopping_prep():
    N = 100
    time_step = timedelta(months=1)
    pop = Population(size=N, start_date=date(5000, 1, 1))
    pop.prep.date_prep_intro = [date(2000), date(3000), date(4000), date(5000)]
    pop.set_present_variable(col.HIV_DIAGNOSED, False)
    pop.set_present_variable(col.PREP_ELIGIBLE, False)
    pop.set_variable_range(col.PREP_ELIGIBLE, True, N * 0.9, N - 1)
    pop.set_present_variable(col.EVER_PREP, True)
    pop.set_variable_range(col.EVER_PREP, False, N * 0.9, N - 1)
    pop.set_present_variable(col.ON_PREP, True)
    pop.set_present_variable(col.LAST_TEST_DATE, pop.date)
    pop.set_present_variable(col.LAST_PREP_STOP_DATE, None)
    pop.set_present_variable(col.PREP_PAUSED, False)
    pop.set_present_variable(col.LAST_TEST_DATE, pop.date)
    pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.CONT_INTENT_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.PREP_TYPE, PrEPType.Oral)
    pop.set_present_variable(col.FAVOURED_PREP_TYPE, PrEPType.Oral)

    # nobody stops prep by choice
    prob_base_prep_stop = 0
    pop.prep.prob_oral_prep_stop = prob_base_prep_stop
    pop.prep.prob_cab_prep_stop = prob_base_prep_stop
    pop.prep.prob_len_prep_stop = prob_base_prep_stop
    pop.prep.prob_vr_prep_stop = prob_base_prep_stop

    pop.prep.prep_usage(pop, time_step)
    # expecting the ineligible to pause prep
    assert sum(pop.get_variable(col.ON_PREP)) == N * 0.1
    assert sum(pop.get_variable(col.PREP_PAUSED)) == N * 0.9
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
    )
    # check continuous prep usage
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.CONT_ON_PREP) == timedelta(months=2))
    )
    assert all(pop.get_variable(col.CONT_INTENT_ON_PREP) == timedelta(months=3))
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.CONT_ACTIVE_ON_PREP) == timedelta(months=0))
    )

    pop.date += time_step
    pop.inc_variable(col.LAST_TEST_DATE, time_step)
    pop.prep.prep_usage(pop, time_step)
    # the ineligible can't restart
    assert sum(pop.get_variable(col.ON_PREP)) == N * 0.1
    assert sum(pop.get_variable(col.PREP_PAUSED)) == N * 0.9
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date - time_step)
    )
    # check continuous prep usage
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.CONT_ON_PREP) == timedelta(months=2))
    )
    assert all(pop.get_variable(col.CONT_INTENT_ON_PREP) == timedelta(months=4))
    assert all(
        pop.get_variable(col.PREP_PAUSED)
        == (pop.get_variable(col.CONT_ACTIVE_ON_PREP) == timedelta(months=0))
    )

    pop.date += time_step
    pop.inc_variable(col.LAST_TEST_DATE, time_step)
    pop.set_present_variable(col.PREP_ELIGIBLE, True)
    pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=2))
    pop.set_present_variable(col.CONT_INTENT_ON_PREP, timedelta(months=4))
    pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=0))
    pop.prep.prep_usage(pop, time_step)
    # expecting everyone to automatically restart from pause
    assert all(pop.get_variable(col.ON_PREP))
    assert sum(pop.get_variable(col.PREP_PAUSED)) == 0
    assert all(pop.get_variable(col.LAST_PREP_STOP_DATE).isnull())
    # check continuous prep usage
    assert all(pop.get_variable(col.CONT_ON_PREP) == timedelta(months=3))
    assert all(pop.get_variable(col.CONT_INTENT_ON_PREP) == timedelta(months=5))
    assert all(pop.get_variable(col.CONT_ACTIVE_ON_PREP) == timedelta(months=1))

    pop.date += time_step
    pop.set_present_variable(col.HIV_DIAGNOSED, True)
    pop.set_variable_range(col.HIV_DIAGNOSED, False, N * 0.9, N - 1)
    pop.prep.prep_usage(pop, time_step)
    # expecting the diagnosed to stop prep
    assert sum(pop.get_variable(col.ON_PREP)) == N * 0.1
    assert all(~pop.get_variable(col.ON_PREP) == pop.get_variable(col.HIV_DIAGNOSED))
    assert sum(pop.get_variable(col.PREP_PAUSED)) == 0
    assert all(
        ~pop.get_variable(col.ON_PREP)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date)
    )
    # check continuous prep usage
    assert all(
        ~pop.get_variable(col.ON_PREP)
        == (pop.get_variable(col.CONT_ON_PREP) == timedelta(months=0))
    )
    assert all(
        ~pop.get_variable(col.ON_PREP)
        == (pop.get_variable(col.CONT_INTENT_ON_PREP) == timedelta(months=0))
    )
    assert all(
        ~pop.get_variable(col.ON_PREP)
        == (pop.get_variable(col.CONT_ACTIVE_ON_PREP) == timedelta(months=0))
    )

    pop.date += time_step
    pop.prep.prep_usage(pop, time_step)
    # expecting those who stopped will not restart because prep is not paused
    assert sum(pop.get_variable(col.ON_PREP)) == N * 0.1
    assert sum(pop.get_variable(col.PREP_PAUSED)) == 0
    assert all(
        ~pop.get_variable(col.ON_PREP)
        == (pop.get_variable(col.LAST_PREP_STOP_DATE) == pop.date - time_step)
    )
