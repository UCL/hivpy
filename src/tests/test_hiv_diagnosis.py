import operator as op
from math import sqrt

import hivpy.column_names as col
from hivpy.common import date
from hivpy.hiv_diagnosis import HIVTestType
from hivpy.population import Population
from hivpy.prep import PrEPType


def test_primary_infection_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.IN_PRIMARY_INFECTION] = True
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = None
    # test sensitivities
    pop.hiv_diagnosis.test_sens_primary_ab = 0.50
    test_sens_primary_na = 0.86
    test_sens_primary_agab = 0.75

    # Ab primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.Ab
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_primary_ab
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_primary_ab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # NA primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.NA
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * test_sens_primary_na
    stdev = sqrt(mean * (1 - test_sens_primary_na))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # AgAb primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.AgAb
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * test_sens_primary_agab
    stdev = sqrt(mean * (1 - test_sens_primary_agab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev


def test_primary_infection_prep_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.IN_PRIMARY_INFECTION] = True
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = PrEPType.Cabotegravir
    pop.data[col.PREP_JUST_STARTED] = False
    # test sensitivities
    pop.hiv_diagnosis.test_sens_prep_inj_primary_ab = 0.1
    pop.hiv_diagnosis.test_sens_prep_inj_primary_na = 0.3
    test_sens_prep_inj_primary_agab = 0

    # Ab + PrEP primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.Ab
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_primary_ab
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_primary_ab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # NA + PrEP primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.NA
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_primary_na
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_primary_na))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # AgAb + PrEP primary infection outcomes
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.AgAb
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * test_sens_prep_inj_primary_agab
    stdev = sqrt(mean * (1 - test_sens_prep_inj_primary_agab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev


def test_general_population_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.HIV_STATUS] = True
    pop.data[col.IN_PRIMARY_INFECTION] = False
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = None

    # general outcomes
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_general
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_general))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev


def test_general_population_prep_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.HIV_STATUS] = True
    pop.data[col.IN_PRIMARY_INFECTION] = False
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = PrEPType.Lenacapavir
    # test sensitivities
    pop.hiv_diagnosis.test_sens_prep_inj_3m_ab = 0.2
    pop.hiv_diagnosis.test_sens_prep_inj_ge6m_ab = 0.5
    pop.hiv_diagnosis.test_sens_prep_inj_3m_na = 0.7
    pop.hiv_diagnosis.test_sens_prep_inj_ge6m_na = 0.8

    # Ab + PrEP general outcomes (recent infection)
    pop.hiv_diagnosis.prep_inj_na = False
    pop.data[col.HIV_INFECTION_GE6M] = False
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_3m_ab
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_3m_ab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # Ab + PrEP general outcomes (older infection)
    pop.data[col.HIV_INFECTION_GE6M] = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_ge6m_ab
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_ge6m_ab))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # NA + PrEP general outcomes (recent infection)
    pop.hiv_diagnosis.prep_inj_na = True
    pop.data[col.HIV_INFECTION_GE6M] = False
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_3m_na
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_3m_na))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # NA + PrEP general outcomes (older infection)
    pop.data[col.HIV_INFECTION_GE6M] = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    diag_pop = len(pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, True)]))
    mean = N * pop.hiv_diagnosis.test_sens_prep_inj_ge6m_na
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.test_sens_prep_inj_ge6m_na))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= diag_pop <= mean + 3 * stdev


def test_primary_loss_at_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.IN_PRIMARY_INFECTION] = True
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = None
    pop.data[col.SEX_WORKER] = False
    # adjust probabilities
    pop.hiv_diagnosis.test_sens_primary_ab = 1  # diagnose everyone
    pop.hiv_diagnosis.prob_loss_at_diag = 0.50
    pop.hiv_diagnosis.sw_incr_prob_loss_at_diag = 1.4
    sw_prob_loss_at_diag = pop.hiv_diagnosis.prob_loss_at_diag * pop.hiv_diagnosis.sw_incr_prob_loss_at_diag

    # Ab primary infection loss of care
    pop.hiv_diagnosis.hiv_test_type = HIVTestType.Ab
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    lost = len(pop.get_sub_pop([(col.UNDER_CARE, op.eq, False)]))
    mean = N * pop.hiv_diagnosis.prob_loss_at_diag
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.prob_loss_at_diag))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= lost <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # Ab primary infection loss of care (sex workers)
    pop.data[col.SEX_WORKER] = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    lost = len(pop.get_sub_pop([(col.UNDER_CARE, op.eq, False)]))
    mean = N * sw_prob_loss_at_diag
    stdev = sqrt(mean * (1 - sw_prob_loss_at_diag))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= lost <= mean + 3 * stdev


def test_general_loss_at_diagnosis():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data[col.HIV_STATUS] = True
    pop.data[col.IN_PRIMARY_INFECTION] = False
    pop.data[col.LAST_TEST_DATE] = pop.date
    pop.data[col.HIV_DIAGNOSED] = False
    pop.data[col.PREP_TYPE] = None
    pop.data[col.SEX_WORKER] = False
    pop.data[col.ADC] = False
    pop.data[col.TB] = False
    pop.data[col.NON_TB_WHO3] = False
    pop.data[col.NUM_PARTNERS] = 2
    # adjust probabilities
    pop.hiv_diagnosis.test_sens_general = 1  # diagnose everyone
    pop.hiv_diagnosis.prob_loss_at_diag = 0.50
    pop.hiv_diagnosis.prob_loss_at_diag_adc_tb = 0.30
    pop.hiv_diagnosis.prob_loss_at_diag_non_tb_who3 = 0.45

    # general outcomes (less engaged)
    pop.hiv_diagnosis.higher_newp_less_engagement = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    lost = len(pop.get_sub_pop([(col.UNDER_CARE, op.eq, False)]))
    mean = N * pop.hiv_diagnosis.prob_loss_at_diag * 1.5
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.prob_loss_at_diag * 1.5))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= lost <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # general outcomes (adc or tb)
    pop.data[col.ADC] = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    lost = len(pop.get_sub_pop([(col.UNDER_CARE, op.eq, False)]))
    mean = N * pop.hiv_diagnosis.prob_loss_at_diag_adc_tb
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.prob_loss_at_diag_adc_tb))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= lost <= mean + 3 * stdev

    # reset diagnosis
    pop.data[col.HIV_DIAGNOSED] = False
    # general outcomes (non-tb who3)
    pop.data[col.ADC] = False
    pop.data[col.NON_TB_WHO3] = True
    pop.hiv_diagnosis.update_HIV_diagnosis(pop)

    # get stats
    lost = len(pop.get_sub_pop([(col.UNDER_CARE, op.eq, False)]))
    mean = N * pop.hiv_diagnosis.prob_loss_at_diag_non_tb_who3
    stdev = sqrt(mean * (1 - pop.hiv_diagnosis.prob_loss_at_diag_non_tb_who3))
    # check tested value is within 3 standard deviations
    assert mean - 3 * stdev <= lost <= mean + 3 * stdev
