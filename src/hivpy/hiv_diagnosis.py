from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import IntEnum

import hivpy.column_names as col

from .common import rng, timedelta
from .hiv_diagnosis_data import HIVDiagnosisData
from .prep import PrEPType


# Ab (default), NA (RNA VL / PCR), Ag/Ab
class HIVTestType(IntEnum):
    Ab = 0  # antibody only assay
    NA = 1  # nucleic acid based
    AgAb = 2  # antigen/antibody combined assay


class HIVDiagnosisModule:

    def __init__(self, **kwargs):

        # init hiv diagnosis data
        with importlib.resources.path("hivpy.data", "hiv_diagnosis.yaml") as data_path:
            self.hd_data = HIVDiagnosisData(data_path)

        # FIXME: consider how to move more variables to data file
        self.hiv_test_type = HIVTestType.Ab
        self.init_prep_inj_na = rng.choice([True, False])
        # FIXME: should this variable affect the value of hiv_test_type or is it completely separate?
        self.prep_inj_na = rng.choice([True, False]) if self.init_prep_inj_na else False

        self.test_sens_general = self.hd_data.test_sens_general
        self.test_sens_primary_ab = self.hd_data.test_sens_primary_ab.sample()
        self.test_sens_prep_inj_primary_ab = self.hd_data.test_sens_prep_inj_primary_ab.sample()
        self.test_sens_prep_inj_3m_ab = self.hd_data.test_sens_prep_inj_3m_ab.sample()
        self.test_sens_prep_inj_ge6m_ab = self.hd_data.test_sens_prep_inj_ge6m_ab.sample()
        self.tests_sens_prep_inj = self.hd_data.tests_sens_prep_inj.sample()
        # test_sens_prep_inj used as index for further sensitivity selection
        self.test_sens_prep_inj_primary_na = self.hd_data.test_sens_prep_inj_primary_na[self.tests_sens_prep_inj]
        self.test_sens_prep_inj_3m_na = self.hd_data.test_sens_prep_inj_3m_na[self.tests_sens_prep_inj]
        self.test_sens_prep_inj_ge6m_na = self.hd_data.test_sens_prep_inj_ge6m_na[self.tests_sens_prep_inj]

        self.prob_loss_at_diag = self.hd_data.prob_loss_at_diag.sample()
        # FIXME: may be 2 or 3 if sw_art_disadv=1
        self.sw_incr_prob_loss_at_diag = 1
        self.higher_newp_less_engagement = rng.choice([True, False], p=[0.2, 0.8])
        self.prob_loss_at_diag_adc_tb = rng.beta(5, 95)
        self.prob_loss_at_diag_non_tb_who3 = rng.beta(15, 85)

    def update_HIV_diagnosis(self, pop: Population):
        """
        Diagnose people that have been tested this time step. The default test type used
        is Ab, but certain policy options make use of NA (RNA VL / PCR) or Ag/Ab tests.
        Accuracy depends on test sensitivity, PrEP usage, as well as CD4 count.
        """
        # tested population in primary infection
        primary_pop = pop.get_sub_pop([(col.IN_PRIMARY_INFECTION, op.eq, True),
                                       (col.LAST_TEST_DATE, op.eq, pop.date),
                                       (col.HIV_DIAGNOSED, op.eq, False)])

        if len(primary_pop) > 0:
            # primary infection diagnosis outcomes
            diagnosed = pop.transform_group([col.PREP_TYPE, col.PREP_JUST_STARTED],
                                            self.calc_primary_diag_outcomes, sub_pop=primary_pop)
            # set outcomes
            pop.set_present_variable(col.HIV_DIAGNOSED, diagnosed, primary_pop)
            pop.set_present_variable(col.HIV_DIAGNOSIS_DATE, pop.date,
                                     sub_pop=pop.apply_bool_mask(diagnosed, primary_pop))

            # some people lost at diagnosis
            lost = pop.transform_group([col.SEX_WORKER], self.calc_primary_loss_at_diag,
                                       sub_pop=pop.apply_bool_mask(diagnosed, primary_pop))
            pop.set_present_variable(col.UNDER_CARE, ~lost, sub_pop=pop.apply_bool_mask(diagnosed, primary_pop))

        # remaining tested general population
        general_pop = pop.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                       (col.IN_PRIMARY_INFECTION, op.eq, False),
                                       (col.LAST_TEST_DATE, op.eq, pop.date),
                                       (col.HIV_DIAGNOSED, op.eq, False)])

        if len(general_pop) > 0:
            # set infection timeframe bool
            general_infection_dates = pop.get_variable(col.DATE_HIV_INFECTION, general_pop)
            pop.set_present_variable(col.HIV_INFECTION_GE6M,
                                     [pop.date]*len(general_infection_dates) - general_infection_dates >=
                                     timedelta(months=6), general_pop)
            # general diagnosis outcomes
            diagnosed = pop.transform_group([col.PREP_TYPE, col.HIV_INFECTION_GE6M],
                                            self.calc_general_diag_outcomes, sub_pop=general_pop)
            # set outcomes
            pop.set_present_variable(col.HIV_DIAGNOSED, diagnosed, general_pop)
            pop.set_present_variable(col.HIV_DIAGNOSIS_DATE, pop.date,
                                     sub_pop=pop.apply_bool_mask(diagnosed, general_pop))

            # FIXME: should also include onart_tm1 and may need to be affected by date_most_recent_tb
            # some people lost at diagnosis
            lost = pop.transform_group([col.SEX_WORKER, col.NUM_PARTNERS, col.ADC, col.TB, col.NON_TB_WHO3],
                                       self.calc_general_loss_at_diag,
                                       sub_pop=pop.apply_bool_mask(diagnosed, general_pop))
            pop.set_present_variable(col.UNDER_CARE, ~lost, sub_pop=pop.apply_bool_mask(diagnosed, general_pop))

    def calc_prob_primary_diag(self, prep_type, prep_just_started):
        """
        Calculates the probability of an individual in primary infection getting
        diagnosed with HIV based on test sensitivity and injectable PrEP usage.
        """
        eff_test_sens_primary = 0
        prep_inj = prep_type == PrEPType.Cabotegravir or prep_type == PrEPType.Lenacapavir
        # default Ab test type
        if self.hiv_test_type == HIVTestType.Ab:
            # injectable PrEP started before this time step
            if prep_inj and not prep_just_started:
                eff_test_sens_primary = self.test_sens_prep_inj_primary_ab
            else:
                eff_test_sens_primary = self.test_sens_primary_ab
        # NA test type
        elif self.hiv_test_type == HIVTestType.NA:
            # injectable PrEP started before this time step
            if prep_inj and not prep_just_started:
                eff_test_sens_primary = self.test_sens_prep_inj_primary_na
            else:
                eff_test_sens_primary = 0.86
        # Ag/Ab test type
        elif self.hiv_test_type == HIVTestType.AgAb:
            # injectable PrEP started before this time step
            if prep_inj and not prep_just_started:
                eff_test_sens_primary = 0
            else:
                eff_test_sens_primary = 0.75

        return eff_test_sens_primary

    def calc_primary_diag_outcomes(self, prep_type, prep_just_started, size):
        """
        Uses HIV test sensitivity and injectable PrEP usage to return
        primary infection diagnosis outcomes.
        """
        prob_diag = self.calc_prob_primary_diag(prep_type, prep_just_started)
        # outcomes
        r = rng.uniform(size=size)
        diagnosed = r < prob_diag

        return diagnosed

    def calc_prob_general_diag(self, prep_type, hiv_infection_ge6m):
        """
        Calculates the probability of an individual not in primary infection getting diagnosed
        with HIV based on test sensitivity, injectable PrEP usage, and infection duration.
        """
        eff_test_sens_general = self.test_sens_general
        # FIXME: does injectable use timing matter for general diagnosis?
        # injectable PrEP in current use
        if prep_type == PrEPType.Cabotegravir or prep_type == PrEPType.Lenacapavir:
            if self.prep_inj_na:
                # infected for 6 months or more
                if hiv_infection_ge6m:
                    eff_test_sens_general = self.test_sens_prep_inj_ge6m_na
                else:
                    eff_test_sens_general = self.test_sens_prep_inj_3m_na
            else:
                # infected for 6 months or more
                if hiv_infection_ge6m:
                    eff_test_sens_general = self.test_sens_prep_inj_ge6m_ab
                else:
                    eff_test_sens_general = self.test_sens_prep_inj_3m_ab

        return eff_test_sens_general

    def calc_general_diag_outcomes(self, prep_type, hiv_infection_ge6m, size):
        """
        Uses HIV test sensitivity, injectable PrEP usage, and infection duration
        to return general diagnosis outcomes.
        """
        prob_diag = self.calc_prob_general_diag(prep_type, hiv_infection_ge6m)
        # outcomes
        r = rng.uniform(size=size)
        diagnosed = r < prob_diag

        return diagnosed

    def calc_prob_loss_at_diag(self, sex_worker):
        """
        Calculates the generic probability of an individual diagnosed with HIV
        exiting care after diagnosis based on sex worker status.
        """
        # FIXME: may need to be affected by lower future ART coverage and/or decr_prob_loss_at_diag_year_i
        eff_prob_loss_at_diag = self.prob_loss_at_diag
        if sex_worker:
            # FIXME: use eff_sw_incr_prob_loss_at_diag after introducing ART and SW programs
            eff_prob_loss_at_diag = min(1, eff_prob_loss_at_diag * self.sw_incr_prob_loss_at_diag)

        return eff_prob_loss_at_diag

    def calc_primary_loss_at_diag(self, sex_worker, size):
        """
        Uses sex worker status in individuals in primary infection after a
        positive HIV diagnosis to return loss of care outcomes.
        """
        prob_loss = self.calc_prob_loss_at_diag(sex_worker)
        # outcomes
        r = rng.uniform(size=size)
        lost = r < prob_loss

        return lost

    def calc_general_loss_at_diag(self, sex_worker, num_stp, adc, tb, non_tb_who3, size):
        """
        Uses sex worker, ADC, TB, and non-TB WHO3 status and number of short term partners in
        individuals not in primary infection after a positive HIV diagnosis
        to return loss of care outcomes.
        """
        # outcomes
        r = rng.uniform(size=size)
        # ADC, non-TB WHO3, and TB not present
        if not adc and not non_tb_who3 and not tb:
            generic_prob_loss = self.calc_prob_loss_at_diag(sex_worker)
            # people with more partners less likely to be engaged with care
            if self.higher_newp_less_engagement and num_stp > 1:
                generic_prob_loss *= 1.5
            lost = r < generic_prob_loss
        # ADC or TB present
        elif adc or tb:
            lost = r < self.prob_loss_at_diag_adc_tb
        # non-TB WHO3 present
        elif non_tb_who3:
            lost = r < self.prob_loss_at_diag_non_tb_who3

        return lost
