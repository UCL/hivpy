from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import IntEnum

import pandas as pd

import hivpy.column_names as col

from .common import AND, COND, OR, SexType, date, rng, timedelta
from .prep_data import PrEPData


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):

        # init prep data
        with importlib.resources.path("hivpy.data", "prep.yaml") as data_path:
            self.p_data = PrEPData(data_path)

        self.prep_strategy = self.p_data.prep_strategy.sample()
        self.date_prep_intro = [date(self.p_data.date_prep_oral_intro),
                                date(self.p_data.date_prep_cab_intro),
                                date(self.p_data.date_prep_len_intro),
                                date(self.p_data.date_prep_vr_intro)]
        self.prob_risk_informed_prep = self.p_data.prob_risk_informed_prep
        self.prob_greater_risk_informed_prep = self.p_data.prob_greater_risk_informed_prep
        self.prob_suspect_risk_prep = self.p_data.prob_suspect_risk_prep

        self.prep_oral_pref_beta = rng.choice([1.1, 1.3, 1.5])
        self.prep_cab_pref_beta = self.prep_oral_pref_beta + 0.3
        self.prep_len_pref_beta = self.prep_cab_pref_beta
        self.prep_vr_pref_beta = self.prep_oral_pref_beta - 0.1
        self.vl_prevalence_affects_prep = rng.choice([True, False], p=[1/3, 2/3])
        self.vl_prevalence_prep_threshold = rng.choice([0.005, 0.01])

        self.rate_test_onprep_any = self.p_data.rate_test_onprep_any
        self.prep_willing_threshold = self.p_data.prep_willing_threshold
        self.prob_test_prep_start = self.p_data.prob_test_prep_start.sample()
        self.prob_prep_restart = self.p_data.prob_prep_restart.sample()

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.PREP_ORAL_PREF, 0)
        pop.init_variable(col.PREP_CAB_PREF, 0)
        pop.init_variable(col.PREP_LEN_PREF, 0)
        pop.init_variable(col.PREP_VR_PREF, 0)
        pop.init_variable(col.PREP_ORAL_WILLING, False)
        pop.init_variable(col.PREP_CAB_WILLING, False)
        pop.init_variable(col.PREP_LEN_WILLING, False)
        pop.init_variable(col.PREP_VR_WILLING, False)
        pop.init_variable(col.PREP_ANY_WILLING, False)
        pop.init_variable(col.R_PREP, 1.0)
        pop.init_variable(col.PREP_ELIGIBLE, False)
        pop.init_variable(col.PREP_TYPE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)
        pop.init_variable(col.LTP_HIV_STATUS, False)
        pop.init_variable(col.LTP_HIV_DIAGNOSED, False)
        pop.init_variable(col.LTP_ON_ART, False)

    def reroll_r_prep(self, pop: Population):
        """
        Reroll the r_prep value for each individual that was ineligible for PrEP last time step.
        """
        ineligible_pop = pop.get_sub_pop(COND(col.PREP_ELIGIBLE, op.eq, False))
        pop.set_present_variable(col.R_PREP, rng.uniform(size=len(ineligible_pop)), sub_pop=ineligible_pop)

    def get_at_risk_pop(self, pop: Population):
        """
        Return the sub-population that either has one or more short-term partners or
        has a diagnosed long-term partner who is not on ART.
        """
        return pop.get_sub_pop(OR(COND(col.NUM_PARTNERS, op.ge, 1),
                                  AND(COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                      COND(col.LTP_ON_ART, op.eq, False))))

    def get_risk_informed_pop(self, pop: Population, prob_risk_informed_prep):
        """
        Return the sub-population that has a long-term partner who is not on ART
        and pass the probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_HIV_STATUS, op.eq, False),
                                   COND(col.R_PREP, op.lt, prob_risk_informed_prep)))

    def get_suspect_risk_pop(self, pop: Population):
        """
        Return the sub-population that has a long-term partner who is not on ART but is infected
        and pass the higher probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_HIV_STATUS, op.eq, True),
                                   COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep)))

    def set_prep_preference(self, pop: Population, date_intro, pref_beta, pref_col, willing_col, sub_pop_mod=None):
        """
        Set preference values for a specific type of PrEP and determine willingness.
        """
        if pop.date >= date_intro:
            # find those who turned 15 this time step
            sub_pop = pop.get_sub_pop([(col.AGE, op.eq, 15)])
            # unless the current date is the introduction date
            if pop.date == date_intro:
                # then find all over 15s
                sub_pop = pop.get_sub_pop([(col.AGE, op.ge, 15)])
            # find intersection if further modifications should be made to the sub-pop
            if sub_pop_mod is not None:
                sub_pop = pop.get_sub_pop_intersection(sub_pop, sub_pop_mod)

            # random preference beta distribution
            pref = rng.beta(pref_beta, 5, size=len(sub_pop))
            pop.set_present_variable(pref_col, pref, sub_pop)
            # determine willingness by comparing to threshold
            willingness = pref > self.prep_willing_threshold
            pop.set_present_variable(willing_col, willingness, sub_pop)
            pop.set_present_variable(col.PREP_ANY_WILLING, True, pop.apply_bool_mask(willingness, sub_pop))

    def prep_willingness(self, pop: Population):
        """
        Determine which individuals are willing to take PrEP, as well as their PrEP preferences.
        """
        # oral prep pref + willingness
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Oral], self.prep_oral_pref_beta,
                                 col.PREP_ORAL_PREF, col.PREP_ORAL_WILLING)
        # injectable prep pref + willingness
        # FIXME: should Cab be controlled by an availability flag instead of introduction date?
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Cabotegravir], self.prep_cab_pref_beta,
                                 col.PREP_CAB_PREF, col.PREP_CAB_WILLING)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Lenacapavir], self.prep_len_pref_beta,
                                 col.PREP_LEN_PREF, col.PREP_LEN_WILLING)
        # vr prep pref + willingness (women only)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.VaginalRing], self.prep_vr_pref_beta,
                                 col.PREP_VR_PREF, col.PREP_VR_WILLING,
                                 sub_pop_mod=pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))

        # FIXME: do we need to keep track of everyone's highest PrEP preference here?
        # having the actual ranking may be more useful depending on availability

        gen_pop = len(pop.get_sub_pop([(col.AGE, op.ge, 15), (col.AGE, op.lt, 50)]))
        # find prevalence of people with a viral load of over 1000
        vl_prevalence = (len(pop.get_sub_pop([(col.VIRAL_LOAD, op.gt, 1000),
                                              (col.AGE, op.ge, 15),
                                              (col.AGE, op.lt, 50)])) / gen_pop
                         if gen_pop > 0 else 0)

        # there's a chance nobody is willing to take PrEP if unsuppressed viral load prevalence is too low
        if self.vl_prevalence_affects_prep and vl_prevalence < self.vl_prevalence_prep_threshold:
            pop.set_present_variable(col.PREP_ORAL_WILLING, False)
            pop.set_present_variable(col.PREP_CAB_WILLING, False)
            pop.set_present_variable(col.PREP_LEN_WILLING, False)
            pop.set_present_variable(col.PREP_VR_WILLING, False)
            pop.set_present_variable(col.PREP_ANY_WILLING, False)

    def prep_eligibility(self, pop: Population):
        """
        Mark people who are eligible for PrEP this time step.
        """
        # start when first type of prep is introduced
        if pop.date >= min(self.date_prep_intro):

            prob_risk_informed_prep = (self.prob_greater_risk_informed_prep
                                       if (8 <= self.prep_strategy <= 11 or self.prep_strategy == 14)
                                       else self.prob_risk_informed_prep)

            # nobody is eligible by default
            prep_eligible_pop = pd.Index([], dtype="int64")
            # reroll r_prep for those that were ineligible last time step
            self.reroll_r_prep(pop)
            # reset old prep eligibility
            pop.set_present_variable(col.PREP_ELIGIBLE, False)

            # female sex workers + adolescent girls and young women
            if self.prep_strategy == 1:
                fsw_agyw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                   COND(col.SEX, op.eq, SexType.Female),
                                                   OR(COND(col.SEX_WORKER, op.eq, True),
                                                      AND(COND(col.AGE, op.ge, 15),
                                                          COND(col.AGE, op.lt, 25)))))
                # fsw_agyw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    fsw_agyw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # female sex workers
            elif self.prep_strategy == 2:
                fsw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              COND(col.SEX_WORKER, op.eq, True)))
                # fsw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    fsw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # adolescent girls and young women
            elif self.prep_strategy == 3:
                agyw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                               COND(col.SEX, op.eq, SexType.Female),
                                               COND(col.AGE, op.ge, 15),
                                               COND(col.AGE, op.lt, 25)))
                # agyw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    agyw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # general at risk population and informed women
            elif self.prep_strategy == 4 or self.prep_strategy == 8:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    pop.get_sub_pop_intersection(
                        pop.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, False)), self.get_at_risk_pop(pop)),
                    pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # general recently active and informed population
            elif self.prep_strategy == 5 or self.prep_strategy == 9:
                gen_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.AGE, op.ge, 15),
                                              COND(col.AGE, op.lt, 50)))
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9)))
                # gen AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_pop, pop.get_sub_pop_union(
                        active_stp_pop,
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # at risk and informed women
            elif self.prep_strategy == 6 or self.prep_strategy == 10:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female)))
                gen_age_pop = pop.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                # gen_fem AND (at_risk OR (gen_age AND (risk_informed OR suspect_risk)))
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_fem_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop), pop.get_sub_pop_intersection(
                            gen_age_pop, pop.get_sub_pop_union(
                                self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                                self.get_suspect_risk_pop(pop)))))
            # recently active and informed women
            elif self.prep_strategy == 7 or self.prep_strategy == 11:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9)))
                # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_fem_pop, pop.get_sub_pop_union(
                        active_stp_pop,
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # general recently active population
            elif self.prep_strategy == 12:
                gen_pop = pop.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, False))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9))))
                # gen AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_pop, active_pop)
            # recently active women
            elif self.prep_strategy == 13:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female)))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9))))
                # gen_fem AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_fem_pop, active_pop)
            # general active at risk population and informed women
            elif self.prep_strategy == 14:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                active_at_risk_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                         OR(COND(col.LAST_STP_DATE, op.gt,
                                                                 pop.date - timedelta(months=6)),
                                                            AND(COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                                                COND(col.LTP_ON_ART, op.eq, False)))))
                # active_at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    active_at_risk_pop, pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # serodiscordant couples
            elif self.prep_strategy == 15:
                gen_ltp_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                  COND(col.LTP_HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50),
                                                  OR(COND(col.R_PREP, op.lt, 0.01),  # (alt) risk informed
                                                     AND(COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep),
                                                         COND(col.LTP_HIV_STATUS, op.eq, True)))))  # (alt) suspect risk
                at_risk_ltp_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                      COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                                      COND(col.LTP_ON_ART, op.eq, False)))
                # at_risk_ltp OR gen_ltp
                prep_eligible_pop = pop.get_sub_pop_union(at_risk_ltp_pop, gen_ltp_pop)
            # pregnant and lactating/breastfeeding women
            elif self.prep_strategy == 16:
                plw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              OR(COND(col.PREGNANT, op.eq, True),
                                                 COND(col.BREASTFEEDING, op.eq, True))))
                # plw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    plw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))

            if len(prep_eligible_pop) > 0:
                pop.set_present_variable(col.PREP_ELIGIBLE, True, prep_eligible_pop)
