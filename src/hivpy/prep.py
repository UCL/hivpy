from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op
from enum import IntEnum

import pandas as pd

import hivpy.column_names as col

from .common import AND, COND, OR, SexType, date, rng, timedelta


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):
        # FIXME: move these to data file
        self.prep_strategy = rng.choice([4, 8, 14])
        self.date_prep_intro = [date(2018, 3), date(2027), date(3000), date(2100)]
        self.prob_risk_informed_prep = 0.05
        self.prob_greater_risk_informed_prep = 0.1
        self.prob_suspect_risk_prep = 0.5

        self.rate_test_onprep_any = 1
        self.prep_willing_threshold = 0.2
        self.prob_test_prep_start = rng.choice([0.25, 0.50, 0.75])
        self.prob_prep_restart = rng.choice([0.05, 0.10, 0.20])

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.PREP_ELIGIBLE, False)
        pop.init_variable(col.PREP_TYPE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)
        pop.init_variable(col.LTP_HIV_STATUS, False)
        pop.init_variable(col.LTP_HIV_DIAGNOSED, False)
        pop.init_variable(col.LTP_ON_ART, False)

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
        risk_informed_pop = pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LTP_ON_ART, op.eq, False),
                                                COND(col.LTP_HIV_STATUS, op.eq, False)))
        r = rng.uniform(size=len(risk_informed_pop))
        rip_mask = r < prob_risk_informed_prep
        return pop.apply_bool_mask(rip_mask, risk_informed_pop)

    def get_suspect_risk_pop(self, pop: Population):
        """
        Return the sub-population that has a long-term partner who is not on ART but is infected
        and pass the higher probability to fulfill the criteria for risk-informed PrEP.
        """
        suspect_risk_pop = pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                               COND(col.LTP_ON_ART, op.eq, False),
                                               COND(col.LTP_HIV_STATUS, op.eq, True)))
        r = rng.uniform(size=len(suspect_risk_pop))
        srp_mask = r < self.prob_suspect_risk_prep
        return pop.apply_bool_mask(srp_mask, suspect_risk_pop)

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
                        self.get_at_risk_pop(pop), pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # female sex workers
            elif self.prep_strategy == 2:
                fsw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              COND(col.SEX_WORKER, op.eq, True)))
                # fsw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    fsw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop), pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # adolescent girls and young women
            elif self.prep_strategy == 3:
                agyw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                               COND(col.SEX, op.eq, SexType.Female),
                                               COND(col.AGE, op.ge, 15),
                                               COND(col.AGE, op.lt, 25)))
                # agyw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    agyw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop), pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # general at risk population and informed women
            elif self.prep_strategy == 4 or self.prep_strategy == 8:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    self.get_at_risk_pop(pop), pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # general recently active and informed population
            elif self.prep_strategy == 5 or self.prep_strategy == 9:
                gen_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.AGE, op.ge, 15),
                                              COND(col.AGE, op.lt, 50)))
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=9)))
                # gen AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_pop, pop.get_sub_pop_union(
                        active_stp_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
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
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=9)))
                # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_fem_pop, pop.get_sub_pop_union(
                        active_stp_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # general recently active population
            elif self.prep_strategy == 12:
                gen_pop = pop.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, False))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=9))))
                # gen AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_pop, active_pop)
            # recently active women
            elif self.prep_strategy == 13:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female)))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=9))))
                # gen_fem AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_fem_pop, active_pop)
            # general active at risk population and informed women
            elif self.prep_strategy == 14:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                active_at_risk_pop = pop.get_sub_pop(OR(COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=6)),
                                                        AND(COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                                            COND(col.LTP_ON_ART, op.eq, False))))
                # active_at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    active_at_risk_pop, pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # pregnant and lactating/breastfeeding women
            elif self.prep_strategy == 16:
                plw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              OR(COND(col.PREGNANT, op.eq, True),
                                                 COND(col.BREASTFEEDING, op.eq, True))))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.ge, pop.date - timedelta(months=9))))
                # plw AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(plw_pop, active_pop)

            if len(prep_eligible_pop) > 0:
                pop.set_present_variable(col.PREP_ELIGIBLE, True, prep_eligible_pop)
